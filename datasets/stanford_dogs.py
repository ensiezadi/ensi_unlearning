import os
import numpy as np
import json
import random
from tqdm import tqdm
import torch
import pickle
import argparse
from collections import OrderedDict
from colorama import Fore, Style, init
from torch.optim.lr_scheduler import CosineAnnealingLR

init(autoreset=True)

# --- Project-specific Imports ---
import datasets.stanford_cars
import datasets.stanford_dogs
import datasets.caltech101
import datasets.oxford_flowers
import datasets.oxford_pets
import datasets.food101
import datasets.pinsfaces
import datasets.plt_net_mini

from dassl.data.datasets.build import build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from dassl.config import get_cfg_default

# Note: The adapter name suggests it's already fixed, ensure it has the latest logic
from bioclip_adapter_fixed import bioclip_tokenize, create_bioclip_model, BioCLIPAdapter
from utils_bioclip import get_model, get_configs, load_results, eval_all_ds
import utils_bioclip as utils
from utils_lora import Linear
from gen_classes import *
from forget_cls import *

# --- Global Configurations ---
torch.set_num_threads(10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
IGNORE_OTHER_DS = False
PRINT_EVERY = 200
EPOCHS = 2000
REDUCTION_THR = 0.17
UNLEARN_TRIALS = 100

CUSTOM_TEMPLATES = {
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "StanfordCars": "a photo of a {}, a type of car.",
    "Caltech101": "a photo of a {}, an object.",
    "StanfordDogs": "a photo of a {}, a breed of dog.",
    "PLTNetMini": "a photo of {}, a type of plant.",
}

# --- Helper Functions ---
def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def initialize_config(args):
    cfg = get_cfg_default()
    if os.path.exists(args.config_file):
        cfg.merge_from_file(args.config_file)
    else:
        print(f"Warning: Config file not found at {args.config_file}. Using default settings.")
    
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.SEED = args.seed
    cfg.DATASET.ROOT = args.dataset_root
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATASET.NUM_SHOTS = -1
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 4
    cfg.DATALOADER.TEST.BATCH_SIZE = 128
    return cfg

# --- MODIFICATION 1: Pass the loaded model to the function ---
def load_test_datasets(cfg, model):
    test_datasets, test_dataloaders, datasets_cls = {}, {}, {}
    for ds in all_ds:
        cfg.DATASET.NAME = ds

        # --- MODIFICATION 2: Dynamically select the correct transform ---
        # If we are using BioCLIP (wrapped in our adapter), use its specific preprocess function.
        # This ensures the correct normalization stats are used.
        if isinstance(model, BioCLIPAdapter):
            print(Fore.CYAN + f"Using BioCLIP's specific preprocessing for {ds}...")
            tfm_test = model.preprocess
        else:
            # Otherwise (e.g., fallback to standard CLIP), build the transform from config.
            print(Fore.YELLOW + f"Building transform for {ds} from config (standard CLIP)...")
            tfm_test = build_transform(cfg, is_train=False)

        dataset = build_dataset(cfg)
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )
        test_datasets[ds] = dataset
        test_dataloaders[ds] = test_loader
        datasets_cls[ds] = dataset
    return test_datasets, test_dataloaders, datasets_cls

hooks = {}

def get_activation(name):
    def hook(model, input, output):
        global hooks
        hooks[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
    return hook

@torch.no_grad()
def precompute_projections(model, classes, template=['a photo of {}']):
    global hooks
    projections_list = []
    hooks_list = []
    model_device = next(model.parameters()).device

    for classname in classes:
        try:
            hooks.clear()
            classname_clean = classname.replace('_', ' ')
            texts = [t.format(classname_clean) for t in template]
            tokenized_texts = bioclip_tokenize(texts).to(model_device)
            class_embeddings = model.encode_text(tokenized_texts)
            projections_list.append(class_embeddings)

            hook_key = 'text_projection' if 'text_projection' in hooks else 'ln_final'
            if hook_key in hooks:
                hook_data = hooks[hook_key]
                eot_indices = tokenized_texts.argmax(dim=-1)
                eot_features = hook_data[range(hook_data.shape[0]), eot_indices]
                avg_eot_feature = eot_features.mean(dim=0)
                hooks_list.append(avg_eot_feature.clone())
            else:
                print(Fore.YELLOW + f"警告: 未找到预期的钩子 '{hook_key}'。使用最终输出的平均值作为替代。")
                hooks_list.append(class_embeddings.mean(dim=0).clone())
        except Exception as e:
            print(Fore.YELLOW + f"警告: 无法为类别 '{classname}' 预计算投影。错误: {e}。已跳过。")
            continue

    if not projections_list:
        print(Fore.RED + "错误: 未能创建任何有效的投影。")
        return None, None

    projections = torch.stack(projections_list, dim=0)
    list_hooks = torch.stack(hooks_list, dim=0)
    return projections, list_hooks

@torch.no_grad()
def register_model_hooks(model):
    for name, module in model.named_modules():
        if 'ln_final' in name or 'text_projection' in name:
            module.register_forward_hook(get_activation(name))
            print(f"Registered hook for: {name}")

def get_preserved_classes(main_ds, forget_label, args, class_lists):
    classes_preserved_list = class_lists.get(main_ds, [])
    forget_set = set(forget_label.split('|')) if args.multiclass_forget else {forget_label}
    preserved = [cl for cl in classes_preserved_list if cl.lower() not in {f.lower() for f in forget_set}]
    return preserved

@torch.no_grad()
def compute_proj_into(model, num_classes_to_forget, device):
    empty_text = bioclip_tokenize("").to(device)
    embed = model.encode_text(empty_text).repeat(num_classes_to_forget, 1)
    proj_into = embed / embed.norm(dim=-1, keepdim=True)
    return proj_into

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/result0", help="Output directory")
    parser.add_argument("--dataset_root", type=str, default="E:\\Others\\DATASETS", help="Root directory for all datasets")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--run_ds", type=str, default="PLTNetMini", help="Comma-separated list of datasets to run unlearning on")
    parser.add_argument("--backbone_arch", type=str, default="ViT-B/16", help="CLIP backbone architecture")
    parser.add_argument("--config_file", type=str, default="configs/trainers/adam_lr2e-4_B256_ep200_ViT16.yaml", help="Path to dassl config file")
    parser.add_argument("--multiclass_forget", action='store_true', help="Enable multiclass forgetting")
    
    args = parser.parse_args()
    print("Arguments:", args)
    
    set_seeds(args.seed)
    cfg = initialize_config(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    configs = get_configs(args)
    
    utils.CUSTOM_TEMPLATES = CUSTOM_TEMPLATES
    all_logs = {}
    
    run_ds = all_ds[:] if not args.run_ds else [item.strip() for item in args.run_ds.split(',')]
    assert all(ds in all_ds for ds in run_ds), "One or more specified run_ds are not valid."
    
    output_base = args.output_dir
    
    print("Loading base model...")
    model = get_model(device=device, arch=args.backbone_arch)
    
    # --- MODIFICATION 3: Load datasets AFTER the model is loaded ---
    test_datasets, test_dataloaders, datasets_cls = load_test_datasets(cfg, model)
    
    original_model_state = model.state_dict()
    register_model_hooks(model)

    cache_filename = f"projections_{args.backbone_arch.replace('/', '-')}.pt"
    projection_cache_file = os.path.join(args.output_dir, cache_filename)

    if os.path.exists(projection_cache_file):
        print(Fore.GREEN + f"Loading precomputed projections from {projection_cache_file}...")
        cached_data = torch.load(projection_cache_file)
        all_projections = cached_data['projections']
        all_hooks = cached_data['hooks']
    else:
        print("Precomputing projections for all classes (will be cached)...")
        all_projections = {}
        all_hooks = {}
        for ds in tqdm(all_ds, desc="Precomputing Projections"):
            template = [CUSTOM_TEMPLATES.get(ds, 'a photo of a {}.')]
            projections, list_hooks = precompute_projections(model, datasets_cls[ds].classnames, template=template)
            if projections is not None:
                all_projections[ds] = projections
                all_hooks[ds] = list_hooks
        print(Fore.GREEN + f"Saving precomputed projections to {projection_cache_file}...")
        torch.save({'projections': all_projections, 'hooks': all_hooks}, projection_cache_file)

    for main_ds in run_ds:
        print(f"\n{'='*30}\nStarting unlearning for dataset: {main_ds}\n{'='*30}")
        kwargs = configs[main_ds]
        all_logs[main_ds] = {'settings': {'kwargs': kwargs}}
        
        forget_classes_list = forget_classes_all[main_ds]
        if not isinstance(forget_classes_list, list):
            forget_classes_list = [forget_classes_list]
        
        for forget_label in forget_classes_list:
            print(f"\n--- Forgetting class: {forget_label} ---\n")
            model.load_state_dict(original_model_state)
            
            preserve_hooks_list = []
            for ds_name, ds_hooks in all_hooks.items():
                if ds_name == main_ds:
                    preserved_class_names = get_preserved_classes(main_ds, forget_label, args, {'PLTNetMini': pltnetmini_list, 'StanfordDogs': stanforddogs_list, 'StanfordCars': stanfordcars_list, 'Caltech101': caltech_list, 'OxfordFlowers': oxfordflowers_list})
                    preserve_hooks_list.append(ds_hooks[[datasets_cls[ds_name].classnames.index(c) for c in preserved_class_names]])
                else:
                    preserve_hooks_list.append(ds_hooks)
            
            preserve_hooks = torch.cat(preserve_hooks_list, dim=0)

            forget_class_idx = datasets_cls[main_ds].classnames.index(forget_label)
            change_hooks = all_hooks[main_ds][forget_class_idx].unsqueeze(0)
            
            preserve_hooks = preserve_hooks.float()
            change_hooks = change_hooks.float()

            in_proj, out_proj = model.text_projection.shape
            proj_into = compute_proj_into(model, 1, device)

            best_weights = None
            
            with torch.no_grad():
                initial_proj = change_hooks @ model.text_projection.float()
                initial_forget_loss_for_trials = torch.norm(proj_into - initial_proj, p=2).item()

            for trial in range(UNLEARN_TRIALS):
                print(f"\n--- Starting Trial {trial+1}/{UNLEARN_TRIALS} ---")
                
                new_text_proj = Linear(in_proj, out_proj, r=kwargs['lora_r'], bias=False, device=device)
                new_text_proj.weight = torch.nn.Parameter(model.text_projection.T.clone())
                new_text_proj.weight.requires_grad = False
                
                optimizer = torch.optim.Adam(new_text_proj.parameters(), lr=1e-4)
                scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

                best_candidate_total_loss = float('inf')
                best_epoch_weights = None

                print(f"Trial {trial+1}: Initial forget loss is {initial_forget_loss_for_trials:.4f}. Current lamb_forget={kwargs['lamb_forget']:.2f}, lamb_preserve={kwargs['lamb_preserve']:.2f}")

                for epoch in range(EPOCHS):
                    new_forget_output = new_text_proj(change_hooks)
                    new_forget_output_2d = new_forget_output[:, -1, :] if new_forget_output.dim() == 3 else new_forget_output

                    delta_w = (new_text_proj.lora_B @ new_text_proj.lora_A)
                    preserve_loss_val = torch.norm(preserve_hooks @ delta_w.T, p=2)
                    
                    forget_loss_val = torch.norm(proj_into - new_forget_output_2d, p=2)
                    weight_loss_val = torch.norm(delta_w, p=2)
                    
                    loss = kwargs['lamb_forget'] * forget_loss_val + \
                           kwargs['lamb_preserve'] * preserve_loss_val + \
                           kwargs['lamb_weight'] * weight_loss_val

                    if forget_loss_val.item() < initial_forget_loss_for_trials:
                        if loss.item() < best_candidate_total_loss:
                            best_candidate_total_loss = loss.item()
                            with torch.no_grad():
                                best_epoch_weights = (new_text_proj.weight + delta_w).clone()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if (epoch + 1) % 100 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        print(f"   Epoch {epoch+1:>{len(str(EPOCHS))}} | Total Loss: {loss.item():.4f} | Raw Forget: {forget_loss_val.item():.4f} | Raw Preserve (Ortho): {preserve_loss_val.item():.4f} | LR: {current_lr:.2e}")

                if best_epoch_weights is None:
                    print(f"Trial {trial+1} failed: No valid candidate weights were found (forget loss never improved).")
                    continue

                with torch.no_grad():
                    final_proj = change_hooks @ best_epoch_weights
                    final_forget_loss = torch.norm(proj_into - final_proj).item()
                    reduction = (initial_forget_loss_for_trials - final_forget_loss) / (initial_forget_loss_for_trials + 1e-8)
                    
                    print(f"Trial {trial+1} Summary: Initial Loss={initial_forget_loss_for_trials:.4f}, Best Final Loss={final_forget_loss:.4f}, Reduction={reduction:.2%}")
                    
                    if reduction >= REDUCTION_THR:
                        print(Fore.GREEN + "Unlearning successful, breaking trial loop.")
                        best_weights = best_epoch_weights
                        break
                    else:
                        if trial < UNLEARN_TRIALS - 1:
                            kwargs['lamb_forget'] *= 1.2 
                            print(f"Reduction insufficient. Increasing lamb_forget aggressively to {kwargs['lamb_forget']:.2f} for the next trial.")
                        else:
                            print(Fore.RED + "All trials failed to meet the reduction threshold. No successful unlearning was achieved.")

            if best_weights is not None:
                model.text_projection = torch.nn.Parameter(best_weights.T)
                model.eval()

                results_ds = eval_all_ds(model, datasets_cls, main_ds, forget_label, test_dataloaders,
                                         None, eval_forgetonly=IGNORE_OTHER_DS, debug=True, device=device)

                all_logs[main_ds][forget_label] = {'final_results': results_ds}
                print(f"*** Final results for forgetting '{forget_label}': {results_ds[main_ds][forget_label]} ***")

                model_save_path = os.path.join(output_base, f"model_{main_ds}_{forget_label}.pth")
                print(Fore.GREEN + f"Saving unlearned model to {model_save_path}...")
                torch.save(model.state_dict(), model_save_path)

                with open(os.path.join(output_base, "logs.json"), "w") as f:
                    json.dump(all_logs, f, indent=4)
            else:
                print(f"Error: Could not find suitable weights for forgetting '{forget_label}'.")

    print("\nAll tasks completed.")
    with open(os.path.join(output_base, "logs.json"), "w") as f:
        json.dump(all_logs, f, indent=4)

