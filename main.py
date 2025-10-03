import os
import sys
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

def convert_to_json_serializable(obj):
    """将numpy数组和torch张量转换为JSON可序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        # 尝试转换其他numpy标量类型
        if hasattr(obj, 'item'):  # numpy标量类型
            return obj.item()
        return obj

# 添加Dassl.pytorch到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
dassl_path = os.path.join(current_dir, 'Dassl.pytorch')
if dassl_path not in sys.path:
    sys.path.insert(0, dassl_path)

# --- Imports from utils_bioclip ---
from sklearn.metrics import confusion_matrix
import pandas as pd
from torchvision.transforms import Normalize

init(autoreset=True)

# --- Project-specific Imports ---
from dassl.data.datasets.build import build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from dassl.config import get_cfg_default

from bioclip_adapter_fixed import bioclip_tokenize, create_bioclip_model, BioCLIPAdapter, clip_classifier
from utils_lora import Linear
from gen_classes import *
from forget_cls import *

# 导入自定义数据集
import datasets

# --- Global Configurations ---
torch.set_num_threads(10)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
IGNORE_OTHER_DS = False
PRINT_EVERY = 200
EPOCHS = 2000
REDUCTION_THR = 0.7
UNLEARN_TRIALS = 100

CUSTOM_TEMPLATES = {
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "StanfordCars": "a photo of a {}, a type of car.",
    "Caltech101": "a photo of a {}, an object.",
    "StanfordDogs": "a photo of a {}, a breed of dog.",
    "PLTNetMini": "a photo of a {}, a type of plant.",
    "Bird525": "a photo of a {}, a type of bird.",
}


# =================================================================================
# --- Utility Functions ---
# =================================================================================

def load_results(backbone):
    filename = "results_zs_all_RN50.pkl" if backbone == "RN50" else "results_zs_all_ViT16.pkl"
    pickle_path = os.path.join("zs_results", filename)
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"Warning: Pickle file not found at {pickle_path}. Returning empty dictionary.")
        return {}

def get_configs(args):
    onecls_configs = {
        'RN50':{
            'StanfordCars': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'StanfordDogs': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'Caltech101': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'OxfordFlowers': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'PLTNetMini': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'Bird525': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
        },
        'ViT-B/16': {
            'StanfordCars': {'lamb_preserve': 0.25, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'StanfordDogs': {'lamb_preserve': 0.3, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'Caltech101': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'OxfordFlowers': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'PLTNetMini': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'Bird525': {'lamb_preserve': 0.0, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 0.}
        }
    }
    multiclass_configs = {
        'RN50': {
            'StanfordCars': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'StanfordDogs': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'Caltech101': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'OxfordFlowers': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'PLTNetMini': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'Bird525': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
        },
        'ViT-B/16': {
            'StanfordCars': {'lamb_preserve': 0.35, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'StanfordDogs': {'lamb_preserve': 0.35, 'lamb_forget': 1.0, 'lora_r': 5, 'lamb_weight': 1.},
            'Caltech101': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'OxfordFlowers': {'lamb_preserve': 0.25, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'PLTNetMini': {'lamb_preserve': 0.25, 'lamb_forget': 1.1, 'lora_r': 8, 'lamb_weight': 1.},
            'Bird525': {'lamb_preserve': 0.25, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
        }
    }
    if args.multiclass_forget:
        print(f"SETTING {args.backbone_arch} MULTICLASS")
        configs = multiclass_configs[args.backbone_arch]
    else:
        print(f"SETTING {args.backbone_arch} ONE CLASS")
        configs = onecls_configs[args.backbone_arch]
    return configs

def get_model(arch="ViT-B/16", device='cpu', load_path=""):
    print("Loading model...")
    model = create_bioclip_model(arch=arch, device=device)
    return model

def cls_acc(output, target, topk=1):
    pred = np.argmax(output, axis=1)
    correct = pred == target
    acc = 100 * correct.sum() / target.shape[0]
    return acc

@torch.no_grad()
def calculate_average_class_similarity(model, loader, classnames, template, device):
    """
    Computes the average similarity between images of each class and their corresponding text description.
    """
    model.eval()
    
    all_image_features = []
    all_labels = []
    for batch in tqdm(loader, desc="Calculating similarities"):
        images = batch['img'].to(device)
        labels = batch['label'].to(device)
        
        image_features = model.encode_image(images.to(next(model.parameters()).dtype))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        all_image_features.append(image_features)
        all_labels.append(labels)
        
    all_image_features = torch.cat(all_image_features)
    all_labels = torch.cat(all_labels)
    
    text_features = clip_classifier(classnames, [template], model).to(device)
    
    if text_features.shape[0] != len(classnames):
        print(Fore.YELLOW + "Transposing text features to match expected shape (num_classes, feature_dim)...")
        text_features = text_features.T
        
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    avg_similarities = {}
    for i, classname in enumerate(classnames):
        class_mask = (all_labels == i)
        if class_mask.sum() == 0:
            continue
            
        class_image_features = all_image_features[class_mask]
        
        similarity_scores = class_image_features @ text_features[i]
        
        avg_similarities[classname] = similarity_scores.mean().item()
        
    return avg_similarities

def evaluate_clip_zs(model, loader, clip_weights, device=None, out_conf=False, output_probs=False):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            images, target = batch['img'].to(device), batch['label'].to(device)
            image_features = model.encode_image(images.to(next(model.parameters()).dtype))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())

    labels, features = torch.cat(labels).numpy(), torch.cat(features)
    clip_weights_tensor = clip_weights.detach().cpu()
    if clip_weights_tensor.shape[0] != features.shape[1]:
        clip_weights_tensor = clip_weights_tensor.T
    
    features_float, clip_weights_float = features.float(), clip_weights_tensor.float()
    raw_similarity_tensor = features_float @ clip_weights_float
    scaled_logits_tensor = 100. * raw_similarity_tensor
    clip_logits_test = scaled_logits_tensor.numpy()
    
    acc = cls_acc(clip_logits_test, labels) / 100.

    if out_conf:
        raw_similarity = raw_similarity_tensor.numpy()
        return acc, (labels, clip_logits_test, raw_similarity)
    return acc

def eval_all_ds(model, datasets_cls, forget_ds, forget_lbl, all_loaders, train_loader=None, eval_forgetonly=False, debug=False, device='cpu', ignore_labels_main=[]):
    results = {ds: {} for ds in all_loaders}
    for ds, test_loader in all_loaders.items():
        if ds not in datasets_cls: continue
        model.eval()
        classnames = datasets_cls[ds].classnames
        template = [CUSTOM_TEMPLATES.get(ds, 'a photo of a {}.')]
        clip_weights = clip_classifier(classnames, [template], model).to(device)

        if ds == forget_ds:
            if debug:
                acc, details = evaluate_clip_zs(model, test_loader, clip_weights, device=device, out_conf=True)
                labels, clip_logits_test, raw_similarities = details
                
                if ignore_labels_main:
                    ignore_ids = [classnames.index(c) for c in ignore_labels_main if c in classnames]
                    mask = ~np.isin(labels, ignore_ids)
                else:
                    forget_id = classnames.index(forget_lbl) if forget_lbl in classnames else -1
                    mask = labels != forget_id
                
                cm = confusion_matrix(labels, clip_logits_test.argmax(1))
                if not ignore_labels_main and forget_lbl in classnames:
                    forget_id = classnames.index(forget_lbl)
                    cls_acc_test = cm[forget_id, forget_id] / (cm[forget_id, :].sum() + 1e-8)
                else:
                    cls_acc_test = -1

                # 计算保留类别的准确率 - 需要从混淆矩阵中排除忘记类别
                if ignore_labels_main:
                    ignore_ids = [classnames.index(c) for c in ignore_labels_main if c in classnames]
                    preserved_class_ids = [i for i in range(len(classnames)) if i not in ignore_ids]
                else:
                    forget_id = classnames.index(forget_lbl) if forget_lbl in classnames else -1
                    preserved_class_ids = [i for i in range(len(classnames)) if i != forget_id]
                
                if preserved_class_ids:
                    preserved_cm = cm[np.ix_(preserved_class_ids, preserved_class_ids)]
                    no_cls_acc = np.diag(preserved_cm).sum() / (preserved_cm.sum() + 1e-8)
                else:
                    no_cls_acc = -1

                key = '|'.join(ignore_labels_main) if ignore_labels_main else forget_lbl
                
                results[ds][key] = {
                    'cls_acc_test' : cls_acc_test, 
                    'no_cls_acc' : no_cls_acc,
                    'raw_similarities': raw_similarities
                }
                print(f"{10*'+++'} Main DS: {ds} | Results: {{'cls_acc_test': {cls_acc_test:.4f}, 'no_cls_acc': {no_cls_acc:.4f}}} {10*'+++'}")
        
        elif not eval_forgetonly:
            acc = evaluate_clip_zs(model, test_loader, clip_weights, device=device)
            results[ds]['all'] = {'all_ds' : acc}
            print(f"{10*'+++'} Other DS: {ds} | Accuracy: {acc:.4f} {10*'+++'}")

    return results

# =================================================================================
# --- Main Script Helper Functions ---
# =================================================================================

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

def load_test_datasets(cfg, model):
    import datasets.stanford_cars
    import datasets.stanford_dogs
    import datasets.caltech101
    import datasets.oxford_flowers
    import datasets.oxford_pets
    import datasets.food101
    # import datasets.pinsfaces
    import datasets.plt_net_mini
    import datasets.bird525

    test_datasets, test_dataloaders, datasets_cls = {}, {}, {}
    for ds in all_ds:
        cfg.DATASET.NAME = ds

        if isinstance(model, BioCLIPAdapter):
            print(Fore.CYAN + f"Using BioCLIP's specific preprocessing for {ds}...")
            tfm_test = model.preprocess
        
        try:
            dataset = build_dataset(cfg)
        except Exception as e:
            print(Fore.RED + f"Error building dataset '{ds}': {e}. Skipping.")
            continue

        test_loader = build_data_loader(
            cfg, sampler_type='SequentialSampler', data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, tfm=tfm_test, is_train=False,
            dataset_wrapper=None
        )
        test_datasets[ds] = dataset
        test_dataloaders[ds] = test_loader
        datasets_cls[ds] = dataset
    return test_datasets, test_dataloaders, datasets_cls

def get_preserved_classes(main_ds, forget_label, args, class_lists):
    classes_preserved_list = class_lists.get(main_ds, [])
    forget_set = set(forget_label.split('|')) if args.multiclass_forget else {forget_label}
    preserved = [cl for cl in classes_preserved_list if cl.lower() not in {f.lower() for f in forget_set}]
    return preserved

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
    model_dtype = next(model.parameters()).dtype

    for classname in tqdm(classes, desc=f"Projections for {template[0][:15]}..."):
        try:
            hooks.clear()
            classname_clean = classname.replace('_', ' ')
            texts = [t.format(classname_clean) for t in template]
            tokenized_texts = bioclip_tokenize(texts).to(model_device)
            class_embeddings = model.encode_text(tokenized_texts)
            projections_list.append(class_embeddings)

            hook_key = 'ln_final'
            if hook_key in hooks:
                hook_data = hooks[hook_key]
                eot_indices = tokenized_texts.argmax(dim=-1)
                eot_features = hook_data[torch.arange(hook_data.shape[0]), eot_indices]
                avg_eot_feature = eot_features.mean(dim=0)
                hooks_list.append(avg_eot_feature.clone())
            else:
                print(Fore.YELLOW + f"Warning: Hook data for '{hook_key}' not found. Skipping hook feature for class '{classname}'.")
        except Exception as e:
            print(Fore.YELLOW + f"Warning: Could not compute projection for class '{classname}'. Error: {e}. Skipping.")
            continue

    if not projections_list:
        return None, None

    projections = torch.stack(projections_list, dim=0).to(dtype=model_dtype) 
    list_hooks = torch.stack(hooks_list, dim=0).to(dtype=model_dtype)
    return projections, list_hooks

@torch.no_grad()
def register_model_hooks(model):
    for name, module in model.named_modules():
        if 'ln_final'  in name:
            module.register_forward_hook(get_activation(name))
            print(Fore.BLUE + f"Registered hook for: {name}")

@torch.no_grad()
def compute_proj_into(model, original_class_projection, device, method="opposite"):
    """
    计算遗忘目标投影，基于原始类别投影
    
    Args:
        model: CLIP模型
        original_class_projection: 原始类别的文本投影 (tensor)
        device: 设备
        method: 遗忘策略
            - "opposite": 原始投影的相反方向
            - "orthogonal": 与原始投影正交的随机方向
            - "noise": 添加噪声后的方向
            - "empty": 原始的空文本方法（备选）
    
    Returns:
        proj_into: 遗忘目标投影
    """
    original_projection_2d = original_class_projection[:, -1, :] if original_class_projection.dim() == 3 else original_class_projection
    
    if method == "opposite":
        # 方法1: 使用原始投影的相反方向作为遗忘目标
        proj_into = -original_projection_2d
        proj_into = proj_into / proj_into.norm(dim=-1, keepdim=True)
        print(f"Using opposite direction as forgetting target")
    
    elif method == "orthogonal":
        # 方法2: 计算与原始投影正交的随机方向
        feature_dim = original_projection_2d.shape[-1]
        random_vector = torch.randn_like(original_projection_2d).to(device)
        
        # 使用Gram-Schmidt过程使随机向量与原始投影正交
        dot_product = (random_vector * original_projection_2d).sum(dim=-1, keepdim=True)
        orthogonal_vector = random_vector - dot_product * original_projection_2d
        proj_into = orthogonal_vector / orthogonal_vector.norm(dim=-1, keepdim=True)
        print(f"Using orthogonal direction as forgetting target")
    
    elif method == "noise":
        # 方法3: 在原始投影上添加强噪声
        noise_scale = 2.0  # 噪声强度
        noise = torch.randn_like(original_projection_2d).to(device) * noise_scale
        noisy_projection = original_projection_2d + noise
        proj_into = noisy_projection / noisy_projection.norm(dim=-1, keepdim=True)
        print(f"Using noisy direction as forgetting target (noise_scale={noise_scale})")
    
    elif method == "empty":
        # 方法4: 原始的空文本方法（作为备选）
        empty_text = bioclip_tokenize("").to(device)
        embed = model.encode_text(empty_text).repeat(original_projection_2d.shape[0], 1)
        embed_2d = embed[:, -1, :] if embed.dim() == 3 else embed
        proj_into = embed_2d / embed_2d.norm(dim=-1, keepdim=True)
        print(f"Using empty text as forgetting target (original method)")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return proj_into

# =================================================================================
# --- Main Execution Block ---
# =================================================================================

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
    
    all_logs = {}
    
    run_ds = all_ds[:] if not args.run_ds else [item.strip() for item in args.run_ds.split(',')]
    assert all(ds in all_ds for ds in run_ds), "One or more specified run_ds are not valid."
    
    output_base = args.output_dir
    
    print("Loading base model...")
    model = get_model(device=device, arch=args.backbone_arch)
    
    test_datasets, test_dataloaders, datasets_cls = load_test_datasets(cfg, model)
    
    original_model_state = {k: v.clone() for k, v in model.state_dict().items()}
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
            if ds not in datasets_cls: 
                continue
            template = [CUSTOM_TEMPLATES.get(ds, 'a photo of a {}, a type of plant.')]
            projections, list_hooks = precompute_projections(model, datasets_cls[ds].classnames, template=template)
            if projections is not None:
                all_projections[ds] = projections
                all_hooks[ds] = list_hooks
        print(Fore.GREEN + f"Saving precomputed projections to {projection_cache_file}...")
        torch.save({'projections': all_projections, 'hooks': all_hooks}, projection_cache_file)

    for main_ds in run_ds:
        print(f"\n{'='*30}\nStarting unlearning for dataset: {main_ds}\n{'='*30}")
        kwargs = configs[main_ds]
        
        forget_classes_list = forget_classes_all[main_ds]
        if not isinstance(forget_classes_list, list):
            forget_classes_list = [forget_classes_list]
        
        for forget_label in forget_classes_list:
            all_logs.setdefault(main_ds, {})
            print(f"\n--- Forgetting class: {forget_label} ---\n")
            model.load_state_dict(original_model_state)
            
            # Calculate and print similarities before unlearning
            print(Fore.CYAN + "--- Calculating Average Class Similarities (Before Unlearning) ---")
            sims_before = calculate_average_class_similarity(
                model,
                test_dataloaders[main_ds],
                datasets_cls[main_ds].classnames,
                CUSTOM_TEMPLATES.get(main_ds, 'a photo of a {}.'),
                device
            )
            print(f"  - Similarity for forget class '{forget_label}': {sims_before.get(forget_label, 'N/A'):.4f}")
            preserved_samples = [c for c in datasets_cls[main_ds].classnames if c != forget_label][:3]
            for preserved_cls in preserved_samples:
                print(f"  - Similarity for preserved class '{preserved_cls}': {sims_before.get(preserved_cls, 'N/A'):.4f}")
            
            all_logs[main_ds][forget_label] = {'similarities_before_unlearn': sims_before}

            
            print("Constructing matched preserve set for hooks and outputs...")
            preserve_hooks_list = []
            preserve_output_list = []

            for ds_name, ds_hooks in all_hooks.items():
                if ds_name == main_ds:
                    preserved_class_names = get_preserved_classes(main_ds, forget_label, args, {'PLTNetMini': pltnetmini_list, 'StanfordDogs': stanforddogs_list, 'StanfordCars': stanfordcars_list, 'Caltech101': caltech_list, 'OxfordFlowers': oxfordflowers_list})
                    if preserved_class_names:
                        indices = [datasets_cls[ds_name].classnames.index(c) for c in preserved_class_names]
                        preserve_hooks_list.append(ds_hooks[indices])
                        preserve_output_list.append(all_projections[ds_name][indices])
            
            if preserve_hooks_list:
                preserve_hooks = torch.cat(preserve_hooks_list, dim=0)
                preserve_output = torch.cat(preserve_output_list, dim=0)
                if preserve_output.dim() == 3:
                    preserve_output = preserve_output.squeeze(1)
            else:
                feature_dim = list(all_hooks.values())[0].shape[-1]
                preserve_hooks = torch.empty(0, feature_dim).to(device)
                preserve_output = torch.empty(0, feature_dim).to(device)

            forget_class_idx = datasets_cls[main_ds].classnames.index(forget_label)
            change_hooks = all_hooks[main_ds][forget_class_idx].unsqueeze(0)
            
            # 获取原始遗忘类别的投影作为参考
            original_forget_projection = all_projections[main_ds][forget_class_idx].unsqueeze(0)
            
            original_dtype = model.text_projection.dtype
            preserve_hooks = preserve_hooks.float().to(device)
            preserve_output = preserve_output.float().to(device)
            change_hooks = change_hooks.float().to(device)

            in_proj, out_proj = model.text_projection.shape
            
            # 使用原始类别投影来计算遗忘目标
            # 可以选择不同的遗忘策略: "opposite", "orthogonal", "noise", "empty"
            forgetting_method = "empty"  # 默认使用空投影
            proj_into = compute_proj_into(model, original_forget_projection.float().to(device), device, method=forgetting_method).float()

            best_weights = None
            current_lambdas = {'forget': kwargs['lamb_forget']}

            for trial in range(UNLEARN_TRIALS):
                print(f"\n--- Starting Trial {trial+1}/{UNLEARN_TRIALS} ---")
                
                new_text_proj = Linear(in_proj, out_proj, r=kwargs['lora_r'], bias=False, device=device)
                new_text_proj.weight = torch.nn.Parameter(model.text_projection.T.clone())
                new_text_proj.weight.requires_grad = False
                
                # 确保LoRA模块处于训练模式
                new_text_proj.train()
                
                # 检查LoRA参数是否正确初始化
                print(f"LoRA A shape: {new_text_proj.lora_A.shape}, requires_grad: {new_text_proj.lora_A.requires_grad}")
                print(f"LoRA B shape: {new_text_proj.lora_B.shape}, requires_grad: {new_text_proj.lora_B.requires_grad}")
                print(f"LoRA scaling factor: {new_text_proj.scaling}")
                
                optimizer = torch.optim.Adam(new_text_proj.parameters(), lr=1e-4)
                scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

                with torch.no_grad():
                    initial_proj = new_text_proj(change_hooks)
                    initial_proj_2d = initial_proj[:, -1, :] if initial_proj.dim() == 3 else initial_proj
                    initial_forget_loss_for_trials = torch.norm(proj_into - initial_proj_2d, p=2).item()

                print(f"Trial {trial+1}: Initial forget loss is {initial_forget_loss_for_trials:.4f}. Current lamb_forget={current_lambdas['forget']:.2f}, lamb_preserve={kwargs['lamb_preserve']:.2f}")
                
                final_forget_loss = None
                initial_lora_a = new_text_proj.lora_A.clone()
                initial_lora_b = new_text_proj.lora_B.clone()
                
                for epoch in range(EPOCHS):
                    # 确保模块在训练模式
                    new_text_proj.train()
                    
                    new_preserve_output = new_text_proj(preserve_hooks)
                    new_forget_output = new_text_proj(change_hooks)
                    new_forget_output_2d = new_forget_output[:, -1, :] if new_forget_output.dim() == 3 else new_forget_output
                    
                    delta_w = (new_text_proj.lora_B @ new_text_proj.lora_A)
                    
                    forget_loss_val = torch.norm(proj_into - new_forget_output_2d, p=2)
                    
                    if preserve_hooks.shape[0] > 0:
                        preserve_loss_val = torch.norm(preserve_output - new_preserve_output, p=2)
                    else:
                        preserve_loss_val = torch.tensor(0.0).to(device)

                    weight_loss = torch.norm(delta_w, p=2) 
                    
                    loss = current_lambdas['forget'] * forget_loss_val + \
                           kwargs['lamb_preserve'] * preserve_loss_val + \
                           kwargs['lamb_weight'] * weight_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Periodic evaluation during training
                    if (epoch + 1) % 1000 == 0 or (epoch + 1) == EPOCHS:
                        with torch.no_grad():
                            # 正确合并LoRA权重，包括scaling因子
                            current_delta_w = (new_text_proj.lora_B @ new_text_proj.lora_A) * new_text_proj.scaling
                            current_weights_fp32 = (new_text_proj.weight.T + current_delta_w.T).clone()
                            current_weights = current_weights_fp32.to(dtype=original_dtype)

                            # 暂存原始权重，评估后恢复
                            original_text_projection = model.text_projection.clone()
                            model.text_projection = torch.nn.Parameter(current_weights)
                            
                            # 同时检查是否需要更新底层模型的text_projection
                            if hasattr(model, 'model') and hasattr(model.model, 'text_projection'):
                                original_underlying_projection = model.model.text_projection.clone()
                                model.model.text_projection = torch.nn.Parameter(current_weights)
                                print(f"  - Updated underlying model text_projection")
                            
                            model.eval()
                            
                            # 验证权重确实被更新
                            updated_norm = torch.norm(model.text_projection).item()
                            original_norm = torch.norm(original_text_projection).item()
                            print(f"  - Weight update verification: {original_norm:.6f} -> {updated_norm:.6f}")

                        print(Fore.MAGENTA + f"\n--- Calculating Intermediate Similarities at Epoch {epoch+1} ---")
                        sims_epoch = calculate_average_class_similarity(
                            model,
                            test_dataloaders[main_ds],
                            datasets_cls[main_ds].classnames,
                            CUSTOM_TEMPLATES.get(main_ds, 'a photo of a {}.'),
                            device
                        )

                        forget_sim_start = sims_before.get(forget_label, float('nan'))
                        forget_sim_current = sims_epoch.get(forget_label, float('nan'))
                        print(f"  - Forget Class '{forget_label}': {forget_sim_current:.4f} (Start: {forget_sim_start:.4f})")
                        print(f"  - Delta magnitude: {torch.norm(current_delta_w).item():.6f}")
                        
                        # 调试：检查ln_final的输出是否发生变化
                        with torch.no_grad():
                            # 获取遗忘类别的当前ln输出
                            forget_class_name_clean = forget_label.replace('_', ' ')
                            test_text = bioclip_tokenize([f"a photo of a {forget_class_name_clean}, a type of plant."]).to(device)
                            
                            # 清除之前的hooks
                            hooks.clear()
                            
                            # 通过模型获取文本特征，这会触发hooks
                            text_features = model.encode_text(test_text)
                            
                            # 检查ln_final的输出
                            if 'ln_final' in hooks:
                                current_ln_output = hooks['ln_final'].mean().item()
                                print(f"  - Current ln_final output mean: {current_ln_output:.6f}")
                            
                            # 检查text_projection权重的变化
                            current_text_proj_norm = torch.norm(model.text_projection).item()
                            print(f"  - Text projection norm: {current_text_proj_norm:.6f}")
                        
                        # 另外，检查原始的change_hooks和当前处理后的输出差异
                        with torch.no_grad():
                            original_output = new_text_proj.weight.T @ change_hooks.T
                            current_output_with_lora = new_text_proj(change_hooks)
                            lora_effect = torch.norm(current_output_with_lora - original_output.T).item()
                            print(f"  - LoRA effect on change_hooks: {lora_effect:.6f}")

                        log_key = 'epoch_similarities'
                        if log_key not in all_logs[main_ds][forget_label]:
                            all_logs[main_ds][forget_label][log_key] = {}
                        all_logs[main_ds][forget_label][log_key][epoch + 1] = sims_epoch
                        
                        # 恢复原始权重，让LoRA训练继续
                        model.text_projection = torch.nn.Parameter(original_text_projection)
                        if hasattr(model, 'model') and hasattr(model.model, 'text_projection'):
                            model.model.text_projection = torch.nn.Parameter(original_underlying_projection)

                # 保存最终的遗忘损失
                final_forget_loss = forget_loss_val.item()

                # 检查LoRA参数是否有更新
                lora_a_change = torch.norm(new_text_proj.lora_A - initial_lora_a).item()
                lora_b_change = torch.norm(new_text_proj.lora_B - initial_lora_b).item()
                print(f"LoRA parameter changes: A={lora_a_change:.6f}, B={lora_b_change:.6f}")

                with torch.no_grad():
                    reduction = (initial_forget_loss_for_trials - final_forget_loss) / (initial_forget_loss_for_trials + 1e-8)
                    
                    print(f"Trial {trial+1} Summary: Initial Loss={initial_forget_loss_for_trials:.4f}, Final Loss={final_forget_loss:.4f}, Reduction={reduction:.2%}")
                    
                    if reduction >= REDUCTION_THR:
                        print(Fore.GREEN + "Unlearning successful, breaking trial loop.")
                        # 正确合并LoRA权重，包括scaling因子
                        delta_w = (new_text_proj.lora_B @ new_text_proj.lora_A) * new_text_proj.scaling
                        final_weights_fp32 = (new_text_proj.weight.T + delta_w.T).clone()
                        best_weights = final_weights_fp32.to(dtype=original_dtype)
                        print(f"Applied LoRA delta with scaling {new_text_proj.scaling:.4f}")
                        break
                    else:
                        best_weights = None
                        if trial < UNLEARN_TRIALS - 1:
                            current_lambdas['forget'] *= 1.2 
                            print(f"Reduction insufficient. Increasing lamb_forget aggressively to {current_lambdas['forget']:.2f} for the next trial.")
                        else:
                            print(Fore.RED + "All trials failed to meet the reduction threshold.")

            if best_weights is not None:
                # 保存更新前的权重，用于验证
                old_weights_norm = torch.norm(model.text_projection).item()
                
                # 检查模型结构
                if hasattr(model, 'model') and hasattr(model.model, 'text_projection'):
                    print("Updating both adapter and underlying model text_projection")
                    model.model.text_projection = torch.nn.Parameter(best_weights)
                
                model.text_projection = torch.nn.Parameter(best_weights)
                new_weights_norm = torch.norm(model.text_projection).item()
                print(f"Text projection weight norm change: {old_weights_norm:.6f} -> {new_weights_norm:.6f}")
                
                # 强制设置为训练模式下不被重置
                if hasattr(model, 'model'):
                    model.model.eval()
                model.eval()
                
                print(Fore.GREEN + "\n--- Calculating Average Class Similarities (After Unlearning) ---")
                sims_after = calculate_average_class_similarity(
                    model,
                    test_dataloaders[main_ds],
                    datasets_cls[main_ds].classnames,
                    CUSTOM_TEMPLATES.get(main_ds, 'a photo of a {}.'),
                    device
                )
                print(f"  - Similarity for forget class '{forget_label}': {sims_after.get(forget_label, 'N/A'):.4f} (Before: {sims_before.get(forget_label, 'N/A'):.4f})")
                for preserved_cls in preserved_samples:
                    print(f"  - Similarity for preserved class '{preserved_cls}': {sims_after.get(preserved_cls, 'N/A'):.4f} (Before: {sims_before.get(preserved_cls, 'N/A'):.4f})")
                
                all_logs[main_ds][forget_label]['similarities_after_unlearn'] = sims_after
                
                results_ds = eval_all_ds(model, datasets_cls, main_ds, forget_label, test_dataloaders,
                                         None, eval_forgetonly=False, debug=True, device=device)
                
                all_logs[main_ds][forget_label]['final_results'] = results_ds
                print(f"*** Final results for forgetting '{forget_label}': {results_ds[main_ds].get(forget_label, 'N/A')} ***")

                model_save_path = os.path.join(output_base, f"model_{main_ds}_{forget_label}.pth")
                print(Fore.GREEN + f"Saving unlearned model to {model_save_path}...")
                torch.save(model.state_dict(), model_save_path)
                
                # 转换为JSON可序列化格式并保存
                serializable_logs = convert_to_json_serializable(all_logs)
                with open(os.path.join(output_base, "logs.json"), "w") as f:
                    json.dump(serializable_logs, f, indent=4)
            else:
                print(f"Error: Could not find suitable weights for forgetting '{forget_label}'.")

    print("\nAll tasks completed.")
    # 转换为JSON可序列化格式并保存最终日志
    serializable_logs = convert_to_json_serializable(all_logs)
    with open(os.path.join(output_base, "logs.json"), "w") as f:
        json.dump(serializable_logs, f, indent=4)