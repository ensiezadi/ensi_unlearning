import os
import numpy as np
import torch
from clip import clip
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import matplotlib.pyplot as plt
import swanlab
from colorama import Fore, init

from dassl.config import get_cfg_default
from dassl.data.datasets.build import build_dataset
from bioclip_adapter_fixed import clip_classifier

import datasets.stanford_cars
import datasets.stanford_dogs
import datasets.caltech101
import datasets.oxford_flowers
_swanlab_initialized = False
init(autoreset=True)

CUSTOM_TEMPLATES = {
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "StanfordCars": "a photo of a {}, a type of car.",
    "Caltech101": "a photo of a {}, an object.",
    "StanfordDogs": "a photo of a {}, a breed of dog.",
    "PLTNetMini": "a photo of a {}, a type of plant.",
    "Bird525": "a photo of a {}, a type of bird.",
}

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
            'Bird525': {'lamb_preserve': 0.0, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 0.0}
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

    # Determine configurations based on arguments
    if args.multiclass_forget:
        print(f"SETTING {args.backbone_arch} MULTICLASS")
        configs = multiclass_configs[args.backbone_arch]
    else:
        print(f"SETTING {args.backbone_arch} ONE CLASS")
        configs = onecls_configs[args.backbone_arch]

    return configs

def eval_all_ds(model, datasets_cls, forget_ds, forget_lbl, all_loaders, train_loader=None, eval_forgetonly=False, debug=False, device='cpu', ignore_labels_main=[]):
        
    results = {ds: {} for ds in all_loaders}
    for ds in all_loaders:
        model.eval()
        test_loader = all_loaders[ds]
        
        classnames = datasets_cls[ds].classnames
        clip_weights = clip_classifier(classnames, [CUSTOM_TEMPLATES[ds]], model).to(device)
        
        if ds == forget_ds:
            cls_acc_test = None
            no_cls_acc = None
            if debug:
                acc, (labels, clip_logits_test, raw_similarity) = evaluate_clip_zs(model, test_loader, clip_weights, device=device, out_conf=True)
                # print("acc", acc)
                if ignore_labels_main:
                    ignore_labels = []
                    for tlbl in ignore_labels_main:
                        ignore_labels.append(classnames.index(tlbl))
                    ignore_labels = np.array(ignore_labels)
                    
                    mask_labels = (torch.tensor((~np.isin(labels, ignore_labels)), dtype=labels.dtype)).bool()
                elif forget_lbl not in classnames:
                    id_lbl = 0
                    mask_labels = labels != -9999
                else:
                    id_lbl = classnames.index(forget_lbl)
                    mask_labels = labels != id_lbl
                
                    cls_acc_test = confusion_matrix(labels, clip_logits_test.argmax(1))[id_lbl]
                    cls_acc_test = cls_acc_test[id_lbl] / cls_acc_test.sum()
                    
                no_cls_acc = confusion_matrix(labels[mask_labels], clip_logits_test.argmax(1)[mask_labels])
                no_cls_acc = np.diag(no_cls_acc).sum() / no_cls_acc.sum()
                
                if ignore_labels_main:
                    out_acc_all = {}
                    for c in ignore_labels_main:
                        c_id = classnames.index(c)
                        cls_acc_test = confusion_matrix(labels, clip_logits_test.argmax(1))[c_id]
                        cls_acc_test = cls_acc_test[c_id] / cls_acc_test.sum()
                        out_acc_all[c] = cls_acc_test
            
            # include accuracy of the train data if not None
            if train_loader is not None:
                acc_train = evaluate_clip_zs(model, train_loader, clip_weights, device=device, out_conf=False)
                acc_train = acc_train 
            else:
                acc_train = None
                
            if ignore_labels_main:
                results[ds]['|'.join(ignore_labels_main)] = {'cls_acc_test' : out_acc_all, 
                                          'no_cls_acc' : no_cls_acc, 
                                          'acc_train' : acc_train}
                print(f"{10*'+++'} Train dataset: {ds} - {results[ds]['|'.join(ignore_labels_main)]} {10*'+++'}")
            else:
                results[ds][forget_lbl] = {'cls_acc_test' : cls_acc_test, 
                                          'no_cls_acc' : no_cls_acc, 
                                          'acc_train' : acc_train}
                
                print(f"{10*'+++'} Train dataset: {ds} - {results[ds][forget_lbl]} {10*'+++'}")
                
        else:
            # continue
            if eval_forgetonly or (not debug): continue
            acc = evaluate_clip_zs(model, test_loader, clip_weights, device=device,out_conf=False)
            results[ds]['all'] = {'all_ds' : acc}     
            print(f"{10*'+++'} {ds} - {acc} {10*'+++'}")
                
    return results

        
def get_model(arch="RN50", device='cpu', load_path="", lr=5e-5):
    url = clip._MODELS[arch]
    model_path = clip._download(url)
    print("Loading model...")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict()).float().to(device).eval()

    if load_path:
        print(f"LOADING FROM {load_path}")
        model.load_state_dict(torch.load(load_path, map_location="cpu"))
    
    return model

def cls_acc(output, target, topk=1):
    pred = np.argmax(output, axis=1)
    correct = pred == target
    acc = 100 * correct.sum() / target.shape[0]
    return acc

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

def acc_certain_cls(acc_all, all_lbls, ids_lbl):
    acc_selected = acc_all[ids_lbl]
    labels_selected = np.unique(all_lbls, return_counts=True)[1][ids_lbl]
    
    return np.average(acc_selected, weights=labels_selected)

def create_results(res_folder, return_logs=False, rn=True, log_name='logs.json', multiclass=False):
    
    if res_folder != "":
        with open(res_folder + f"/{log_name}", "r") as f:
            all_logs = json.load(f)

        with open(res_folder + "/args.txt", "r") as f:
            args = f.read()
    else:
        all_logs = log_name
        args = ""
        
    if rn:
        results_zs = load_results("RN50")
    else:
        results_zs = load_results("ViT16")
        
    full_df = []
    final_results = {}
    add_cols = []
    for jj, file in enumerate(all_logs):

        cols = ['cls_forget', 'full_forget', 'acc_train']
        
        single_df = pd.DataFrame(columns=cols)

        if file == 'settings': continue
        final_results[file] = {}
        
        if multiclass:
            all_cls_sameds = []
            all_cls_sameds_ids = []
            cfg = get_cfg_default()
            cfg.DATASET.NAME = file
            cfg.DATASET.SUBSAMPLE_CLASSES = "all"
            cfg.DATASET.ROOT = "/app/datasets/"
            cfg.DATASET.NUM_SHOTS = -1
            dataset = build_dataset(cfg)
            all_lbls = torch.tensor([d.label for d in dataset.test])
            all_accuracies = torch.tensor([results_zs[file][key]['cls_acc_test'] for key in results_zs[file]])

        for ii, k in enumerate(all_logs[file]):
            if not all_logs[file][k]: continue
            if 'kwargs' in all_logs[file][k]: continue
            
            final_results[file][k] = all_logs[file][k]['final_results'][file][k]
            
            single_df.loc[ii, cols] = pd.DataFrame(final_results[file][k].items())[1].values
            single_df.loc[ii, 'name'] = k
            single_df.loc[ii, 'ds'] = file
            
            if multiclass:
                key_splitted = k.split("|")
                all_cls_sameds_ids = [dataset.classnames.index(key) for key in key_splitted]
                remaining_ids = torch.tensor([dataset.classnames.index(cln) for cln in dataset.classnames if cln not in key_splitted])
                
                single_df.loc[ii, 'full_Noforget'] = acc_certain_cls(all_accuracies, all_lbls, remaining_ids)
                single_df.loc[ii, 'cls_Noforget'] = acc_certain_cls(all_accuracies, all_lbls, all_cls_sameds_ids)
            else:
                single_df.loc[ii, 'full_Noforget'] = results_zs[file][k]['no_cls_acc']
                single_df.loc[ii, 'cls_Noforget'] = results_zs[file][k]['cls_acc_test']
                            
            for k1 in list(all_logs[file][k]['final_results'].keys()):
                if 'all' not in all_logs[file][k]['final_results'][k1]: continue
                single_df.loc[ii, f'res_{k1}'] = all_logs[file][k]['final_results'][k1]['all']['all_ds']
                single_df.loc[ii, f'full_{k1}'] = results_zs[k1][list(results_zs[k1].keys())[0]]['full_acc']
                if f'res_{k1}' not in add_cols:
                    add_cols.append(f'res_{k1}')
                if f'full_{k1}' not in add_cols:
                    add_cols.append(f'full_{k1}')
                
        full_df.append(single_df)
    
    full_df = pd.concat(full_df)[['ds', 'name', 'full_Noforget', 'cls_Noforget'] + ['cls_forget', 'full_forget'] + add_cols]
        
    if multiclass:
        full_df = full_df.reset_index(drop=True)
        full_df['cls_forget_all'] = full_df['cls_forget']
        for indx, row in enumerate(full_df.iterrows()):
            full_df.loc[indx, 'cls_forget'] = np.mean([val for key, val in row[1]['cls_forget'].items()])
        
        return full_df, args
    return full_df, args 
    

def compute_avg_gain(df):
    forget_perc = 1.- (df['cls_Noforget'] - df['cls_forget'])/df['cls_Noforget']
    list_main_perc = ((df['full_Noforget'] - df['full_forget'])/df['full_Noforget']).clip(0)
    forget_perc_scars = ((df['full_StanfordCars'] - df['res_StanfordCars'])/df['full_StanfordCars']).clip(0).fillna(0)
    forget_perc_caltech = ((df['full_Caltech101'] - df['res_Caltech101'])/df['full_Caltech101']).clip(0).fillna(0)
    forget_perc_oxflow = ((df['full_OxfordFlowers'] - df['res_OxfordFlowers'])/df['full_OxfordFlowers']).clip(0).fillna(0)
    forget_perc_sdogs = ((df['full_StanfordDogs'] - df['res_StanfordDogs'])/df['full_StanfordDogs']).clip(0).fillna(0)
    # divide by 5 as we have 4 datasets + forget_perc (we have 6 elements below but in each row one element is 0 as it's NA)
    scores = (forget_perc + list_main_perc + forget_perc_scars + forget_perc_caltech + forget_perc_oxflow + forget_perc_sdogs)/5
    return scores.astype(float)


def safe_swanlab_log(data):
    """安全的SwanLab日志记录函数"""
    global _swanlab_initialized
    if _swanlab_initialized:
        try:
            swanlab.log(data)
        except Exception as e:
            print(f"⚠️ SwanLab logging error: {e}")

def initialize_swanlab(project_name, experiment_name, config_dict):
    """初始化SwanLab实验跟踪"""
    global _swanlab_initialized
    if not _swanlab_initialized:
        try:
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config=config_dict
            )
            _swanlab_initialized = True
            return True
        except Exception as e:
            print(f"⚠️ SwanLab initialization error: {e}")
            print("Continuing without SwanLab tracking...")
            return False
    else:
        print("📊 SwanLab already initialized, skipping...")
        return True

def finish_swanlab():
    """安全结束SwanLab实验"""
    global _swanlab_initialized
    if _swanlab_initialized:
        try:
            swanlab.finish()
            print("📊 SwanLab experiment saved successfully!")
        except Exception as e:
            print(f"⚠️ SwanLab finish error: {e}")
    else:
        print("📊 SwanLab was not initialized, skipping finish.")

def plot_forget_loss_curve(forget_loss_history, trial_num, task_name, save_dir="assets/plots"):
    """
    绘制遗忘损失曲线
    
    Args:
        forget_loss_history: 遗忘损失历史列表
        trial_num: 试验编号
        task_name: 任务名称
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(forget_loss_history) + 1))
    plt.plot(epochs, forget_loss_history, 'b-', linewidth=2, label='Forget Loss')
    
    # 添加关键点标记
    if len(forget_loss_history) > 0:
        min_idx = forget_loss_history.index(min(forget_loss_history))
        plt.plot(min_idx + 1, forget_loss_history[min_idx], 'ro', markersize=8, label=f'Min Loss: {forget_loss_history[min_idx]:.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Forget Loss')
    plt.title(f'Forget Loss Curve - {task_name} (Trial {trial_num})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plot_path = os.path.join(save_dir, f"forget_loss_{task_name}_trial{trial_num}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def plot_experiment_summary(all_logs, save_dir="assets/plots"):
    """
    绘制整个实验的汇总图表
    
    Args:
        all_logs: 所有实验日志
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 收集所有任务的相似度变化数据
    tasks = []
    forget_sim_before = []
    forget_sim_after = []
    preserved_sim_before = []
    preserved_sim_after = []
    
    for dataset, dataset_logs in all_logs.items():
        for forget_class, class_logs in dataset_logs.items():
            if 'similarities_before_unlearn' in class_logs and 'similarities_after_unlearn' in class_logs:
                task_name = f"{dataset}_{forget_class}"
                tasks.append(task_name)
                
                sims_before = class_logs['similarities_before_unlearn']
                sims_after = class_logs['similarities_after_unlearn']
                
                forget_sim_before.append(sims_before.get(forget_class, 0.0))
                forget_sim_after.append(sims_after.get(forget_class, 0.0))
                
                # 计算保持类别的平均相似度
                preserved_classes = [k for k in sims_before.keys() if k != forget_class]
                if preserved_classes:
                    preserved_sim_before.append(np.mean([sims_before.get(cls, 0.0) for cls in preserved_classes[:5]]))  # 取前5个保持类别
                    preserved_sim_after.append(np.mean([sims_after.get(cls, 0.0) for cls in preserved_classes[:5]]))
                else:
                    preserved_sim_before.append(0.0)
                    preserved_sim_after.append(0.0)
    
    if tasks:
        # 创建相似度变化对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 遗忘类别相似度变化
        x_pos = np.arange(len(tasks))
        ax1.bar(x_pos - 0.2, forget_sim_before, 0.4, label='Before Unlearning', alpha=0.7, color='red')
        ax1.bar(x_pos + 0.2, forget_sim_after, 0.4, label='After Unlearning', alpha=0.7, color='blue')
        ax1.set_xlabel('Tasks')
        ax1.set_ylabel('Similarity')
        ax1.set_title('Forget Class Similarity Change')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(tasks, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 保持类别相似度变化
        ax2.bar(x_pos - 0.2, preserved_sim_before, 0.4, label='Before Unlearning', alpha=0.7, color='green')
        ax2.bar(x_pos + 0.2, preserved_sim_after, 0.4, label='After Unlearning', alpha=0.7, color='orange')
        ax2.set_xlabel('Tasks')
        ax2.set_ylabel('Similarity')
        ax2.set_title('Preserved Classes Average Similarity')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(tasks, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_path = os.path.join(save_dir, "experiment_similarity_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Experiment summary plot saved to: {summary_path}")
        return summary_path
    
    return None

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

