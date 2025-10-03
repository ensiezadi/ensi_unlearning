# 建议将此文件命名为 bioclip_utils.py 或类似名称

import torch
import torch.nn as nn
import open_clip
from typing import List

# 1. 首先定义好分词器，以便全局使用
try:
    bioclip_tokenize = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
except Exception as e:
    print(f"无法加载 BioCLIP tokenizer: {e}")

# 2. 修正并简化 BioCLIPAdapter
class BioCLIPAdapter(nn.Module):
    """
    一个正确的、继承自 nn.Module 的 BioCLIP 适配器。
    它只负责包裹模型，所有功能（如 .to, .eval, .parameters）都自动继承。
    """
    def __init__(self, model_name='hf-hub:imageomics/bioclip', device='cuda'):
        self.device = device
        import open_clip
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Store original text projection for compatibility
        self.text_projection = self.model.text_projection

        # Add CLIP-compatible attributes for hooks and LoRA
        self.visual = self.model.visual

        # Ensure visual has conv1 for compatibility with existing hooks
        if not hasattr(self.visual, 'conv1'):
            # Create a dummy conv1 attribute for compatibility
            if hasattr(self.visual, 'trunk') and hasattr(self.visual.trunk, 'patch_embed'):
                # For ViT models, map to patch embedding
                self.visual.conv1 = self.visual.trunk.patch_embed.proj
            else:
                # Create a dummy conv1 attribute
                self.visual.conv1 = type('DummyConv1', (), {'weight': torch.nn.Parameter(torch.randn(1, 3, 1, 1).to(device))})()

        # Add transformer attribute for text encoder compatibility
        if hasattr(self.model, 'text'):
            self.transformer = self.model.text
        elif hasattr(self.model, 'text_encoder'):
            self.transformer = self.model.text_encoder
        else:
            # Create minimal transformer structure
            self.transformer = type('Transformer', (), {
                'width': self.model.text_projection.shape[0] if hasattr(self.model, 'text_projection') else 512
            })()


    def encode_image(self, image):
        return self.model.encode_image(image)

    def encode_text(self, text):
        return self.model.encode_text(text)
        
    # 我们不再需要手动实现 .to, .eval, .parameters 等方法，nn.Module 会自动处理

# 3. 修正 create_bioclip_model 工厂函数
def create_bioclip_model(arch="ViT-B/16", device='cuda'):
    """
    使用正确的参数创建并加载 BioCLIP 模型，然后用适配器包裹。
    """
    print(f"正在加载 BioCLIP 模型, 架构: {arch}")
    
    # 核心修正：为 create_model_and_transforms 提供正确的参数
    model, _, _ = open_clip.create_model_and_transforms(
        # model_name=arch,  # 例如 "ViT-B/16" -> "ViT-B-16"
        model_name='hf-hub:imageomics/bioclip'
    )
    # # --- 新增修正：更稳健地获取模型架构 ---
    # vision_arch = '未知'
    # try:
    #     # 优先尝试 .config 属性 (如果存在)
    #     vision_arch = model.config.get('vision_cfg', {}).get('model_name', '未知')
    # except AttributeError:
    #     # 如果 .config 不存在, 则通过检查模型结构来推断
    #     print("'.config' 属性不存在，尝试从模型结构推断架构...")
    #     if hasattr(model, 'visual') and 'VisionTransformer' in str(type(model.visual)):
    #         try:
    #             # 从 Vision Transformer 的关键参数推断
    #             width = model.visual.ln_post.normalized_shape[0]
    #             layers = len(model.visual.transformer.resblocks)
    #             patch_size = model.visual.conv1.kernel_size[0]
                
    #             if layers == 12 and width == 768:
    #                 vision_arch = f"ViT-B/{patch_size}"
    #             elif layers == 24 and width == 1024:
    #                 vision_arch = f"ViT-L/{patch_size}"
    #             else: # 通用情况
    #                 vision_arch = f"ViT (Layers={layers}, Width={width}, Patch={patch_size})"
    #         except Exception as e:
    #             vision_arch = f"VisionTransformer (无法解析详情: {e})"
    #     else:
    #         vision_arch = "架构类型无法自动识别"

    # print(f"模型加载成功! 实际视觉架构: {vision_arch}")
    model.to(device)
    model.eval()
    
    print("BioCLIP 模型加载成功!")
    
    # 用我们简化的适配器包裹模型
    adapter = BioCLIPAdapter(model)
    return adapter

# 4. 修正并简化 clip_classifier 函数
def clip_classifier(classnames: List[str], templates: List[str], model: nn.Module):
    """
    为给定的类别名称和模板创建分类器权重。
    """ 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            # 将类别名中的下划线替换为空格，以获得更自然的句子
            classname_natural = classname.replace('_', ' ')
            texts = [template.format(classname_natural) for template in templates]
            
            tokens = bioclip_tokenize(texts).to(device)
            
            # 调用适配器的 encode_text 方法，现在它应该能为不同文本返回不同向量
            class_embeddings = model.encode_text(tokens)
            
            # 标准的归一化和平均流程
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights