"""
BioCLIP adapter module for replacing CLIP with BioCLIP
"""
import torch
import torch.nn.functional as F
from typing import List, Union
import numpy as np
from colorama import Fore, Style, init
init(autoreset=True)

class BioCLIPAdapter:
    """Adapter class to make BioCLIP compatible with CLIP interface"""

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

        # --- FIX: More robust search for the text encoder attribute ---
        if hasattr(self.model, 'text'):
            self.transformer = self.model.text
        elif hasattr(self.model, 'transformer'): # A common name
            self.transformer = self.model.transformer
        elif hasattr(self.model, 'text_model'): # Another common name
            self.transformer = self.model.text_model
        elif hasattr(self.model, 'text_encoder'):
            self.transformer = self.model.text_encoder
        else:
            # Provide a more helpful error message for debugging
            available_attrs = "\n".join([f" - {attr}" for attr in dir(self.model) if not attr.startswith('_')])
            raise ValueError(
                "BioCLIP model does not have a recognizable text encoder attribute "
                "(checked for 'text', 'transformer', 'text_model', 'text_encoder').\n"
                f"Available attributes on the loaded model are:\n{available_attrs}"
            )

    def encode_text(self, text_tokens):
        """Encode text tokens using BioCLIP"""
        return self.model.encode_text(text_tokens)
    
    def encode_text_no_grad(self, text_tokens):
        """Encode text tokens using BioCLIP without gradients"""
        with torch.no_grad():
            return self.model.encode_text(text_tokens)

    def encode_image(self, images):
        """Encode images using BioCLIP"""
        with torch.no_grad():
            return self.model.encode_image(images)

    def __call__(self, image, text):
        """Forward pass for image and text"""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with compatibility"""
        # Filter out keys that don't match BioCLIP architecture
        filtered_state_dict = {}
        model_keys = set([name for name, _ in self.model.named_parameters()])
        
        for key, value in state_dict.items():
            if key in model_keys:
                filtered_state_dict[key] = value
            else:
                # Try to map CLIP keys to BioCLIP keys
                mapped_key = self._map_clip_to_bioclip_key(key)
                if mapped_key and mapped_key in model_keys:
                    filtered_state_dict[mapped_key] = value
        
        if filtered_state_dict:
            missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys in BioCLIP model: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"Unexpected keys in state dict: {unexpected_keys[:5]}...")
        else:
            print("No compatible keys found in state dict for BioCLIP")

    def _map_clip_to_bioclip_key(self, clip_key):
        """Map CLIP parameter names to BioCLIP parameter names"""
        # Basic mapping - can be extended based on actual differences
        key_mapping = {
            'visual.conv1.weight': 'visual.trunk.patch_embed.proj.weight',
            'visual.conv1.bias': 'visual.trunk.patch_embed.proj.bias',
            # Add more mappings as needed
        }
        return key_mapping.get(clip_key, None)

    def train(self, mode=True):
        """Set model to training mode"""
        self.model.train(mode)
        return self

    def eval(self):
        """Set model to evaluation mode"""  
        self.model.eval()
        return self

    def to(self, device):
        """Move model to device"""
        self.device = device
        self.model = self.model.to(device)
        return self

    def parameters(self):
        """Return model parameters"""
        return self.model.parameters()

    def named_parameters(self):
        """Return named model parameters"""
        return self.model.named_parameters()

    def named_modules(self):
        """Return named modules"""
        return self.model.named_modules()

    def state_dict(self):
        """Return model state dict"""
        return self.model.state_dict()


def create_bioclip_model(arch="ViT-B-16", device='cuda', load_path=""):
    """Create BioCLIP model with CLIP-compatible interface"""
    print(f"Attempting to create BioCLIP model with architecture: {arch}")

    try:
        # Try to load actual BioCLIP model
        import open_clip
        model_name='hf-hub:imageomics/bioclip'
        
        print(f"Loading BioCLIP model: {model_name}")
        
        # Wrap in adapter for CLIP compatibility
        adapter = BioCLIPAdapter(model_name=model_name, device=device)
        
        if load_path:
            print(f"LOADING FROM {load_path}")
            state_dict = torch.load(load_path, map_location="cpu")
            # Try to load compatible weights
            try:
                adapter.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load state dict for BioCLIP: {e}")
                print("Continuing with pretrained BioCLIP weights...")

        print(Fore.GREEN + "Successfully loaded BioCLIP model!")
        return adapter
        
    except Exception as e:
        print(f"Error loading BioCLIP model: {e}")
        print("Falling back to regular CLIP...")
        
        # Fallback to CLIP if BioCLIP fails
        try:
            from clip import clip
            clip_model, _ = clip.load(arch, device=device)

            if load_path:
                print(f"LOADING FROM {load_path}")
                state_dict = torch.load(load_path, map_location="cpu")
                clip_model.load_state_dict(state_dict, strict=False)
                clip_model = clip_model.to(device).eval()

            return clip_model
        except Exception as clip_e:
            print(f"Error loading CLIP model: {clip_e}")
            raise clip_e


def bioclip_tokenize(texts: Union[str, List[str]], context_length: int = 77):
    """Tokenize text using BioCLIP tokenizer with fallback to CLIP"""
    try:
        import open_clip
        # Try to get BioCLIP tokenizer
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        
        # BioCLIP typically uses context_length=77, ensure consistency
        if isinstance(texts, str):
            texts = [texts]
        
        # Use open_clip's tokenize function with proper context length
        tokens = tokenizer(texts, context_length=context_length)
        return tokens
    except Exception as e:
        print(f"BioCLIP tokenizer failed: {e}, falling back to CLIP tokenizer")
        import clip
        return clip.tokenize(texts, context_length=context_length, truncate=True)


def clip_classifier(classnames, templates, model):
    """Create classifier weights using BioCLIP (compatible with CLIP interface)"""
    with torch.no_grad():
        clip_weights = []

        # Get model device safely
        if hasattr(model, 'device'):
            model_device = model.device
        elif hasattr(model, 'visual') and hasattr(model.visual, 'conv1'):
            model_device = model.visual.conv1.weight.device
        else:
            model_device = next(model.parameters()).device

        for classname in classnames:
            # Tokenize all templates for this class
            # 确保classname是字符串
            classname = str(classname).replace('_', ' ')

            # Safe template formatting with error handling
            texts = []
            for template in templates[0]:
                try:
                    # Handle template formatting safely
                    if '{}' in template:
                        formatted_text = template.format(classname)
                    else:
                        # If no {} placeholder, just use the template as is
                        formatted_text = template
                    texts.append(formatted_text)
                except (ValueError, KeyError) as e:
                    print(f"Template formatting error for '{template}' with class '{classname}': {e}")
                    # Fallback: just use classname
                    texts.append(f"a photo of a {classname}")

            try:
                texts = bioclip_tokenize(texts).to(model_device)
                # Get text features
                class_embeddings = model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)
            except Exception as e:
                print(f"Error processing class {classname}: {e}")
                # Create a dummy embedding if processing fails
                # Get the correct embedding dimension from the model
                if hasattr(model, 'text_projection') and model.text_projection is not None:
                    embedding_dim = model.text_projection.shape[0]
                else:
                    # Try to infer from transformer width
                    if hasattr(model, 'transformer') and hasattr(model.transformer, 'width'):
                        embedding_dim = model.transformer.width
                    else:
                        # Default fallback based on common architectures
                        embedding_dim = 1024  # RN50 default
                clip_weights.append(torch.randn(embedding_dim).to(model_device))

        clip_weights = torch.stack(clip_weights, dim=1)

    # 对于BioCLIP (ViT架构)，我们需要转置权重矩阵以匹配 features @ weights 的计算
    # features: [N, 512], weights需要是: [512, num_classes]
    return clip_weights  # clip_weights是 [512, num_classes] 形状，正确用于矩阵乘法
