import os
import torch
import clip
import numpy as np
from PIL import Image
from torch import nn
from typing import List, Dict, Any, Tuple

class CLIPAdapter(nn.Module):
    """
    Adapter for OpenAI's CLIP model to extract visual and textual features
    for RxR heatmap generation.
    """
    
    def __init__(self, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the CLIP adapter.
        
        Args:
            model_name (str): CLIP model variant to use
            device (str): Device to run the model on
        """
        super(CLIPAdapter, self).__init__()
        
        self.device = device
        self.model_name = model_name
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        
        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"CLIP model loaded successfully on {device}")
        
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Encode images using CLIP.
        
        Args:
            images (List[Image.Image]): List of PIL images
            
        Returns:
            torch.Tensor: Image features
        """
        # Preprocess images
        preprocessed_images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = self.model.encode_image(preprocessed_images)
            
        return image_features
    
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text instructions using CLIP.
        
        Args:
            texts (List[str]): List of text instructions
            
        Returns:
            torch.Tensor: Text features
        """
        # Tokenize and encode text
        text_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            
        return text_features
    
    def extract_attention_maps(self, images: List[Image.Image], texts: List[str]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract attention maps from CLIP for text-to-image grounding.
        Inspired by the approach in "Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding"
        
        Args:
            images (List[Image.Image]): List of PIL images
            texts (List[str]): List of text instructions
            
        Returns:
            Tuple[torch.Tensor, Dict]: Attention maps and additional metadata
        """
        # Preprocess images and text
        preprocessed_images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_tokens = clip.tokenize(texts).to(self.device)
        
        # Get attention maps
        # Note: This is a simplified implementation as direct access to CLIP attention maps
        # requires model modifications. In a full implementation, we would need to modify
        # the CLIP model to extract attention maps from specific heads.
        
        # Placeholder for attention maps
        batch_size = len(images)
        img_size = 224  # Standard CLIP input size
        patch_size = 32 if "ViT-B/32" in self.model_name else 16
        num_patches = (img_size // patch_size) ** 2
        
        # In a real implementation, we would extract these from the model
        attention_maps = torch.zeros(batch_size, num_patches, dtype=torch.float32)
        
        # For demonstration purposes, create dummy attention maps
        # In practice, these would come from specific attention heads in the CLIP model
        for i in range(batch_size):
            # Create a simple Gaussian attention pattern centered on the image
            center_x, center_y = num_patches // 2, num_patches // 2
            for x in range(int(np.sqrt(num_patches))):
                for y in range(int(np.sqrt(num_patches))):
                    idx = x * int(np.sqrt(num_patches)) + y
                    dist = np.sqrt((x - center_x/np.sqrt(num_patches))**2 + (y - center_y/np.sqrt(num_patches))**2)
                    attention_maps[i, idx] = np.exp(-dist**2 / 4)
        
        # Normalize attention maps
        attention_maps = attention_maps / attention_maps.sum(dim=1, keepdim=True)
        
        metadata = {
            "patch_size": patch_size,
            "num_patches": num_patches,
            "img_size": img_size
        }
        
        return attention_maps, metadata
    
    def reshape_attention_to_image(self, attention_maps: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Reshape attention maps to image space.
        
        Args:
            attention_maps (torch.Tensor): Attention maps [batch_size, num_patches]
            metadata (Dict): Metadata from extract_attention_maps
            
        Returns:
            torch.Tensor: Reshaped attention maps [batch_size, H, W]
        """
        batch_size = attention_maps.shape[0]
        patch_size = metadata["patch_size"]
        img_size = metadata["img_size"]
        
        # Reshape to square grid
        grid_size = int(np.sqrt(attention_maps.shape[1]))
        reshaped = attention_maps.view(batch_size, grid_size, grid_size)
        
        # Upsample to full image resolution
        upsampled = nn.functional.interpolate(
            reshaped.unsqueeze(1),
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return upsampled
