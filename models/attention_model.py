import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

class MultiHeadAttentionLocalization(nn.Module):
    """
    Multi-Head Attention Localization (MHAL) module for generating heatmaps
    from vision-language features, inspired by the approach in
    "Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding"
    """
    
    def __init__(self, 
                 embed_dim=512, 
                 num_heads=8, 
                 dropout=0.1,
                 use_clip_features=True):
        """
        Initialize the MHAL module.
        
        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            use_clip_features (bool): Whether to use CLIP features
        """
        super(MultiHeadAttentionLocalization, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_clip_features = use_clip_features
        
        # Multi-head attention for text-to-image localization
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection layers
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        self.image_projection = nn.Linear(embed_dim, embed_dim)
        
        # Output layers for heatmap generation
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(num_heads, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                text_features: torch.Tensor, 
                image_features: torch.Tensor, 
                image_size: Tuple[int, int] = (224, 224)) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate heatmaps.
        
        Args:
            text_features (torch.Tensor): Text features [batch_size, seq_len, embed_dim]
            image_features (torch.Tensor): Image features [batch_size, num_patches, embed_dim]
            image_size (Tuple[int, int]): Output image size (height, width)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing heatmaps and attention weights
        """
        batch_size = text_features.shape[0]
        
        # Project features
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        
        # Apply multi-head attention (text as query, image as key/value)
        attn_output, attn_weights = self.multihead_attn(
            query=text_proj,
            key=image_proj,
            value=image_proj,
            need_weights=True,
            average_attn_weights=False  # Get per-head attention weights
        )
        
        # Extract attention weights for each head
        # Shape: [batch_size, num_heads, seq_len, num_patches]
        head_attns = attn_weights.view(batch_size, self.num_heads, text_features.shape[1], -1)
        
        # Average attention across text tokens for each head
        # Shape: [batch_size, num_heads, num_patches]
        head_attns = head_attns.mean(dim=2)
        
        # Reshape attention maps to 2D spatial grid
        patch_size = int(image_features.shape[1] ** 0.5)  # Assuming square grid of patches
        head_attns_spatial = head_attns.view(batch_size, self.num_heads, patch_size, patch_size)
        
        # Generate heatmaps from attention maps
        # Upsample to desired output size
        head_attns_upsampled = F.interpolate(
            head_attns_spatial, 
            size=image_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Process through convolutional layers to get final heatmap
        heatmap = self.heatmap_conv(head_attns_upsampled)
        
        # Apply sigmoid to get values in [0, 1] range
        heatmap = torch.sigmoid(heatmap)
        
        return {
            'heatmap': heatmap.squeeze(1),  # [batch_size, H, W]
            'head_attentions': head_attns,  # [batch_size, num_heads, num_patches]
            'head_attentions_spatial': head_attns_spatial,  # [batch_size, num_heads, patch_size, patch_size]
            'attention_upsampled': head_attns_upsampled,  # [batch_size, num_heads, H, W]
        }
    
    def get_localization_heads(self, 
                              text_features: torch.Tensor, 
                              image_features: torch.Tensor,
                              num_heads_to_select: int = 3) -> List[int]:
        """
        Identify the most effective localization heads based on spatial entropy.
        
        Args:
            text_features (torch.Tensor): Text features
            image_features (torch.Tensor): Image features
            num_heads_to_select (int): Number of heads to select
            
        Returns:
            List[int]: Indices of selected localization heads
        """
        batch_size = text_features.shape[0]
        
        # Project features
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        
        # Apply multi-head attention
        _, attn_weights = self.multihead_attn(
            query=text_proj,
            key=image_proj,
            value=image_proj,
            need_weights=True,
            average_attn_weights=False
        )
        
        # Extract attention weights for each head
        head_attns = attn_weights.view(batch_size, self.num_heads, text_features.shape[1], -1)
        head_attns = head_attns.mean(dim=2)  # Average across text tokens
        
        # Calculate spatial entropy for each head
        # Lower entropy indicates more focused attention
        head_entropies = []
        for h in range(self.num_heads):
            head_attn = head_attns[:, h, :]  # [batch_size, num_patches]
            
            # Normalize to sum to 1
            head_attn = head_attn / (head_attn.sum(dim=1, keepdim=True) + 1e-9)
            
            # Calculate entropy: -sum(p * log(p))
            entropy = -(head_attn * torch.log(head_attn + 1e-9)).sum(dim=1).mean()
            head_entropies.append(entropy.item())
        
        # Select heads with lowest entropy (most focused attention)
        selected_heads = sorted(range(len(head_entropies)), key=lambda i: head_entropies[i])[:num_heads_to_select]
        
        return selected_heads
