import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def visualize_heatmap(heatmap, target=None, title=None):
    """
    Visualize a heatmap with optional target for comparison.
    
    Args:
        heatmap (np.ndarray): Predicted heatmap
        target (np.ndarray, optional): Target heatmap for comparison
        title (str, optional): Title for the visualization
        
    Returns:
        np.ndarray: Visualization image
    """
    # Create figure
    if target is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot predicted heatmap
    axes[0].imshow(heatmap, cmap='jet')
    axes[0].set_title('Predicted Heatmap')
    axes[0].axis('off')
    
    # Plot heatmap overlay on a placeholder image
    # In a real implementation, this would overlay on the actual panorama
    placeholder = np.ones((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.float32) * 0.7
    
    # Create overlay
    heatmap_rgb = plt.cm.jet(heatmap)[:, :, :3]
    overlay = placeholder.copy()
    mask = heatmap > 0.1
    overlay[mask] = heatmap_rgb[mask] * 0.7 + placeholder[mask] * 0.3
    
    axes[1].imshow(overlay)
    axes[1].set_title('Heatmap Overlay')
    axes[1].axis('off')
    
    # Plot target heatmap if provided
    if target is not None:
        axes[2].imshow(target, cmap='jet')
        axes[2].set_title('Target Heatmap')
        axes[2].axis('off')
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    fig.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    visualization = np.array(fig.canvas.renderer.buffer_rgba())
    
    plt.close(fig)
    
    return visualization

def save_visualization(visualization, output_path):
    """
    Save visualization to file.
    
    Args:
        visualization (np.ndarray): Visualization image
        output_path (str): Output file path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert RGBA to RGB
    if visualization.shape[2] == 4:
        visualization = visualization[:, :, :3]
    
    # Save image
    Image.fromarray(visualization).save(output_path)

def create_attention_visualization(attention_maps, image, text, output_path=None):
    """
    Create visualization of attention maps on an image.
    
    Args:
        attention_maps (torch.Tensor): Attention maps [num_heads, H, W]
        image (PIL.Image): Input image
        text (str): Input text
        output_path (str, optional): Output file path
        
    Returns:
        np.ndarray: Visualization image
    """
    num_heads = attention_maps.shape[0]
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_heads)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    # Plot each attention head
    for i in range(num_heads):
        if i < len(axes):
            attention = attention_maps[i].cpu().numpy()
            
            # Normalize attention
            if attention.max() > 0:
                attention = attention / attention.max()
            
            # Convert image to numpy array
            img_np = np.array(image)
            
            # Create heatmap overlay
            heatmap_rgb = plt.cm.jet(attention)[:, :, :3]
            overlay = img_np.copy() / 255.0
            mask = attention > 0.2
            overlay[mask] = heatmap_rgb[mask] * 0.7 + overlay[mask] * 0.3
            
            axes[i].imshow(overlay)
            axes[i].set_title(f'Head {i}')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')
    
    # Set overall title
    fig.suptitle(f'Attention Maps for: "{text}"', fontsize=16)
    
    fig.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    visualization = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Save if output path is provided
    if output_path:
        save_visualization(visualization, output_path)
    
    plt.close(fig)
    
    return visualization

def create_comparison_visualization(pred_heatmaps, target_heatmaps, instructions, output_path=None):
    """
    Create visualization comparing multiple predicted and target heatmaps.
    
    Args:
        pred_heatmaps (list): List of predicted heatmaps
        target_heatmaps (list): List of target heatmaps
        instructions (list): List of instructions
        output_path (str, optional): Output file path
        
    Returns:
        np.ndarray: Visualization image
    """
    num_samples = len(pred_heatmaps)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    
    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each sample
    for i in range(num_samples):
        # Plot predicted heatmap
        axes[i, 0].imshow(pred_heatmaps[i], cmap='jet')
        axes[i, 0].set_title(f'Predicted: {instructions[i][:50]}...')
        axes[i, 0].axis('off')
        
        # Plot target heatmap
        axes[i, 1].imshow(target_heatmaps[i], cmap='jet')
        axes[i, 1].set_title('Target')
        axes[i, 1].axis('off')
    
    fig.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    visualization = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Save if output path is provided
    if output_path:
        save_visualization(visualization, output_path)
    
    plt.close(fig)
    
    return visualization
