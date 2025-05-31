import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import yaml
import time
from tqdm import tqdm
import logging
from datetime import datetime

# Import project modules
from data.rxr_dataset import RxRDataset, create_dataloader
from models.clip_adapter import CLIPAdapter
from models.attention_model import MultiHeadAttentionLocalization
from models.heatmap_generator import HeatmapGenerator
from utils.visualization import visualize_heatmap, save_visualization

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RxR Heatmap Generator')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--data_root', type=str, default=None, help='Root directory of RxR dataset')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='CLIP model variant')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(output_dir):
    """Set up logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_checkpoint(state, is_best, output_dir):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(state, best_path)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        outputs = model(batch)
        heatmaps = outputs['heatmap']
        
        # Calculate loss
        # In a real implementation, this would use ground truth heatmaps
        # For demonstration, we use a placeholder target
        target_heatmaps = torch.zeros_like(heatmaps)
        for i in range(len(target_heatmaps)):
            # Create a simple target heatmap (this would be based on ground truth in real implementation)
            h, w = target_heatmaps[i].shape
            center_y, center_x = h // 2, w // 2
            for y in range(h):
                for x in range(w):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    target_heatmaps[i][y, x] = np.exp(-dist**2 / (2 * (w/8)**2))
            
            # Normalize
            if target_heatmaps[i].max() > 0:
                target_heatmaps[i] = target_heatmaps[i] / target_heatmaps[i].max()
        
        loss = criterion(heatmaps, target_heatmaps)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Visualize and save sample heatmaps periodically
        if batch_idx % 50 == 0:
            sample_idx = 0
            sample_heatmap = heatmaps[sample_idx].detach().cpu().numpy()
            sample_target = target_heatmaps[sample_idx].detach().cpu().numpy()
            
            visualization = visualize_heatmap(
                sample_heatmap, 
                sample_target,
                title=f'Epoch {epoch}, Batch {batch_idx}'
            )
            
            save_path = os.path.join(args.output_dir, 'visualizations', f'epoch_{epoch}_batch_{batch_idx}.png')
            save_visualization(visualization, save_path)
    
    # Calculate average loss
    avg_loss = epoch_loss / len(dataloader)
    logger.info(f'Epoch {epoch} - Training Loss: {avg_loss:.4f}')
    
    return avg_loss

def validate(model, dataloader, criterion, device, epoch, logger):
    """Validate the model."""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation')):
            # Move data to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch)
            heatmaps = outputs['heatmap']
            
            # Calculate loss (same placeholder as in training)
            target_heatmaps = torch.zeros_like(heatmaps)
            for i in range(len(target_heatmaps)):
                h, w = target_heatmaps[i].shape
                center_y, center_x = h // 2, w // 2
                for y in range(h):
                    for x in range(w):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        target_heatmaps[i][y, x] = np.exp(-dist**2 / (2 * (w/8)**2))
                
                # Normalize
                if target_heatmaps[i].max() > 0:
                    target_heatmaps[i] = target_heatmaps[i] / target_heatmaps[i].max()
            
            loss = criterion(heatmaps, target_heatmaps)
            val_loss += loss.item()
            
            # Visualize and save sample validation heatmaps
            if batch_idx % 10 == 0:
                sample_idx = 0
                sample_heatmap = heatmaps[sample_idx].detach().cpu().numpy()
                sample_target = target_heatmaps[sample_idx].detach().cpu().numpy()
                
                visualization = visualize_heatmap(
                    sample_heatmap, 
                    sample_target,
                    title=f'Validation - Epoch {epoch}, Batch {batch_idx}'
                )
                
                save_path = os.path.join(args.output_dir, 'visualizations', f'val_epoch_{epoch}_batch_{batch_idx}.png')
                save_visualization(visualization, save_path)
    
    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    logger.info(f'Epoch {epoch} - Validation Loss: {avg_val_loss:.4f}')
    
    return avg_val_loss

def main(args):
    """Main training function."""
    # Set up
    set_seed(args.seed)
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_root:
        config['data']['root'] = args.data_root
    
    # Set up output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info(f"Starting training with config: {config}")
    logger.info(f"Arguments: {args}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        data_root=config['data']['root'],
        split='train',
        batch_size=args.batch_size,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    val_dataloader = create_dataloader(
        data_root=config['data']['root'],
        split='val_seen',
        batch_size=args.batch_size,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_dataloader)}, Val: {len(val_dataloader)}")
    
    # Initialize models
    clip_adapter = CLIPAdapter(
        model_name=args.clip_model,
        device=args.device
    )
    
    attention_model = MultiHeadAttentionLocalization(
        embed_dim=config['model'].get('embed_dim', 512),
        num_heads=config['model'].get('num_heads', 8),
        dropout=config['model'].get('dropout', 0.1)
    )
    
    model = HeatmapGenerator(
        clip_adapter=clip_adapter,
        attention_model=attention_model,
        embed_dim=config['model'].get('embed_dim', 512),
        use_pose_traces=config['model'].get('use_pose_traces', True),
        use_ground_truth_path=config['model'].get('use_ground_truth_path', True)
    )
    
    model = model.to(args.device)
    logger.info(f"Model initialized and moved to {args.device}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at {args.resume}")
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=args.device,
            epoch=epoch,
            logger=logger
        )
        
        # Validate
        val_loss = validate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=args.device,
            epoch=epoch,
            logger=logger
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        save_checkpoint(
            state={
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss
            },
            is_best=is_best,
            output_dir=os.path.join(args.output_dir, 'checkpoints')
        )
        
        logger.info(f"Epoch {epoch} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Checkpoint saved. Best Val Loss: {best_val_loss:.4f}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
