import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import yaml

class RxRDataUtils:
    """
    Utility functions for RxR dataset processing.
    """
    
    @staticmethod
    def load_config(config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @staticmethod
    def create_ground_truth_heatmap(pose_trace, panorama_size, path_width=10):
        """
        Create ground truth heatmap from pose trace.
        
        Args:
            pose_trace (dict): Pose trace data
            panorama_size (tuple): Size of panorama (height, width)
            path_width (int): Width of the path in pixels
            
        Returns:
            np.ndarray: Ground truth heatmap
        """
        height, width = panorama_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        if pose_trace is None:
            return heatmap
        
        # Extract pose information
        pano_ids = pose_trace['pano']
        positions = pose_trace.get('position', None)
        headings = pose_trace.get('heading', None)
        
        if positions is None or headings is None:
            return heatmap
        
        # Create path corridor
        for i in range(len(positions) - 1):
            start_pos = positions[i]
            end_pos = positions[i+1]
            
            # Convert 3D positions to 2D panorama coordinates
            # This is a simplified version - actual implementation would depend on
            # the specific panorama projection used in RxR
            start_x = int((headings[i] / (2 * np.pi) + 0.5) * width) % width
            end_x = int((headings[i+1] / (2 * np.pi) + 0.5) * width) % width
            
            # Draw path with Gaussian smoothing
            for t in np.linspace(0, 1, 100):
                x = int(start_x * (1 - t) + end_x * t) % width
                y_center = height // 2  # Simplified - actual y would depend on elevation
                
                # Apply Gaussian around the path
                for y in range(max(0, y_center - path_width), min(height, y_center + path_width)):
                    for dx in range(-path_width, path_width + 1):
                        x_pos = (x + dx) % width
                        dist = np.sqrt(dx**2 + (y - y_center)**2)
                        if dist <= path_width:
                            weight = np.exp(-(dist**2) / (2 * (path_width/3)**2))
                            heatmap[y, x_pos] = max(heatmap[y, x_pos], weight)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap
    
    @staticmethod
    def identify_obstacles(pose_trace, follower_trace, panorama_size):
        """
        Identify obstacle regions by comparing guide and follower traces.
        
        Args:
            pose_trace (dict): Guide pose trace
            follower_trace (dict): Follower pose trace
            panorama_size (tuple): Size of panorama (height, width)
            
        Returns:
            np.ndarray: Obstacle heatmap (lower values indicate obstacles)
        """
        height, width = panorama_size
        obstacle_map = np.ones((height, width), dtype=np.float32)
        
        if pose_trace is None or follower_trace is None:
            return obstacle_map
        
        # Extract pose information
        guide_positions = pose_trace.get('position', None)
        follower_positions = follower_trace.get('position', None)
        
        if guide_positions is None or follower_positions is None:
            return obstacle_map
        
        # Calculate deviation between guide and follower paths
        # Areas with high deviation are likely obstacles
        # This is a simplified implementation
        
        # Normalize heatmap
        if np.min(obstacle_map) < 1:
            obstacle_map = (obstacle_map - np.min(obstacle_map)) / (1 - np.min(obstacle_map))
            
        return obstacle_map
    
    @staticmethod
    def combine_heatmaps(path_heatmap, obstacle_map, alpha=0.7):
        """
        Combine path and obstacle heatmaps.
        
        Args:
            path_heatmap (np.ndarray): Path heatmap
            obstacle_map (np.ndarray): Obstacle map
            alpha (float): Weighting factor
            
        Returns:
            np.ndarray: Combined heatmap
        """
        combined = alpha * path_heatmap + (1 - alpha) * obstacle_map
        return combined
