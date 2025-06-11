import torch
import numpy as np
from typing import List, Dict, Any

def rxr_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for RxR dataset that handles variable-sized elements.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        dict: Batched samples with proper handling of variable-sized elements
    """
    # Initialize batch dictionary
    batch_dict = {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    for key in keys:
        # Handle different types of data
        if key == 'panoramas':
            # For panoramas, we need special handling since they are lists of different lengths
            # We'll pad shorter lists with None
            max_panos = max([len(sample[key]) for sample in batch])
            padded_panos = []
            for sample in batch:
                # Pad with None if needed
                padded = sample[key] + [None] * (max_panos - len(sample[key]))
                padded_panos.append(padded)
            
            # Transpose to get lists of panoramas at each position
            # This creates a list where each element contains all panoramas at that position
            transposed_panos = list(zip(*padded_panos))
            
            # Filter out None values from each position
            filtered_panos = []
            for pos_panos in transposed_panos:
                filtered_panos.append([p for p in pos_panos if p is not None])
            
            batch_dict[key] = filtered_panos
        elif key == 'pose_trace':
            # For pose traces, we can't easily batch them, so keep as list
            batch_dict[key] = [sample[key] for sample in batch]
        elif key == 'path':
            # For paths, keep as list of lists
            batch_dict[key] = [sample[key] for sample in batch]
        elif key == 'instruction_id':
            # For instruction IDs, convert to tensor if they're integers
            if all(isinstance(sample[key], int) for sample in batch):
                batch_dict[key] = torch.tensor([sample[key] for sample in batch])
            else:
                batch_dict[key] = [sample[key] for sample in batch]
        elif key == 'instruction':
            # For instructions, keep as list of strings
            batch_dict[key] = [sample[key] for sample in batch]
        elif key == 'scan_id':
            # For scan IDs, keep as list of strings
            batch_dict[key] = [sample[key] for sample in batch]
        elif key == 'heading':
            # For headings, convert to tensor
            batch_dict[key] = torch.tensor([sample[key] for sample in batch], dtype=torch.float32)
        elif isinstance(batch[0][key], torch.Tensor):
            # For tensors, use standard stacking
            batch_dict[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], (int, float)):
            # For scalars, convert to tensor
            batch_dict[key] = torch.tensor([sample[key] for sample in batch])
        elif isinstance(batch[0][key], str):
            # For strings, keep as list
            batch_dict[key] = [sample[key] for sample in batch]
        else:
            # For other types, keep as list
            batch_dict[key] = [sample[key] for sample in batch]
    
    return batch_dict