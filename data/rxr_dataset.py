# import os
# import yaml
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import json
# from data.rxr_colat_fn import rxr_collate_fn
# class RxRDataset(Dataset):
#     """
#     Dataset class for loading and preprocessing RxR dataset for heatmap generation.
#     Integrates with CLIP for feature extraction and prepares data for training.
#     """
    
#     def __init__(self, data_root, split='train', transform=None, max_length=80):
#         """
#         Initialize the RxR dataset.
        
#         Args:
#             data_root (str): Root directory of the RxR dataset
#             split (str): Dataset split ('train', 'val_seen', 'val_unseen', 'test')
#             transform: Optional transforms to apply to images
#             max_length (int): Maximum length of instruction tokens
#         """
#         self.data_root = data_root
#         self.split = split
#         self.transform = transform
#         self.max_length = max_length
        
#         # Load dataset annotations
#         self.annotations = self._load_annotations()

#         # Load pose traces
#         self.pose_traces = self._load_pose_traces()
#         print(f"Loaded {len(self.annotations)} samples from {split} split")
    
#     def _load_annotations(self):
#         """
#         Load guide annotations from the RxR dataset.
        
#         Returns:
#             list: List of annotation dictionaries
#         """
#         annotation_file = os.path.join(self.data_root, f"rxr_{self.split}_guide_unzipped.jsonl")
        
#         if not os.path.exists(annotation_file):
#             raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
#         annotations = []
#         with open(annotation_file, 'r') as f:
#             i = 0
#             for line in f:
#                 i+=1
#                 annotations.append(json.loads(line.strip()))
#                 if i ==300:
#                     break
#         return annotations
    
#     def _load_pose_traces(self):
#         """
#         Load pose traces for the annotations.
        
#         Returns:
#             dict: Dictionary mapping instruction_id to pose trace
#         """
#         pose_traces = {}
#         pose_trace_dir = os.path.join(self.data_root, "pose_traces")

#         if not os.path.exists(pose_trace_dir):
#             print(f"Warning: Pose trace directory not found: {pose_trace_dir}")
#             return pose_traces
        
#         for annotation in self.annotations:
#             instruction_id = annotation['instruction_id']
#             pose_trace_file = os.path.join(
#                 pose_trace_dir, 
#                 f"rxr_{self.split}",
#                 f"{instruction_id:06}_guide_pose_trace.npz"
#             )
            
#             if os.path.exists(pose_trace_file):
#                 # Load the data and convert to dict to avoid keeping file handles open
#                 with np.load(pose_trace_file, allow_pickle=True) as data:
#                     # Convert to a regular dictionary to avoid file handle issues
#                     pose_traces[instruction_id] = {key: data[key] for key in data.keys()}
        
#         print(f"Loaded {len(pose_traces)} pose traces")
#         return pose_traces

    
#     def _load_panorama(self, scan_id, viewpoint_id):
#         """
#         Load panorama image for a given scan and viewpoint.
        
#         Args:
#             scan_id (str): Scan identifier
#             viewpoint_id (str): Viewpoint identifier
            
#         Returns:
#             Image: Panorama image
#         """
#         panorama_path = os.path.join(
#             self.data_root,
#             scan_id,
#             'matterport_skybox_images',
#             f"{viewpoint_id}.jpg"
#         )
#         if not os.path.exists(panorama_path):
#             raise FileNotFoundError(f"Panorama image not found: {panorama_path}")
            
        
#         # Load and preprocess panorama
#         from PIL import Image
#         panorama = Image.open(panorama_path)
        
#         if self.transform:
#             panorama = self.transform(panorama)
            
#         return panorama
    
#     def __len__(self):
#         return len(self.annotations)
    
#     def __getitem__(self, idx):
#         """
#         Get a sample from the dataset.
        
#         Args:
#             idx (int): Index
            
#         Returns:
#             dict: Sample dictionary containing images, instructions, and metadata
#         """
#         annotation = self.annotations[idx]
        
#         # Extract metadata
#         instruction_id = annotation['instruction_id']
#         scan_id = annotation['scan']
#         path = annotation['path']
#         instruction = annotation['instruction']
        
#         # Load panoramas along the path
#         panoramas = []
#         for viewpoint_id in path:
#             try:
#                 panorama = self._load_panorama(scan_id, viewpoint_id)
#                 panoramas.append(panorama)
#             except FileNotFoundError as e:
#                 print(f"Warning: {e}")
        
#         # Get pose trace if available
#         pose_trace = self.pose_traces.get(instruction_id, None)
#         # Create sample dictionary
#         sample = {
#             'instruction_id': instruction_id,
#             'scan_id': scan_id,
#             'path': path,
#             'instruction': instruction,
#             'panoramas': panoramas,
#             'pose_trace': pose_trace,
#             'heading': annotation.get('heading', 0.0),
#         }
        
#         return sample


# def create_dataloader(data_root, split, batch_size=8, num_workers=4, transform=None):
#     """
#     Create a dataloader for the RxR dataset.
    
#     Args:
#         data_root (str): Root directory of the RxR dataset
#         split (str): Dataset split
#         batch_size (int): Batch size
#         num_workers (int): Number of workers for data loading
#         transform: Optional transforms to apply to images
        
#     Returns:
#         DataLoader: PyTorch dataloader
#     """
#     dataset = RxRDataset(data_root, split, transform)
#     print(dataset.__len__)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(split == 'train'),
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=(split == 'train'),
#         collate_fn=rxr_collate_fn
#     )
    
#     return dataloader


import os
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from data.rxr_colat_fn import rxr_collate_fn

class RxRDataset(Dataset):
    """
    Dataset class for loading and preprocessing RxR dataset for heatmap generation.
    Integrates with CLIP for feature extraction and prepares data for training.
    """
    
    def __init__(self, data_root, split='train', transform=None, max_length=80, max_scans=5):
        """
        Initialize the RxR dataset.
        
        Args:
            data_root (str): Root directory of the RxR dataset
            split (str): Dataset split ('train', 'val_seen', 'val_unseen', 'test')
            transform: Optional transforms to apply to images
            max_length (int): Maximum length of instruction tokens
            max_scans (int): Maximum number of unique scan IDs to include (None for all)
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.max_scans = max_scans
        
        # Load dataset annotations
        self.annotations = self._load_annotations()
        
        # Filter annotations to include only the first max_scans scan IDs
        if self.max_scans is not None:
            self.annotations = self._filter_by_scan_count(self.annotations, self.max_scans)
        
        # Load pose traces
        self.pose_traces = self._load_pose_traces()
        print(f"Loaded {len(self.annotations)} samples from {split} split")
    
    def _filter_by_scan_count(self, annotations, max_scans):
        """
        Filter annotations to include only the first max_scans unique scan IDs.
        
        Args:
            annotations (list): List of annotation dictionaries
            max_scans (int): Maximum number of unique scan IDs to include
            
        Returns:
            list: Filtered list of annotation dictionaries
        """
        unique_scans = set()
        filtered_annotations = []
        
        for annotation in annotations:
            scan_id = annotation['scan']
            if scan_id not in unique_scans:
                if len(unique_scans) >= max_scans:
                    continue
                unique_scans.add(scan_id)
            
            if scan_id in unique_scans:
                filtered_annotations.append(annotation)
        
        print(f"Filtered to {len(filtered_annotations)} annotations from {len(unique_scans)} unique scan IDs: {unique_scans}")
        return filtered_annotations
    
    def _load_annotations(self):
        """
        Load guide annotations from the RxR dataset.
        
        Returns:
            list: List of annotation dictionaries
        """
        annotation_file = os.path.join(self.data_root, f"rxr_{self.split}_guide_unzipped.jsonl")
        
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        annotations = []
        with open(annotation_file, 'r') as f:
            for line in f:
                annotations.append(json.loads(line.strip()))
        
        return annotations
    
    def _load_pose_traces(self):
        """
        Load pose traces for the annotations.
        
        Returns:
            dict: Dictionary mapping instruction_id to pose trace
        """
        pose_traces = {}
        pose_trace_dir = os.path.join(self.data_root, "pose_traces")

        if not os.path.exists(pose_trace_dir):
            print(f"Warning: Pose trace directory not found: {pose_trace_dir}")
            return pose_traces
        
        for annotation in self.annotations:
            instruction_id = annotation['instruction_id']
            pose_trace_file = os.path.join(
                pose_trace_dir, 
                f"rxr_{self.split}",
                f"{instruction_id:06}_guide_pose_trace.npz"
            )
            
            if os.path.exists(pose_trace_file):
                # Load the data and convert to dict to avoid keeping file handles open
                with np.load(pose_trace_file, allow_pickle=True) as data:
                    # Convert to a regular dictionary to avoid file handle issues
                    pose_traces[instruction_id] = {key: data[key] for key in data.keys()}
        
        print(f"Loaded {len(pose_traces)} pose traces")
        return pose_traces

    
    def _load_panorama(self, scan_id, viewpoint_id):
        """
        Load panorama image for a given scan and viewpoint.
        
        Args:
            scan_id (str): Scan identifier
            viewpoint_id (str): Viewpoint identifier
            
        Returns:
            Image: Panorama image
        """
        panorama_path = os.path.join(
            self.data_root,
            scan_id,
            'matterport_skybox_images',
            f"{viewpoint_id}.jpg"
        )
        if not os.path.exists(panorama_path):
            raise FileNotFoundError(f"Panorama image not found: {panorama_path}")
            
        
        # Load and preprocess panorama
        from PIL import Image
        panorama = Image.open(panorama_path)
        
        if self.transform:
            panorama = self.transform(panorama)
            
        return panorama
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Sample dictionary containing images, instructions, and metadata
        """
        annotation = self.annotations[idx]
        
        # Extract metadata
        instruction_id = annotation['instruction_id']
        scan_id = annotation['scan']
        path = annotation['path']
        instruction = annotation['instruction']
        
        # Load panoramas along the path
        panoramas = []
        for viewpoint_id in path:
            try:
                panorama = self._load_panorama(scan_id, viewpoint_id)
                panoramas.append(panorama)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        
        # Get pose trace if available
        pose_trace = self.pose_traces.get(instruction_id, None)
        # Create sample dictionary
        sample = {
            'instruction_id': instruction_id,
            'scan_id': scan_id,
            'path': path,
            'instruction': instruction,
            'panoramas': panoramas,
            'pose_trace': pose_trace,
            'heading': annotation.get('heading', 0.0),
        }
        
        return sample


def create_dataloader(data_root, split, batch_size=8, num_workers=0, transform=None, max_scans=5):
    """
    Create a dataloader for the RxR dataset.
    
    Args:
        data_root (str): Root directory of the RxR dataset
        split (str): Dataset split
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        transform: Optional transforms to apply to images
        max_scans (int): Maximum number of unique scan IDs to include (None for all)
        
    Returns:
        DataLoader: PyTorch dataloader
    """
    dataset = RxRDataset(data_root, split, transform, max_scans=max_scans)
    print(f"Dataset size: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,  # Start with 0 for debugging
        pin_memory=True,
        drop_last=(split == 'train'),
        collate_fn=rxr_collate_fn
    )
    
    return dataloader