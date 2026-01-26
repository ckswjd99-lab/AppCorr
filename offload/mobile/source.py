import multiprocessing
import time
import torch
from torchvision import datasets, transforms
import numpy as np
from offload.common import ExperimentConfig
from offload.policies import get_transmission

def load_imagenet1k_val(root, image_size=256, batch_size=32, num_workers=4):
    """Load ImageNet with specified batch size."""
    if image_size == 256:
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
    else:
        val_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    val_dataset = datasets.ImageFolder(root=root, transform=val_transforms)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False 
    )
    return val_loader

class SourceModule(multiprocessing.Process):
    """
    Experiment Loop: Request Batch -> Wait Response -> Calculate Metrics.
    Handles partial batches via padding and tracks data transfer size.
    """

    def __init__(self, output_queue, feedback_queue, config: ExperimentConfig, data_root: str, loader_batch_size: int):
        super().__init__()
        self.output_queue = output_queue
        self.feedback_queue = feedback_queue
        self.config = config
        self.data_root = data_root
        self.loader_batch_size = loader_batch_size

    def run(self):
        print(f"[Source] Loading ImageNet from {self.data_root} (Loader Batch: {self.loader_batch_size})...")
        try:
            # Assuming load_imagenet1k_val is defined above in the file
            loader = load_imagenet1k_val(self.data_root, batch_size=self.loader_batch_size)
        except Exception as e:
            print(f"[Source] Failed to load ImageNet: {e}")
            return

        # 1. Handshake
        self.output_queue.put(self.config)
        time.sleep(1)

        policy = get_transmission(self.config.transmission_policy_name)
        
        # Initialize metrics
        total_top1 = 0
        total_top5 = 0
        total_samples = 0
        total_bytes = 0 

        # Constants for padding
        SERVER_BATCH_SIZE = self.config.batch_size
        IMG_H, IMG_W, IMG_C = self.config.image_shape

        print("[Source] Starting Batch Evaluation Loop...")

        for batch_idx, (images, labels) in enumerate(loader):
            curr_bs = images.size(0)
            
            # --- Step A: Send Phase ---
            # 1. Prepare Full Batch Container (Pad with zeros)
            full_batch_np = np.zeros((SERVER_BATCH_SIZE, IMG_H, IMG_W, IMG_C), dtype=np.uint8)
            
            # 2. Fill with real data
            real_imgs_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            full_batch_np[:curr_bs] = real_imgs_np
            
            # 3. Encode the full batch
            all_patches = policy.encode(full_batch_np, self.config)
            
            # 4. Measure payload size
            batch_bytes = sum(len(p.data) for p in all_patches)
            total_bytes += batch_bytes
            batch_kb = batch_bytes / 1024.0

            # 5. Send patches
            for p in all_patches:
                self.output_queue.put(p)
            
            # --- Step B: Wait Phase ---
            result = self.feedback_queue.get()
            
            # --- Step C: Calculate Metrics ---
            # Slice results to ignore padding
            valid_preds = result.output[:curr_bs]
            valid_labels = labels.tolist()
            
            # Calculate Top-1 and Top-5 using list comprehension
            batch_top1 = sum(p[0] == l for p, l in zip(valid_preds, valid_labels))
            batch_top5 = sum(l in p for p, l in zip(valid_preds, valid_labels))
            
            latency = time.time() - result.timestamp

            # Update globals
            total_top1 += batch_top1
            total_top5 += batch_top5
            total_samples += curr_bs
            
            acc1 = total_top1 / total_samples * 100
            acc5 = total_top5 / total_samples * 100
            
            if (batch_idx+1) % 100 == 0:
                print(f"[Source] Batch {(batch_idx+1)}/{len(loader)} | Acc@1: {acc1:.2f}% | Acc@5: {acc5:.2f}% | Latency: {latency:.4f}s")
                break # TEMP

        print(f"[Source] Final | Top-1: {acc1:.2f}% | Top-5: {acc5:.2f}% | Avg. Transfer: {total_bytes/1024/total_samples:.2f} KB/image")
        self.output_queue.put('STOP')
            
