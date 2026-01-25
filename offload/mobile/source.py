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
        pin_memory=True
    )
    return val_loader

class SourceModule(multiprocessing.Process):
    """
    Experiment Loop: Request Batch -> Wait Batch Response -> Calculate Accuracy.
    """

    def __init__(self, output_queue, feedback_queue, config: ExperimentConfig, data_root: str, loader_batch_size: int):
        super().__init__()
        self.output_queue = output_queue
        self.feedback_queue = feedback_queue
        self.config = config
        self.data_root = data_root
        self.loader_batch_size = loader_batch_size

    def run(self):
        print(f"[Source] Loading ImageNet from {self.data_root} (Batch: {self.loader_batch_size})...")
        try:
            loader = load_imagenet1k_val(self.data_root, batch_size=self.loader_batch_size)
        except Exception as e:
            print(f"[Source] Failed to load ImageNet: {e}")
            return

        # 1. Handshake
        self.output_queue.put(self.config)
        time.sleep(1)

        policy = get_transmission(self.config.transmission_policy_name)
        total_correct = 0
        total_samples = 0

        print("[Source] Starting Batch Evaluation Loop...")

        for batch_idx, (images, labels) in enumerate(loader):
            # images: (B, 3, H, W), labels: (B)
            curr_bs = images.size(0)
            
            # --- Step A: Send Phase (Request Batch) ---
            # Optimize: Convert entire batch to numpy at once
            images_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            
            # Encode entire batch
            all_patches = policy.encode(images_np, self.config)
            
            # Send all patches
            for p in all_patches:
                self.output_queue.put(p)
            
            # --- Step B: Wait Phase (Response Batch) ---
            result = self.feedback_queue.get()
            
            # Output is expected to be an array/list of predictions of size (Batch_Size,)
            preds = result.output 
            
            # Calculate accuracy for this batch
            batch_correct = 0
            
            # Iterate through the batch predictions locally
            for i in range(curr_bs):
                label = labels[i].item()
                
                # Safe access in case partial batch returned (though not expected in Mock)
                if i < len(preds):
                    pred = preds[i]
                    if pred == label:
                        batch_correct += 1
            
            latency = time.time() - result.timestamp

            # --- Step C: Update Metrics ---
            total_correct += batch_correct
            total_samples += curr_bs
            
            acc = total_correct / total_samples * 100
            print(f"[Source] Batch {batch_idx} Done. Batch Acc: {batch_correct}/{curr_bs}, Global Acc: {acc:.2f}% | Latency: {latency:.4f}s")

            if total_samples >= 128: 
                print("[Source] Test limit reached.")
                break

        print(f"[Source] Final Accuracy: {total_correct}/{total_samples} ({total_correct/total_samples*100:.2f}%)")
        self.output_queue.put('STOP')