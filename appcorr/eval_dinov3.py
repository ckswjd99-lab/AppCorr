import torch
from evaluators.imnet1k import eval_imagenet1k
from models.dinov3.hub.classifiers import dinov3_vit7b16_lc

REPO_DIR = "/home/nxc/cjpark/dinov3"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PRECISION = torch.bfloat16 if DEVICE == "cuda:0" else torch.float32

@torch.inference_mode()
def eval_classifier():
    model = dinov3_vit7b16_lc(
        pretrained=True,
        weights="~/cjpark/weights/dinov3/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth",
        backbone_weights="~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    model = model.to(dtype=PRECISION).to(device=DEVICE)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        eval_result = eval_imagenet1k(model, DEVICE, image_size=256, batch_size=32, dtype=PRECISION)
    
    print(f"Top-1 Accuracy: {eval_result['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {eval_result['top5_accuracy']:.2f}%")


if __name__ == "__main__":
    eval_classifier()