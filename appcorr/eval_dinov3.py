import torch
import argparse
from evaluators.imnet1k import eval_imagenet1k
from models.dinov3.hub.classifiers import dinov3_vit7b16_lc

REPO_DIR = "/home/nxc/cjpark/dinov3"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PRECISION = torch.bfloat16 if DEVICE == "cuda:0" else torch.float32

# --- PLANS DEFINITION ---
PLAN_APPROX_ONLY = [
    # op_type, level, layers, group_idx
    ("A", 0, range( 0, 10), None),
    ("A", 0, range(10, 20), None),
    ("A", 0, range(20, 30), None),
    ("A", 0, range(30, 40), None),
]

PLAN_INTERLEAVED_CORRECT_S2 = [
    ("A", 0, range( 0, 20), None),
    ("C", 1, range( 0, 20),    1),
    ("A", 0, range(20, 40), None),
    ("C", 1, range( 0, 40),    2),
]

PLAN_INTERLEAVED_CORRECT_S4 = [
    ("A", 0, range( 0, 10), None),
    ("C", 1, range( 0, 10),    1),
    ("A", 0, range(10, 20), None),
    ("C", 1, range( 0, 20),    2),
    ("A", 0, range(20, 30), None),
    ("C", 1, range( 0, 30),    3),
    ("A", 0, range(30, 40), None),
    ("C", 1, range( 0, 40),    4),
]

PLAN_INTERLEAVED_CORRECT_S4_LDROP = [
    ("A", 0, range( 0, 10), None),
    ("C", 1, range( 0, 10),    1),
    ("A", 0, range(10, 20), None),
    ("C", 1, range( 0, 20),    2),
    ("A", 0, range(20, 30), None),
    ("C", 1, [i for i in range( 0, 30) if i % 3 in [0, 1]],    3),
    ("A", 0, range(30, 40), None),
    ("C", 1, [i for i in range( 0, 40) if i % 4 in [0, 2]],    4),
]

PLAN_INTERLEAVED_CORRECT_S4_LDROP_V2 = [
    ("A", 0, range( 0, 10), None),
    ("C", 1, [i for i in range( 0, 10) if i % 3 in [0]],    1),     # 4 layers alive, 6 dropped
    ("A", 0, range(10, 20), None),
    ("C", 1, [i for i in range( 0, 20) if i % 2 in [0]],    2),     # 10 layers alive, 10 dropped
    ("A", 0, range(20, 30), None),
    ("C", 1, [i for i in range( 0, 30) if i % 3 in [0, 1]],    3),  # 20 layers alive, 10 dropped
    ("A", 0, range(30, 40), None),
    ("C", 1, [i for i in range( 0, 40) if i % 4 in [0, 1, 2]],    4),   # 30 layers alive, 10 dropped
]

PLANS = {
    "approx_only": PLAN_APPROX_ONLY,
    "interleaved_correct_s2": PLAN_INTERLEAVED_CORRECT_S2,
    "interleaved_correct_s4": PLAN_INTERLEAVED_CORRECT_S4,
    "interleaved_correct_s4_ldrop": PLAN_INTERLEAVED_CORRECT_S4_LDROP,
    "interleaved_correct_s4_ldrop_v2": PLAN_INTERLEAVED_CORRECT_S4_LDROP_V2,
}

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 AppCorr Plans")
    
    # Levels: 0 (Original), 2 (1/4 scale)
    parser.add_argument("--levels", type=int, nargs='+', required=True, 
                        help="Pyramid levels list (e.g., 0 for orig, 2 for 1/4 scale)")
    
    # Plan selection
    parser.add_argument("--plan", type=str, required=True, choices=PLANS.keys(),
                        help="Plan key from PLANS dict")
    
    # Groups (for interleaved correct)
    parser.add_argument("--groups", type=int, default=4, help="Number of groups for correction")
    
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    return parser.parse_args()

@torch.inference_mode()
def eval_classifier(args):
    print(f"\n>>> CONFIG: Levels={args.levels} | Plan={args.plan} | Groups={args.groups} | Batch Size={args.batch_size}")
    
    model = dinov3_vit7b16_lc(
        pretrained=True,
        weights="~/cjpark/weights/dinov3/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth",
        backbone_weights="~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    model.eval()
    model = model.to(dtype=PRECISION).to(device=DEVICE)
    print("Model loaded.")

    # Configure AppCorr Mode
    current_plan = PLANS[args.plan]
    
    model.backbone.set_appcorr_mode(
        enabled=True, 
        update_attn=True, 
        pyramid_levels=args.levels,
        token_res = [0.5, 1],
        plan=current_plan,
        num_groups=args.groups,
    )

    # Eval
    with torch.autocast('cuda', dtype=torch.bfloat16):
        eval_result = eval_imagenet1k(
            model, 
            DEVICE, 
            image_size=256, 
            batch_size=args.batch_size, 
            dtype=PRECISION,
            max_samples=args.batch_size * 10,  # For quick eval
        )
    
    print(f"[{args.plan} | Levels {args.levels}] Top-1 Accuracy: {eval_result['top1_accuracy']:.2f}%")
    print(f"[{args.plan} | Levels {args.levels}] Top-5 Accuracy: {eval_result['top5_accuracy']:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    args = get_args()
    eval_classifier(args)