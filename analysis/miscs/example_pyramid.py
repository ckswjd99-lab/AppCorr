import argparse
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="mysquirell.png")
    args = parser.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to read {args.input}"
    stem = args.input.rsplit(".", 1)[0]
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Custom colormap: SUMMER feel with 0 = black
    summer_lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_SUMMER)
    lut = np.zeros_like(summer_lut)
    for i in range(256):
        lut[i, 0] = (summer_lut[i, 0].astype(np.float32) * (i / 255)).astype(np.uint8)

    # Build Gaussian pyramid
    L0 = img.astype(np.float32)
    L1 = cv2.pyrDown(L0)
    L2 = cv2.pyrDown(L1)

    # Upsample for residual computation
    L1_up = cv2.pyrUp(L1, dstsize=(L0.shape[1], L0.shape[0]))
    L2_up = cv2.pyrUp(L2, dstsize=(L1.shape[1], L1.shape[0]))

    # Residuals (absdiff, grayscale, normalized to [0, 255])
    r01 = np.abs(L0 - L1_up).mean(axis=2)
    r01 = cv2.applyColorMap((r01 * 5).clip(0, 255).astype(np.uint8), lut)
    r12 = np.abs(L1 - L2_up).mean(axis=2)
    r12 = cv2.applyColorMap((r12 * 5).clip(0, 255).astype(np.uint8), lut)

    cv2.imwrite(f"{stem}_L0.png", img)
    cv2.imwrite(f"{stem}_L1.png", L1.astype(np.uint8))
    cv2.imwrite(f"{stem}_L2.png", L2.astype(np.uint8))
    cv2.imwrite(f"{stem}_residual_L0_L1.png", r01)
    cv2.imwrite(f"{stem}_residual_L1_L2.png", r12)

    print(f"Saved: {stem}_L0.png, {stem}_L1.png, {stem}_L2.png, {stem}_residual_L0_L1.png, {stem}_residual_L1_L2.png")


if __name__ == "__main__":
    main()
