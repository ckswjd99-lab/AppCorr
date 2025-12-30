import torch
from torchvision import datasets, transforms
from tqdm import tqdm

def load_imagenet1k_val(image_size=224, batch_size=64, num_workers=8):
    if image_size == 224:
        val_transforms = transforms.Compose([
            transforms.Resize(256),              # 크기를 256으로 조절
            transforms.CenterCrop(224),          # 224x224로 중앙 크롭
            transforms.ToTensor(),               # PyTorch Tensor로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 표준 정규화
                                std=[0.229, 0.224, 0.225])
        ])
    elif image_size == 256:
        val_transforms = transforms.Compose([
            transforms.Resize(256),              # 크기를 256으로 조절
            transforms.CenterCrop(256),          # 256x256로 중앙 크롭
            transforms.ToTensor(),               # PyTorch Tensor로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 표준 정규화
                                std=[0.229, 0.224, 0.225])
        ])
    elif image_size == 384:
        val_transforms = transforms.Compose([
            transforms.Resize(432),              # 크기를 432로 조절
            transforms.CenterCrop(384),          # 384x384로 중앙 크롭
            transforms.ToTensor(),               # PyTorch Tensor로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 표준 정규화
                                std=[0.229, 0.224, 0.225])
        ])
    elif image_size == 512:
        val_transforms = transforms.Compose([
            transforms.Resize(512),              # 크기를 572로 조절
            transforms.CenterCrop(512),          # 512x512로 중앙 크롭
            transforms.ToTensor(),               # PyTorch Tensor로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 표준 정규화
                                std=[0.229, 0.224, 0.225])
        ])
    elif image_size == 518:
        val_transforms = transforms.Compose([
            transforms.Resize(582),              # 크기를 582로 조절
            transforms.CenterCrop(518),          # 518x518로 중앙 크롭
            transforms.ToTensor(),               # PyTorch Tensor로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 표준 정규화
                                std=[0.229, 0.224, 0.225])
        ])
    elif image_size == 768:
        val_transforms = transforms.Compose([
            transforms.Resize(822),              # 크기를 822로 조절
            transforms.CenterCrop(768),          # 768x768로 중앙 크롭
            transforms.ToTensor(),               # PyTorch Tensor로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 표준 정규화
                                std=[0.229, 0.224, 0.225])
        ])
    elif image_size == 1024:
        val_transforms = transforms.Compose([
            transforms.Resize(1124),              # 크기를 1124로 조절
            transforms.CenterCrop(1024),          # 1024x1024로 중앙 크롭
            transforms.ToTensor(),               # PyTorch Tensor로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 표준 정규화
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Unsupported image size. Supported sizes are 224, 256, 384, 512, 768 and 1024.")

    # 정리된 폴더를 사용하여 ImageFolder 객체 생성
    val_dataset = datasets.ImageFolder(
        root="~/data/imagenet_val",
        transform=val_transforms
    )

    # DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return val_loader

def eval_imagenet1k(model, device="cuda" if torch.cuda.is_available() else "cpu", image_size=224, batch_size=64, max_samples=50000, dtype=torch.float32):
    val_loader = load_imagenet1k_val(image_size=image_size, batch_size=batch_size)

    model = model.to(device)
    model.eval()

    num_total = 0
    num_correct_top1 = 0
    num_correct_top5 = 0

    pbar = tqdm(val_loader)
    for images, labels in pbar:
        images = images.to(device).to(dtype)
        labels = labels.to(device).to(dtype)

        outputs = model(images)
        predicted = outputs.argmax(dim=-1)

        num_total += labels.size(0)
        num_correct_top1 += (predicted == labels).sum().item()

        # top5 accuracy
        _, predicted_top5 = outputs.topk(5, dim=-1)
        for i in range(labels.size(0)):
            if labels[i] in predicted_top5[i]:
                num_correct_top5 += 1
        
        pbar.set_description(f"Top-1 Acc: {num_correct_top1 / num_total * 100.0:.2f}%, Top-5 Acc: {num_correct_top5 / num_total * 100.0:.2f}%")

        if num_total >= max_samples: break

    accuracy_top1 = num_correct_top1 / num_total * 100.0
    accuracy_top5 = num_correct_top5 / num_total * 100.0

    return {
        "top1_accuracy": accuracy_top1,
        "top5_accuracy": accuracy_top5
    }