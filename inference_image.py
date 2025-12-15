import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import models  # 确保 models.py 在同一目录下

# === 配置部分 (复用原脚本的 Mean/Std) ===
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}

def get_args():
    parser = argparse.ArgumentParser(description="Single Image Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the .pth checkpoint")
    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.GAPLModel(
        fe_path=None,
        proto_path=None,
        freeze_backbone=False
    )

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    print(f"[INFO] Loading checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.load_prototype(checkpoint['prototype'])
    model.to(device)
    model.eval()

    stat_from = "imagenet" 
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
    ])
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found at {args.image_path}")

    img = Image.open(args.image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(input_tensor).sigmoid().item()

    threshold = 0.5
    is_fake = score > threshold
    label_str = "Fake (AI-Generated)" if is_fake else "Real (Natural)"
    confidence = score if is_fake else 1 - score

    print("\n" + "="*40)
    print(f"🖼️  Target Image: {os.path.basename(args.image_path)}")
    print(f"🤖 Prediction:   {label_str}")
    print(f"📊 Confidence:   {confidence:.2%}")
    print(f"🔢 Raw Score:    {score:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()