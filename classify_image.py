#!/usr/bin/env python3
"""Classify one or more mushroom images with a trained classifier."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision import models, transforms


def adjust_brightness(image: Image.Image) -> Image.Image:
    """Brighten a dark image without requiring an additional model."""
    image = ImageOps.autocontrast(image)
    return ImageEnhance.Brightness(image).enhance(1.35)


def is_dark(image: Image.Image, threshold: float) -> bool:
    grayscale = ImageOps.grayscale(image)
    histogram = grayscale.histogram()
    mean_brightness = sum(value * count for value, count in enumerate(histogram))
    mean_brightness /= 255 * grayscale.width * grayscale.height
    return mean_brightness < threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("classifier_output/best_model.pth"))
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    enhancement = parser.add_mutually_exclusive_group()
    enhancement.add_argument(
        "--enhance",
        action="store_true",
        help="Apply brightness correction to every image.",
    )
    enhancement.add_argument(
        "--auto-enhance",
        action="store_true",
        help="Correct only images whose average brightness is below the threshold.",
    )
    parser.add_argument(
        "--dark-threshold",
        type=float,
        default=0.40,
        help="Average brightness threshold for --auto-enhance (default: 0.40).",
    )
    args = parser.parse_args()
    if not 0 < args.dark_threshold < 1:
        raise ValueError("--dark-threshold must be between 0 and 1.")

    checkpoint = torch.load(args.model, map_location="cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    enhancer = None
    if args.enhance or args.auto_enhance:
        enhancer = adjust_brightness

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    image_transform = transforms.Compose(
        [
            transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    with torch.no_grad():
        for image_path in args.input:
            image = Image.open(image_path).convert("RGB")
            corrected = args.enhance or (
                args.auto_enhance and is_dark(image, args.dark_threshold)
            )
            if corrected:
                image = enhancer(image)
            probabilities = model(image_transform(image).unsqueeze(0).to(device)).softmax(dim=1)[0]
            index = int(probabilities.argmax())
            confidence = probabilities[index].item() * 100
            mode = "明るさ補正あり" if corrected else "明るさ補正なし"
            print(
                f"{image_path} [{mode}]："
                f"{checkpoint['classes'][index]}（信頼度: {confidence:.2f}%）"
            )


if __name__ == "__main__":
    main()
