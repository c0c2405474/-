#!/usr/bin/env python3
"""Compare classifier performance with and without brightness correction."""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision import models, transforms


def adjust_brightness(image: Image.Image) -> Image.Image:
    image = ImageOps.autocontrast(image)
    return ImageEnhance.Brightness(image).enhance(1.35)


def predict(model, image, image_transform, device):
    with torch.no_grad():
        probabilities = model(image_transform(image).unsqueeze(0).to(device)).softmax(dim=1)[0]
    index = int(probabilities.argmax())
    return index, probabilities[index].item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("classifier_output/best_model.pth"))
    parser.add_argument("--test-dir", type=Path, default=Path("ANNOTATION_BATCHES/test_datelist"))
    parser.add_argument("--output", type=Path, default=Path("classifier_output/brightness_comparison.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    classes = checkpoint["classes"]
    class_to_index = {name: index for index, name in enumerate(classes)}
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    image_transform = transforms.Compose(
        [
            transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    rows = []
    for label_dir in sorted(args.test_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        if label_dir.name not in class_to_index:
            raise RuntimeError(f"Unknown class folder: {label_dir}")
        label_index = class_to_index[label_dir.name]
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            original = Image.open(image_path).convert("RGB")
            original_index, original_confidence = predict(
                model, original, image_transform, device
            )
            corrected_index, corrected_confidence = predict(
                model, adjust_brightness(original), image_transform, device
            )
            rows.append(
                {
                    "image": str(image_path),
                    "label": label_dir.name,
                    "original_prediction": classes[original_index],
                    "original_confidence": f"{original_confidence:.6f}",
                    "corrected_prediction": classes[corrected_index],
                    "corrected_confidence": f"{corrected_confidence:.6f}",
                    "original_correct": original_index == label_index,
                    "corrected_correct": corrected_index == label_index,
                    "prediction_changed": original_index != corrected_index,
                }
            )

    if not rows:
        raise RuntimeError(f"No supported images found in {args.test_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    original_accuracy = sum(row["original_correct"] for row in rows) / len(rows)
    corrected_accuracy = sum(row["corrected_correct"] for row in rows) / len(rows)
    changed = sum(row["prediction_changed"] for row in rows)
    original_confidence = sum(float(row["original_confidence"]) for row in rows) / len(rows)
    corrected_confidence = sum(float(row["corrected_confidence"]) for row in rows) / len(rows)
    print(f"Images: {len(rows)}")
    print(f"Original accuracy: {original_accuracy:.2%}")
    print(f"Corrected accuracy: {corrected_accuracy:.2%}")
    print(f"Original average confidence: {original_confidence:.2%}")
    print(f"Corrected average confidence: {corrected_confidence:.2%}")
    print(f"Predictions changed: {changed}")
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
