#!/usr/bin/env python3
"""Train a two-class mushroom image classifier from folder-organized data."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("ANNOTATION_BATCHES/hiratake_dateset"))
    parser.add_argument("--test-dir", type=Path, default=Path("ANNOTATION_BATCHES/test_datelist"))
    parser.add_argument("--output", type=Path, default=Path("classifier_output"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        raise RuntimeError("The test dataset contains no images.")
    return total_loss / total, correct / total


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is not available.")
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    if args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")

    train_transform, test_transform = build_transforms()
    train_data = datasets.ImageFolder(args.train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(args.test_dir, transform=test_transform)
    if train_data.classes != test_data.classes:
        raise RuntimeError(
            "Train/test class folders differ: "
            f"{train_data.classes!r} vs {test_data.classes!r}"
        )
    if len(train_data.classes) != 2:
        raise RuntimeError(f"Expected exactly two class folders, found {train_data.classes!r}.")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    args.output.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    history = []
    print(
        f"Training {len(train_data)} images, testing {len(test_data)} images, "
        f"classes={train_data.classes}, device={device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        test_loss, test_accuracy = evaluate(model, test_loader, device, criterion)
        train_loss = running_loss / len(train_data)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "test_accuracy": test_accuracy}
        )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"test_accuracy={test_accuracy:.4f}"
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": train_data.classes,
                    "image_size": 224,
                    "model": "resnet18",
                },
                args.output / "best_model.pth",
            )

    (args.output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Best model saved to {args.output / 'best_model.pth'}")


if __name__ == "__main__":
    main()
