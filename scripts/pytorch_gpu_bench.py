#!/usr/bin/env python3
"""
PyTorch GPU Benchmark — trains a ResNet-18 on synthetic data to stress the GPU.

Usage: python pytorch_gpu_bench.py <num_epochs>

Runs the specified number of training epochs.
Prints throughput (images/sec) and GPU utilisation stats.
"""

import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Configuration ────────────────────────────────────────────
BATCH_SIZE = 64
IMAGE_SIZE = 224
NUM_CLASSES = 1000
NUM_SAMPLES = 2048  # synthetic dataset size (kept in GPU memory)


def build_model():
    """Build a ResNet-18 and move to GPU."""
    # Use torchvision if available, else a simple conv model
    try:
        from torchvision.models import resnet18

        model = resnet18(num_classes=NUM_CLASSES)
    except ImportError:
        # Fallback: lightweight conv network that still stresses GPU
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, NUM_CLASSES),
        )
    return model.cuda()


def build_dataloader():
    """Create a synthetic dataset of random images + labels on GPU."""
    images = torch.randn(NUM_SAMPLES, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)


def main():
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"=== PyTorch GPU Benchmark — running {num_epochs} epochs ===")

    model = build_model()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loader = build_dataloader()

    model.train()
    start = time.time()
    total_images = 0

    for epoch in range(1, num_epochs + 1):
        epoch_images = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_images += images.size(0)

        total_images += epoch_images
        elapsed = time.time() - start
        throughput = total_images / elapsed if elapsed > 0 else 0
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"[Epoch {epoch}/{num_epochs}] {epoch_images} images | "
            f"Total: {total_images} | {throughput:.1f} img/s | "
            f"Peak GPU mem: {mem_used:.2f} GB | "
            f"Elapsed: {elapsed:.0f}s"
        )

    elapsed = time.time() - start
    print(f"\n=== Completed {num_epochs} epochs, {total_images} images in {elapsed:.1f}s ===")
    print(f"=== Throughput: {total_images / elapsed:.1f} images/sec ===")


if __name__ == "__main__":
    main()
