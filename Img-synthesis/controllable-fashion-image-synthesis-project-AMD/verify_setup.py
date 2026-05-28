#!/usr/bin/env python3
"""Verification script - run after setup.sh"""
import sys, os
print("=" * 70)
print("🔍 Verifying Setup")
print("=" * 70)
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except:
    print("❌ PyTorch not installed")
print("=" * 70)
