#!/usr/bin/env python3
"""
Phase 3.1 • Model Architecture for BFRB Detection
Defines a compact, high-performance MLP for 893-dim feature inputs.
"""

import torch.nn as nn

class BFRBMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512,       256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256,       128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
