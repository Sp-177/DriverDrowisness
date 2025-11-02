import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Squeeze-and-Excitation (Attention Block) ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


# --- Main CNN Model ---
class DrowsinessCNN_Best(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super(DrowsinessCNN_Best, self).__init__()

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        # --- Feature extractor with attention ---
        self.features = nn.Sequential(
            conv_block(3, 32),
            SEBlock(32),
            conv_block(32, 64),
            SEBlock(64),
            conv_block(64, 128),
            SEBlock(128),
            conv_block(128, 256),
            SEBlock(256)
        )

        # --- Global Average Pooling ---
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = DrowsinessCNN_Best(num_classes=6)
    dummy = torch.randn(4, 3, 128, 128)
    out = model(dummy)
    print("✅ Output shape:", out.shape)
    print("✅ Total parameters:", sum(p.numel() for p in model.parameters()) // 1_000, "K")
