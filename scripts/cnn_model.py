import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Squeeze-and-Excitation (Attention Block)
# ==========================================================
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


# ==========================================================
# CBAM (Convolutional Block Attention Module)
# ==========================================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel_att(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.spatial_att(spatial)
        return x * spatial


# ==========================================================
# Enhanced CNN Model for Drowsiness Detection
# ==========================================================
class DrowsinessCNN(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super(DrowsinessCNN_Pro, self).__init__()

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        # Feature extraction backbone with attention
        self.features = nn.Sequential(
            conv_block(3, 32),
            CBAM(32),
            conv_block(32, 64),
            CBAM(64),
            conv_block(64, 128),
            CBAM(128),
            conv_block(128, 256),
            CBAM(256)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
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


# ==========================================================
# Model Summary Test
# ==========================================================
if __name__ == "__main__":
    model = DrowsinessCNN_Pro(num_classes=6)
    dummy = torch.randn(4, 3, 128, 128)
    out = model(dummy)
    total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
    print("✅ Output shape:", out.shape)
    print(f"✅ Total parameters: {total_params:.2f}M")
