import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A convolutional block with optional batch normalization.
    """
    def __init__(self, in_ch, out_ch, n_convs, use_bn):
        super().__init__()
        layers = []
        for i in range(n_convs):
            in_c = in_ch if i == 0 else out_ch
            bias = not use_bn
            layers.append(nn.Conv2d(in_c, out_ch, kernel_size=3, padding=1, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class PlainCNN(nn.Module):
    def __init__(self, num_classes, use_bn=False, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, n_convs=1, use_bn=use_bn),
            ConvBlock(32, 64, n_convs=1, use_bn=use_bn),
            ConvBlock(64, 128, n_convs=1, use_bn=use_bn)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def gradcam_target_layer(self):
        return self.features[-1].block[-2]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
    def describe(self):
        return [
            {
                "name": "ConvBlock 1",
                "type": "Sequential",
                "output_shape": "(32, 32, 32)",
                "parameter_count": sum(p.numel() for p in self.features[0].parameters()),
                "description": "Extracts basic features like edges and colors."
            },
            {
                "name": "ConvBlock 2",
                "type": "Sequential",
                "output_shape": "(64, 16, 16)",
                "parameter_count": sum(p.numel() for p in self.features[1].parameters()),
                "description": "Combines basic features into simpler shapes and textures."
            },
            {
                "name": "ConvBlock 3",
                "type": "Sequential",
                "output_shape": "(128, 8, 8)",
                "parameter_count": sum(p.numel() for p in self.features[2].parameters()),
                "description": "Builds complex patterns representing traffic sign parts."
            },
            {
                "name": "Flatten",
                "type": "Flatten",
                "output_shape": "(8192,)",
                "parameter_count": 0,
                "description": "Transforms the 3D feature maps into a 1D vector."
            },
            {
                "name": "Dense 1",
                "type": "Linear + ReLU + Dropout",
                "output_shape": "(256,)",
                "parameter_count": sum(p.numel() for p in self.classifier[1].parameters()),
                "description": "Learns high-level combinations of features, discarding some to prevent memorisation."
            },
            {
                "name": "Dense 2",
                "type": "Linear",
                "output_shape": f"({self.classifier[-1].out_features},)",
                "parameter_count": sum(p.numel() for p in self.classifier[-1].parameters()),
                "description": "Outputs the final classification scores for each traffic sign class."
            }
        ]

class MargNet(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, n_convs=2, use_bn=True),
            ConvBlock(32, 64, n_convs=2, use_bn=True),
            ConvBlock(64, 128, n_convs=2, use_bn=True),
            ConvBlock(128, 256, n_convs=2, use_bn=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def gradcam_target_layer(self):
        return self.features[-1].block[-2]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def describe(self):
        return [
            {
                "name": "ConvBlock 1",
                "type": "Sequential",
                "output_shape": "(32, 32, 32)",
                "parameter_count": sum(p.numel() for p in self.features[0].parameters()),
                "description": "Extracts primary visual elements like lines, edges and basic colours."
            },
            {
                "name": "ConvBlock 2",
                "type": "Sequential",
                "output_shape": "(64, 16, 16)",
                "parameter_count": sum(p.numel() for p in self.features[1].parameters()),
                "description": "Detects intermediate shapes such as circles, triangles and corners."
            },
            {
                "name": "ConvBlock 3",
                "type": "Sequential",
                "output_shape": "(128, 8, 8)",
                "parameter_count": sum(p.numel() for p in self.features[2].parameters()),
                "description": "Identifies complex symbols and patterns specific to traffic signs."
            },
            {
                "name": "ConvBlock 4",
                "type": "Sequential",
                "output_shape": "(256, 4, 4)",
                "parameter_count": sum(p.numel() for p in self.features[3].parameters()),
                "description": "Captures highly abstract, deep semantic features of the entire sign."
            },
            {
                "name": "Global Average Pooling",
                "type": "AdaptiveAvgPool2d + Flatten",
                "output_shape": "(256,)",
                "parameter_count": 0,
                "description": "Summarises each feature map into a single number, greatly reducing parameters and preventing overfitting."
            },
            {
                "name": "Dense",
                "type": "Dropout + Linear",
                "output_shape": f"({self.classifier[-1].out_features},)",
                "parameter_count": sum(p.numel() for p in self.classifier[-1].parameters()),
                "description": "Computes the final probabilities for each traffic sign class."
            }
        ]
