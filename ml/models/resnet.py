import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    @property
    def gradcam_target_layer(self):
        # Target the last block of layer4
        return self.backbone.layer4[-1]

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers except the newly added head (the fc replacement)."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, lr, lr_backbone=None, lr_head=None, weight_decay=0.0):
        """Return parameter groups for different learning rates in backbone vs head."""
        if lr_backbone is None:
            lr_backbone = lr
        if lr_head is None:
            lr_head = lr
            
        backbone_params = []
        head_params = []
        
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc."):
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        return [
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay}
        ]

    def describe(self):
        return [
            {
                "name": "Stem",
                "type": "Conv2d + BatchNorm + ReLU + MaxPool",
                "output_shape": "(64, 56, 56)",
                "parameter_count": sum(p.numel() for p in self.backbone.conv1.parameters()) + sum(p.numel() for p in self.backbone.bn1.parameters()),
                "description": "Extracts basic initial features from the 224x224 input image."
            },
            {
                "name": "Layer 1 (3 blocks)",
                "type": "Residual Blocks",
                "output_shape": "(256, 56, 56)",
                "parameter_count": sum(p.numel() for p in self.backbone.layer1.parameters()),
                "description": "First set of residual blocks. A residual (skip) connection adds the block's input to its output, which prevents the vanishing gradient problem and lets very deep networks train effectively."
            },
            {
                "name": "Layer 2 (4 blocks)",
                "type": "Residual Blocks",
                "output_shape": "(512, 28, 28)",
                "parameter_count": sum(p.numel() for p in self.backbone.layer2.parameters()),
                "description": "Downsamples the spatial dimensions while doubling the number of feature channels."
            },
            {
                "name": "Layer 3 (6 blocks)",
                "type": "Residual Blocks",
                "output_shape": "(1024, 14, 14)",
                "parameter_count": sum(p.numel() for p in self.backbone.layer3.parameters()),
                "description": "The deepest feature extraction stage that captures complex structural parts of the traffic sign."
            },
            {
                "name": "Layer 4 (3 blocks)",
                "type": "Residual Blocks",
                "output_shape": "(2048, 7, 7)",
                "parameter_count": sum(p.numel() for p in self.backbone.layer4.parameters()),
                "description": "Produces the final high-level semantic feature maps."
            },
            {
                "name": "Global Average Pooling",
                "type": "AdaptiveAvgPool2d",
                "output_shape": "(2048, 1, 1)",
                "parameter_count": 0,
                "description": "Averages each 7x7 feature map into a single value, summarising the whole image."
            },
            {
                "name": "Head",
                "type": "Dropout + Linear",
                "output_shape": f"({self.backbone.fc[-1].out_features},)",
                "parameter_count": sum(p.numel() for p in self.backbone.fc.parameters()),
                "description": "Our custom classification head that maps the 2048 features to the final traffic sign classes."
            }
        ]
