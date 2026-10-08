import torch
import torch.nn as nn
from ml.models.resnet import ResNet50Classifier

def test_resnet_output_shape():
    model = ResNet50Classifier(num_classes=75, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 75)

def test_resnet_frozen_stage():
    model = ResNet50Classifier(num_classes=75, pretrained=False)
    model.freeze_backbone()
    
    # Check that only fc parameters are trainable
    for name, param in model.named_parameters():
        if name.startswith("backbone.fc."):
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False
            
    # Check that BatchNorm layers would be put in eval mode
    # Actually, we rely on the trainer for this, but we can check the logic:
    def set_frozen_batchnorm_to_eval(m):
        for name, module in m.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if all(not p.requires_grad for p in module.parameters()):
                    module.eval()
                    
    model.train() # default
    set_frozen_batchnorm_to_eval(model)
    
    # Check a specific BN layer
    assert model.backbone.bn1.training is False

def test_resnet_unfreeze_all():
    model = ResNet50Classifier(num_classes=75, pretrained=False)
    model.freeze_backbone()
    model.unfreeze_all()
    
    for param in model.parameters():
        assert param.requires_grad is True

def test_resnet_parameter_groups():
    model = ResNet50Classifier(num_classes=75, pretrained=False)
    groups = model.get_parameter_groups(lr=0.1, lr_backbone=0.01, lr_head=0.05, weight_decay=1e-4)
    
    assert len(groups) == 2
    assert groups[0]["lr"] == 0.01
    assert groups[0]["weight_decay"] == 1e-4
    assert groups[1]["lr"] == 0.05
    assert groups[1]["weight_decay"] == 1e-4
    
    # Head params are in the second group
    assert any(param.shape == model.backbone.fc[-1].weight.shape for param in groups[1]["params"])
