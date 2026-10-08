import os
import torch
from .custom_cnn import PlainCNN, MargNet

def build_model(name, variant, num_classes, **kwargs):
    """
    Registry for building models.
    custom v1 -> PlainCNN(use_bn=False)
    custom v2, v3, v4 -> PlainCNN(use_bn=True)
    custom v5 -> MargNet
    resnet50 -> Not implemented yet
    """
    if name == "custom":
        if variant == "v1":
            return PlainCNN(num_classes=num_classes, use_bn=False, dropout=kwargs.get("dropout", 0.5))
        elif variant in ["v2", "v3", "v4"]:
            return PlainCNN(num_classes=num_classes, use_bn=True, dropout=kwargs.get("dropout", 0.5))
        elif variant == "v5":
            return MargNet(num_classes=num_classes, dropout=kwargs.get("dropout", 0.4))
        else:
            raise ValueError(f"Unknown custom variant: {variant}")
    elif name == "resnet50":
        from .resnet import ResNet50Classifier
        pretrained = kwargs.get("pretrained", True)
        return ResNet50Classifier(num_classes=num_classes, dropout=kwargs.get("dropout", 0.3), pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {name}")

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_size_mb(model):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
    return size_mb
