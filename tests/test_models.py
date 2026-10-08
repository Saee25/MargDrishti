import torch
import pytest
from ml.models.factory import build_model, count_parameters
from ml.models.custom_cnn import PlainCNN, MargNet

def test_custom_models_output_shape():
    batch_size = 2
    for num_classes in [10, 75]:
        for variant in ["v1", "v2", "v3", "v4", "v5"]:
            model = build_model("custom", variant, num_classes)
            x = torch.randn(batch_size, 3, 64, 64)
            out = model(x)
            assert out.shape == (batch_size, num_classes)

def test_plaincnn_parameter_count():
    model = build_model("custom", "v1", num_classes=75)
    total, _ = count_parameters(model)
    
    # Hand calculation for PlainCNN(num_classes=75, use_bn=False):
    # Conv1: (3*3*3 + 1)*32 = 28 * 32 = 896
    # Conv2: (3*3*32 + 1)*64 = 289 * 64 = 18496
    # Conv3: (3*3*64 + 1)*128 = 577 * 128 = 73856
    # Linear1: (8192 + 1)*256 = 2097408
    # Linear2: (256 + 1)*75 = 19275
    # Total = 896 + 18496 + 73856 + 2097408 + 19275 = 2209931
    assert total == 2209931
    assert total > 2000000 and total < 2500000

def test_backward_pass():
    model = build_model("custom", "v5", num_classes=75)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"

def test_eval_mode_batch_1():
    model = build_model("custom", "v2", num_classes=75)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 75)

def test_gradcam_target_layer():
    model_plain = build_model("custom", "v1", num_classes=75)
    model_marg = build_model("custom", "v5", num_classes=75)
    
    assert hasattr(model_plain, "gradcam_target_layer")
    assert hasattr(model_marg, "gradcam_target_layer")
    
    # Should be a ReLU or similar activation layer inside the last block
    assert isinstance(model_plain.gradcam_target_layer, torch.nn.Module)
    assert isinstance(model_marg.gradcam_target_layer, torch.nn.Module)

def test_feature_map_shapes():
    # v1
    model = build_model("custom", "v1", num_classes=75)
    x = torch.randn(1, 3, 64, 64)
    out1 = model.features[0](x)
    assert out1.shape == (1, 32, 32, 32)
    out2 = model.features[1](out1)
    assert out2.shape == (1, 64, 16, 16)
    out3 = model.features[2](out2)
    assert out3.shape == (1, 128, 8, 8)

    # v5 (MargNet)
    model = build_model("custom", "v5", num_classes=75)
    x = torch.randn(1, 3, 64, 64)
    out1 = model.features[0](x)
    assert out1.shape == (1, 32, 32, 32)
    out2 = model.features[1](out1)
    assert out2.shape == (1, 64, 16, 16)
    out3 = model.features[2](out2)
    assert out3.shape == (1, 128, 8, 8)
    out4 = model.features[3](out3)
    assert out4.shape == (1, 256, 4, 4)
