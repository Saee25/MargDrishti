import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = self.model.gradcam_target_layer()
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self.save_activation)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()
        
        # Get activations and gradients
        gradients = self.gradients[0].cpu().data.numpy() # (C, H, W)
        activations = self.activations[0].cpu().data.numpy() # (C, H, W)
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2)) # (C,)
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32) # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU on CAM
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
            
        # Resize to match input image size (H, W)
        img_h, img_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam_resized = F.interpolate(cam_tensor, size=(img_h, img_w), mode='bilinear', align_corners=False)
        cam = cam_resized.squeeze().numpy()
        
        return cam

def apply_colormap_on_image(org_im: Image.Image, activation: np.ndarray) -> Image.Image:
    """
    Applies a custom lavender-to-purple-to-ochre colormap on the activation map
    and blends it with the original image.
    """
    # Create a custom colormap (lavender -> purple -> ochre)
    # lavender-200: #CDCBF0 (205, 203, 240)
    # purple-500: #7C5FA6 (124, 95, 166)
    # ochre: #C39A45 (195, 154, 69)
    
    heatmap = np.zeros((activation.shape[0], activation.shape[1], 3), dtype=np.uint8)
    
    for i in range(activation.shape[0]):
        for j in range(activation.shape[1]):
            v = activation[i, j]
            if v < 0.5:
                # Interpolate between lavender and purple
                f = v * 2.0
                r = 205 + f * (124 - 205)
                g = 203 + f * (95 - 203)
                b = 240 + f * (166 - 240)
            else:
                # Interpolate between purple and ochre
                f = (v - 0.5) * 2.0
                r = 124 + f * (195 - 124)
                g = 95 + f * (154 - 95)
                b = 166 + f * (69 - 166)
            heatmap[i, j] = [int(r), int(g), int(b)]
            
    heatmap_img = Image.fromarray(heatmap)
    
    # Resize just in case it doesn't match perfectly, though it should
    if heatmap_img.size != org_im.size:
        heatmap_img = heatmap_img.resize(org_im.size, Image.Resampling.LANCZOS)
        
    # Blend with original image
    # Use alpha blending, say 0.6 for original, 0.4 for heatmap
    org_rgb = org_im.convert('RGB')
    blended = Image.blend(org_rgb, heatmap_img, alpha=0.5)
    return blended
