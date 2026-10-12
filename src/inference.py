
import numpy as np
import torch
import matplotlib

from torchvision.models import ResNet50_Weights, resnet50

from src.gradcam import GradCAM
from src.visualization import normalize_cam, resize_cam


# Load pretrained ResNet50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Define preprocessing and target layer
preprocess = weights.transforms()
target_layer = model.layer4[2].conv3

# Initialize Grad-CAM
gradcam = GradCAM(model, target_layer)

# ImageNet class names
class_names = weights.meta["categories"]

# Normalization parameters
mean = torch.tensor(preprocess.mean).view(3, 1, 1)
std = torch.tensor(preprocess.std).view(3, 1, 1)


def predict_and_explain(image):
    """
    Predict an image class and generate a Grad-CAM explanation.

    Args:
        image: PIL.Image.Image

    Returns:
        predicted_class: str
        confidence: float
        heatmap: np.ndarray
        overlay: np.ndarray
    """

    # 1. Prepare input image
    image = image.convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    # 2. Run classification and Grad-CAM
    # Gradients must remain enabled for Grad-CAM.
    with torch.enable_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)
        target_class = output.argmax(dim=1).item()

        confidence = probabilities[0, target_class].item()

        model.zero_grad(set_to_none=True)

        # Raw CAM: [1, 7, 7]
        cam = gradcam.generate_cam(
            input_tensor,
            target_class
        )

    # 3. Prepare CAM for visualization
    cam = cam.detach()

    cam = resize_cam(
        cam,
        size=(input_tensor.shape[-2], input_tensor.shape[-1])
    )

    cam = normalize_cam(cam)

    heatmap = cam.cpu().numpy()

    # 4. Recover the image seen by ResNet50
    display_tensor = (
        input_tensor[0].detach().cpu() * std + mean
    )

    display_image = (
        display_tensor
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    # 5. Create colored heatmap
    colormap = matplotlib.colormaps["jet"]

    colored_heatmap = colormap(heatmap)[..., :3]

    # 6. Create intensity-dependent overlay
    # Low CAM values remain close to the original image.
    alpha = 0.6 * heatmap[..., np.newaxis]

    overlay = (
        (1 - alpha) * display_image
        + alpha * colored_heatmap
    )

    overlay = np.clip(overlay, 0, 1)

    # 7. Convert results to uint8
    heatmap_image = (
        colored_heatmap * 255
    ).astype(np.uint8)

    overlay_image = (
        overlay * 255
    ).astype(np.uint8)

    # 8. Get predicted class name
    predicted_class = class_names[target_class]

    return (
        predicted_class,
        confidence,
        heatmap_image,
        overlay_image
    )
