import numpy as np
import torch

from torchvision.models import ResNet50_Weights, resnet50
from src.gradcam import GradCAM
from src.visualization import normalize_cam, resize_cam


# 1. Cargar modelo preentrenado ResNet50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# 2. Definir la capa objetivo para Grad-CAM y preprocesamiento de la imagen
target_layer = model.layer4[2].conv3
preprocess = weights.transforms()

# 3. Inicializar Grad-CAM con el modelo y la capa objetivo
gradcam = GradCAM(model, target_layer)

# 4. Etiquetas o Labels de ImageNet
class_names = weights.meta["categories"]


}
}
def predict_and_explain(image):
    """
    Predict the image class and generate a Grad-CAM explanation.

    Args:
        image: PIL.Image.Image

    Returns:
        predicted_class: str
        confidence: float
        heatmap: np.ndarray
        overlay: np.ndarray
    """

    # 5. Preprocesar la imagen y convertirla a tensor
    image = image.convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    # 6. Predecir clase }
    with torch.enable_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)

        target_class = output.argmax(dim=1).item()
        confidence = probabilities[0, target_class].item()

        # 7. Generate raw Grad-CAM
        model.zero_grad(set_to_none=True)
        cam = gradcam.generate_cam(input_tensor, target_class)

    # 8. Normalize and resize CAM
    cam = cam.detach()
    cam = resize_cam(cam, size=(224, 224))
    cam = normalize_cam(cam)

    heatmap = cam.cpu().numpy()

    # 9. Reconstruct the image in the model's input coordinates
    mean = torch.tensor(weights.transforms().mean).view(3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(3, 1, 1)

    display_tensor = input_tensor[0].detach().cpu() * std + mean
    display_image = (
        display_tensor.clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    # 10. Create colored heatmap and overlay
    import matplotlib
    colormap = matplotlib.colormaps["jet"]

    colored_heatmap = colormap(heatmap)[..., :3]

    overlay = (
        0.5 * display_image
        + 0.5 * colored_heatmap
    )
    overlay = np.clip(overlay, 0, 1)

    # 11. Return results
    predicted_class = class_names[target_class]

    return (
        predicted_class,
        confidence,
        (colored_heatmap * 255).astype(np.uint8),
        (overlay * 255).astype(np.uint8),
    )