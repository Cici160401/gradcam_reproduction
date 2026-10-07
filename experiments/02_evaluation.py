#¿Mi implementación es correcta respecto a la referencia? 
# (la referencia es la implementación de la librería pytorch-grad-cam)

import torch

from src.gradcam import GradCAM as CustomGradCAM
from pytorch_grad_cam import GradCAM as ReferenceGradCAM
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.visualization import normalize_cam, resize_cam
import torch.nn.functional as F

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)


model.eval()

target_layer = model.layer4[2].conv3

#image = Image.open("../assets/examples/.jpg").convert("RGB")
image = Image.open("assets/examples/iguana.jpg").convert("RGB")


preprocess = weights.transforms()


input_tensor = preprocess(image).unsqueeze(0)

output = model(input_tensor)

target_class = output.argmax(dim=1).item()

targets = [ClassifierOutputTarget(target_class)]

custom_gradcam = CustomGradCAM(model, target_layer)
custom_cam = custom_gradcam.generate_cam(input_tensor, target_class)

reference_gradcam = ReferenceGradCAM(
    model=model,
    target_layers=[target_layer]
)
reference_cam = reference_gradcam(input_tensor, targets)

reference_activations = reference_gradcam.activations_and_grads.activations[0]
reference_gradients = reference_gradcam.activations_and_grads.gradients[0]


weights_custom = custom_gradcam.gradients.mean(dim=(2, 3))
weights_reference = reference_gradcam.activations_and_grads.gradients[0].mean(dim=(2, 3))
weights_difference = torch.abs(weights_custom - weights_reference)

max_weight_difference = weights_difference.max()

weights_final = weights_custom.unsqueeze(-1).unsqueeze(-1)
weights_reference_final = weights_reference.unsqueeze(-1).unsqueeze(-1)

custom_cam_final = (weights_final * reference_activations).sum(dim=1)
reference_cam_final = (weights_reference_final * reference_activations).sum(dim=1)

reference_raw_cam = F.relu(reference_cam_final)

raw_difference = torch.abs(custom_cam_final - reference_cam_final)
mae = raw_difference.mean()

custom_flat = custom_cam_final.flatten()
reference_flat = reference_cam_final.flatten()

raw_correlation = torch.corrcoef(
    torch.stack([
        custom_flat,
        reference_flat
    ])
)[0, 1]

# ---------------------------------------------------------
#post procesamiento de los CAMs para compararlos con la salida de pytorch-grad-cam
#--------------------------------------------------------

# Process custom Grad-CAM
custom_normalized_cam = normalize_cam(custom_cam_final)
custom_processed_cam = resize_cam(custom_normalized_cam)

# pytorch-grad-cam already returns the processed CAM
# with the spatial resolution of the input image.
reference_processed_cam = torch.from_numpy(reference_cam[0])

print("Custom processed CAM shape:", custom_processed_cam.shape)
print("Reference processed CAM shape:", reference_processed_cam.shape)

# Compare both processed CAMs
processed_difference = torch.abs(
    custom_processed_cam - reference_processed_cam
)

processed_mae = processed_difference.mean()
processed_max_difference = processed_difference.max()

# Pearson correlation
custom_processed_flat = custom_processed_cam.flatten()
reference_processed_flat = reference_processed_cam.flatten()

processed_correlation = torch.corrcoef(
    torch.stack([
        custom_processed_flat,
        reference_processed_flat
    ])
)[0, 1]

print("\n--- Processed CAM comparison ---")
print("Processed CAM MAE:", processed_mae.item())
print("Processed CAM max difference:", processed_max_difference.item())
print("Processed CAM correlation:", processed_correlation.item())


print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
print("Target class:", target_class)
print("Custom CAM shape:", custom_cam.shape)
print("Reference CAM shape:", reference_cam.shape)
print("Reference activations:", reference_activations.shape)
print("Reference gradients:", reference_gradients.shape)
print("Custom weights shape:", weights_custom.shape)
print("Reference weights shape:", weights_reference.shape)
print("Reference weights final shape:", weights_reference_final.shape)
print("Custom weights final shape:", weights_final.shape)
print("Difference in weights:", weights_difference.shape)
print("Maximum weight difference:", max_weight_difference.item())
print("Custom CAM final shape:", custom_cam_final.shape)
print("Reference CAM final shape:", reference_cam_final.shape)
print("Mean absolute error (MAE):", mae.item())
print("Maximum raw difference:", raw_difference.max().item())
print("Raw CAM correlation:", raw_correlation.item())
