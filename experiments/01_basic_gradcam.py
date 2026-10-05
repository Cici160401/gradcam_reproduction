from src.gradcam import GradCAM
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image
from src.visualization import create_heatmap, normalize_cam,resize_cam,  overlay_heatmap, show_gradcam


weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

model.eval()

target_layer = model.layer4[2].conv3

#image = Image.open("../assets/examples/cat.jpg").convert("RGB")
image = Image.open("assets/examples/cat.jpg").convert("RGB")


preprocess = weights.transforms()


input_tensor = preprocess(image).unsqueeze(0)

output = model(input_tensor)

target_class = output.argmax(dim=1).item()

grad_cam = GradCAM(model, target_layer)

cam = grad_cam.generate_cam(input_tensor, target_class)

print("Input shape:", input_tensor.shape)
print("Target class:", target_class)
print("CAM shape:", cam.shape)

normalized_cam = normalize_cam(cam)

resized_cam = resize_cam(normalized_cam)

heatmap = create_heatmap(resized_cam)

overlay = overlay_heatmap(image, heatmap)

show_gradcam(image, heatmap, overlay)

