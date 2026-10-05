import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def normalize_cam(cam, eps=1e-8):
    """
    Normalize a CAM to the range [0, 1].

    Parameters
    ----------
    cam : torch.Tensor
        Raw CAM tensor.
    eps : float
        Small value used to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Normalized CAM.
    """
    cam_min = cam.min()
    cam_max = cam.max()

    normalized_cam = (cam - cam_min) / (cam_max - cam_min + eps)

    return normalized_cam


def resize_cam(cam, size=(224, 224)):
    """
    Resize a CAM using bilinear interpolation.

    Parameters
    ----------
    cam : torch.Tensor
        CAM with shape [1, H, W].
    size : tuple
        Desired spatial output size.

    Returns
    -------
    torch.Tensor
        Resized CAM with shape [H_out, W_out].
    """

    # F.interpolate expects [B, C, H, W].
    cam = cam.unsqueeze(1)

    resized_cam = F.interpolate(
        cam,
        size=size,
        mode="bilinear",
        align_corners=False
    )

    # Remove batch and channel dimensions.
    resized_cam = resized_cam[0, 0]

    return resized_cam


def create_heatmap(cam, cmap="jet"):
    """
    Convert a normalized CAM into an RGB heatmap.

    Parameters
    ----------
    cam : torch.Tensor
        Normalized 2D CAM with values in [0, 1].
    cmap : str
        Matplotlib colormap.

    Returns
    -------
    numpy.ndarray
        RGB heatmap with values in [0, 1].
    """

    cam_numpy = cam.detach().cpu().numpy()

    colormap = plt.get_cmap(cmap)

    # Colormap returns RGBA, so keep only RGB.
    heatmap = colormap(cam_numpy)[..., :3]

    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.5):
    """
    Overlay a heatmap on an RGB image.

    Parameters
    ----------
    image : PIL.Image.Image
        Original RGB image.
    heatmap : numpy.ndarray
        RGB heatmap with values in [0, 1].
    alpha : float
        Heatmap opacity.

    Returns
    -------
    numpy.ndarray
        RGB overlay with values in [0, 1].
    """

    image = image.resize(
        (heatmap.shape[1], heatmap.shape[0])
    )

    image_numpy = np.asarray(image).astype(np.float32) / 255.0

    overlay = (
        (1 - alpha) * image_numpy
        + alpha * heatmap
    )

    overlay = np.clip(overlay, 0, 1)

    return overlay


def show_gradcam(image, heatmap, overlay, class_name=None):
    """
    Display the original image, Grad-CAM heatmap and overlay.
    """

    title = "Grad-CAM"

    if class_name is not None:
        title = f"Grad-CAM — {class_name}"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(title)
    axes[2].axis("off")

    print("Showing Grad-CAM...")
    plt.show()
    print("Visualization closed.")
    

    plt.tight_layout()
    plt.show()

   
# TODO: Visualization and experiment engineering improvements
#
# 1. Make visualization independent of a fixed image size
#    - Do not assume that the visualization must always be 224x224.
#    - Allow resize_cam() to receive the desired output size dynamically.
#
# 2. Preserve the exact image used by the model
#    - The model preprocessing may resize and center-crop the original image.
#    - The Grad-CAM overlay should ideally correspond to the actual spatial
#      input seen by the model, not simply the original image resized afterward.
#
# 3. Add validation for CAM shapes
#    - Check that normalize_cam() and resize_cam() receive valid tensor shapes.
#    - Produce clear errors for incompatible inputs.
#
# 4. Handle constant CAMs safely
#    - If cam.max() == cam.min(), min-max normalization becomes degenerate.
#    - Define explicitly how this case should be handled.
#
# 5. Validate alpha values
#    - Ensure overlay alpha stays in the valid [0, 1] range.
#
# 6. Support configurable colormaps
#    - Keep "jet" as an option, but allow other matplotlib colormaps.
#
# 7. Separate visualization from display
#    - Functions that generate heatmaps/overlays should return data.
#    - show_gradcam() should only be responsible for displaying results.
#
# 8. Add a function for saving results
#    - Save heatmap and overlay without requiring an interactive matplotlib window.
#    - Useful for experiments and automated evaluation.
#
# 9. Add type hints
#    - Specify expected types for torch.Tensor, PIL.Image and numpy.ndarray.
#
# 10. Improve docstrings
#     - Document expected shapes, value ranges and return types for every function.
#
# 11. Add tests for visualization utilities
#     - Verify normalization produces values in [0, 1].
#     - Verify resize output dimensions.
#     - Verify heatmap and overlay shapes.
#
# 12. Avoid unnecessary tensor/NumPy conversions
#     - Keep operations in PyTorch until NumPy is actually required for
#       visualization.