import torch
import torch.nn.functional as F 

class GradCAM:
    def __init__(self,model,target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self.save_activation)

    #aquí guardaremos Aᵏ
    def save_activation(self, module, input, output):
        self.activations = output
        #register hook to save gradients during backward pass
        output.register_hook(self.save_gradient)

    #aquí guardaremos ∂y/∂Aᵏ
    def save_gradient(self, gradient):
        self.gradients = gradient

    def generate_cam(self, input_tensor, target_class):
        #forward pass
        output = self.model(input_tensor)
        #score yc for the target class
        yc= output[0,target_class]
        #backward pass to get activation and gradients
        yc.backward()
        #compute weights αᵏ, which are the global average of the gradients
        #we reduce the gradients across the spatial dimensions (H, W) to get a single weight for each channel
        weights = self.gradients.mean(dim=(2, 3))
        #we need to multiply the weights with the activations and sum across the channels to get the CAM
        #in order to do this, we need to unsqueeze the weights to match the dimensions of the activations
        #we use the -1 index to unsqueeze the weights to the last dimension, so that we can multiply them with the activations
        #we do it 2 times so we can add two 1s to the dimensions of the weights, so that we can multiply them with the activations
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        weighted_activations = weights * self.activations
        #now we eliminate the feature maps dimension by summing all the 2048 feature maps, so we sum across the channel dimension (1)
        cam = weighted_activations.sum(dim=1)
        #we apply ReLU to the CAM to keep only positive values, as negative values are not informative for the class activation
        cam = F.relu(cam)
        return cam

# TODO: Engineering improvements
#
# 1. Clear model gradients before each backward pass
#    - Use model.zero_grad() before computing gradients for a new input.
#    - Prevent gradients stored in model parameters from accumulating
#      across multiple Grad-CAM calls.
#
# 2. Detach the final CAM from the computation graph
#    - The returned CAM does not need to remain connected to autograd
#      once Grad-CAM has been computed.
#
# 3. Store the forward-hook handle
#    - register_forward_hook() returns a handle.
#    - Store it so the hook can be removed when Grad-CAM is no longer used.
#
# 4. Add a method to remove hooks
#    - For example, a remove_hooks() method.
#    - Prevent old hooks from remaining attached to the model.
#
# 5. Reset stored activations and gradients between calls
#    - Avoid accidentally using tensors captured from a previous forward/
#      backward pass.
#
# 6. Support target_class=None
#    - If no target class is provided, automatically use the model's
#      predicted class (argmax).
#
# 7. Handle device consistency
#    - Make sure the model and input tensor are on the same device
#      (CPU/CUDA).
#
# 8. Decide how batch inputs should be handled
#    - The current implementation assumes a single image because it uses
#      output[0, target_class].
#    - Either explicitly enforce batch size = 1 or extend the implementation
#      to support multiple images.
#
# 9. Add input and state validation
#    - Check that target_class is valid.
#    - Check that activations and gradients were actually captured.
#    - Produce clear errors when an incompatible target layer is used.
#
# 10. Separate raw Grad-CAM from visualization/post-processing
#     - Keep generate_cam() responsible for the core Grad-CAM computation.
#     - Put normalization, resizing, colormap and image overlay in separate
#       utilities.
#
# 11. Add type hints and docstrings
#     - Document model, target_layer, input_tensor, target_class and
#       the returned CAM.
#     - Make the implementation easier to understand and reuse.
#
# 12. Add automated tests
#     - Verify expected tensor shapes.
#     - Compare channel weights and raw CAM against pytorch-grad-cam.
#     - Test multiple images/classes instead of validating only one example.