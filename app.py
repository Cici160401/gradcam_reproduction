
import gradio as gr
import os

from src.inference import predict_and_explain


def run_gradcam(image):
    """
    Execute Grad-CAM and format results for the interface.
    """

    if image is None:
        raise gr.Error("Please upload an image.")

    predicted_class, confidence, heatmap, overlay = (
        predict_and_explain(image)
    )

    return (
        predicted_class,
        f"{confidence * 100:.2f}%",
        heatmap,
        overlay
    )


with gr.Blocks(title="Grad-CAM Explorer") as demo:

    # Header
    gr.Markdown(
        """
        # Grad-CAM Explorer

        ### Visual Explanations for Convolutional Neural Networks

        Explore how a pretrained ResNet50 classifies images
        using a from-scratch implementation of Grad-CAM.
        """
    )

    # Main section
    with gr.Row():

        # Image input
        with gr.Column(scale=1):

            input_image = gr.Image(
                type="pil",
                label="Upload an Image",
                height=350
            )

            generate_button = gr.Button(
                "Generate Grad-CAM",
                variant="primary"
            )

        # Prediction results
        with gr.Column(scale=1):

            predicted_class = gr.Textbox(
                label="Predicted Class",
                interactive=False
            )

            confidence = gr.Textbox(
                label="Prediction Confidence",
                interactive=False
            )

            gr.Markdown(
                """
                **How to interpret the results**

                Grad-CAM highlights regions that contribute
                positively to the predicted class score.

                Warmer colors indicate stronger positive
                contributions within the generated map.
                """
            )

    # Visualization section
    gr.Markdown("## Visual Explanations")

    with gr.Row():

        heatmap = gr.Image(
            label="Grad-CAM Heatmap",
            height=300,
            interactive=False
        )

        overlay = gr.Image(
            label="Grad-CAM Overlay",
            height=300,
            interactive=False
        )

    # Connect button to inference function
    generate_button.click(
        fn=run_gradcam,
        inputs=[input_image],
        outputs=[
            predicted_class,
            confidence,
            heatmap,
            overlay
        ],
        concurrency_limit=1
    )

    # Project information
    gr.Markdown(
        """
        ---

        ### About this project

        This application reproduces the **Grad-CAM** method
        introduced by Selvaraju et al. (2017).

        **Model:** ResNet50 pretrained on ImageNet.

        **Target layer:** `layer4[2].conv3`

        **Implementation:** Grad-CAM implemented from scratch
        using PyTorch forward and backward hooks.

        The visualization highlights spatial regions that
        contribute positively to the selected class score.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )