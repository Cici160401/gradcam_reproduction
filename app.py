import gradio as gr
from src.inference import predict_and_explain


def run_gradcam(image):
    if image is None:
        raise gr.Error("Por favor suba uma imagem para continuar.")

    return predict_and_explain(image)


demo = gr.Interface(
    fn=run_gradcam,
    inputs=gr.Image(type="pil", label="Subir una imagen"),
    outputs=[
        gr.Textbox(label="Clase Predicha"),
        gr.Number(label="Confianza de la Predicción"),
        gr.Image(label="Grad-CAM Heatmap"),
        gr.Image(label="Grad-CAM Overlay"),
    ],
    title="Grad-CAM: Explicaciones Visuales para CNNs",
    description=(
        "Explora como ResNet50 toma decisiones para clasificación de imágenes usando una implementación desde cero de Grad-CAM."
    ),
)

if __name__ == "__main__":
    demo.launch()