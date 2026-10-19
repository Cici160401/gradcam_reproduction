# Grad-CAM from Scratch: Paper Reproduction

## Overview

This repository is a reproduction of the paper Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. The Grad-CAM method was implemented from scratch to obtain the class-discriminative localization map built by weighting feature maps according to their importance for the target class score. The method was implemented step by step from the equations stated in the paper. I used the pre-trained ResNet50 with ImageNet to understand how Grad-CAM works internally, and the implementation was validated against the pytorch-grad-cam reference implementation.

## Paper

> Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).  
> [*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*](https://arxiv.org/abs/1610.02391).  
> Proceedings of the IEEE International Conference on Computer Vision (ICCV).

CNNs achieve strong performance but their predictions are difficult to interpret, this makes failures/biases harder to understand, so Grad-CAM provides spatial information about which regions contribute to a particular class.

## Objective of this reproduction

## Method
### 1. Model and input

I used a ResNet50 model pretrained on ImageNet. The model was used only for inference and was not fine-tuned or retrained. The input image was preprocessed using the transformations associated with the pretrained weights. After preprocessing, the image tensor had a shape of `[3, 224, 224]`. `unsqueeze(0)` was used to add the batch dimension to the input tensor because ResNet50 expects its input in batched form, resulting in a shape of `[1, 3, 224, 224]`.

![alt text](assets/examples/image-1.png)

### 2. Target convolutional 

I searched for one of the last convolutional layers of ResNet50 because deep convolutional layers keep spatial information  while they capture high-level semantic carachteristics. I selected model.layer4[2].conv3, which corresponds to the last convolutional layer of the final Bottleneck in layer4. The ouput is [1, 2048, 7, 7], where 2048 are the amount of total feature maps and 7x7 represents the spatial resolution of each feature map.

### 3. Forward pass and target class

I performed a forward pass, obtaining an output with shape [1, 1000], where 1000 corresponds to the number of classes in ImageNet. I used \(c\) to represent the index of the target class and \(y^c\) as the score assigned to that class by the model. Since Grad-CAM is class-discriminative, \(y^c\) is selected so that the gradients can be computed specifically with respect to the score of class \(c\). Following the Grad-CAM formulation, \(y^c\) corresponds to the class score before the softmax operation.

### 4. Capturing feature maps

To capture the activations from the target convolutional layer, I registered a forward hook to capture the output of model.layer4[2].conv3, which I stored in activations. These activations correspond to the feature maps \(A^k\) and are later weighted according to their importance for the target class to build the Grad-CAM localization map. The captured activations had a shape of [1, 2048, 7, 7], corresponding to 2048 feature maps with a spatial resolution of 7x7. 

### **5. Computing Gradients**

After selecting the target class score $y^c$, I performed backpropagation to obtain the gradients of the class score with respect to the activations of the target convolutional layer:

$$
\frac{\partial y^c}{\partial A_{ij}^{k}}
$$

These gradients represent how sensitive the score of class $c$ is to each activation in the feature maps. The resulting gradients have a shape of `[1, 2048, 7, 7]`, the same shape as the activations, because each spatial activation has its corresponding gradient.


### **6. Computing Channel Importance Weights**

After obtaining the gradients, I calculated the importance of each feature map for the target class. Following the Grad-CAM equation, the gradients of each feature map are averaged over its spatial dimensions:

$$
\alpha_k^c =
\frac{1}{Z}
\sum_i \sum_j
\frac{\partial y^c}{\partial A_{ij}^{k}}
$$

Since every feature map has a spatial size of $7 \times 7$, its 49 gradient values are averaged to obtain one importance weight $\alpha_k^c$ for each feature map.

This changes the shape from `[1, 2048, 7, 7]` to `[1, 2048]`, giving one importance weight for each of the 2048 feature maps.


### **7. Building the Grad-CAM Localization Map**

Once the importance weights were calculated, each feature map $A^k$ was multiplied by its corresponding weight $\alpha_k^c$. This gives more influence to the feature maps that are more important for the target class.

The weighted feature maps were then summed across the 2048 channels and ReLU was applied:

$$
L_{\mathrm{Grad-CAM}}^c =
\mathrm{ReLU}\left(
\sum_k \alpha_k^c A^k
\right)
$$

ReLU keeps the positive contributions to the target class and removes the negative values. After summing the 2048 weighted feature maps, the resulting Grad-CAM localization map has a spatial size of $7 \times 7$.


### **8. Visualization**

The $7 \times 7$ Grad-CAM localization map was first normalized between 0 and 1. It was then resized to $224 \times 224$, the same spatial size as the image used as input by ResNet50.

I used bilinear interpolation for the resize because the Grad-CAM output is a continuous heatmap rather than a discrete class mask. Finally, the resized heatmap was overlaid on the input image to visualize which regions had a higher positive contribution to the target class score.

It is important to note that resizing the heatmap from $7 \times 7$ to $224 \times 224$ does not add new spatial information. It only makes the original Grad-CAM map easier to visualize over the input image.


## **Results**

For the first experiment, ResNet50 predicted the input image as a `golden retriever`. The Grad-CAM localization map showed the strongest positive contributions mainly around regions of the dog's face.

The final heatmap was obtained from the original $7 \times 7$ Grad-CAM map and resized to $224 \times 224$ for visualization. The result shows where the model obtained positive evidence for the selected class, but it should not be interpreted as an exact segmentation of the object.

![Grad-CAM visualization for the golden retriever example](assets/examples/image-2.png)


## **Validation**

To verify the implementation, I compared my Grad-CAM result with the `pytorch-grad-cam` reference implementation using the same ResNet50 model, input image, target class, and target convolutional layer.

The first comparison between the final $224 \times 224$ heatmaps produced:

- Mean Absolute Difference (MAE): approximately `0.0012`
- Correlation: approximately `1.0`

Although there was a small numerical difference between the maps, their spatial patterns were almost perfectly correlated.

To investigate this difference, I applied the same min-max normalization to both heatmaps. After using the same normalization, the results were:

- Mean Absolute Difference (MAE): approximately `9.29e-09`
- Maximum difference: approximately `1.79e-07`
- Correlation: approximately `1.0`

This showed that the initial difference was related to scaling or post-processing rather than a meaningful difference in the spatial pattern produced by Grad-CAM.

I also compared the channel importance weights $\alpha_k^c$ calculated by my implementation with the weights calculated from the gradients captured by the reference implementation. Both contained 2048 importance weights and the maximum difference was:

`Alpha max difference: 0.0`

This means that, for this experiment, the calculation of the Grad-CAM channel importance weights was identical between both implementations.

The final validation step will compare the raw $7 \times 7$ Grad-CAM maps before normalization, resizing, colormap application, or overlay. This will allow the core Grad-CAM calculation to be compared independently from the visualization and post-processing steps.

> **Complete raw $7 \times 7$ CAM validation.** 

After comparing the raw Grad-CAM maps from my implementation and pytorch-grad-cam before post-processing, using MAE, maximum difference, and correlation, I can conclude that the small differences observed in the previous comparison were associated with scaling/post-processing of the final maps. For this case, the raw $7 \times 7$ Grad-CAM produced by my implementation matched the raw map reconstructed from the reference implementation exactly (MAE = 0.0, maximum difference = 0.0, correlation = 1.0). This confirms that the core Grad-CAM computation is numerically equivalent to the reference for the tested model, input, target class, and target layer.

## Experiments

### `01_basic_gradcam.py`

Runs the complete Grad-CAM pipeline using the custom implementation:

1. loads pretrained ResNet50;
2. preprocesses an input image using the transformations associated with the
   ImageNet weights;
3. selects the predicted class;
4. generates the raw Grad-CAM map;
5. normalizes and resizes the map;
6. creates the heatmap and visualization overlay.

### `02_evaluation.py`

PENDING, TO DO.

## Repository structure

```text
gradcam-reproduction/
├── assets/
│   └── examples/              # Example input images and README assets
│
├── experiments/
│   ├── 01_basic_gradcam.py    # Basic experiment using the custom implementation
│   └── 02_evaluation.py       # Validation against the reference implementation
│
├── notebooks/
│   └── exploration.ipynb      # Step-by-step exploration and initial reproduction
│
├── results/
│   ├── figures/               # Generated visualizations
│   └── tables/                # Evaluation results
│
├── src/
│   ├── gradcam.py             # From-scratch Grad-CAM implementation
│   ├── model.py               # Model-related utilities
│   └── visualization.py      # CAM post-processing and visualization utilities
│
├── README.md
└── requirements.txt
```

## Installation and usage

Instalación y Uso
## Instalación y uso

### 1. Requisitos previos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- Python 3.12
- Git
- Una conexión a internet para descargar los pesos preentrenados de ResNet50 durante la primera ejecución.

La aplicación utiliza PyTorch y puede ejecutarse en CPU, por lo que no es obligatorio contar con una GPU dedicada.

### 2. Clonar el repositorio

Clona el repositorio desde GitHub y accede al directorio del proyecto:

```bash
git clone https://github.com/Cici160401/gradcam_reproduction.git
cd gradcam_reproduction
```

### 3. Crear un entorno virtual

Se recomienda utilizar un entorno virtual para mantener aisladas las dependencias del proyecto.

```bash
python -m venv .venv
```

Activa el entorno virtual según tu sistema operativo.

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Instalar las dependencias

Instala las bibliotecas necesarias mediante:

```bash
python -m pip install -r requirements.txt
```

Las principales dependencias son PyTorch, TorchVision, NumPy, Matplotlib, Pillow y Gradio.

### 5. Ejecutar Grad-CAM Explorer

Desde la raíz del proyecto, ejecuta:

```bash
python app.py
```

Una vez iniciado el servidor, abre la siguiente dirección en tu navegador:

**http://127.0.0.1:7860**

### 6. Generar una explicación con Grad-CAM

1. Carga una imagen desde la interfaz.
2. Haz clic en **Generate Grad-CAM**.
3. Espera a que ResNet50 clasifique la imagen y genere la explicación visual.
4. Examina los resultados:
   - **Predicted Class:** clase predicha por el modelo.
   - **Prediction Confidence:** confianza asociada a la predicción.
   - **Grad-CAM Heatmap:** mapa de calor que representa las contribuciones positivas a la clase seleccionada.
   - **Grad-CAM Overlay:** superposición del mapa de calor sobre la imagen procesada.

La aplicación utiliza ResNet50 preentrenada en ImageNet y extrae las activaciones de la capa convolucional `layer4[2].conv3`.

**Importante:** los mapas Grad-CAM no representan segmentaciones exactas de los objetos ni demuestran relaciones causales. Son visualizaciones de las regiones que contribuyen positivamente al score de una clase según el método utilizado.

### 7. Ejecutar los experimentos

El repositorio también incluye experimentos independientes para reproducir y evaluar la implementación de Grad-CAM.

**Experimento básico:**

```bash
python -m experiments.01_basic_gradcam
```

**Experimento de evaluación:**

```bash
python -m experiments.02_evaluation
```

Estos scripts permiten explorar el funcionamiento de Grad-CAM y comparar sus resultados con una implementación de referencia.

Dependiendo del experimento, pueden necesitarse imágenes de ejemplo y bibliotecas adicionales.

### Consideraciones adicionales

- Los pesos preentrenados de ResNet50 se descargan automáticamente durante la primera ejecución.
- El modelo no se entrena ni se ajusta nuevamente; se utiliza para clasificación y generación de explicaciones.
- Aunque ResNet50 se encuentra en modo evaluación, Grad-CAM requiere calcular gradientes mediante backpropagation.
- El tiempo de ejecución dependerá del hardware disponible.
- La aplicación se ejecuta localmente y no requiere servicios de alojamiento externos.

## Key takeaways

- Grad-CAM does not directly produce a high-resolution explanation. For the
  selected ResNet50 layer, the actual localization map is only `7×7`.

- The class-discriminative behavior comes from computing gradients with respect
  to a specific target-class score $y^c$.

- Global Average Pooling of the gradients produces one importance weight
  $\alpha_k^c$ for each feature map.

- ReLU keeps the spatial regions that make a positive contribution to the
  target class.

- Resizing the `7×7` map to `224×224` improves visualization but does not add
  new spatial information.

- Visualization and Grad-CAM computation are different stages. Normalization,
  interpolation, colormaps, and overlays can introduce numerical differences
  even when the underlying Grad-CAM maps are equivalent.

- Comparing raw intermediate results is therefore important when validating
  explainability implementations.

## References

To do

## Reflexión Personal

Lo que más disfruté de reproducir el paper fue haberlo entendido muchísimo mejor luego de ponerlo en práctica, no es lo mismo que solo leerlo, reproducirlo te permite entenderlo a un nivel superior. Leer las ecuaciones es chévere, pero ponerlas en práctica en código es muchísimo mejor! :)



