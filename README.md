# 🧠 CNN Project – Implementación de ResNet-34 con PyTorch y Modal

Este proyecto implementa y entrena una **Red Neuronal Convolucional (CNN)** del tipo **ResNet-34**, desarrollada en **Python** con **PyTorch**.  
El trabajo se divide en una parte teórica (documento PDF) y una parte práctica (código).  

El objetivo principal es entrenar y desplegar un modelo de clasificación de imágenes utilizando **recursos en la nube (Modal)** para aprovechar GPUs, y posteriormente realizar inferencia tanto en la nube como en local con herramientas de visualización.

---

## 📚 Estructura del proyecto

| Archivo / Carpeta | Descripción |
|--------------------|-------------|
| **CNNProject.pdf** | Documento teórico introductorio. Explica los fundamentos de las CNN y la arquitectura ResNet-34. |
| **model.py** | Implementación de la arquitectura **ResNet-34** en PyTorch. Define las capas, bloques residuales y estructura del modelo. |
| **train.py** | Script de **entrenamiento** del modelo en la nube utilizando **Modal** (permite usar GPU). Incluye configuración de TensorBoard para el seguimiento de métricas. |
| **main.py** | Código de **inferencia remota**. Conecta con la API de Modal para recuperar el modelo entrenado y realizar predicciones desde la nube. |
| **local_inference.py** | Alternativa de inferencia **local**. Carga el modelo entrenado (`best_model.pth`) y genera una **visualización** del procesamiento en las primeras capas convolucionales. |
| **best_model.pth** | Archivo con los pesos del modelo entrenado. |
| **perrete.webp** | Imagen de ejemplo utilizada para realizar pruebas de inferencia local. |
| **requirements.txt** | Lista de dependencias necesarias para ejecutar el proyecto. |
| **tensorboard_logs/** | Carpeta donde se almacenan los logs generados por **TensorBoard** durante el entrenamiento. |

---

## ⚙️ Herramientas y Tecnologías utilizadas

- **Python 3.10+**
- **PyTorch** → Framework de deep learning para implementar la ResNet-34.
- **Modal** → Plataforma de computación en la nube para ejecutar código con acceso a GPU.
- **TensorBoard** → Monitoreo del proceso de entrenamiento (loss, accuracy, etc.).
- **Matplotlib / NumPy / PIL** → Procesamiento y visualización de imágenes.
- **Git & GitHub** → Control de versiones y despliegue del proyecto.

---

## 🚀 Flujo general del proyecto

1. **Diseño teórico:**  
   Se elabora el documento `CNNProject.pdf` con una explicación sobre las CNN y la arquitectura ResNet.

2. **Implementación del modelo:**  
   El archivo `model.py` define la arquitectura ResNet-34 en PyTorch.

3. **Entrenamiento en la nube:**  
   En `train.py`, el entrenamiento se ejecuta en Modal, aprovechando GPUs.  
   Durante este proceso, se generan métricas registradas en **TensorBoard**.

4. **Inferencia:**
   - **Remota:** `main.py` obtiene el modelo desde la nube a través de la API de Modal.  
   - **Local:** `local_inference.py` utiliza el archivo `best_model.pth` para hacer inferencia en local y visualizar las activaciones intermedias.

---

## 📊 Resultados y visualización

El proyecto permite:
- Visualizar el entrenamiento con **TensorBoard** (loss, accuracy, etc.).
- Observar las **activaciones de las primeras kernels** del modelo durante la inferencia local, lo que ayuda a entender cómo la red procesa las imágenes.

---

## 💡 Aprendizajes y objetivos

Este proyecto me ha permitido:
- Profundizar en la **arquitectura ResNet** y sus bloques residuales.  
- Aprender a **entrenar modelos en la nube** utilizando **Modal**.  
- Integrar **TensorBoard** para el seguimiento de métricas.  
- Desarrollar herramientas de **visualización de convoluciones** para analizar el comportamiento interno del modelo.  
- Gestionar un proyecto completo con **Git y GitHub**, documentando y estructurando el código de forma profesional.

---