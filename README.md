# ♻️ EcoSort AI · Clasificación Inteligente de Residuos con Modelos Preentrenados e IA Generativa

Aplicación web interactiva desarrollada para el certamen del curso **IA Generativa para la Gestión Industrial** (Ingeniería Civil Industrial, Universidad del Desarrollo).

## 👥 Integrantes del Grupo
* Darío Ampuero
* Rubén Urrutia
* Nicolás Véliz

---

## 1. Problema y Motivación
En las plantas de clasificación de residuos y puntos limpios industriales, la separación manual es lenta y propensa a errores que provocan **contaminación cruzada de lotes reciclables**. **EcoSort AI** automatiza la identificación visual de residuos en **6 categorías**, reportando el nivel de confianza en tiempo real y conectándose con **Google Gemini (`gemini-2.5-flash`)** para entregar protocolos operativos de reciclaje y alertas de inspección humana cuando la confianza es baja.

---

## 2. Ficha de Datos (Dataset)
* **Nombre y Enlace:** [Garbage Classification (6 classes) - Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
* **Origen y Autores:** Creado por Gary Thung y Mindy Yang (*TrashNet*, Stanford University) y publicado en Kaggle por `asdasdasasdas`.
* **Licencia:** MIT / Dominio Público.
* **Distribución por clase (2.527 imágenes en total):**
  * `paper` (Papel): 594 imágenes
  * `glass` (Vidrio): 501 imágenes
  * `plastic` (Plástico): 482 imágenes
  * `metal` (Metal): 410 imágenes
  * `cardboard` (Cartón): 403 imágenes
  * `trash` (Basura general / No reciclable): 137 imágenes *(clase minoritaria desbalanceada: 5,4% del total)*.
* **División utilizada:** Separación estratificada (`seed=42`) en **70% Entrenamiento (1.766 fotos)**, **15% Validación (378 fotos)** y **15% Prueba (383 fotos)**.
* **Limitaciones:** Imágenes tomadas con un solo objeto sobre fondos claros e iluminación controlada; en entornos industriales con objetos superpuestos o sucios se requiere reentrenamiento local.

---

## 3. Comparación de Modelos y Justificación
Se evaluaron tres enfoques vistos en clases sobre las mismas **383 imágenes del conjunto de prueba (`test`)**:

| Modelo / Enfoque | Librería | Accuracy Balanceada (Test) | Accuracy Normal (Test) | Tamaño de Pesos |
|---|---|---|---|---|
| **1 · CLIP (Zero-Shot sin entrenar)** | `transformers` | 71,6% | 73,4% | ~605 MB |
| **2 · EfficientNet-B0 (Transfer Learning, 5 épocas)** | `timm` | 75,3% | 78,3% | ~16 MB |
| **3 · YOLO11n-cls (Elegido, 10 épocas)** | `ultralytics` | **89,7%** | **89,0%** | **~3,2 MB** |

**Justificación:** Se seleccionó **YOLO11n-cls** por obtener la mayor Accuracy Balanceada (**89,7%**) y un **Recall de 95,2%** en la clase crítica *Basura General*, además de pesar solo **3,2 MB**, lo que permite subirlo directamente a GitHub y ejecutarlo con baja latencia en *Streamlit Community Cloud*.

---

## 4. Estructura del Repositorio
* `app.py`: Aplicación web interactiva en Streamlit
* `requirements.txt`: Librerías necesarias para Streamlit Cloud
* `README.md`: Documentación completa del proyecto
* `notebooks/entrenamiento.ipynb`: Notebook ejecutado con comparación de modelos y métricas
* `modelo/`: Pesos del modelo YOLO11n-cls entrenado (`best.pt`) y métricas (`info_modelo.json`)
* `ejemplos/`: 6 imágenes de prueba del conjunto Test
* `.streamlit/config.toml`: Paleta visual temática de la aplicación

---

## 5. Instrucciones de Ejecución y Configuración de Clave API
1. Instalar dependencias: `pip install -r requirements.txt`
2. Configurar la clave `GEMINI_API_KEY` en los Secrets de Streamlit Cloud.
3. Ejecutar la aplicación: `streamlit run app.py`

---

## 6. Declaración de Uso de Inteligencia Artificial
De acuerdo con las normas del curso, declaramos el uso de **Google Gemini** como asistente de programación y diseño durante el desarrollo de este proyecto para estructurar el código de evaluación en PyTorch/Ultralytics, diseñar la interfaz en Streamlit (`.streamlit/config.toml` y gráficos Plotly) e integrar el SDK `google-genai` con manejo seguro de credenciales (`st.secrets`). Todo el código fue ejecutado, validado, adaptado y comprendido íntegramente por los miembros del grupo.
