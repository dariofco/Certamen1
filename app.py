# =============================================================================
#  EcoSort AI · Sistema Inteligente de Clasificación de Residuos e IA Generativa
#  Certamen IA Generativa para la Gestión Industrial · UDD
#  Ejecutar: streamlit run app.py
# =============================================================================

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. Configuración visual de la página y paleta de colores ejecutiva
# -----------------------------------------------------------------------------
st.set_page_config(page_title="EcoSort AI · Gestión Industrial de Residuos", page_icon="♻️", layout="wide")

VERDE_PRINCIPAL = "#2E7D32"
VERDE_SUAVE     = "#E8F5E9"
AZUL_INFO       = "#1F5A96"
NARANJA_ALERTA  = "#E65100"
GRIS_FONDO      = "#F4F7F4"
GRIS_BORDE      = "#DCE2DC"
TINTA           = "#1B2430"
TENUE           = "#5F6B7A"

# Diccionario rico de metadatos por clase para ambientar la aplicación
INFO_CLASES = {
    "cardboard": {
        "nombre": "Cartón (Cardboard)",
        "icono": "📦",
        "contenedor": "Contenedor Azul / Prensa de Cartón Corrugado",
        "color": "#8D6E63",
        "riesgo": "Baja peligrosidad. Evitar humedad, grasas o restos de comida que arruinan la fibra celulósica."
    },
    "glass": {
        "nombre": "Vidrio (Glass)",
        "icono": "🍾",
        "contenedor": "Contenedor Verde / Tolva de Calcín de Vidrio",
        "color": "#2E7D32",
        "riesgo": "Riesgo físico de corte y contaminación cerámica. Retirar tapas metálicas o plásticas antes del triturado."
    },
    "metal": {
        "nombre": "Metal y Aluminio (Metal)",
        "icono": "🥫",
        "contenedor": "Contenedor Amarillo / Separador Magnético y Eddy Current",
        "color": "#546E7A",
        "riesgo": "Riesgo medio por bordes filosos o envases presurizados (aerosoles). Compactar solo tras verificar vaciado."
    },
    "paper": {
        "nombre": "Papel (Paper)",
        "icono": "📄",
        "contenedor": "Contenedor Azul / Línea de Pulpa Blanca",
        "color": "#1E88E5",
        "riesgo": "Alta sensibilidad a contaminación cruzada con líquidos, aceites o servilletas usadas (no reciclables)."
    },
    "plastic": {
        "nombre": "Plástico (Plastic)",
        "icono": "🧴",
        "contenedor": "Contenedor Amarillo / Clasificador Óptico PET-HDPE",
        "color": "#FB8C00",
        "riesgo": "Suele confundirse visualmente con vidrio transparente. Verificar ausencia de residuos químicos o tóxicos."
    },
    "trash": {
        "nombre": "Basura General / No Reciclable (Trash)",
        "icono": "🗑️",
        "contenedor": "Contenedor Gris Oscuro / Descarte a Relleno Sanitario",
        "color": "#C62828",
        "riesgo": "ALTO RIESGO DE CONTAMINACIÓN DE LOTE: Si ingresa por error a una línea reciclable, invalida el lote industrial."
    }
}

UMBRAL_CONFIANZA = 65.0  # Por debajo de 65% se solicita revisión humana obligatoria

# Estilos CSS ejecutivos inspirados en el dashboard visto en clases (con ajuste para textos métricos)
st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.4rem; max-width: 1350px; }}
  .encabezado {{ border-left: 6px solid {VERDE_PRINCIPAL}; padding: 6px 0 6px 18px; margin-bottom: 14px; background: {GRIS_FONDO}; border-radius: 0 8px 8px 0; }}
  .encabezado h1 {{ font-size: 1.95rem; margin: 0; color: {TINTA}; }}
  .encabezado p {{ margin: 4px 0 0; color: {TENUE}; font-size: 0.95rem; }}
  [data-testid="stMetric"] {{ background: {GRIS_FONDO}; border: 1px solid {GRIS_BORDE}; border-radius: 8px; padding: 12px 16px; }}
  [data-testid="stMetricValue"] {{ font-size: 1.35rem !important; }}
  .caja-ok {{ background: {VERDE_SUAVE}; border-left: 5px solid {VERDE_PRINCIPAL}; padding: 14px 18px; border-radius: 6px; margin: 10px 0; }}
  .caja-alerta {{ background: #FFF3E0; border-left: 5px solid {NARANJA_ALERTA}; padding: 14px 18px; border-radius: 6px; margin: 10px 0; color: #BF360C; }}
  .caja-etica {{ background: #ECEFF1; border: 1px solid #CFD8DC; padding: 10px 14px; border-radius: 6px; font-size: 0.86rem; color: {TENUE}; margin-top: 14px; }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Carga en caché del modelo YOLO entrenado y métricas del conjunto de prueba
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo YOLO11n-cls entrenado...")
def cargar_modelo():
    ruta_modelo = os.path.join("modelo", "best.pt")
    return YOLO(ruta_modelo)

@st.cache_data
def cargar_info_modelo():
    ruta_info = os.path.join("modelo", "info_modelo.json")
    if os.path.exists(ruta_info):
        with open(ruta_info, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "modelo_seleccionado": "YOLO11n-cls (Ultralytics)",
        "accuracy_test": 89.0,
        "balanced_accuracy_test": 89.7,
        "comparacion": {
            "CLIP (Zero-Shot)": 71.6,
            "EfficientNet-B0 (timm)": 75.3,
            "YOLO11n-cls (Ultralytics)": 89.7
        }
    }

modelo = cargar_modelo()
info_metricas = cargar_info_modelo()

# -----------------------------------------------------------------------------
# 3. Barra lateral: Controles, ejemplos de prueba y configuración de API
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ♻️ Panel de Control")
    st.caption("Sistema de inspección visual asistido por IA para plantas de clasificación de residuos.")
    
    modo_entrada = st.radio(
        "Fuente de la imagen:",
        ["📂 Subir imagen propia", "🧪 Usar imágenes de ejemplo (Test)"],
        index=1
    )
    
    st.markdown("---")
    st.markdown("### 📊 Desempeño del Modelo (Test)")
    st.metric("Accuracy Balanceada (Prueba)", f"{info_metricas['balanced_accuracy_test']:.1f}%")
    st.metric("Accuracy Global (Prueba)", f"{info_metricas['accuracy_test']:.1f}%")
    st.caption("Nota: El *Accuracy en prueba* mide el acierto general del modelo en 383 imágenes históricas no vistas. La *Confianza* mide la seguridad del modelo en la foto actual.")
    
    st.markdown("---")
    st.markdown("### 🤖 Configuración LLM (Gemini)")
    api_key_secreta = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key_secreta = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    if not api_key_secreta:
        api_key_secreta = os.environ.get("GEMINI_API_KEY", "")
        
    if api_key_secreta:
        st.success("🔒 Clave API cargada de forma segura desde Secrets.")
        api_key_final = api_key_secreta
    else:
        api_key_final = st.text_input("Ingresa tu GEMINI_API_KEY (opcional para pruebas):", type="password")
        st.caption("En producción la clave se guarda oculta en *Streamlit Secrets*.")

# -----------------------------------------------------------------------------
# 4. Encabezado principal y pestañas de navegación
# -----------------------------------------------------------------------------
st.markdown("""
<div class="encabezado">
  <h1>♻️ EcoSort AI · Clasificación Inteligente de Residuos Industriales</h1>
  <p>De la imagen a la decisión operativa: visión por computador con <b>YOLO11n-cls</b> + recomendaciones con <b>IA Generativa (Google Gemini)</b></p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "🔍 Clasificador e IA Generativa",
    "📈 Desempeño y Justificación del Modelo",
    "🛡️ Datos, Limitaciones y Ética"
])

# =============================================================================
# PESTAÑA 1: CLASIFICADOR INTERACTIVO E IA GENERATIVA
# =============================================================================
with tabs[0]:
    col_izq, col_der = st.columns([1.05, 1.35], gap="large")
    imagen_actual = None
    nombre_archivo = ""

    with col_izq:
        st.subheader("1. Selecciona o sube una imagen")
        if modo_entrada == "📂 Subir imagen propia":
            archivo_subido = st.file_uploader(
                "Sube la foto de un residuo (JPG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"]
            )
            if archivo_subido is not None:
                imagen_actual = Image.open(archivo_subido).convert("RGB")
                nombre_archivo = archivo_subido.name
        else:
            carpeta_ej = "ejemplos"
            archivos_ej = sorted(os.listdir(carpeta_ej)) if os.path.exists(carpeta_ej) else []
            if archivos_ej:
                etiquetas_ej = {
                    f: f"{INFO_CLASES.get(f.replace('ejemplo_', '').split('.')[0], {}).get('icono', '📷')} Ejemplo: {INFO_CLASES.get(f.replace('ejemplo_', '').split('.')[0], {}).get('nombre', f)}"
                    for f in archivos_ej
                }
                seleccion = st.selectbox("Elige una imagen de prueba del conjunto Test:", archivos_ej, format_func=lambda x: etiquetas_ej[x])
                ruta_ej = os.path.join(carpeta_ej, seleccion)
                imagen_actual = Image.open(ruta_ej).convert("RGB")
                nombre_archivo = seleccion
            else:
                st.warning("No se encontraron imágenes en la carpeta `ejemplos/`.")

        if imagen_actual is not None:
            st.image(imagen_actual, caption=f"Imagen cargada: {nombre_archivo}", use_container_width=True)
        else:
            st.info("👆 Sube una imagen o cambia a la opción de ejemplos en el panel izquierdo para comenzar.")

    with col_der:
        st.subheader("2. Resultado de Clasificación y Protocolo IA")
        if imagen_actual is not None:
            resultado = modelo(imagen_actual, verbose=False)[0]
            probs = resultado.probs.data.cpu().numpy()
            nombres_modelo = resultado.names

            idx_top1 = int(resultado.probs.top1)
            clase_en = nombres_modelo[idx_top1]
            confianza = float(resultado.probs.top1conf.item()) * 100.0
            meta_clase = INFO_CLASES.get(clase_en, {
                "nombre": clase_en, "icono": "♻️", "contenedor": "General", "color": VERDE_PRINCIPAL, "riesgo": "N/A"
            })

            k1, k2, k3 = st.columns(3)
            k1.metric("Clase Predicha", f"{meta_clase['icono']} {meta_clase['nombre'].split(' (')[0]}")
            k2.metric("Confianza en esta foto", f"{confianza:.1f}%")
            k3.metric("Accuracy Modelo (Test)", f"{info_metricas['balanced_accuracy_test']:.1f}%", "Métrica global")

            if confianza < UMBRAL_CONFIANZA:
                st.markdown(f"""
                <div class="caja-alerta">
                  <b>⚠️ ALERTA DE CONFIANZA BAJA ({confianza:.1f}% &lt; {UMBRAL_CONFIANZA:.0f}%): Se requiere revisión humana</b><br>
                  El modelo presenta incertidumbre entre materiales visualmente parecidos. <b>No envíe este ítem automáticamente a la línea de compactación</b>; desvíelo a inspección manual para evitar contaminación cruzada del lote.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="caja-ok">
                  <b>✅ Clasificación de Alta Confianza ({confianza:.1f}%)</b><br>
                  <b>Destino recomendado:</b> {meta_clase['contenedor']}<br>
                  <b>Consideración operativa:</b> {meta_clase['riesgo']}
                </div>
                """, unsafe_allow_html=True)

            df_probs = pd.DataFrame({
                "Clase_EN": [nombres_modelo[i] for i in range(len(probs))],
                "Material": [f"{INFO_CLASES[nombres_modelo[i]]['icono']} {INFO_CLASES[nombres_modelo[i]]['nombre']}" for i in range(len(probs))],
                "Probabilidad": probs * 100.0,
                "Color": [INFO_CLASES[nombres_modelo[i]]["color"] for i in range(len(probs))]
            }).sort_values("Probabilidad", ascending=True)

            fig = go.Figure(go.Bar(
                x=df_probs["Probabilidad"],
                y=df_probs["Material"],
                orientation="h",
                marker_color=df_probs["Color"],
                text=[f"{p:.1f}%" for p in df_probs["Probabilidad"]],
                textposition="outside",
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>"
            ))
            fig.update_layout(
                title="Probabilidades asignadas por el modelo a cada clase (%)",
                height=290,
                template="plotly_white",
                margin=dict(l=10, r=40, t=40, b=10),
                xaxis=dict(range=[0, 112], ticksuffix="%", title=None),
                yaxis=dict(title=None)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 💬 Recomendación Operativa Generada por IA")
            top2_idx = int(np.argsort(probs)[::-1][1])
            segunda_clase = INFO_CLASES[nombres_modelo[top2_idx]]["nombre"]
            segunda_prob = float(probs[top2_idx]) * 100.0

            if st.button("✨ Generar protocolo de reciclaje y recomendación con IA", type="primary", use_container_width=True):
                if not api_key_final:
                    st.error("🔑 Para consultar al LLM en vivo, ingresa tu `GEMINI_API_KEY` en la barra lateral izquierda (o configúrala en *Streamlit Secrets*).")
                else:
                    with st.spinner("Consultando a Google Gemini con prompt responsable..."):
                        try:
                            from google import genai
                            cliente = genai.Client(api_key=api_key_final)
                            
                            estado_confianza = (
                                "BAJA CONFIANZA (exigir inspección humana manual obligatoria antes de procesar)"
                                if confianza < UMBRAL_CONFIANZA
                                else "ALTA CONFIANZA (procesamiento estándar con verificación visual rápida)"
                            )
                            
                            prompt_responsable = f"""
                            Actúa como un Ingeniero Experto en Gestión de Residuos Industriales y Economía Circular en Chile.
                            Un sistema de visión artificial (YOLO11n-cls) acaba de analizar la fotografía de un residuo en una planta de clasificación con estos resultados técnicos:
                            - Material detectado (Top 1): {meta_clase['nombre']}
                            - Nivel de confianza de la predicción: {confianza:.1f}% ({estado_confianza})
                            - Segunda clase más probable (posible confusión visual): {segunda_clase} ({segunda_prob:.1f}%)
                            - Línea/Contenedor sugerido: {meta_clase['contenedor']}

                            Redacta una recomendación breve, estructurada y accionable en español (máximo 160 palabras) con estos 3 puntos:
                            1. **Acción inmediata en planta / punto limpio**: Qué hacer con el residuo y cómo prepararlo (limpieza, compactación, retiro de tapas/etiquetas).
                            2. **Prevención del error más costoso**: Qué riesgo existe si se confunde con {segunda_clase} o con basura no reciclable (contaminación cruzada de lotes).
                            3. **Protocolo de seguridad según confianza**: Si la confianza es menor a 65% ({confianza:.1f}%), enfatiza que un operario humano DEBE revisarlo manualmente.

                            Reglas estrictas: Limítate exclusivamente a gestión de residuos, reciclaje y seguridad industrial. Incluye al final una breve nota indicando que esta es una recomendación orientativa de uso educativo.
                            """
                            respuesta = cliente.models.generate_content(
                                model="gemini-3.8-flash",
                                contents=prompt_responsable
                            )
                            st.markdown("#### 📋 Protocolo Recomendado por Gemini:")
                            st.info(respuesta.text)
                        except Exception as e:
                            st.error(f"Error al conectar con la API de Gemini: {e}")

        st.markdown("""
        <div class="caja-etica">
          <b>🛡️ Advertencia de Uso Responsable:</b> Esta aplicación es un prototipo académico desarrollado para el curso <i>IA Generativa para la Gestión Industrial (UDD)</i>. Sus resultados son orientativos y no reemplazan protocolos normativos de manejo de residuos peligrosos ni la supervisión de un operario humano, especialmente cuando la confianza del modelo es inferior al 65%.
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PESTAÑA 2: DESEMPEÑO Y JUSTIFICACIÓN DEL MODELO
# =============================================================================
with tabs[1]:
    st.subheader("Comparación de Modelos Preentrenados en el Conjunto de Prueba (Test = 383 imágenes)")
    st.markdown("""
    Para seleccionar el modelo de producción evaluamos **tres enfoques vistos en clases** exactamente sobre el mismo conjunto de prueba (`test`, 15% de los datos nunca vistos durante el entrenamiento), utilizando **Accuracy Balanceada** para corregir el desbalance de la clase *Basura General (trash)*.
    """)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        comp = info_metricas.get("comparacion", {})
        df_comp = pd.DataFrame({
            "Enfoque / Modelo": list(comp.keys()),
            "Accuracy Balanceada en Prueba (%)": list(comp.values()),
            "Peso del Archivo": ["~605 MB (Zero-Shot)", "~16 MB (.pth)", "~3,2 MB (.pt)"]
        })
        st.dataframe(df_comp, hide_index=True, use_container_width=True)

        st.markdown("""
        **¿Por qué elegimos YOLO11n-cls para la aplicación web?**
        * **Máxima precisión en prueba:** Superó ampliamente a CLIP sin entrenar (**89,7% vs. 71,6%**) y a EfficientNet-B0 (**75,3%**), logrando un **Recall de 95,2%** en la clase crítica *Basura General*.
        * **Eficiencia para producción:** Su archivo de pesos (`best.pt`) pesa solo **3,2 MB**, lo que permite alojarlo directamente en GitHub y ejecutar inferencias en milisegundos en *Streamlit Community Cloud*.
        """)

    with c2:
        fig_comp = go.Figure(go.Bar(
            x=list(comp.values()),
            y=list(comp.keys()),
            orientation="h",
            marker_color=["#78909C", "#1E88E5", "#2E7D32"],
            text=[f"{v:.1f}%" for v in comp.values()],
            textposition="outside"
        ))
        fig_comp.add_vline(x=16.7, line_dash="dash", line_color="red", annotation_text="Azar (16,7%)")
        fig_comp.update_layout(
            title="Accuracy Balanceada en Prueba por Modelo (%)",
            height=300,
            template="plotly_white",
            xaxis=dict(range=[0, 108], ticksuffix="%"),
            margin=dict(l=10, r=30, t=40, b=10)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.subheader("Análisis del Error Más Costoso en Gestión Industrial")
    st.markdown("""
    * **¿Cuál es el error más costoso?** Clasificar un residuo no reciclable o contaminante (**Basura General / `trash`**) como si fuera un material reciclable limpio (**Papel, Cartón, Plástico o Vidrio**).
    * **Impacto operativo y económico:** En una planta recuperadora, introducir basura orgánica, envases con químicos o cerámica en una paca de papel o en un horno de vidrio genera **contaminación cruzada**, obligando a rechazar lotes completos de toneladas de material o dañando equipos de trituración.
    * **Desempeño de nuestro modelo:** En las 383 imágenes de prueba, de los **21 casos reales de Basura General**, nuestro modelo detectó **20 correctamente (95,2% de Recall)** y solo confundió 1 con vidrio. Las confusiones más frecuentes ocurren entre **Vidrio y Plástico** (objetos transparentes), lo cual mitigamos mediante el umbral de revisión humana (< 65%) y las alertas del modelo generativo.
    """)

# =============================================================================
# PESTAÑA 3: DATOS, LIMITACIONES Y CONSIDERACIONES ÉTICAS
# =============================================================================
with tabs[2]:
    st.subheader("Ficha Técnica del Dataset y Consideraciones Éticas")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        **1. Origen y Licencia del Dataset**
        * **Dataset:** *Garbage Classification (TrashNet - 6 classes)*.
        * **Autores:** Gary Thung y Mindy Yang (Stanford University, CS229), publicado en Kaggle por `asdasdasasdas`.
        * **Licencia:** Dominio Público / MIT License (uso académico libre, sin fotos de personas ni datos sensibles).
        * **Total de imágenes:** 2.527 fotografías RGB divididas estratificadamente en **70% Entrenamiento (1.766)**, **15% Validación (378)** y **15% Prueba (383)**.
        """)
    with col_d2:
        st.markdown("""
        **2. Limitaciones y Sesgos Identificados**
        * **Desbalance de clases:** La clase `trash` (137 fotos, 5,4%) tiene considerablemente menos ejemplos que `paper` (594 fotos, 23,5%).
        * **Sesgo de fondo e iluminación:** Las fotografías fueron capturadas con objetos aislados sobre fondos uniformes y luz controlada. En cintas transportadoras reales con suciedad, sombras, objetos abollados o múltiples residuos superpuestos, la confianza del clasificador puede disminuir, requiriendo reentrenamiento con imágenes locales de la planta.
        """)
