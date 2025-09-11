import gdown
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import ultralytics
from ultralytics import YOLO
import json

# ----------------------------
# Configuración y descarga de modelos
# ----------------------------
@st.cache_resource
def descargar_y_cargar_modelo_frutas():
    """Descargar y cargar modelo de frutas"""
    ruta_modelo = "W_FRUTA.pt" 
    if not os.path.exists(ruta_modelo):
        st.info("📥 Descargando modelo de frutas...")
        url = "https://drive.google.com/uc?id=1lfq0_VK9DZsgR-TVGraBrYDUNu-P-hTl"
    
    return YOLO(ruta_modelo)

@st.cache_resource
def descargar_y_cargar_modelo_placas():
    """Descargar y cargar modelo de placas"""
    ruta_modelo = "W_PLACA.pt" 
    if not os.path.exists(ruta_modelo):
        st.info("📥 Descargando modelo de placas...")
        url = "https://drive.google.com/uc?id=12KSiZvxS262NPQ1s-hdsOxJliHSMS3tS"
        gdown.download(url, ruta_modelo, quiet=False)
    
    return YOLO(ruta_modelo)

# Diccionario para caracteres de placas
ID_TO_CHAR = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z', 36: 'placa'
}

# ----------------------------
# Funciones de procesamiento
# ----------------------------
def procesar_imagen_frutas(modelo, imagen, confianza_min=0.5):
    """Procesar imagen para detectar frutas"""
    try:
        resultados = modelo.predict(
            source=imagen,
            conf=confianza_min,
            imgsz=640,
            verbose=False
        )
        
        detecciones = []
        img_resultado = None
        
        for r in resultados:
            img_resultado = r.plot()
            for box in r.boxes:
                clase_id = int(box.cls[0].item())
                clase = modelo.names[clase_id]
                confianza = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detecciones.append({
                    "clase": clase,
                    "confianza": round(confianza, 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tipo": "fruta"
                })
        
        return img_resultado, sorted(detecciones, key=lambda x: x['confianza'], reverse=True)
    
    except Exception as e:
        st.error(f"Error procesando frutas: {str(e)}")
        return None, []

def procesar_imagen_placas(modelo, imagen, confianza_min=0.5):
    """Procesar imagen para detectar placas"""
    try:
        resultados = modelo.predict(
            source=imagen,
            conf=confianza_min,
            imgsz=640,
            verbose=False
        )
        
        detecciones = []
        img_resultado = None
        texto_placa = ""
        
        for r in resultados:
            img_resultado = r.plot()
            caracteres_detectados = []
            
            for box in r.boxes:
                clase_id = int(box.cls[0].item())
                char = ID_TO_CHAR.get(clase_id, '')
                confianza = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detecciones.append({
                    "clase": char,
                    "confianza": round(confianza, 3),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "tipo": "placa"
                })
                
                if char != 'placa' and char:
                    caracteres_detectados.append({
                        'caracter': char,
                        'x': x1,
                        'confianza': confianza
                    })
            
            # Ordenar caracteres por posición X y formar texto
            if caracteres_detectados:
                caracteres_ordenados = sorted(caracteres_detectados, key=lambda x: x['x'])
                texto_placa = ''.join([c['caracter'] for c in caracteres_ordenados])
        
        return img_resultado, detecciones, texto_placa
    
    except Exception as e:
        st.error(f"Error procesando placas: {str(e)}")
        return None, [], ""

# ----------------------------
# Funciones de exportación JSON
# ----------------------------
def generar_datos_json():
    """Generar datos para exportar en formato JSON"""
    # Obtener placa actual
    placa = st.session_state.texto_placa_actual if st.session_state.texto_placa_actual else "No detectada"
    
    # Obtener frutas del historial
    frutas_detectadas = [d for d in st.session_state.detecciones_historial if d.get('tipo') == 'fruta']
    
    # Contar cantidad total de frutas
    cantidad_fruta = len(frutas_detectadas)
    
    # Clasificar frutas por estado usando las clases YOLO reales
    clasificacion_por_estado = {}
    
    for fruta in frutas_detectadas:
        estado_fruta = fruta['clase']  # Usar directamente la clase de YOLO (maduro, sobremaduro, p:largo, etc.)
        confianza = fruta['confianza']
        
        # Agregar al conteo por estado
        if estado_fruta not in clasificacion_por_estado:
            clasificacion_por_estado[estado_fruta] = {
                "cantidad": 0,
                "confianza_promedio": 0,
                "confianzas": []
            }
        
        clasificacion_por_estado[estado_fruta]["cantidad"] += 1
        clasificacion_por_estado[estado_fruta]["confianzas"].append(confianza)
    
    # Calcular confianza promedio para cada estado
    for estado in clasificacion_por_estado:
        confianzas = clasificacion_por_estado[estado]["confianzas"]
        clasificacion_por_estado[estado]["confianza_promedio"] = round(np.mean(confianzas), 3)
        # Remover la lista de confianzas del JSON final para mantenerlo limpio
        del clasificacion_por_estado[estado]["confianzas"]
    
    # Estructura JSON final
    datos_json = {
        "placa": placa,
        "cantidad_fruta": cantidad_fruta,
        "clasificacion_fruta_por_estado": clasificacion_por_estado,
        "timestamp": datetime.now().isoformat(),
        "resumen": {
            "total_detecciones": len(st.session_state.detecciones_historial),
            "estados_fruta_detectados": len(clasificacion_por_estado),
            "confianza_promedio_general": round(
                np.mean([d['confianza'] for d in frutas_detectadas]) if frutas_detectadas else 0, 3
            )
        }
    }
    
    return datos_json

# ----------------------------
# Funciones de visualización
# ----------------------------
def crear_grafico_frutas(detecciones):
    """Crear gráfico de barras para frutas"""
    frutas = [d for d in detecciones if d.get('tipo') == 'fruta']
    if not frutas:
        return None
    
    # Contar frutas por clase
    clases = [d['clase'] for d in frutas]
    conteo = pd.Series(clases).value_counts().reset_index()
    conteo.columns = ['clase', 'cantidad']
    
    fig = px.bar(
        conteo,
        x='clase',
        y='cantidad',
        title="🌴 Frutas detectadas por clase",
        text='cantidad',
        color='clase',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title="Tipo de fruta",
        yaxis_title="Cantidad",
        showlegend=False,
        height=400
    )
    
    return fig

def crear_grafico_confianza(detecciones, tipo_filtro=None):
    """Crear gráfico de confianza"""
    if tipo_filtro:
        detecciones_filtradas = [d for d in detecciones if d.get('tipo') == tipo_filtro]
    else:
        detecciones_filtradas = detecciones
    
    if not detecciones_filtradas:
        return None
    
    df = pd.DataFrame(detecciones_filtradas)
    
    fig = px.bar(
        df,
        x='clase',
        y='confianza',
        title=f"📊 Confianza de detecciones",
        color='confianza',
        color_continuous_scale='Viridis',
        text='confianza'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Clase detectada",
        yaxis_title="Nivel de confianza",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

# ----------------------------
# Configuración de la aplicación
# ----------------------------
def init_session_state():
    """Inicializar variables de sesión de manera segura"""
    # Inicializar solo si no existen
    if "imagen_actual" not in st.session_state:
        st.session_state.imagen_actual = None
    if "detecciones_historial" not in st.session_state:
        st.session_state.detecciones_historial = []
    if "resultado_actual" not in st.session_state:
        st.session_state.resultado_actual = None
    if "texto_placa_actual" not in st.session_state:
        st.session_state.texto_placa_actual = ""

def configurar_pagina():
    """Configurar página de Streamlit"""
    st.set_page_config(
        page_title="🌴🚗 Sistema Dual CNN",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# ----------------------------
# Interfaz principal
# ----------------------------
def main():
    configurar_pagina()
    init_session_state()
    
    # Título principal
    st.title("🔬 Sistema Dual CNN - Detección Inteligente")
    st.markdown("Sistema de detección con dos redes neuronales especializadas")
    
    # Cargar modelos de manera segura
    try:
        modelo_frutas = descargar_y_cargar_modelo_frutas()
        modelo_placas = descargar_y_cargar_modelo_placas()
        modelos_ok = True
    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        modelos_ok = False
    
    if not modelos_ok:
        st.stop()
    
    # Sidebar con configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Parámetros
        confianza = st.slider("🎚️ Confianza mínima", 0.0, 1.0, 0.5, 0.01)
        
        # Estado del sistema
        st.subheader("📊 Estado del Sistema")
        st.success("✅ Modelo frutas cargado")
        st.success("✅ Modelo placas cargado")
        
        total_detecciones = len(st.session_state.detecciones_historial)
        st.metric("Detecciones totales", total_detecciones)
        
        # Botón de limpieza
        if st.button("🗑️ Limpiar historial", key="btn_limpiar"):
            st.session_state.detecciones_historial = []
            st.session_state.resultado_actual = None
            st.session_state.texto_placa_actual = ""
            st.success("Historial limpiado")
            time.sleep(1)
            st.rerun()
    
    # Layout principal con tabs
    tab1, tab2, tab3 = st.tabs(["📸 Cargar Imagen", "🌴 Detectar Frutas", "🚗 Detectar Placas"])
    
    with tab1:
        st.header("📸 Cargar imagen")
        
        # Métodos de carga
        metodo = st.radio(
            "Método de entrada:",
            ["📁 Subir archivo", "📷 Cámara web", "🎯 Imagen de ejemplo"],
            key="metodo_carga"
        )
        
        imagen_cargada = False
        
        if metodo == "📁 Subir archivo":
            archivo = st.file_uploader(
                "Selecciona una imagen",
                type=['jpg', 'jpeg', 'png'],
                key="uploader_imagen"
            )
            if archivo is not None:
                try:
                    imagen_pil = Image.open(archivo)
                    imagen = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
                    st.session_state.imagen_actual = imagen
                    imagen_cargada = True
                except Exception as e:
                    st.error(f"Error cargando imagen: {str(e)}")
        
        elif metodo == "📷 Cámara web":
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📸 Capturar desde cámara", type="primary", key="btn_camara"):
                    try:
                        cap = cv2.VideoCapture(0)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            st.session_state.imagen_actual = frame
                            imagen_cargada = True
                            st.success("✅ Imagen capturada correctamente")
                        else:
                            st.error("❌ No se pudo acceder a la cámara")
                    except Exception as e:
                        st.error(f"❌ Error de cámara: {str(e)}")
        
        elif metodo == "🎯 Imagen de ejemplo":
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("Generar imagen de ejemplo", key="btn_ejemplo"):
                    # Generar imagen de ejemplo más realista
                    imagen = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
                    st.session_state.imagen_actual = imagen
                    imagen_cargada = True
                    st.info("📸 Imagen de ejemplo generada")
        
        # Mostrar imagen actual
        if st.session_state.imagen_actual is not None:
            st.subheader("🖼️ Imagen cargada")
            st.image(st.session_state.imagen_actual, channels="BGR", use_column_width=True)
    
    with tab2:
        st.header("🌴 Detección de Frutas")
        
        if st.session_state.imagen_actual is None:
            st.warning("⚠️ Primero carga una imagen en la pestaña 'Cargar Imagen'")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔍 Detectar Frutas", type="primary", key="btn_frutas", use_container_width=True):
                    with st.spinner("🧠 Analizando frutas..."):
                        img_resultado, detecciones = procesar_imagen_frutas(
                            modelo_frutas, 
                            st.session_state.imagen_actual, 
                            confianza
                        )
                        
                        if detecciones and img_resultado is not None:
                            st.session_state.resultado_actual = img_resultado
                            st.session_state.detecciones_historial.extend(detecciones)
                            
                            st.success(f"✅ {len(detecciones)} frutas detectadas")
                        else:
                            st.warning("🔍 No se detectaron frutas con la confianza especificada")
            
            # Mostrar resultados de frutas
            if st.session_state.resultado_actual is not None:
                st.subheader("🎯 Resultado de la detección")
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)
                
                # Obtener solo las detecciones de frutas más recientes
                frutas_detectadas = [d for d in st.session_state.detecciones_historial if d.get('tipo') == 'fruta']
                if frutas_detectadas:
                    # Mostrar gráfico
                    fig = crear_grafico_frutas(frutas_detectadas[-10:])  # Últimas 10
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="grafico_frutas")
    
    with tab3:
        st.header("🚗 Detección de Placas")
        
        if st.session_state.imagen_actual is None:
            st.warning("⚠️ Primero carga una imagen en la pestaña 'Cargar Imagen'")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔍 Detectar Placas", type="primary", key="btn_placas", use_container_width=True):
                    with st.spinner("🧠 Analizando placas..."):
                        img_resultado, detecciones, texto_placa = procesar_imagen_placas(
                            modelo_placas,
                            st.session_state.imagen_actual,
                            confianza
                        )
                        
                        if detecciones and img_resultado is not None:
                            st.session_state.resultado_actual = img_resultado
                            st.session_state.texto_placa_actual = texto_placa
                            st.session_state.detecciones_historial.extend(detecciones)
                            
                            if texto_placa:
                                st.success(f"✅ Placa detectada: **{texto_placa}**")
                                
                                # Guardar resultado en JSON
                                resultado_json = {
                                    "imagen": "imagen_procesada",
                                    "placa": texto_placa,
                                    "timestamp": datetime.now().isoformat(),
                                    "confianza_promedio": np.mean([d['confianza'] for d in detecciones])
                                }
                                
                                try:
                                    with open("resultado_placa.json", "w", encoding="utf-8") as f:
                                        json.dump(resultado_json, f, indent=4, ensure_ascii=False)
                                    st.info("💾 Resultado guardado en resultado_placa.json")
                                except Exception as e:
                                    st.warning(f"No se pudo guardar el archivo: {str(e)}")
                            else:
                                st.success("✅ Elementos de placa detectados pero no se pudo formar texto completo")
                        else:
                            st.warning("🔍 No se detectaron placas con la confianza especificada")
            
            # Mostrar resultados de placas
            if st.session_state.resultado_actual is not None and st.session_state.texto_placa_actual:
                st.subheader("🎯 Resultado de la detección")
                st.image(st.session_state.resultado_actual, channels="BGR", use_column_width=True)
                
                # Mostrar texto de placa en formato destacado
                st.subheader("🚗 Placa detectada")
                st.code(st.session_state.texto_placa_actual, language="text")
    
    # Sección de historial y estadísticas
    if st.session_state.detecciones_historial:
        st.markdown("---")
        st.header("📈 Historial y Estadísticas")
        
        # Métricas generales
        frutas_total = len([d for d in st.session_state.detecciones_historial if d.get('tipo') == 'fruta'])
        placas_total = len([d for d in st.session_state.detecciones_historial if d.get('tipo') == 'placa'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌴 Frutas detectadas", frutas_total)
        with col2:
            st.metric("🚗 Placas detectadas", placas_total)
        with col3:
            st.metric("📊 Total detecciones", len(st.session_state.detecciones_historial))
        
        # Tabla de historial (últimas 10 detecciones)
        st.subheader("🕒 Últimas detecciones")
        ultimas_detecciones = st.session_state.detecciones_historial[-10:]
        if ultimas_detecciones:
            df_historial = pd.DataFrame(ultimas_detecciones)[['tipo', 'clase', 'confianza', 'timestamp']]
            st.dataframe(df_historial, use_container_width=True, hide_index=True)
        
        # Sección de descarga mejorada
        st.subheader("💾 Descargar Resultados")
        
        # Mostrar vista previa de los datos JSON
        if st.button("👁️ Vista previa JSON", key="preview_json"):
            datos_json = generar_datos_json()
            st.json(datos_json)
        
        # Botones de descarga
        col1, col2 = st.columns(2)
        
        with col1:
            # Botón de descarga JSON
            if st.button("📥 Descargar Reporte JSON", type="primary", key="btn_download_json"):
                datos_json = generar_datos_json()
                json_str = json.dumps(datos_json, indent=4, ensure_ascii=False)
                
                st.download_button(
                    label="📄 Descargar archivo JSON",
                    data=json_str,
                    file_name=f"reporte_detecciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json_btn"
                )
        
        with col2:
            # Botón de descarga CSV (historial completo)
            if st.button("📊 Descargar Historial CSV", key="btn_download_csv"):
                if st.session_state.detecciones_historial:
                    df_export = pd.DataFrame(st.session_state.detecciones_historial)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="📈 Descargar archivo CSV",
                        data=csv,
                        file_name=f"historial_detecciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv_btn"
                    )

if __name__ == "__main__":

    main()
