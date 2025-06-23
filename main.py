import streamlit as st
import torch
import numpy as np
import os
import openai
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(
    page_title="Selector de imagenes",
    page_icon=":camera:",
    layout="wide"
)

@st.cache_data(show_spinner="Generando imagen....")
def generar_imagen(prompt):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1
    )
    return response.data[0].url

@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    #defino dispositivo a utilizar cpu o cuda(gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Configurar CLIP
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") #Procesa las imagenes
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")#hace la inferencia

    model.to(device)#le asignamos al modelo el disposito a utilizar

    return model, processor, device

@st.cache_data(show_spinner="Realizando inferencia...")
def inferencia_CLIP(imagenes,concepto, _model, _processor, device) -> np.ndarray:
    #Crear tokens del concepto
    label_token = _processor( 
        text=[concepto], 
        padding=True, 
        images=None, 
        return_tensors="pt"
    ).to(device)

    #codificamos los token a embeddings
    label_embeddings = _model.get_text_features(**label_token)

    # extraemos el tensor del dispositivo a la CPU, para liberar la GPU y lo convertimos a un array de numpy
    label_embeddings = label_embeddings.detach().cpu().numpy()

    #normalizamos los embeddings
    label_embeddings = label_embeddings / np.linalg.norm(label_embeddings, axis=1, keepdims=True)

    #Procesar las imagenes
    images_token = _processor(
        text=None, 
        images=imagenes, 
        return_tensors="pt", 
        padding=True
    ).to(device)

    #codificamos las imagenes a embeddings
    image_embeddings = _model.get_image_features(**images_token)

    # extraemos el tensor del dispositivo a la CPU, para liberar la GPU y lo convertimos a un array de numpy
    image_embeddings = image_embeddings.detach().cpu().numpy()

    #normalizamos los embeddings
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    #obtenemos la similitud entre los embeddings de las imagenes y el concepto
    scores = np.dot(image_embeddings, label_embeddings.T)

    return scores

if __name__ == "__main__":

    # Inicializar session_state si no existen
    for key in ["resultados", "concepto", "url_dalle"]:
        if key not in st.session_state:
            st.session_state[key] = None

    model, processor, device = cargar_modelo()  

    col1,col2,col3 = st.columns([1, 2, 1])  # Crear tres columnas con proporciones

    with col2:
        st.title("Selector de imagenes")
        
        with st.form("subir_informacion", width=800): 
            #subida de imagenes, acepta multiples archivos
            imagenes = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"],accept_multiple_files=True)
            #ingreso del concepto a comparar
            concepto = st.text_input("Concepto de la imagen")
            #ingreso umbral para generar imagen con DALL-E 3
            umbral = st.number_input("Umbral para generar imagen con DALL-E 3", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
            #envio del formulario
            submit_button = st.form_submit_button("Enviar")

        if submit_button: 
            if imagenes and concepto: #si ingresaron imagenes y concepto
                #Mostrar mensaje de exito
                st.success("Imagenes y concepto enviados correctamente.")
                imagenes = [Image.open(imagen).convert("RGB") for imagen in imagenes]  # Convertir
                scores = inferencia_CLIP(imagenes, concepto, model, processor, device)
                result_ordenados = sorted(zip(imagenes, scores), key=lambda x: x[1], reverse=True)

                #guardar en session state
                st.session_state["resultados"] = result_ordenados
                st.session_state["mejor_score"] = result_ordenados[0][1][0] if result_ordenados else None
                st.session_state["concepto"] = concepto
                st.session_state["url_dalle"] = None  # inicializar la URL de la imagen de dalle
                st.session_state["umbral"] = umbral
            else:
                st.error("Por favor, sube al menos una imagen y proporciona un concepto.")


    if st.session_state["resultados"]:
        colA, colB = st.columns(2)  # Crear dos columnas 
                
        with colA:
            st.subheader("Ranking según el concepto")
                    
                    
            # Mostrar las imágenes y sus scores
            if "resultados" in st.session_state:
                for i, (img, score) in enumerate(st.session_state["resultados"]):
                    if score[0] > 0.2:
                        st.markdown(f"#### <span style='color: green;'>{i+1}.- Score: {score[0]:.4f}</span>", unsafe_allow_html=True)
                        st.image(img,width=300)
                    else:
                        st.markdown(f"#### <span style='color: red;'>{i+1}.- Score: {score[0]:.4f}</span>", unsafe_allow_html=True)
                        st.image(img,width=300)  
        with colB:
            st.subheader("Generar imagen con DALL-E 3")
            if st.session_state["mejor_score"] < st.session_state["umbral"]:
                # Generar imagen con DALL-E 3
                if st.button("Generar imagen con DALL-E 3", key="generar_imagen_dalle"):
                    #url = generar_imagen(concepto)
                    url = "imagenDalle.png" #simular la generacion de imagen
                    st.session_state["url_dalle"] = url

                if st.session_state.get("url_dalle"):
                    st.image(st.session_state["url_dalle"], caption="Imagen generada por DALL-E 3", width=300)
                    #st.image("imagenDalle.png", caption="Imagen generada por DALL-E 3", width=300)
            else:
                st.warning("El score de la mejor imagen es mayor al umbral, no se generará una imagen con DALL-E 3.")
            st.session_state["url_dalle"] = None
