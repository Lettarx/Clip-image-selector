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

st.title("Selector de imagenes")
openai.api_key = os.getenv("OPENAI_API_KEY")

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
def inferencia_CLIP(imagenes,concepto, _model, _processor, device):


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
    model, processor, device = cargar_modelo()  

    with st.form("subir_informacion"):
        #subida de imagenes, acepta multiples archivos
        imagenes = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"],accept_multiple_files=True)
        #ingreso del concepto a comparar
        concepto = st.text_input("Concepto de la imagen")
        #envio del formulario
        submit_button = st.form_submit_button("Enviar")

    if submit_button: 
        if imagenes and concepto: #si ingresaron imagenes y concepto
            #Mostrar mensaje de exito
            st.success("Imagenes y concepto enviados correctamente.")

            #generar_img = st.button("Generar Imagen")

            # if generar_img:
            #     url = generar_imagen(concepto)
            #     st.image(url, caption="Imagen generada por DALL-E 3", width=300)
            
            imagenes = [Image.open(imagen).convert("RGB") for imagen in imagenes]  # Convertir
            scores = inferencia_CLIP(imagenes, concepto, model, processor, device)

            for i, (img, score) in enumerate(zip(imagenes, scores)):
                #classname image container si score supera umbral el container es color verde 
                if score[0] > 0.2:
                    st.image(img, caption=f"Imagen {i+1} - Score: {score[0]:.4f}",width=300)
                else:
                    st.image(img, caption=f"Imagen {i+1} - Score: {score[0]:.4f}",width=300)  # Mostrar la imagen con el score
            
            url = generar_imagen(concepto)
            st.image(url, caption="Imagen generada por DALL-E 3", width=300)
    
        else:
            st.error("Por favor, sube al menos una imagen y proporciona un concepto.")
