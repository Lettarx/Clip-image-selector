### **Proyecto en Desarrollo**

# Selector de imagenes

### Instalación

1. Clonar el repositorio

```

git clone https://github.com/Lettarx/Clip-image-selector.git

```
2. Instala las dependencias


```

pip install -r requirements.txt

```

3. Configurar la generación de imagenes

Para generar imagenes necesitaras una API key de OPENAI

En el archivo .env remplaza "your_openai_api_key_here" por tu API key y comenta la linea 152 de main y descomenta la 151

**Si no quieres generar imagenes no es necesario realizar los pasos anteriores, ya que la generación se simula con una imagen precargada**

4. Ejecuta el programa

En la carpeta raiz ejecuta el siguiente comando:

```

streamlit run main.py

```

