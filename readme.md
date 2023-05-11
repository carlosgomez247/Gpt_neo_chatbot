# Proyecto de Chatbot

Este proyecto contiene dos scripts principales: `chatbot.py` y `entrenar_modelo.py`.

modelo base : EleutherAI/gpt-neo-125M

El script `chatbot.py` inicializa un modelo preentrenado de GPT-Neo y permite al usuario interactuar con el chatbot. El chatbot responde a las entradas del usuario basándose en el conocimiento que ha aprendido durante su entrenamiento.

El script `entrenar_modelo.py` entrena el modelo GPT-Neo en un conjunto de datos de texto especificado. El modelo y el tokenizer son guardados en un directorio especificado.

# Version de python y modulos
Python 3.11.3
torch==2.0.0
transformers==4.28.1
datasets==2.12.0

## Uso

Antes de usar estos scripts, asegúrate de instalar las dependencias requeridas con:

```bash
pip install -r requirements.txt
```

El modelo ajustado se puede descargar del siguiente enlace: [falta por subir]

Después de descargar el modelo, colócalo en la carpeta ./modelo_ajustado del directorio principal del proyecto.
