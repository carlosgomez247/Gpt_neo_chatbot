import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from preguntas_respuestas import preguntas_respuestas
from util import procesar_entrada

GPTNEO_MODEL_PATH = "./modelo_ajustado_GPTNEO"
# GPTNEO_MODEL_PATH = "EleutherAI/gpt-neo-125M" 
def inicializar_modelo(modelo_nombre):
    # Carga el modelo y el tokenizador
    # un modelo es una estructura que ha sido entrenada para realizar una tarea específica. En este caso, el modelo es una versión de GPT-Neo, que es una red neuronal de transformadores diseñada para generar texto. El modelo ha sido preentrenado en un gran conjunto de datos y ha aprendido a predecir la próxima palabra en una secuencia de texto.
    model = AutoModelForCausalLM.from_pretrained(modelo_nombre)
    # Un tokenizador es una herramienta que se utiliza para dividir el texto en "tokens", que son las unidades mínimas de significado que la red neuronal puede entender. Por ejemplo, un token podría ser una palabra, un carácter, o una subpalabra dependiendo del tokenizador. En el caso de GPT-Neo, el tokenizador divide el texto en subpalabras.
    tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
    # Determina si hay un dispositivo de GPU disponible y, si no, usa la CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mueve el modelo al dispositivo correspondiente
    model.to(device)
    return model, tokenizer, device

def generar_respuesta(prompt, model, tokenizer, device):
    entrada_procesada = procesar_entrada(prompt)
    # Comprueba si la entrada del usuario está en el diccionario de preguntas y respuestas
    if entrada_procesada in preguntas_respuestas:
        return preguntas_respuestas[entrada_procesada]
    else:
        try:
            # Codifica la entrada del usuario y la pasa al modelo
            input_ids = tokenizer(procesar_entrada(prompt), return_tensors="pt").input_ids.to(device)
            # Genera una respuesta a partir del modelo
            # max_length define la longitud máxima de la respuesta
            # temperature afecta la aleatoriedad de la respuesta. Un valor más alto hace que la respuesta sea más aleatoria
            # do_sample determina si se debe muestrear aleatoriamente según las probabilidades de las palabras
            # top_k  Durante el proceso de generación de texto, el modelo seleccionará la próxima palabra de un conjunto de opciones. El parámetro top_k limita estas opciones a las k palabras más probables. Esto puede ayudar a prevenir respuestas poco probables y sin sentido. Sin embargo, un valor demasiado bajo puede hacer que las respuestas sean predecibles y aburridas.
            # top_p Este es otro método para limitar las opciones de la próxima palabra, conocido como muestreo "nucleus". En lugar de seleccionar de las k palabras más probables, selecciona el conjunto más pequeño de palabras tal que la suma de sus probabilidades es al menos p. Esto puede permitir una mayor diversidad que top_k mientras todavía evita las palabras extremadamente improbables.
            # num_return_sequences: Este parámetro determina cuántas secuencias de texto independientes se deben generar a partir de cada entrada. En este caso, solo se está generando una secuencia para cada entrada del usuario.
            # pad_token_id: Este es el ID del token que el tokenizador usa para rellenar secuencias de texto que son más cortas que la longitud máxima permitida. En el caso de GPT-Neo, este es el mismo que el ID del token EOS (End Of Sentence), que indica el final de una secuencia de texto.
            output = model.generate(
                input_ids=input_ids,
                max_length=100,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            # Decodifica la salida del modelo en texto legible
            return tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Ocurrió un error: {e}")
            return "Lo siento, no pude generar una respuesta."

def main():
    model, tokenizer, device = inicializar_modelo(GPTNEO_MODEL_PATH)
    while True:        
        prompt = input("Usuario: ")        
        if prompt.lower() == 'salir':
            break        
        rta = generar_respuesta(prompt, model, tokenizer, device)
        print(f"Chatbot: {rta}")

if __name__ == '__main__':
    main()
