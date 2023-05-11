import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from preguntas_respuestas import preguntas_respuestas
from util import procesar_entrada

def inicializar_modelo(modelo_nombre):
    model = AutoModelForCausalLM.from_pretrained(modelo_nombre)
    tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def generar_respuesta(prompt, model, tokenizer, device):
    entrada_procesada = procesar_entrada(prompt)
    if entrada_procesada in preguntas_respuestas:
        return preguntas_respuestas[entrada_procesada]
    else:
        try:
            input_ids = tokenizer(procesar_entrada(prompt), return_tensors="pt").input_ids.to(device)
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
            return tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Ocurrió un error: {e}")
            return "Lo siento, no pude generar una respuesta."

def main():
    model, tokenizer, device = inicializar_modelo("./modelo_ajustado")
    while True:
        prompt = input("Usuario: ")
        if prompt.lower() == 'salir':
            break
        rta = generar_respuesta(prompt, model, tokenizer, device)
        print(f"Chatbot: {rta}")

if __name__ == '__main__':
    main()
