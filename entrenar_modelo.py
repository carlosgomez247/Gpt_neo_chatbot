from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

MODEL_PATH = "./modelo_ajustado"
DATA_PATH = './datasets/horario_sof_noche_prosa.txt'
OUTPUT_PATH = "./resultados_de_entrenamiento"

# Subclase de Trainer con una función de pérdida personalizada
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def inicializar_modelo_y_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def preparar_datasets(tokenizer, data_path, train_size=0.9):
    def tokenize(examples):
        # Tokenizando y rellenando a la longitud máxima de 128
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)
    
    # Cargando y tokenizando el conjunto de datos
    dataset = load_dataset('text', data_files={'train': data_path})
    dataset = dataset['train'].map(tokenize, batched=True)
    
    # Dividiendo el conjunto de datos en entrenamiento y validación
    full_dataset = dataset.train_test_split(train_size=train_size)
    
    # Preparando los conjuntos de datos finales
    train_dataset = full_dataset['train'].map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    eval_dataset = full_dataset['test'].map(lambda examples: {'labels': examples['input_ids']}, batched=True)

    return train_dataset, eval_dataset

# Función para entrenar el modelo
def entrenar_modelo(model, tokenizer, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,           # Directorio de salida
        overwrite_output_dir=True,        # Sobreescribir el directorio de salida
        num_train_epochs=1,               # Número de épocas de entrenamiento
        per_device_train_batch_size=1,    # Tamaño del lote de entrenamiento por dispositivo
        per_device_eval_batch_size=1,     # Tamaño del lote de evaluación por dispositivo
        eval_steps=500,                   # Realizar una evaluación cada 500 pasos
        save_steps=1000,                  # Guardar el modelo cada 1000 pasos
        save_total_limit=2,               # Límite para el número total de puntos de control
        prediction_loss_only=False,       # Si es verdadero, solo la pérdida de predicción se usa para la evaluación
        evaluation_strategy="steps",      # Estrategia de evaluación
        optim="adamw_torch"               # Optimizador a utilizar
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

def guardar_modelo_y_tokenizer(model, tokenizer, model_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

def main():
    model, tokenizer = inicializar_modelo_y_tokenizer(MODEL_PATH)
    train_dataset, eval_dataset = preparar_datasets(tokenizer, DATA_PATH)
    entrenar_modelo(model, tokenizer, train_dataset, eval_dataset)
    guardar_modelo_y_tokenizer(model, tokenizer, MODEL_PATH)

if __name__ == "__main__":
    main()
