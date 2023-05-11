from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

MODEL_PATH = "./modelo_ajustado"
DATA_PATH = './datasets/horario_sof_noche_prosa.txt'
OUTPUT_PATH = "./resultados_de_entrenamiento"

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
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)
    
    dataset = load_dataset('text', data_files={'train': data_path})
    dataset = dataset['train'].map(tokenize, batched=True)
    full_dataset = dataset.train_test_split(train_size=train_size)
    train_dataset = full_dataset['train'].map(lambda examples: {'labels': examples['input_ids']}, batched=True)
    eval_dataset = full_dataset['test'].map(lambda examples: {'labels': examples['input_ids']}, batched=True)

    return train_dataset, eval_dataset

def entrenar_modelo(model, tokenizer, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_steps=500,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=False,
        evaluation_strategy="steps",
        optim="adamw_torch"  
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
