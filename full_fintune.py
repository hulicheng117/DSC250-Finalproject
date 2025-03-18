import time
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer, EarlyStoppingCallback


def load_data(dataset, tokenizer):
    inputs=[data["en"] for data in dataset["translation"]]
    targets=[data["zh"] for data in dataset["translation"]]
    model_inputs=tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels=tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"]=labels["input_ids"]
    return model_inputs

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    dataset=load_dataset("Helsinki-NLP/opus-100", "en-zh")
    model_name="google-t5/t5-base" 
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    tokenized_datasets=dataset.map(lambda x: load_data(x, tokenizer), batched=True, remove_columns=["translation"])
    train_subset= tokenized_datasets["train"].shuffle(seed=42).select(range(int(len(tokenized_datasets["train"]) * 0.01)))
    
    training_args = TrainingArguments(
        output_dir="./full_finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=1e-5,
        num_train_epochs=5,
        weight_decay=0.02, 
        warmup_steps=500,
        fp16=True if torch.cuda.is_available() else False,
        logging_dir="./logs_full_finetune", 
        logging_strategy="steps",  
        logging_steps=50,
        save_safetensors=False
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=train_subset,
        eval_dataset=tokenized_datasets["validation"],
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    trainer.save_model("./model_fullfinetune")
    tokenizer.save_pretrained("./model_fullfinetune")
    print("Training complete!")
    
if __name__ == "__main__":
    main()