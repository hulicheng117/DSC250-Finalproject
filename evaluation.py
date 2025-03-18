import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import sacrebleu
from tqdm import tqdm


def bleu_evaluation(model_path, base_model_name, dataset, max_length=128, is_peft=False):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to("cuda")
    if is_peft:
        base_finetuned = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to("cuda")
        finetuned_model = PeftModel.from_pretrained(base_finetuned, model_path).to("cuda")
    else:
        finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    test_data = dataset["test"]

    def generate_translation(model, text, max_length):
        """Generates a translation using the model."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_length)
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    subset_data = test_data[:50]
    references = [example["zh"] for example in subset_data["translation"]]

    base_predictions = [
        generate_translation(base_model, example["en"], max_length)
        for example in tqdm(subset_data["translation"], desc="Base model predictions")
    ]

    ft_predictions = [
        generate_translation(finetuned_model, example["en"], max_length)
        for example in tqdm(subset_data["translation"], desc="Fine-tuned model predictions")
    ]
    base_bleu = sacrebleu.corpus_bleu(base_predictions, [references]).score
    ft_bleu = sacrebleu.corpus_bleu(ft_predictions, [references]).score

    return base_bleu, ft_bleu


def bleu_evaluation_other_model(base_model_name, dataset, max_length=128):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    test_data = dataset["test"]

    def generate_translation(model, text, max_length):
        """Generates a translation using the model."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_length)
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    subset_data = test_data[:50]
    references = [example["zh"] for example in subset_data["translation"]]

    base_predictions = [
        generate_translation(base_model, example["en"], max_length)
        for example in tqdm(subset_data["translation"], desc="Base model predictions")
    ]
    base_bleu = sacrebleu.corpus_bleu(base_predictions, [references]).score

    return base_bleu


def ter_evaluation(model_path, base_model_name, dataset, max_length=128, is_peft=False):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to("cuda")
    if is_peft:
        base_finetuned = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to("cuda")
        finetuned_model = PeftModel.from_pretrained(base_finetuned, model_path).to("cuda")
    else:
        finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    test_data = dataset["test"]

    def generate_translation(model, text, max_length):
        """Generates a translation using the model."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_length)
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    subset_data = test_data[:50]
    references = [example["zh"] for example in subset_data["translation"]]

    base_predictions = [
        generate_translation(base_model, example["en"], max_length)
        for example in tqdm(subset_data["translation"], desc="Base model predictions")
    ]

    ft_predictions = [
        generate_translation(finetuned_model, example["en"], max_length)
        for example in tqdm(subset_data["translation"], desc="Fine-tuned model predictions")
    ]


    base_ter = sacrebleu.corpus_ter(base_predictions, [references]).score
    ft_ter = sacrebleu.corpus_ter(ft_predictions, [references]).score

    return base_ter, ft_ter
def ter_evaluation_other_model(base_model_name, dataset, max_length=128):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    test_data = dataset["test"]

    def generate_translation(model, text, max_length):
        """Generates a translation using the model."""
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_length=max_length)
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    subset_data = test_data[:50]
    references = [example["zh"] for example in subset_data["translation"]]

    base_predictions = [
        generate_translation(base_model, example["en"], max_length)
        for example in tqdm(subset_data["translation"], desc="Base model predictions")
    ]

    base_ter = sacrebleu.corpus_ter(base_predictions, [references]).score
    return base_ter




if __name__ == "__main__":
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh")
    print("Starting BLEU evaluation on Helsinki-NLP/opus-mt-en-zh")
    base_bleu = bleu_evaluation_other_model("Helsinki-NLP/opus-mt-en-zh", dataset)
    print(f"Base BLEU: {base_bleu:.2f}")
    
    model_paths=['./model_lora_r8', './model_lora_r4','./model_fullfinetune']
    for model_path in model_paths: 
        print(f"Evaluating model at {model_path}...")
        if model_path == './model_fullfinetune':
            is_peft=False
        else:
            is_peft=True
        base_ter, ft_ter = bleu_evaluation(
            model_path=model_path, 
            base_model_name='t5-base', 
            dataset=dataset, 
            is_peft=is_peft
        )
        print(f"Model at {model_path} - Base BLEU: {base_ter}")
        print(f"Model at {model_path} - Fine-tuned BLEU: {ft_ter}")
        print('-' * 50)
        
    
    for model_path in model_paths: 
        print(f"Evaluating model at {model_path}...")
        if model_path == './model_fullfinetune':
            is_peft=False
        else:
            is_peft=True
        base_ter, ft_ter = ter_evaluation(
            model_path=model_path, 
            base_model_name='t5-base', 
            dataset=dataset, 
            is_peft=is_peft
        )
        print(f"Model at {model_path} - Base TER: {base_ter}")
        print(f"Model at {model_path} - Fine-tuned TER: {ft_ter}")
        print('-' * 50)
        
        
    print("Starting TER evaluation on Helsinki-NLP/opus-mt-en-zh")
    base_bleu = ter_evaluation_other_model("Helsinki-NLP/opus-mt-en-zh", dataset)
    print(f"Base TER: {base_bleu:.2f}")
    

        
    
    
    


    
    

