import time
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from peft import PeftModel
import matplotlib.pyplot as plt

def measure_inference_time(model, tokenizer, sentences, device="cuda"):
    model.to(device)
    total_time = 0
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        start = time.time()
        _ = model.generate(**inputs)
        total_time += time.time() - start
    return total_time / len(sentences)  

def measure_gpu_memory(model, tokenizer, sentences, device="cuda"):
    torch.cuda.empty_cache()
    model.to(device)
    torch.cuda.reset_peak_memory_stats()
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        _ = model.generate(**inputs)

    peak_memory = torch.cuda.max_memory_allocated(device) / 1e6  # Convert to MB
    return peak_memory

def main():
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh")
    test_data = dataset["test"]

    sentences = [item['translation']['en'] for item in test_data][:100] 

    base_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base").to("cuda")
    lora_r8_finetuned = PeftModel.from_pretrained(base_model, "./model_lora_r8").to("cuda")
    lora_r4_finetuned = PeftModel.from_pretrained(base_model, "./model_lora_r4").to("cuda")
    full_finetuned = AutoModelForSeq2SeqLM.from_pretrained("./model_fullfinetune", use_safetensors=False).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

    full_model_time = measure_inference_time(full_finetuned, tokenizer, sentences)
    lora_r8_model_time = measure_inference_time(lora_r8_finetuned, tokenizer, sentences)
    lora_r4_model_time = measure_inference_time(lora_r4_finetuned, tokenizer, sentences)

    print(f"Full fine-tune Avg Inference Time: {full_model_time:.4f} sec/sentence")
    print(f"LoRA fine-tune with rank 8 Avg Inference Time: {lora_r8_model_time:.4f} sec/sentence")
    print(f"LoRA fine-tune with rank 4 Avg Inference Time: {lora_r4_model_time:.4f} sec/sentence")
    full_memory = measure_gpu_memory(full_finetuned, tokenizer, sentences)
    lora_r8_memory = measure_gpu_memory(lora_r8_finetuned, tokenizer, sentences)
    lora_r4_memory = measure_gpu_memory(lora_r4_finetuned, tokenizer, sentences)
    
    print(f"Full Fine-tune Peak Memory: {full_memory:.2f} MB")
    print(f"LoRA Fine-tune with rank 8 Peak Memory: {lora_r8_memory:.2f} MB")
    print(f"LoRA Fine-tune with rank 4 Peak Memory: {lora_r4_memory:.2f} MB")
    methods = ["Full Fine-tune", "LoRA Fine-tune with rank 8", "LoRA Fine-tune with rank 4"]
    times = [full_model_time, lora_r8_model_time, lora_r4_model_time]
    memory_usage = [full_memory, lora_r8_memory, lora_r4_memory]
    plt.bar(methods, times, color=["red", "blue", 'green'])
    plt.xlabel("Method")
    plt.ylabel("Avg Inference Time (sec)")
    plt.title("Inference Speed Comparison")
    plt.show()

    plt.bar(methods, memory_usage, color=["red", "blue", 'green'])
    plt.xlabel("Method")
    plt.ylabel("GPU Memory (MB)")
    plt.title("GPU Memory Usage Comparison")
    plt.show()

if __name__ == "__main__":
    main()

