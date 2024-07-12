import os
import json
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import csv
from nltk.translate.bleu_score import sentence_bleu
import sys

# Set file paths
data_path = "/Users/sanvishukla/Desktop/SRIP/HintDroid-main/EvaluationData/analyze.jsonl"
save_path = "/Users/sanvishukla/Desktop/SRIP"

# Load the dataset
with open(data_path, 'r') as f:
    data = [json.loads(line) for line in f]

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a padding token

model = GPT2LMHeadModel.from_pretrained(model_name)

# Custom Dataset Class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"Q: {item['prompt']}\nA: {item['completion']}"
        inputs = self.tokenizer(prompt, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")
        return inputs

# Create Datasets
train_dataset = CustomDataset(train_data, tokenizer)
test_dataset = CustomDataset(test_data, tokenizer)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir=os.path.join(save_path, "fine-tuned-model"),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    logging_dir=os.path.join(save_path, "logs"),
    logging_steps=10,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=1000,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

trainer.train()

# Save the fine-tuned model and tokenizer
fine_tuned_model_path = os.path.join(save_path, "fine-tuned-model")
model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)

print(f"Fine-tuned model saved at: {fine_tuned_model_path}")

# Function to generate hints
def generate_hints(model, tokenizer, prompts, device='cpu'):
    generated_hints = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        generated_hint = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_hints.append(generated_hint)
    return generated_hints

data_path = "/Users/sanvishukla/Desktop/SRIP/HintDroid-main/EvaluationData/analyze.jsonl"
save_path = "/Users/sanvishukla/Desktop/SRIP"
model_path = "/Users/sanvishukla/Desktop/SRIP/fine-tuned-model"

# Load the dataset
with open(data_path, 'r') as f:
    data = [json.loads(line) for line in f]

# Split the dataset into training and testing sets (ensure the same split as during training)
_, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a padding token
model = GPT2LMHeadModel.from_pretrained(model_path)

# Function to generate hints
def generate_hints(model, tokenizer, prompts, device='cpu'):
    model.to(device)
    generated_hints = []
    max_new_tokens = 50  # Adjust as needed
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=len(input_ids[0]) + max_new_tokens,  # Adjust max_length dynamically
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        generated_hint = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_hints.append(generated_hint)
        
        print(f"Generated hint: {generated_hint}")  # Add debugging prints
        
    return generated_hints

# Function to calculate BLEU scores
def calculate_bleu_scores(reference, hypothesis):
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0))
    bleu_3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.3, 0.3, 0.3, 0))
    bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu_1, bleu_2, bleu_3, bleu_4

# Generate hints for test dataset
prompts = [item['prompt'] for item in test_data]
generated_hints = generate_hints(model, tokenizer, prompts, device='cpu')

# Save BLEU scores to a CSV file
csv_file = os.path.join(save_path, "bleu_scores.csv")
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "ground_truth", "generated_hint", "bleu_1", "bleu_2", "bleu_3", "bleu_4"])
    
    for idx, item in enumerate(test_data):
        prompt = item['prompt'].replace('\n', ' ')
        ground_truth = item['completion'].replace('\n', ' ')
        generated_hint = generated_hints[idx]
        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_scores(ground_truth, generated_hint)
        
        writer.writerow([prompt, ground_truth, generated_hint, bleu_1, bleu_2, bleu_3, bleu_4])
        print(f"Processed prompt: {prompt}")
        print(f"Ground truth: {ground_truth}")
        print(f"Generated hint: {generated_hint}")
        print(f"BLEU scores - 1: {bleu_1}, 2: {bleu_2}, 3: {bleu_3}, 4: {bleu_4}")
        sys.stdout.flush()  # Ensure the print statements are flushed

print(f"BLEU scores saved to: {csv_file}")

