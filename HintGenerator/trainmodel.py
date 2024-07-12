import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch

# Function to tokenize dataset with truncation and padding
def tokenize_dataset(dataset_path):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    with open(dataset_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
    tokenized_datasets = []

    # Add a padding token to the tokenizer if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    max_length = 512  # Reduced max length for tokens
    
    for line in lines:
        data = json.loads(line.strip())
        prompt = data['prompt']
        completion = data['completion']
        
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')
        tokenized_completion = tokenizer(completion, truncation=True, max_length=max_length, padding="max_length", return_tensors='pt')

        tokenized_datasets.append({'input_ids': tokenized_prompt['input_ids'], 'labels': tokenized_completion['input_ids']})
        
    return tokenizer, tokenized_datasets

# Example dataset path
dataset_path = "/Users/sanvishukla/Desktop/SRIP/HintDroid-main/EvaluationData/analyze.jsonl"

# Tokenize the dataset
tokenizer, tokenized_datasets = tokenize_dataset(dataset_path)

# Training arguments with higher learning rate
training_args = TrainingArguments(
    output_dir="/Users/sanvishukla/Desktop/SRIP/fine-tuned-model",
    per_device_train_batch_size=1,  # Adjust based on memory availability
    num_train_epochs=3,
    logging_dir='./logs',
    fp16=False,  # Ensure fp16 is explicitly set to False
    learning_rate=1e-4  # Set a higher learning rate, adjust as necessary
)

# Initialize model and trainer
model = GPT2LMHeadModel.from_pretrained('gpt2')
device = torch.device('cpu')  # Explicitly set device to CPU
model.to(device)  # Move model to CPU

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=lambda data: {'input_ids': torch.stack([item['input_ids'] for item in data]),
                                'labels': torch.stack([item['labels'] for item in data])}
)

# Train the model on CPU
trainer.train()

# Save the fine-tuned model
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

print(f"Fine-tuned model and tokenizer saved to {training_args.output_dir}")
