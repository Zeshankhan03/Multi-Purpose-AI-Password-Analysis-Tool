
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words

print("Loading words from file...")
words = load_words('ML_Algorithms/Transformer/small.txt')  # Update with your file path
print(f"Loaded {len(words)} words.")

print("Initializing tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("Tokenizing words...")
tokens = tokenizer(words, return_tensors='pt', padding=True, truncation=True)
dataset = torch.utils.data.TensorDataset(tokens['input_ids'], tokens['attention_mask'])

# Define data loader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print("Data loader created.")

# Define model
print("Loading pre-trained GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Resize model embeddings to account for the new special tokens
model.resize_token_embeddings(len(tokenizer))

# Move model to GPU if available
model.to(device)

# Training setup
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()
print("Training setup complete.")

# Determine the number of steps per epoch
steps_per_epoch = len(dataloader)
print(f"Steps per epoch: {steps_per_epoch}")

# Training loop
num_epochs = 3
print("Starting training...")
for epoch in range(num_epochs):  # number of epochs
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_iterator = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(epoch_iterator):
        optimizer.zero_grad()
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)  # Move batch to GPU
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_iterator.set_postfix({'loss': loss.item(), 'step': step+1})
    print(f"Epoch {epoch+1} completed.")

# Save model
print("Saving model...")
model.save_pretrained('ML_Algorithms/Transformer/model') 
tokenizer.save_pretrained('ML_Algorithms/Transformer/model')
print("Model saved.")
