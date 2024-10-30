import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model and tokenizer
print("Loading model and tokenizer...")
model = GPT2LMHeadModel.from_pretrained('ML_Algorithms/Transformer/model')
tokenizer = GPT2Tokenizer.from_pretrained('ML_Algorithms/Transformer/model')

# Move model to GPU if available
model.to(device)
model.eval()
print("Model and tokenizer loaded.")

# Generate new words
def generate_words(seed_text, max_length=50):
    print(f"Generating new words for seed text: '{seed_text}'")
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    input_ids = input_ids.to(device)  # Move input to GPU
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
seed_text = "e"
generated_text = generate_words(seed_text)
print("Generated text:", generated_text)
