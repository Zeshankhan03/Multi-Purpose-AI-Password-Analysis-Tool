import os
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from datetime import datetime
import platform
import psutil
from collections import defaultdict, Counter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RNN Model Definition
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out.reshape(out.size(0) * out.size(1), out.size(2)))
        return out, h

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

# Load model and character mappings
char_to_idx = torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\char_to_idx.pth')
idx_to_char = torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\idx_to_char.pth')
chars = sorted(char_to_idx.keys())
input_size = len(chars)
hidden_size = 64
num_layers = 2

model = RNNModel(input_size, hidden_size, num_layers, len(chars)).to(device)
model.load_state_dict(torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\rnn_model.pth'))

# System information display
print(f"Device: {device}")
print(f"Platform: {platform.system()} {platform.release()} ({platform.processor()})")
print(f"Total Memory: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

# Directory setup for output
output_dir = "SourceCode/Generated_Dict"
os.makedirs(output_dir, exist_ok=True)

# Function to generate a single word
def generate_word(model, start_char, length, temperature=1.0):
    model.eval()
    h = model.init_hidden(1)
    input = torch.eye(len(chars))[char_to_idx[start_char]].unsqueeze(0).unsqueeze(0).to(device)
    word = start_char

    with torch.no_grad():
        for _ in range(length - 1):
            output, h = model(input, h)
            output = output[-1] / temperature
            probabilities = torch.softmax(output, dim=0).cpu().numpy()
            generated_idx = np.random.choice(len(chars), p=probabilities)
            next_char = idx_to_char[generated_idx]
            word += next_char
            input = torch.eye(len(chars))[char_to_idx[next_char]].unsqueeze(0).unsqueeze(0).to(device)
    
    return word

# Configuration
word_length = 8
temperature = 0.5
total_words = 1000000
most_common_count = 10  # Number of top starting characters to create sets

# Generate words with progress monitoring and track starting characters
start_char_count = Counter()
generated_words = defaultdict(list)

print(f"Total words to generate: {total_words}")

with tqdm.tqdm(total=total_words, desc="Generating Words") as total_progress:
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_word, model, start_char, word_length, temperature) for _ in range(total_words)]
        
        for future in as_completed(futures):
            word = future.result()
            start_char = word[0]
            start_char_count[start_char] += 1
            generated_words[start_char].append(word)
            total_progress.update(1)

# Select the most common starting characters and equally divide words among them
most_common_starts = [char for char, _ in start_char_count.most_common(most_common_count)]
output_files = {}

# Save divided sets to separate files
for start_char in most_common_starts:
    set_filename = os.path.join(output_dir, f"generated_words_{start_char}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    output_files[start_char] = set_filename
    with open(set_filename, "w") as file:
        for word in generated_words[start_char]:
            file.write(word + '\n')

print(f"Word generation complete. Output saved to individual files for each starting character.")
print("Files created for each set:")
for char, filepath in output_files.items():
    print(f" - {char}: {filepath}")
