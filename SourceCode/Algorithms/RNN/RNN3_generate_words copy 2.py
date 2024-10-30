import os
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from datetime import datetime
import platform
import psutil

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

# Directory setup
output_dir = "SourceCode/Generated_Dict"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"generated_words_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

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

# Main word generation with multithreading for individual words
start_char = 'o'
word_length = 8
temperature = 0.5
total_words = 1000000
#max_concurrent_threads = 128  # Max number of threads to run at a time

# Display thread and generation information
print(f"Total words to generate: {total_words}")
#print(f"Max concurrent threads: {max_concurrent_threads}")

# File output with combined progress monitoring
with open(output_file, "w") as file:
    with tqdm.tqdm(total=total_words, desc="Total Progress") as total_progress:
        #with ThreadPoolExecutor(max_workers=max_concurrent_threads) as executor:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_word, model, start_char, word_length, temperature) for _ in range(total_words)]
            
            # Process each generated word as it completes
            for future in as_completed(futures):
                word = future.result()
                file.write(word + '\n')  # Write each word to file
                total_progress.update(1)  # Update combined progress bar

print(f"Word generation complete. Output saved to {output_file}")
