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

# Character groups for password generation
uppercase_chars = [chr(i) for i in range(65, 91)]  # A-Z
lowercase_chars = [chr(i) for i in range(97, 123)] # a-z
special_chars = ['!', '@', '#', '$', '%', '^', '&', '(', ')', '_', '+', '-', '=']

# Combined set of characters for iteration
starting_chars = uppercase_chars + lowercase_chars

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

# Generate words for each starting character and save them in separate files
word_length = 8
temperature = 0.5
words_per_char = 10000  # Number of words per starting character 65*count
#max_concurrent_threads = 128  # Max number of threads to run at a time


# Generate words starting with uppercase and lowercase characters separately
for start_char in starting_chars:
    output_file = os.path.join(output_dir, f"generated_words_{start_char}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    print(f"Generating words starting with '{start_char}'...")
    
    with open(output_file, "w") as file:
        with tqdm.tqdm(total=words_per_char, desc=f"Generating for {start_char}") as progress_bar:
            #with ThreadPoolExecutor(max_workers=max_concurrent_threads) as executor:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(generate_word, model, start_char, word_length, temperature) for _ in range(words_per_char)]
                
                for future in as_completed(futures):
                    word = future.result()
                    file.write(word + '\n')
                    progress_bar.update(1)

    print(f"Words starting with '{start_char}' saved to {output_file}")

# Generate all special character words in a single file
special_output_file = os.path.join(output_dir, f"generated_words_special_chars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
print("Generating words starting with special characters...")

with open(special_output_file, "w") as special_file:
    with tqdm.tqdm(total=words_per_char * len(special_chars), desc="Generating for special characters") as progress_bar:
        with ThreadPoolExecutor() as executor:
            for start_char in special_chars:
                futures = [executor.submit(generate_word, model, start_char, word_length, temperature) for _ in range(words_per_char)]
                
                for future in as_completed(futures):
                    word = future.result()
                    special_file.write(word + '\n')
                    progress_bar.update(1)

print(f"Words starting with special characters saved to {special_output_file}")
print("Word generation complete for all specified starting characters.")
