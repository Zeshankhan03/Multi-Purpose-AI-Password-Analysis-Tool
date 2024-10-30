import torch
import torch.nn as nn
import numpy as np

torch.cuda.is_available()

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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
print(device)
char_to_idx = torch.load(R'SourceCode\Algorithms\RNN\char_to_idx.pth')
idx_to_char = torch.load(R'SourceCode\Algorithms\RNN\idx_to_char.pth')
chars = sorted(char_to_idx.keys())

input_size = len(chars)
hidden_size = 64
num_layers = 2

model = RNNModel(input_size, hidden_size, num_layers, len(chars)).to(device)
model.load_state_dict(torch.load(R'SourceCode\Algorithms\RNN\rnn_model.pth'))

def generate_word(model, start_char, length, temperature=1.0):
    model.eval()
    h = model.init_hidden(1)
    input = torch.eye(len(chars))[char_to_idx[start_char]].unsqueeze(0).unsqueeze(0).to(device)
    word = start_char

    with torch.no_grad():
        for _ in range(length - 1):
            output, h = model(input, h)
            output = output[-1] / temperature  # Apply temperature
            probabilities = torch.softmax(output, dim=0).cpu().numpy()
            generated_idx = np.random.choice(len(chars), p=probabilities)  # Sample from distribution
            
            next_char = idx_to_char[generated_idx]
            word += next_char
            input = torch.eye(len(chars))[char_to_idx[next_char]].unsqueeze(0).unsqueeze(0).to(device)

    return word

start_char = 'o'  # Starting character for word generation
word_length = 8   # Length of each word to generate
temperature = 0.5  # Adjust temperature to control randomness

# Loop to generate 40 words
num_words = 10000
generated_words = []
for _ in range(num_words):
    new_word = generate_word(model, start_char, word_length, temperature)
    generated_words.append(new_word)

# Print all generated words
for i, word in enumerate(generated_words, 1):
    print(f'Word {i}: {word}')
