import torch
import torch.nn as nn
import numpy as np

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

char_to_idx = torch.load('ML_Algorithms/RNN/Human/char_to_idx.pth')
idx_to_char = torch.load('ML_Algorithms/RNN/Human/idx_to_char.pth')
chars = sorted(char_to_idx.keys())

print(f"Size of char_to_idx: {len(char_to_idx)}")
print(f"Size of idx_to_char: {len(idx_to_char)}")

input_size = len(chars)
hidden_size = 64
num_layers = 2

model = RNNModel(input_size, hidden_size, num_layers, len(chars)).to(device)
model.load_state_dict(torch.load('ML_Algorithms/RNN/Human/rnn_model.pth'))

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

start_char = 's'  # Starting character for word generation
word_length = 8   # Length of the word to generate
temperature = 0.8  # Adjust temperature to control randomness
new_word = generate_word(model, start_char, word_length, temperature)
print(f'Generated word: {new_word}')
