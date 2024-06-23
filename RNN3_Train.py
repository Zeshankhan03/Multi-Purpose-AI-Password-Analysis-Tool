# train_rnn.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Parameters
hidden_size = 64
num_layers = 2
num_epochs = 1  # Reduced epochs for demonstration; increase as needed
learning_rate = 0.003

print("Reading the Dataset.")
def read_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words

words = read_words('ML_Algorithms\\RNN\\small.txt')
print("Reading the Dataset Completed.")

# Create character vocabulary
chars = sorted(list(set(''.join(words))))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

input_size = len(chars)

print(f"Size of char_to_idx: {len(char_to_idx)}")
print(f"Size of idx_to_char: {len(idx_to_char)}")

print("Preprocessing the Dataset.")
def preprocess(words, char_to_idx):
    sequences = []
    for word in words:
        sequences.append([char_to_idx[char] for char in word])
    return sequences

sequences = preprocess(words, char_to_idx)
print("Preprocessing the Dataset completed.")

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
model = RNNModel(input_size, hidden_size, num_layers, len(chars)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    h = model.init_hidden(1)
    loss_avg = 0
    with tqdm(total=len(sequences), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
        for seq in sequences:
            inputs = torch.eye(len(chars))[seq[:-1]].unsqueeze(0).to(device)
            targets = torch.tensor(seq[1:], dtype=torch.long).to(device)

            h = h.detach()
            outputs, h = model(inputs, h)

            loss = criterion(outputs, targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()
            pbar.update(1)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_avg/len(sequences):.4f}')

torch.save(model.state_dict(), 'ML_Algorithms/RNN/rnn_model.pth')
torch.save(char_to_idx, 'ML_Algorithms/RNN/char_to_idx.pth')
torch.save(idx_to_char, 'ML_Algorithms/RNN/idx_to_char.pth')
