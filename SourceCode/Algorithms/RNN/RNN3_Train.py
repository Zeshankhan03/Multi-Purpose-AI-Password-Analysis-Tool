import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# Parameters
hidden_size = 64
num_layers = 2
num_epochs = 50  # Reduced epochs for demonstration; increase as needed
learning_rate = 0.003
batch_size = 222000  # Adjust batch size here

print("Reading the Dataset.")
def read_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words

words = read_words('ML_Algorithms\RNN\Human\length_8\\filtered_words_drealhuman_phill.txt')
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
    targets = []
    for word in words:
        sequence = [char_to_idx[char] for char in word[:-1]]
        target = [char_to_idx[char] for char in word[1:]]
        sequences.append(sequence)
        targets.append(target)
    return sequences, targets

sequences, targets = preprocess(words, char_to_idx)
sequences = torch.tensor(sequences, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

dataset = TensorDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
#device = torch.device('cpu')
model = RNNModel(input_size, hidden_size, num_layers, len(chars)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    loss_avg = 0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
        for batch_sequences, batch_targets in dataloader:
            batch_size_actual = batch_sequences.size(0)  # In case the last batch is smaller
            h = model.init_hidden(batch_size_actual)

            inputs = torch.eye(len(chars))[batch_sequences].to(device)
            targets = batch_targets.to(device)

            h = h.detach()
            outputs, h = model(inputs, h)

            loss = criterion(outputs, targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()
            pbar.update(1)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_avg/len(dataloader):.4f}')

torch.save(model.state_dict(), 'ML_Algorithms/RNN/Human/rnn_model.pth')
torch.save(char_to_idx, 'ML_Algorithms/RNN/Human/char_to_idx.pth')
torch.save(idx_to_char, 'ML_Algorithms/RNN/Human/idx_to_char.pth')
