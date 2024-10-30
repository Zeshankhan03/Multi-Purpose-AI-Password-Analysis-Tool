# train_gan.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Parameters
latent_size = 100
hidden_size = 128
num_layers = 2
num_epochs = 10
learning_rate = 0.0002
batch_size = 64

print("Reading the Dataset.")
def read_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()
    return words

words = read_words('ML_Algorithms/GANS_Algorithms/small.txt')
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
max_length = max(len(seq) for seq in sequences)
print("Preprocessing the Dataset completed.")

# Padding sequences to the same length
def pad_sequences(sequences, max_length, pad_value=0):
    padded_sequences = []
    for seq in sequences:
        padded_seq = seq + [pad_value] * (max_length - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences

sequences = pad_sequences(sequences, max_length)

class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, num_layers, num_classes):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create models
G = Generator(latent_size, hidden_size, num_layers, len(chars)).to(device)
D = Discriminator(len(chars), hidden_size, num_layers).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=learning_rate)
D_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

def one_hot_encode(sequences, num_classes):
    batch_size = len(sequences)
    max_length = len(sequences[0])
    one_hot = np.zeros((batch_size, max_length, num_classes), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, idx in enumerate(seq):
            one_hot[i, j, idx] = 1.0
    return one_hot

real_labels = torch.ones(batch_size, 1).to(device)
fake_labels = torch.zeros(batch_size, 1).to(device)

sequences = one_hot_encode(sequences, len(chars))
sequences = torch.tensor(sequences, dtype=torch.float32).to(device)

# Train the GAN
for epoch in range(num_epochs):
    for i in range(0, len(sequences), batch_size):
        real_data = sequences[i:i+batch_size]
        if real_data.size(0) != batch_size:
            continue
        
        # Train Discriminator
        D_optimizer.zero_grad()
        h_real = D.init_hidden(batch_size)
        outputs = D(real_data, h_real)
        D_loss_real = criterion(outputs, real_labels)
        D_loss_real.backward()
        
        z = torch.randn(batch_size, max_length, latent_size).to(device)
        h_fake = G.init_hidden(batch_size)
        fake_data, _ = G(z, h_fake)
        fake_data = torch.softmax(fake_data, dim=-1)
        
        h_fake = D.init_hidden(batch_size)
        outputs = D(fake_data.detach(), h_fake)
        D_loss_fake = criterion(outputs, fake_labels)
        D_loss_fake.backward()
        
        D_optimizer.step()
        
        D_loss = D_loss_real + D_loss_fake
        
        # Train Generator
        G_optimizer.zero_grad()
        h_fake = D.init_hidden(batch_size)
        outputs = D(fake_data, h_fake)
        G_loss = criterion(outputs, real_labels)
        G_loss.backward()
        
        G_optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {D_loss.item():.4f}, G Loss: {G_loss.item():.4f}')

# Save the model
torch.save(G.state_dict(), 'ML_Algorithms/GANS_Algorithms/gan_generator.pth')
torch.save(D.state_dict(), 'ML_Algorithms/GANS_Algorithms/gan_discriminator.pth')
torch.save(char_to_idx, 'ML_Algorithms/GANS_Algorithms/char_to_idx.pth')
torch.save(idx_to_char, 'ML_Algorithms/GANS_Algorithms/idx_to_char.pth')
