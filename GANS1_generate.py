# generate_gan_words.py

import torch
import torch.nn as nn
import numpy as np

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

char_to_idx = torch.load('ML_Algorithms/GANS_Algorithms/char_to_idx.pth')
idx_to_char = torch.load('ML_Algorithms/GANS_Algorithms/idx_to_char.pth')
chars = sorted(char_to_idx.keys())

print(f"Size of char_to_idx: {len(char_to_idx)}")
print(f"Size of idx_to_char: {len(idx_to_char)}")

latent_size = 100
hidden_size = 128
num_layers = 2
max_length = 10  # Maximum length of generated words

G = Generator(latent_size, hidden_size, num_layers, len(chars)).to(device)
G.load_state_dict(torch.load('ML_Algorithms/GANS_Algorithms/gan_generator.pth'))

def generate_word(G, length):
    G.eval()
    z = torch.randn(1, length, latent_size).to(device)
    h = G.init_hidden(1)
    with torch.no_grad():
        generated_data, _ = G(z, h)
        generated_data = torch.softmax(generated_data, dim=-1)
        generated_data = torch.argmax(generated_data, dim=-1).squeeze().cpu().numpy()
    
    word = ''.join([idx_to_char[idx] for idx in generated_data])
    return word

start_char = 'a'  # Starting character for word generation
word_length = 5   # Length of the word to generate
new_word = generate_word(G, word_length)
print(f'Generated word: {new_word}')
