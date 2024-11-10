import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import os
from typing import List, Dict, Tuple
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import the RNN model architecture
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

def load_model_and_mappings() -> Tuple[RNNModel, Dict, Dict]:
    """Load the trained model and character mappings"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load character mappings
    char_to_idx = torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\char_to_idx.pth')
    idx_to_char = torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\idx_to_char.pth')
    
    # Initialize model
    input_size = len(char_to_idx)
    hidden_size = 64
    num_layers = 2
    model = RNNModel(input_size, hidden_size, num_layers, input_size).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\rnn_model.pth'))
    model.eval()
    
    return model, char_to_idx, idx_to_char

def analyze_generated_words(directory: str) -> List[str]:
    """Read and combine all generated words"""
    all_words = []
    for filename in os.listdir(directory):
        if filename.startswith('generated_words_'):
            with open(os.path.join(directory, filename), 'r') as f:
                all_words.extend(f.read().splitlines())
    return all_words

def plot_character_distribution(words: List[str], save_path: str):
    """Plot character distribution in generated words"""
    char_counts = Counter(''.join(words))
    
    # Increased DPI and figure size
    plt.figure(figsize=(20, 12), dpi=300)
    chars, counts = zip(*sorted(char_counts.items()))
    
    # Enhanced plot styling
    plt.bar(chars, counts, color='skyblue', edgecolor='black')
    plt.title('Character Distribution in Generated Words', fontsize=16, pad=20)
    plt.xlabel('Characters', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save in high resolution
    plt.savefig(os.path.join(save_path, 'character_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_word_length_distribution(words: List[str], save_path: str):
    """Plot word length distribution"""
    lengths = [len(word) for word in words]
    
    # Increased DPI and figure size
    plt.figure(figsize=(16, 10), dpi=300)
    
    # Enhanced plot styling
    plt.hist(lengths, bins=20, color='lightgreen', edgecolor='black')
    plt.title('Word Length Distribution', fontsize=16, pad=20)
    plt.xlabel('Word Length', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save in high resolution
    plt.savefig(os.path.join(save_path, 'word_length_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_transition_probabilities(words: List[str], save_path: str):
    """Analyze and visualize character transition probabilities"""
    unique_chars = sorted(list(set(''.join(words))))
    n_chars = len(unique_chars)
    transitions = np.zeros((n_chars, n_chars))
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    
    for word in words:
        for c1, c2 in zip(word[:-1], word[1:]):
            i, j = char_to_index[c1], char_to_index[c2]
            transitions[i][j] += 1
    
    # Normalize
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transitions, row_sums, where=row_sums!=0)
    
    # Increased DPI and figure size
    plt.figure(figsize=(20, 20), dpi=300)
    
    # Enhanced heatmap styling
    sns.heatmap(transition_probs, 
                xticklabels=unique_chars, 
                yticklabels=unique_chars,
                cmap='YlOrRd',
                annot=True,  # Add number annotations
                fmt='.2f',   # Format annotations to 2 decimal places
                square=True)
    
    plt.title('Character Transition Probabilities', fontsize=16, pad=20)
    plt.xlabel('To Character', fontsize=14)
    plt.ylabel('From Character', fontsize=14)
    plt.xticks(fontsize=10, rotation=45)
    plt.yticks(fontsize=10, rotation=0)
    
    # Save in high resolution
    plt.savefig(os.path.join(save_path, 'transition_probabilities.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return transition_probs

def calculate_model_statistics(words: List[str]) -> Dict:
    """Calculate various statistics about the generated words"""
    stats = {
        'total_words': len(words),
        'unique_words': len(set(words)),
        'avg_word_length': np.mean([len(word) for word in words]),
        'std_word_length': np.std([len(word) for word in words]),
        'unique_chars': len(set(''.join(words))),
    }
    return stats

def main():
    # Change output directory to the same folder
    output_dir = "SourceCode/Model_Evaluation_Metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and generated words
    print("Loading model and data...")
    model, char_to_idx, idx_to_char = load_model_and_mappings()
    generated_words = analyze_generated_words("SourceCode/Generated_Dict")
    
    # Generate and save plots
    print("Generating plots...")
    plot_character_distribution(generated_words, output_dir)
    plot_word_length_distribution(generated_words, output_dir)
    transition_probs = analyze_transition_probabilities(generated_words, output_dir)
    
    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_model_statistics(generated_words)
    
    # Save statistics to file with more detailed formatting
    with open(os.path.join(output_dir, 'model_statistics.txt'), 'w') as f:
        f.write("=== Model Statistics ===\n\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    
    # Save transition probabilities to CSV with better formatting
    unique_chars = sorted(list(set(''.join(generated_words))))
    df_transitions = pd.DataFrame(
        transition_probs,
        index=unique_chars,
        columns=unique_chars
    )
    df_transitions.to_csv(os.path.join(output_dir, 'transition_probabilities.csv'), 
                         float_format='%.4f')
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
