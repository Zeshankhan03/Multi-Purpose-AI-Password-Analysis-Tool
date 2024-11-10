import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import platform
import psutil

def setup_logging() -> Optional[str]:
    """
    Sets up logging configuration without affecting main script output
    Returns the path to the log file
    """
    try:
        # Create logs directory
        log_dir = "SourceCode/Logs/RNN_Training_Logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'rnn_training_{timestamp}.log')
        
        # Configure file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Set format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Configure logger
        logger = logging.getLogger('RNNTrainer')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        return log_file
    except Exception as e:
        print(f"Warning: Could not set up logging: {str(e)}")
        return None

def log_event(message: str, level: str = 'info'):
    """
    Logs a message without interfering with console output
    """
    logger = logging.getLogger('RNNTrainer')
    if level.lower() == 'error':
        logger.error(message)
    elif level.lower() == 'warning':
        logger.warning(message)
    else:
        logger.info(message)

# Initialize logging
log_file = setup_logging()
if log_file:
    log_event(f"Logging initialized. Log file: {log_file}")

# Log system information
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_event(f"Using device: {device}")
log_event(f"Platform: {platform.system()} {platform.release()} ({platform.processor()})")
log_event(f"Total Memory: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

# Parameters
hidden_size = 64
num_layers = 2
num_epochs = 50
learning_rate = 0.003
batch_size = 5000

# Log training parameters
log_event("Training Parameters:")
log_event(f"Hidden Size: {hidden_size}")
log_event(f"Number of Layers: {num_layers}")
log_event(f"Number of Epochs: {num_epochs}")
log_event(f"Learning Rate: {learning_rate}")
log_event(f"Batch Size: {batch_size}")

print("Reading the Dataset.")
log_event("Starting to read dataset")

def read_words(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            words = file.read().splitlines()
        log_event(f"Successfully read {len(words)} words from dataset")
        return words
    except Exception as e:
        log_event(f"Error reading dataset: {str(e)}", 'error')
        raise

words = read_words(r'SourceCode\Datasets_Filtered\Test_DataSet.txt')
print("Reading the Dataset Completed.")
log_event("Dataset reading completed")

# Create character vocabulary
chars = sorted(list(set(''.join(words))))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

input_size = len(chars)

log_event(f"Vocabulary size: {len(chars)}")
log_event(f"Character set: {chars}")

print(f"Size of char_to_idx: {len(char_to_idx)}")
print(f"Size of idx_to_char: {len(idx_to_char)}")

print("Preprocessing the Dataset.")
log_event("Starting dataset preprocessing")

def preprocess(words, char_to_idx):
    try:
        sequences = []
        targets = []
        for word in words:
            sequence = [char_to_idx[char] for char in word[:-1]]
            target = [char_to_idx[char] for char in word[1:]]
            sequences.append(sequence)
            targets.append(target)
        log_event(f"Successfully preprocessed {len(sequences)} sequences")
        return sequences, targets
    except Exception as e:
        log_event(f"Error during preprocessing: {str(e)}", 'error')
        raise

sequences, targets = preprocess(words, char_to_idx)
sequences = torch.tensor(sequences, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

dataset = TensorDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
log_event("Dataset preprocessing completed")
print("Preprocessing the Dataset completed.")

# Model definition remains the same
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

model = RNNModel(input_size, hidden_size, num_layers, len(chars)).to(device)
log_event("Model created and moved to device")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
log_event("Optimizer and criterion initialized")

# Training loop with logging
best_loss = float('inf')
log_event("Starting training")

for epoch in range(num_epochs):
    model.train()
    loss_avg = 0
    batch_count = 0
    
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
        for batch_sequences, batch_targets in dataloader:
            try:
                batch_size_actual = batch_sequences.size(0)
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
                batch_count += 1
                pbar.update(1)

            except Exception as e:
                log_event(f"Error in training batch: {str(e)}", 'error')
                raise

    epoch_loss = loss_avg/len(dataloader)
    log_event(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Track best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        log_event(f"New best loss: {best_loss:.4f}")

log_event("Training completed")

# Save model and mappings
try:
    save_dir = r'SourceCode\Algorithms\RNN\RNN_Test_Modules'
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), f'{save_dir}/rnn_model.pth')
    torch.save(char_to_idx, f'{save_dir}/char_to_idx.pth')
    torch.save(idx_to_char, f'{save_dir}/idx_to_char.pth')
    
    log_event("Model and mappings saved successfully")
except Exception as e:
    log_event(f"Error saving model: {str(e)}", 'error')
    raise

log_event("Script execution completed successfully")
