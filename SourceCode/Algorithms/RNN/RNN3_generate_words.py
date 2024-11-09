import os
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
from datetime import datetime
import platform
import psutil
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate words for each starting character and save them in separate files
word_length = 8
temperature = 0.5
words_per_char = 10  # Number of words per starting character 65*count
#max_concurrent_threads = 128  # Max number of threads to run at a time


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

def setup_logging() -> Optional[str]:
    """
    Sets up logging configuration without affecting main script output
    Returns the path to the log file
    """
    try:
        # Create logs directory
        log_dir = "SourceCode/Logs/RNN_Generation_Logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'rnn_generation_{timestamp}.log')
        
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
        logger = logging.getLogger('RNNGenerator')
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
    logger = logging.getLogger('RNNGenerator')
    if level.lower() == 'error':
        logger.error(message)
    elif level.lower() == 'warning':
        logger.warning(message)
    else:
        logger.info(message)

log_file = setup_logging()
if log_file:
    log_event(f"Logging initialized. Log file: {log_file}")

# Add logging to device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_event(f"Using device: {device}")

# Add logging to model loading
try:
    char_to_idx = torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\char_to_idx.pth')
    idx_to_char = torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\idx_to_char.pth')
    log_event("Character mappings loaded successfully")
    
    chars = sorted(char_to_idx.keys())
    input_size = len(chars)
    hidden_size = 64
    num_layers = 2
    
    model = RNNModel(input_size, hidden_size, num_layers, len(chars)).to(device)
    model.load_state_dict(torch.load(R'SourceCode\Algorithms\RNN\RNN_Modules\rnn_model.pth'))
    log_event("Model loaded successfully")
except Exception as e:
    log_event(f"Error loading model or character mappings: {str(e)}", 'error')
    raise

# Add this function to control what gets logged
def should_log_word(word: str, sample_rate: float = 0.01) -> bool:
    """
    Determines if a generated word should be logged based on sampling rate
    """
    return np.random.random() < sample_rate


# Function to generate a single word
def generate_word(model, start_char, length, temperature=1.0):
    try:
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
        
        # Only log a sample of generated words
        if should_log_word(word):
            log_event(f"Sample generated word: {word}")
        return word
    except Exception as e:
        log_event(f"Error generating word starting with '{start_char}': {str(e)}", 'error')
        raise



# Generate words starting with uppercase and lowercase characters separately
for start_char in starting_chars:
    output_file = os.path.join(output_dir, f"generated_words_{start_char}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_event(f"Starting generation for character '{start_char}'")
    
    try:
        words_generated = 0
        with open(output_file, "w") as file:
            with tqdm.tqdm(total=words_per_char, desc=f"Generating for {start_char}") as progress_bar:
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(generate_word, model, start_char, word_length, temperature) 
                             for _ in range(words_per_char)]
                    
                    for future in as_completed(futures):
                        try:
                            word = future.result()
                            file.write(word + '\n')
                            words_generated += 1
                            progress_bar.update(1)
                        except Exception as e:
                            log_event(f"Error processing generated word: {str(e)}", 'error')
        
        # Log summary instead of individual words
        log_event(f"Completed generation for '{start_char}'. Generated {words_generated} words")
    except Exception as e:
        log_event(f"Error in generation loop for '{start_char}': {str(e)}", 'error')

# Generate all special character words in a single file
special_output_file = os.path.join(output_dir, f"generated_words_special_chars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
log_event("Starting special characters generation")

try:
    with open(special_output_file, "w") as special_file:
        with tqdm.tqdm(total=words_per_char * len(special_chars), desc="Generating for special characters") as progress_bar:
            with ThreadPoolExecutor() as executor:
                total_special_words = 0
                for start_char in special_chars:
                    log_event(f"Generating words for special character '{start_char}'")
                    futures = [executor.submit(generate_word, model, start_char, word_length, temperature) 
                             for _ in range(words_per_char)]
                    
                    for future in as_completed(futures):
                        try:
                            word = future.result()
                            special_file.write(word + '\n')
                            total_special_words += 1
                            progress_bar.update(1)
                        except Exception as e:
                            log_event(f"Error processing special character word: {str(e)}", 'error')
    
    log_event(f"Completed special characters generation. Total words: {total_special_words}")
except Exception as e:
    log_event(f"Error in special characters generation: {str(e)}", 'error')

log_event("Word generation complete for all specified starting characters.")
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
