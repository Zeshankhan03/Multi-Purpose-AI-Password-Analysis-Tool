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
import sys
from typing import Optional, Tuple
import time
import torch.cuda as cuda
import cpuinfo
import pkg_resources
import subprocess
from typing import Dict, Any

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up logging
log_dir = "SourceCode\Logs\RNN_AlogrithmGenerate_Logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'rnn_generation.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()  # This will also print to console
    ]
)

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
logging.info("=== New Generation Session Started ===")
logging.info(f"Device: {device}")
logging.info(f"Platform: {platform.system()} {platform.release()} ({platform.processor()})")
logging.info(f"Total Memory: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")
logging.info(f"Model Configuration - Input Size: {input_size}, Hidden Size: {hidden_size}, Num Layers: {num_layers}")

# Directory setup
output_dir = "SourceCode/Generated_Dict"
os.makedirs(output_dir, exist_ok=True)

# Character groups for password generation
uppercase_chars = [chr(i) for i in range(65, 91)]  # A-Z
lowercase_chars = [chr(i) for i in range(97, 123)] # a-z
special_chars = ['!', '@', '#', '$', '%', '^', '&', '(', ')', '_', '+', '-', '=']

# Combined set of characters for iteration
starting_chars = uppercase_chars + lowercase_chars

# Custom exception classes
class ModelLoadError(Exception):
    """Raised when there's an error loading the model."""
    pass

class GenerationError(Exception):
    """Raised when there's an error during word generation."""
    pass

class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""
    pass

# Add error handling for model loading
def load_model(model_path: str, device: str) -> Tuple[nn.Module, dict]:
    """
    Load the RNN model with error handling
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        model_state = checkpoint['model_state']
        model_config = checkpoint.get('config', {})
        
        if not model_state or not model_config:
            raise ModelLoadError("Invalid model checkpoint structure")
        
        model = RNNModel(model_config['input_size'], 
                   model_config['hidden_size'], 
                   model_config['num_layers']).to(device)
        model.load_state_dict(model_state)
        
        logging.info(f"Successfully loaded model from {model_path}")
        return model, model_config
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logging.error(error_msg)
        raise ModelLoadError(error_msg) from e

# Function to generate a single word
def generate_word(model: nn.Module, 
                 start_char: str, 
                 length: int, 
                 temperature: float = 1.0) -> Optional[str]:
    """
    Generate a single word with comprehensive error handling
    """
    try:
        if not isinstance(model, nn.Module):
            raise TypeError("Invalid model type")
        if not isinstance(start_char, str) or len(start_char) != 1:
            raise ValueError("start_char must be a single character")
        if not isinstance(length, int) or length < 1:
            raise ValueError("length must be a positive integer")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError("temperature must be a positive number")
        
        model.eval()
        h = model.init_hidden(1)
        
        # Validate character
        if start_char not in char_to_idx:
            raise ValueError(f"Invalid start character: {start_char}")
        
        input = torch.eye(len(chars))[char_to_idx[start_char]].unsqueeze(0).unsqueeze(0).to(device)
        word = start_char

        with torch.no_grad():
            for i in range(length - 1):
                try:
                    output, h = model(input, h)
                    output = output[-1] / temperature
                    probabilities = torch.softmax(output, dim=0).cpu().numpy()
                    
                    # Validate probabilities
                    if not np.isfinite(probabilities).all():
                        raise GenerationError("Invalid probability distribution")
                    
                    generated_idx = np.random.choice(len(chars), p=probabilities)
                    next_char = idx_to_char[generated_idx]
                    word += next_char
                    input = torch.eye(len(chars))[char_to_idx[next_char]].unsqueeze(0).unsqueeze(0).to(device)
                
                except Exception as e:
                    raise GenerationError(f"Error during character generation: {str(e)}")
        
        return word
        
    except Exception as e:
        logging.error(f"Error generating word starting with '{start_char}': {str(e)}")
        return None

# Generate words for each starting character and save them in separate files
word_length = 8
temperature = 0.5
words_per_char = 100  # Number of words per starting character 65*count
#max_concurrent_threads = 128  # Max number of threads to run at a time

# Generate words starting with uppercase and lowercase characters separately
for start_char in starting_chars:
    output_file = os.path.join(output_dir, f"generated_words_{start_char}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.info(f"Starting generation for character '{start_char}' -> {output_file}")
    
    words_generated = 0
    errors = 0
    
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
                        if words_generated % 1000 == 0:  # Log every 1000 words
                            logging.info(f"Character '{start_char}': Generated {words_generated}/{words_per_char} words")
                    except Exception as e:
                        errors += 1
                        logging.error(f"Error in word generation for '{start_char}': {str(e)}")
                    finally:
                        progress_bar.update(1)

    logging.info(f"Completed generation for '{start_char}'. Total: {words_generated}, Errors: {errors}")

# Generate all special character words in a single file
special_output_file = os.path.join(output_dir, f"generated_words_special_chars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.info("Starting special characters generation")

total_special_words = 0
total_special_errors = 0

with open(special_output_file, "w") as special_file:
    with tqdm.tqdm(total=words_per_char * len(special_chars), desc="Generating for special characters") as progress_bar:
        with ThreadPoolExecutor() as executor:
            for start_char in special_chars:
                logging.info(f"Starting generation for special character '{start_char}'")
                futures = [executor.submit(generate_word, model, start_char, word_length, temperature) 
                          for _ in range(words_per_char)]
                
                words_generated = 0
                for future in as_completed(futures):
                    try:
                        word = future.result()
                        special_file.write(word + '\n')
                        words_generated += 1
                        total_special_words += 1
                        if words_generated % 1000 == 0:
                            logging.info(f"Special character '{start_char}': Generated {words_generated}/{words_per_char} words")
                    except Exception as e:
                        total_special_errors += 1
                        logging.error(f"Error in special character word generation for '{start_char}': {str(e)}")
                    finally:
                        progress_bar.update(1)
                
                logging.info(f"Completed generation for special character '{start_char}'")

logging.info(f"Special characters generation complete. Total: {total_special_words}, Errors: {total_special_errors}")
logging.info("=== Generation Session Completed ===")

def get_system_info() -> Dict[str, Any]:
    """
    Collect comprehensive system information
    """
    try:
        # CPU Information
        cpu_info = cpuinfo.get_cpu_info()
        
        # GPU Information
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    'name': torch.cuda.get_device_name(i),
                    'capability': torch.cuda.get_device_capability(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_free': torch.cuda.memory_allocated(i),
                    'memory_used': torch.cuda.memory_reserved(i)
                })

        # System Memory
        memory = psutil.virtual_memory()
        
        # Collect all dependencies
        dependencies = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        
        # CUDA version
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "Not available"
        
        system_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'cpu': {
                'brand': cpu_info.get('brand_raw', 'Unknown'),
                'cores_physical': psutil.cpu_count(logical=False),
                'cores_logical': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            },
            'gpu': gpu_info,
            'python': {
                'version': sys.version,
                'implementation': platform.python_implementation()
            },
            'dependencies': {
                'pytorch': torch.__version__,
                'cuda': cuda_version,
                'numpy': np.__version__,
                **dependencies
            }
        }
        
        return system_info
        
    except Exception as e:
        logging.error(f"Error collecting system information: {str(e)}")
        return {}

def log_system_info(system_info: Dict[str, Any]) -> None:
    """
    Log detailed system information
    """
    logging.info("=== System Information ===")
    
    # Platform
    logging.info("Platform Information:")
    for key, value in system_info['platform'].items():
        logging.info(f"  {key}: {value}")
    
    # CPU
    logging.info("\nCPU Information:")
    for key, value in system_info['cpu'].items():
        logging.info(f"  {key}: {value}")
    
    # Memory
    logging.info("\nMemory Information:")
    for key, value in system_info['memory'].items():
        if 'total' in key or 'available' in key or 'used' in key:
            logging.info(f"  {key}: {value / (1024**3):.2f} GB")
        else:
            logging.info(f"  {key}: {value}")
    
    # GPU
    logging.info("\nGPU Information:")
    if system_info['gpu']:
        for i, gpu in enumerate(system_info['gpu']):
            logging.info(f"  GPU {i}:")
            for key, value in gpu.items():
                if 'memory' in key:
                    logging.info(f"    {key}: {value / (1024**3):.2f} GB")
                else:
                    logging.info(f"    {key}: {value}")
    else:
        logging.info("  No GPU detected")
    
    # Dependencies
    logging.info("\nDependencies:")
    for key, value in system_info['dependencies'].items():
        logging.info(f"  {key}: {value}")

# Update the main generation loop with performance metrics
def main():
    try:
        # Log system information
        system_info = get_system_info()
        log_system_info(system_info)
        
        # Initialize performance metrics
        performance_metrics = {
            'start_time': time.time(),
            'total_words': 0,
            'total_chars': 0,
            'generation_times': []
        }

        for start_char in starting_chars:
            char_start_time = time.time()
            words_generated = 0
            chars_generated = 0
            
            # ... existing generation code ...
            
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(generate_word, model, start_char, word_length, temperature) 
                          for _ in range(words_per_char)]
                
                for future in as_completed(futures):
                    try:
                        word = future.result()
                        if word:
                            words_generated += 1
                            chars_generated += len(word)
                            performance_metrics['total_words'] += 1
                            performance_metrics['total_chars'] += len(word)
                            
                            # Log performance every 1000 words
                            if words_generated % 1000 == 0:
                                current_time = time.time()
                                elapsed_time = current_time - char_start_time
                                words_per_second = words_generated / elapsed_time
                                chars_per_second = chars_generated / elapsed_time
                                
                                logging.info(
                                    f"Performance metrics for '{start_char}' after {words_generated} words:\n"
                                    f"  Words per second: {words_per_second:.2f}\n"
                                    f"  Characters per second: {chars_per_second:.2f}"
                                )
                    
                    except Exception as e:
                        logging.error(f"Error in word generation: {str(e)}")
            
            # Log performance metrics for this character
            char_end_time = time.time()
            char_elapsed_time = char_end_time - char_start_time
            performance_metrics['generation_times'].append({
                'char': start_char,
                'time': char_elapsed_time,
                'words': words_generated,
                'chars': chars_generated
            })
            
            logging.info(
                f"\nPerformance Summary for '{start_char}':\n"
                f"  Total time: {char_elapsed_time:.2f} seconds\n"
                f"  Average words per second: {words_generated/char_elapsed_time:.2f}\n"
                f"  Average chars per second: {chars_generated/char_elapsed_time:.2f}"
            )
        
        # Log final performance metrics
        end_time = time.time()
        total_time = end_time - performance_metrics['start_time']
        
        logging.info("\n=== Final Performance Metrics ===")
        logging.info(f"Total execution time: {total_time:.2f} seconds")
        logging.info(f"Total words generated: {performance_metrics['total_words']}")
        logging.info(f"Total characters generated: {performance_metrics['total_chars']}")
        logging.info(f"Overall words per second: {performance_metrics['total_words']/total_time:.2f}")
        logging.info(f"Overall chars per second: {performance_metrics['total_chars']/total_time:.2f}")
        
        # Log performance by character
        logging.info("\nPerformance by Character:")
        for metric in performance_metrics['generation_times']:
            logging.info(
                f"  Character '{metric['char']}':\n"
                f"    Time: {metric['time']:.2f} seconds\n"
                f"    Words/second: {metric['words']/metric['time']:.2f}\n"
                f"    Chars/second: {metric['chars']/metric['time']:.2f}"
            )
        
    except Exception as e:
        logging.critical(f"Critical error in main execution: {str(e)}")
        raise
