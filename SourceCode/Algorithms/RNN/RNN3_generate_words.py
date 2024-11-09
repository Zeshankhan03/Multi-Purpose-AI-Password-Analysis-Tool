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
from typing import Optional, Tuple, Dict, Any
import time
import torch.cuda as cuda
import cpuinfo
from importlib.metadata import distributions, version  # Replace pkg_resources
from logging.handlers import RotatingFileHandler

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

def setup_logging():
    """
    Configure logging with both file and console handlers
    """
    # Create logs directory if it doesn't exist
    log_dir = "SourceCode/Logs/RNN_Generation_Logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'rnn_generation_{timestamp}.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - Rotating file handler to manage file size
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log the initialization
    logging.info("Logging initialized")
    logging.info(f"Log file created at: {log_file}")

# Function to get installed packages and versions
def get_installed_packages() -> Dict[str, str]:
    """
    Get all installed packages and their versions using importlib.metadata
    """
    try:
        return {dist.metadata['Name']: dist.version for dist in distributions()}
    except Exception as e:
        logging.error(f"Error getting package information: {str(e)}")
        return {}

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
        
        # Collect all dependencies using importlib.metadata
        dependencies = get_installed_packages()
        
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
    try:
        logging.info("=" * 50)
        logging.info("SYSTEM INFORMATION")
        logging.info("=" * 50)

        # Python Version
        logging.info("\nPython Environment:")
        logging.info(f"Python Version: {sys.version}")
        logging.info(f"Python Implementation: {platform.python_implementation()}")

        # OS Information
        logging.info("\nOperating System:")
        logging.info(f"System: {platform.system()}")
        logging.info(f"Release: {platform.release()}")
        logging.info(f"Version: {platform.version()}")
        logging.info(f"Machine: {platform.machine()}")
        logging.info(f"Processor: {platform.processor()}")

        # CPU Information
        cpu_info = cpuinfo.get_cpu_info()
        logging.info("\nCPU Information:")
        logging.info(f"CPU Brand: {cpu_info.get('brand_raw', 'Unknown')}")
        logging.info(f"CPU Cores (Physical): {psutil.cpu_count(logical=False)}")
        logging.info(f"CPU Cores (Logical): {psutil.cpu_count(logical=True)}")

        # Memory Information
        mem = psutil.virtual_memory()
        logging.info("\nMemory Information:")
        logging.info(f"Total Memory: {mem.total / (1024**3):.2f} GB")
        logging.info(f"Available Memory: {mem.available / (1024**3):.2f} GB")
        logging.info(f"Memory Usage: {mem.percent}%")

        # GPU Information
        logging.info("\nGPU Information:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logging.info(f"\nGPU {i}:")
                logging.info(f"Name: {torch.cuda.get_device_name(i)}")
                logging.info(f"CUDA Version: {torch.version.cuda}")
                props = torch.cuda.get_device_properties(i)
                logging.info(f"Total Memory: {props.total_memory / (1024**2):.2f} MB")
                logging.info(f"CUDA Capability: {props.major}.{props.minor}")
        else:
            logging.info("No GPU detected")

        # PyTorch Information
        logging.info("\nPyTorch Information:")
        logging.info(f"PyTorch Version: {torch.__version__}")
        logging.info(f"CUDA Available: {torch.cuda.is_available()}")
        logging.info(f"CUDA Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

        # Dependencies
        logging.info("\nInstalled Dependencies:")
        dependencies = get_installed_packages()
        for package_name, package_version in sorted(dependencies.items()):
            logging.info(f"{package_name}: {package_version}")

        logging.info("=" * 50)

    except Exception as e:
        logging.error(f"Error logging system information: {str(e)}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        model = RNNModel(
            model_config['input_size'], 
            model_config['hidden_size'], 
            model_config['num_layers'],
            model_config['num_classes']
        ).to(device)
        model.load_state_dict(model_state)
        
        logging.info(f"Successfully loaded model from {model_path}")
        return model, model_config
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logging.error(error_msg)
        raise ModelLoadError(error_msg) from e

def log_performance_metrics(metrics: dict):
    """
    Log detailed performance metrics with safety checks
    """
    try:
        logging.info("=" * 50)
        logging.info("PERFORMANCE METRICS")
        logging.info("=" * 50)

        # Overall metrics with safety checks
        logging.info("\nOverall Performance:")
        logging.info(f"Total Execution Time: {metrics.get('total_time', 0):.2f} seconds")
        logging.info(f"Total Words Generated: {metrics.get('total_words', 0)}")
        
        # Safe calculation of generation speed
        total_time = metrics.get('total_time', 0)
        if total_time > 0:
            words_per_second = metrics.get('total_words', 0) / total_time
            logging.info(f"Average Generation Speed: {words_per_second:.2f} words/second")
        else:
            logging.info("Average Generation Speed: N/A (no time elapsed)")

        # Memory usage
        try:
            mem = psutil.virtual_memory()
            logging.info("\nMemory Usage:")
            logging.info(f"Current Memory Usage: {mem.percent}%")
            
            if torch.cuda.is_available():
                logging.info("\nGPU Memory Usage:")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**2)
                    reserved = torch.cuda.memory_reserved(i) / (1024**2)
                    logging.info(f"GPU {i}:")
                    logging.info(f"  Allocated: {allocated:.2f} MB")
                    logging.info(f"  Reserved: {reserved:.2f} MB")
        except Exception as e:
            logging.error(f"Error getting memory usage: {str(e)}")

        logging.info("=" * 50)

    except Exception as e:
        logging.error(f"Error logging performance metrics: {str(e)}")

def generate_word(model: nn.Module, 
                 start_char: str, 
                 length: int, 
                 temperature: float = 1.0,
                 char_to_idx: dict = None,
                 idx_to_char: dict = None) -> Optional[str]:
    """
    Generate a single word with comprehensive error handling
    """
    try:
        # Input validation
        if not isinstance(model, nn.Module):
            raise TypeError("Invalid model type")
        if not isinstance(start_char, str) or len(start_char) != 1:
            raise ValueError("start_char must be a single character")
        if not isinstance(length, int) or length < 1:
            raise ValueError("length must be a positive integer")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError("temperature must be a positive number")
        if not char_to_idx or not idx_to_char:
            raise ValueError("Character mappings must be provided")

        # Prepare model
        model.eval()
        word_start_time = time.time()
        
        try:
            h = model.init_hidden(1)
            input_char_idx = char_to_idx.get(start_char)
            if input_char_idx is None:
                raise ValueError(f"Start character '{start_char}' not in vocabulary")
            
            input_tensor = torch.zeros(1, 1, len(char_to_idx))
            input_tensor[0, 0, input_char_idx] = 1
            input_tensor = input_tensor.to(device)
            
            word = start_char

            with torch.no_grad():
                for i in range(length - 1):
                    output, h = model(input_tensor, h)
                    output = output[-1] / temperature
                    probabilities = torch.softmax(output, dim=0).cpu().numpy()
                    
                    if not np.isfinite(probabilities).all():
                        raise GenerationError("Invalid probability distribution")
                    
                    next_char_idx = np.random.choice(len(char_to_idx), p=probabilities)
                    next_char = idx_to_char[next_char_idx]
                    word += next_char
                    
                    input_tensor = torch.zeros(1, 1, len(char_to_idx))
                    input_tensor[0, 0, char_to_idx[next_char]] = 1
                    input_tensor = input_tensor.to(device)

            generation_time = time.time() - word_start_time
            logging.debug(f"Generated word '{word}' in {generation_time:.4f} seconds")
            return word

        except Exception as e:
            raise GenerationError(f"Error during word generation: {str(e)}")

    except Exception as e:
        logging.error(f"Error generating word starting with '{start_char}': {str(e)}")
        return None

def main():
    """
    Main execution function with comprehensive error handling and logging
    """
    try:
        # Initialize logging
        setup_logging()
        
        # Log system information
        system_info = get_system_info()
        log_system_info(system_info)
        
        # Configuration parameters
        model_path = r"SourceCode\Algorithms\RNN\RNN_Modules\rnn_model.pth"
        output_dir = r"SourceCode/Generated_Dict/RNN_Generated"
        word_length = 12
        temperature = 0.5
        words_per_char = 1000
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize performance tracking
        performance_metrics = {
            'start_time': time.time(),
            'total_time': 0.0,
            'total_words': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'char_metrics': []
        }

        try:
            # Load model
            logging.info(f"Loading model from {model_path}")
            model, model_config = load_model(model_path, device)
            logging.info("Model loaded successfully")
            
            # Define character sets
            lowercase_chars = 'abcdefghijklmnopqrstuvwxyz'
            uppercase_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            digits = '0123456789'
            special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
            
            all_chars = lowercase_chars + uppercase_chars + digits + special_chars
            char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
            idx_to_char = {idx: char for char, idx in char_to_idx.items()}
            
            # Generate words for each character set
            for char_set_name, char_set in [
                ("lowercase", lowercase_chars),
                ("uppercase", uppercase_chars),
                ("digits", digits),
                ("special", special_chars)
            ]:
                char_set_start_time = time.time()
                output_file = os.path.join(
                    output_dir, 
                    f"generated_words_{char_set_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
                
                logging.info(f"\nStarting generation for {char_set_name} characters")
                char_set_metrics = {
                    'char_set': char_set_name,
                    'start_time': char_set_start_time,
                    'words_generated': 0,
                    'failures': 0
                }

                with open(output_file, 'w') as f:
                    for start_char in char_set:
                        word = generate_word(model, start_char, word_length, temperature, char_to_idx, idx_to_char)
                        if word:
                            f.write(word + '\n')
                            char_set_metrics['words_generated'] += 1
                        else:
                            char_set_metrics['failures'] += 1

                char_set_end_time = time.time()
                char_set_metrics['end_time'] = char_set_end_time
                char_set_metrics['total_time'] = char_set_end_time - char_set_start_time
                char_set_metrics['total_words'] = char_set_metrics['words_generated'] + char_set_metrics['failures']
                char_set_metrics['char_metrics'] = {
                    'words_generated': char_set_metrics['words_generated'],
                    'failures': char_set_metrics['failures'],
                    'total_time': char_set_metrics['total_time'],
                    'total_words': char_set_metrics['total_words']
                }

                logging.info(f"Generation for {char_set_name} characters completed")
                logging.info(f"Total words generated: {char_set_metrics['total_words']}")
                logging.info(f"Total time taken: {char_set_metrics['total_time']:.2f} seconds")
                logging.info(f"Words generated: {char_set_metrics['words_generated']}")
                logging.info(f"Failures: {char_set_metrics['failures']}")

                # Log character metrics
                logging.info("Character Metrics:")
                for char, metrics in char_set_metrics['char_metrics'].items():
                    logging.info(f"{char}: {metrics}")

                # Log performance metrics
                log_performance_metrics(performance_metrics)

        except Exception as e:
            logging.error(f"Error during main execution: {str(e)}")

    except Exception as e:
        logging.error(f"Error during main execution: {str(e)}")
if __name__ == "__main__":
    print("Starting RNN word generation script...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        logging.critical(f"Unhandled exception: {str(e)}")
        sys.exit(1)