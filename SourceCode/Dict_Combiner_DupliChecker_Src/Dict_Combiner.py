import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
import platform
import psutil
from tqdm import tqdm

def setup_logging() -> Optional[str]:
    """
    Sets up logging configuration without affecting main script output
    Returns the path to the log file
    """
    try:
        # Create logs directory
        log_dir = "SourceCode/Logs/Dict_Combiner_Logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'dict_combiner_{timestamp}.log')
        
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
        logger = logging.getLogger('DictCombiner')
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
    logger = logging.getLogger('DictCombiner')
    if level.lower() == 'error':
        logger.error(message)
    elif level.lower() == 'warning':
        logger.warning(message)
    else:
        logger.info(message)

def read_file(file_path):
    """
    Read a file and return its content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().splitlines()  # Read lines into a list
            log_event(f"Successfully read {len(content)} lines from: {file_path}")
            return content
    except Exception as e:
        log_event(f"Error reading {file_path}: {str(e)}", 'error')
        return []

def combine_files(input_folder: str, output_file: str):
    """
    Combine all text files from input folder into a single output file
    """
    try:
        if not os.path.exists(input_folder):
            log_event(f"Input folder does not exist: {input_folder}", 'error')
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

        # Get list of text files
        txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
        log_event(f"Found {len(txt_files)} text files in input folder")

        # Initialize counters
        total_lines = 0
        total_files_processed = 0
        unique_words = set()

        # Process each file
        print(f"Processing {len(txt_files)} files...")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in tqdm(txt_files, desc="Combining files"):
                try:
                    file_path = os.path.join(input_folder, filename)
                    content = read_file(file_path)
                    
                    # Add words to set and write to file
                    for line in content:
                        line = line.strip()
                        if line and line not in unique_words:
                            unique_words.add(line)
                            outfile.write(line + '\n')
                            total_lines += 1
                    
                    total_files_processed += 1
                    
                    # Log progress periodically
                    if total_files_processed % 10 == 0:
                        log_event(f"Processed {total_files_processed}/{len(txt_files)} files. "
                                f"Total unique lines: {total_lines}")
                
                except Exception as e:
                    log_event(f"Error processing file {filename}: {str(e)}", 'error')

        # Log final statistics
        log_event("=" * 50)
        log_event("Combination Complete")
        log_event(f"Total files processed: {total_files_processed}")
        log_event(f"Total unique lines written: {total_lines}")
        log_event(f"Output file: {output_file}")
        log_event("=" * 50)

        return total_files_processed, total_lines

    except Exception as e:
        log_event(f"Error in combine_files: {str(e)}", 'error')
        raise

def main():
    """
    Main execution function
    """
    try:
        # Initialize logging
        log_file = setup_logging()
        if log_file:
            log_event("Dictionary Combiner Started")
            log_event(f"Log file: {log_file}")

        # Log system information
        log_event(f"Platform: {platform.system()} {platform.release()} ({platform.processor()})")
        log_event(f"Total Memory: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

        # Configuration
        input_folder = "SourceCode/Datasets_Filtered/length_8"
        output_file = "SourceCode/Main_Dict/Main_Dict_8.txt"
        
        # Log configuration
        log_event("Configuration:")
        log_event(f"Input folder: {input_folder}")
        log_event(f"Output file: {output_file}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Process files
        print("Starting dictionary combination process...")
        total_files, total_lines = combine_files(input_folder, output_file)

        print(f"\nProcess completed successfully!")
        print(f"Total files processed: {total_files}")
        print(f"Total unique lines written: {total_lines}")
        print(f"Output saved to: {output_file}")
        
        log_event("Dictionary Combiner completed successfully")

    except Exception as e:
        log_event(f"Critical error in main execution: {str(e)}", 'error')
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()