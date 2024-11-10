import os
from tqdm import tqdm
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple
from datetime import datetime
import platform
import psutil
import sys

def setup_logging() -> Optional[str]:
    """
    Sets up logging configuration without affecting main script output
    Returns the path to the log file
    """
    try:
        # Create base logs directory
        base_log_dir = "SourceCode/Logs"
        os.makedirs(base_log_dir, exist_ok=True)
        
        # Create specific logs directory for DataSplitter
        splitter_log_dir = os.path.join(base_log_dir, "DataSplitter")
        os.makedirs(splitter_log_dir, exist_ok=True)
        
        # Create run-specific directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_log_dir = os.path.join(splitter_log_dir, f"run_{timestamp}")
        os.makedirs(run_log_dir, exist_ok=True)
        
        # Create log file
        log_file = os.path.join(run_log_dir, 'datasplitter.log')
        
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
        logger = logging.getLogger('DataSplitter')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        # Log initial setup information
        log_event("Logging initialized")
        log_event(f"Log directory: {run_log_dir}")
        
        return log_file
    except Exception as e:
        print(f"Warning: Could not set up logging: {str(e)}")
        return None

def log_event(message: str, level: str = 'info'):
    """
    Logs a message without interfering with console output
    """
    logger = logging.getLogger('DataSplitter')
    if level.lower() == 'error':
        logger.error(message)
    elif level.lower() == 'warning':
        logger.warning(message)
    else:
        logger.info(message)

def create_new_output_file(output_dir: str, file_counter: int) -> Tuple[object, str]:
    """Create a new output file with error handling"""
    try:
        output_file_path = os.path.join(output_dir, f"{file_counter:08}.txt")
        output_file = open(output_file_path, 'w', encoding='utf-8')
        log_event(f"Created new output file: {output_file_path}")
        return output_file, output_file_path
    except Exception as e:
        log_event(f"Error creating output file: {str(e)}", 'error')
        raise

def log_processed_file(log_file: str, input_file_path: str):
    """Log processed files with error handling"""
    try:
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"{input_file_path}\n")
        log_event(f"Logged processed file: {input_file_path}")
    except Exception as e:
        log_event(f"Error logging processed file: {str(e)}", 'error')
        raise

def is_file_processed(log_file: str, input_file_path: str) -> bool:
    """Check if file has been processed with error handling"""
    try:
        if not os.path.exists(log_file):
            return False
        with open(log_file, 'r', encoding='utf-8') as log:
            processed_files = log.readlines()
        return input_file_path + "\n" in processed_files
    except Exception as e:
        log_event(f"Error checking processed file status: {str(e)}", 'error')
        return False

def write_buffer_to_file(buffer: list, output_dir: str, file_counter: int):
    """Write buffer to file with error handling"""
    try:
        output_file, output_file_path = create_new_output_file(output_dir, file_counter)
        output_file.write('\n'.join(buffer))
        output_file.close()
        log_event(f"Successfully wrote buffer to file {output_file_path}")
    except Exception as e:
        log_event(f"Error writing buffer to file: {str(e)}", 'error')
        raise

def validate_input_file(input_path: str) -> bool:
    """
    Validates the input file:
    - Checks if file exists
    - Checks if it's a .txt file
    - Checks if it's readable
    """
    try:
        # Check if path exists
        if not os.path.exists(input_path):
            log_event(f"Error: Input file does not exist: {input_path}", 'error')
            return False
        
        # Check if it's a file (not a directory)
        if not os.path.isfile(input_path):
            log_event(f"Error: Input path is not a file: {input_path}", 'error')
            return False
            
        # Check file extension
        if not input_path.lower().endswith('.txt'):
            log_event(f"Error: Input file is not a .txt file: {input_path}", 'error')
            return False
            
        # Check if file is readable
        if not os.access(input_path, os.R_OK):
            log_event(f"Error: Input file is not readable: {input_path}", 'error')
            return False
            
        # Check if file is not empty
        if os.path.getsize(input_path) == 0:
            log_event(f"Error: Input file is empty: {input_path}", 'error')
            return False
            
        log_event(f"Input file validation successful: {input_path}")
        return True
        
    except Exception as e:
        log_event(f"Error validating input file: {str(e)}", 'error')
        return False

def process_files(input_dir: str, output_dir: str, log_file: str, max_chunk_size: int = 2*1024*1024*1024):
    """Process files with comprehensive error handling and logging"""
    try:
        # Validate input file first
        if not validate_input_file(input_dir):
            raise ValueError(f"Invalid input file: {input_dir}")
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_event(f"Created output directory: {output_dir}")

        log_event(f"Starting file processing with max chunk size: {max_chunk_size} bytes")
        file_counter = 1
        buffer = []
        current_size = 0
        total_files_processed = 0
        total_lines_processed = 0

        # Since we're processing a single file, we'll modify the loop
        try:
            input_file_size = os.path.getsize(input_dir)
            log_event(f"Processing file: {input_dir} (Size: {input_file_size/1024/1024:.2f} MB)")

            with open(input_dir, 'r', encoding='utf-8', errors='ignore') as input_file:
                with tqdm(total=input_file_size, unit='B', unit_scale=True, desc=os.path.basename(input_dir)) as pbar:
                    while True:
                        chunk = input_file.read(max_chunk_size)
                        if not chunk:
                            break

                        for line in chunk.splitlines():
                            try:
                                buffer.append(line)
                                current_size += len(line) + 1
                                total_lines_processed += 1

                                if current_size >= max_chunk_size:
                                    write_buffer_to_file(buffer, output_dir, file_counter)
                                    file_counter += 1
                                    buffer.clear()
                                    current_size = 0
                            except Exception as e:
                                log_event(f"Error processing line: {str(e)}", 'error')
                                continue

                        pbar.update(len(chunk))
                        chunk = None  # Free memory after processing

            log_processed_file(log_file, input_dir)
            total_files_processed += 1
            
            # Log progress
            log_event(f"Progress: File processed, {total_lines_processed} lines processed")

        except Exception as e:
            log_event(f"Error processing file {input_dir}: {str(e)}", 'error')
            raise

        # Write remaining buffer
        if buffer:
            write_buffer_to_file(buffer, output_dir, file_counter)

        # Log final statistics
        log_event("=" * 50)
        log_event("Processing Complete")
        log_event(f"Input file processed: {input_dir}")
        log_event(f"Total lines processed: {total_lines_processed}")
        log_event(f"Total output files created: {file_counter}")
        log_event("=" * 50)

    except Exception as e:
        log_event(f"Critical error in process_files: {str(e)}", 'error')
        raise

def main():
    """Main execution function with error handling"""
    try:
        # Initialize logging
        log_file = setup_logging()
        if log_file:
            log_event("Data Splitter Started")
            log_event(f"Log file: {log_file}")

        # Log system information
        log_event(f"Platform: {platform.system()} {platform.release()} ({platform.processor()})")
        log_event(f"Total Memory: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB")

        # Configuration
        input_dir = "D:/Osama Khalid/Osama Khalid BSCY 7/FYP Part 1/AI Codes/PasswrodsBIG/rockyou.txt"
        output_dir = "H:\\PartsOutputs"
        process_log_file = "DataSpliter_log_file.log"
        
        # Log configuration
        log_event("Configuration:")
        log_event(f"Input file: {input_dir}")
        log_event(f"Output directory: {output_dir}")
        log_event(f"Process log file: {process_log_file}")

        # Process files
        print("Starting data splitting process...")
        process_files(input_dir, output_dir, process_log_file)

        print("\nProcess completed successfully!")
        log_event("Data Splitter completed successfully")

    except Exception as e:
        log_event(f"Critical error in main execution: {str(e)}", 'error')
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()