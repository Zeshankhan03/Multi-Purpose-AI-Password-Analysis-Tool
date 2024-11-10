import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageFilter
import csv
import subprocess
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

class CustomGUI:
    def __init__(self, image_path):
        self.setup_logging()  # Initialize logging first
        self.root = tk.Tk()
        self.root.title("Hashcat GUI")
        
        # Log application start
        logging.info("Starting Hashcat GUI application")
        
        # Set the size of the window
        self.root.geometry("800x600")
        logging.info("Window initialized with size 800x600")
        
        try:
            # Load the background image
            self.background_image = Image.open(image_path)
            logging.info(f"Background image loaded from: {image_path}")
        except Exception as e:
            logging.error(f"Failed to load background image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load background image: {str(e)}")
            
        # Rest of your initialization code...

    def setup_logging(self):
        """Setup logging configuration"""
        try:
            # Create base logs directory
            base_log_dir = "SourceCode/Logs"
            os.makedirs(base_log_dir, exist_ok=True)
            
            # Create specific logs directory for Hashcat GUI
            hashcat_log_dir = os.path.join(base_log_dir, "Hashcat_GUI_logs")
            os.makedirs(hashcat_log_dir, exist_ok=True)
            
            # Create run-specific directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_log_dir = os.path.join(hashcat_log_dir, f"run_{timestamp}")
            os.makedirs(run_log_dir, exist_ok=True)
            
            # Create log file
            log_file = os.path.join(run_log_dir, 'hashcat_gui.log')
            
            # Configure logging with RotatingFileHandler
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
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            logger.addHandler(file_handler)
            
            logging.info("Logging initialized")
            logging.info(f"Log directory: {run_log_dir}")
            
            # Store log configuration
            self.log_dir = run_log_dir
            
        except Exception as e:
            print(f"Warning: Could not set up logging: {str(e)}")

    def load_hash_types(self, file_path):
        try:
            hash_types = {}
            logging.info(f"Loading hash types from: {file_path}")
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) == 2 and row[1].strip():
                        hash_types[row[0]] = int(row[1])
            logging.info(f"Successfully loaded {len(hash_types)} hash types")
            return hash_types
        except Exception as e:
            logging.error(f"Failed to load hash types: {str(e)}")
            messagebox.showerror("Error", f"Failed to load hash types: {str(e)}")
            return {}

    def run_hashcat(self):
        """Run the Hashcat software with the selected options."""
        try:
            logging.info("Starting Hashcat execution")
            
            hash_file = self.get_hash_file()
            if hash_file is None:
                logging.error("Invalid hash file")
                return

            selected_hash_type = self.get_hash_type_number()
            if selected_hash_type is None:
                logging.error("Invalid hash type")
                return

            dict_args = self.get_dictionary_files()
            if dict_args is None:
                logging.error("Invalid dictionary configuration")
                return

            output_file = self.get_output_file()
            if output_file is None:
                logging.error("Invalid output file")
                return

            workload_profile = self.get_workload_profile()
            if workload_profile is None:
                logging.error("Invalid workload profile")
                return

            abort_temp = self.get_abort_temperature()

            # Construct the command
            command = [
                "SourceCode\\hashcat-6.2.6\\hashcat.exe",
                "-m", str(selected_hash_type),
                hash_file,
                "-w", workload_profile,
                "-O",
                f"--hwmon-temp-abort={abort_temp}"
            ] + dict_args + ["-o", output_file]

            logging.info(f"Executing command: {' '.join(command)}")

            # Execute Hashcat
            process = subprocess.Popen(
                command,
                cwd="SourceCode\\hashcat-6.2.6",
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            logging.info("Hashcat process started successfully")
            
        except Exception as e:
            error_msg = f"Failed to execute Hashcat: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Error", error_msg)

    def run(self):
        try:
            logging.info("Starting main application loop")
            self.root.mainloop()
        except Exception as e:
            logging.error(f"Application crashed: {str(e)}")
            raise
        finally:
            logging.info("Application shutting down") 