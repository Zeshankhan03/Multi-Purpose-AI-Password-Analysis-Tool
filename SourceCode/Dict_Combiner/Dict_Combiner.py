import os
from datetime import datetime

# Function to read a file and log its content
def read_file(file_path, log_file):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            log_message(log_file, f"Read file: {file_path}")
            return content
    except Exception as e:
        log_message(log_file, f"Error reading {file_path}: {e}")
        return ''

# Function to process each file and check for duplicates
def process_file(file_path, log_file, unique_words):
    content = read_file(file_path, log_file)
    words = content.split()
    unique_words.update(words)  # Use a set to keep unique words

# Function to combine files and log statistics
def combine_files(input_folder, log_file):
    unique_words = set()  # Use a set to store unique words
    file_count = 0

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            process_file(file_path, log_file, unique_words)
            file_count += 1

    log_message(log_file, f"Total files read: {file_count}")
    return unique_words

# Function to log messages with timestamps
def log_message(log_file, message):
    with open(log_file, 'a', encoding='utf-8') as log:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.write(f"{timestamp} - {message}\n")

# Main function to execute the script
def main(input_folder, output_folder, log_folder):
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    output_file = os.path.join(output_folder, 'combined_output.txt')
    log_file = os.path.join(log_folder, 'Dict_combiner.log')

    # Log the start of the process
    log_message(log_file, "Starting the file combining process.")

    unique_words = combine_files(input_folder, log_file)

    # Write unique words to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(' '.join(unique_words))

    # Log the completion of the process
    log_message(log_file, "File combining process completed.")

# Example usage
if __name__ == "__main__":
    input_folder = r'SourceCode\Datasets_Filtered\length_8'  # Replace with your input folder path
    output_folder = r'SourceCode\Main_Dict'  # Replace with your output folder path
    log_folder = r'SourceCode\Logs\Dict_Combiner_Logs'  # Replace with your log folder path
    main(input_folder, output_folder, log_folder)