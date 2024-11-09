import os
import threading
from queue import Queue
from datetime import datetime

# Function to read files and add their contents to a queue
def read_file(file_path, queue, log_file):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            queue.put(content)
            log_message(log_file, f"Read file: {file_path}")
    except Exception as e:
        log_message(log_file, f"Error reading {file_path}: {e}")

# Function to combine contents from the queue into a single string
def combine_files(input_folder, log_file):
    queue = Queue()
    threads = []
    file_count = 0

    # Read all TXT files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            thread = threading.Thread(target=read_file, args=(file_path, queue, log_file))
            threads.append(thread)
            thread.start()
            file_count += 1

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    combined_content = ''
    while not queue.empty():
        combined_content += queue.get()

    log_message(log_file, f"Total files read: {file_count}")
    return combined_content

# Function to log messages with timestamps
def log_message(log_file, message):
    with open(log_file, 'a', encoding='utf-8') as log:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.write(f"{timestamp} - {message}\n")

# Function to remove duplicates and log statistics
def process_combined_content(combined_content, output_file, log_file):
    words = combined_content.split()
    total_words = len(words)
    unique_words = set(words)
    total_unique = len(unique_words)
    total_duplicates = total_words - total_unique

    # Write unique words to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(' '.join(unique_words))

    # Log statistics
    log_message(log_file, f"Total number of words: {total_words}")
    log_message(log_file, f"Total number of unique words: {total_unique}")
    log_message(log_file, f"Total number of duplicate words removed: {total_duplicates}")
    log_message(log_file, "Processing complete. Statistics logged.")

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

    combined_content = combine_files(input_folder, log_file)
    process_combined_content(combined_content, output_file, log_file)

    # Log the completion of the process
    log_message(log_file, "File combining process completed.")

# Example usage
if __name__ == "__main__":
    input_folder = r'SourceCode\Datasets_Filtered\length_8'  # Replace with your input folder path
    output_folder = r'SourceCode\Main_Dict'  # Replace with your output folder path
    log_folder = r'SourceCode\Logs\Dict_Combiner_Logs'  # Replace with your log folder path
    main(input_folder, output_folder, log_folder)
