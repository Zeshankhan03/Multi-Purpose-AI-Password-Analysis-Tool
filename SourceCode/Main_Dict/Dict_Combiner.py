import os
import threading
from queue import Queue
from collections import Counter

# Function to read files and add their contents to a queue
def read_file(file_path, queue):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            queue.put(content)
            print(f"Read file: {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Function to combine contents from the queue into a single string
def combine_files(input_folder):
    queue = Queue()
    threads = []

    # Read all TXT files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            thread = threading.Thread(target=read_file, args=(file_path, queue))
            threads.append(thread)
            thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    combined_content = ''
    while not queue.empty():
        combined_content += queue.get()

    return combined_content

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
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"Total number of words: {total_words}\n")
        log.write(f"Total number of unique words: {total_unique}\n")
        log.write(f"Total number of duplicate words removed: {total_duplicates}\n")

    print("Processing complete. Statistics logged.")

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

    combined_content = combine_files(input_folder)
    process_combined_content(combined_content, output_file, log_file)

# Example usage
if __name__ == "__main__":
    input_folder = 'SourceCode\Generated_Dict'  # Replace with your input folder path
    output_folder = 'SourceCode\Main_Dict'  # Replace with your output folder path
    log_folder = 'SourceCode\Logs\Dict_Combiner_Logs'  # Replace with your log folder path
    main(input_folder, output_folder, log_folder)
