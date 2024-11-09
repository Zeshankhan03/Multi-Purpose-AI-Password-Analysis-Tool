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

# Function to log messages with timestamps
def log_message(log_file, message):
    with open(log_file, 'a', encoding='utf-8') as log:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.write(f"{timestamp} - {message}\n")

# Function to remove duplicates from the output file using chunking
def remove_duplicates_in_chunks(output_file, log_file, chunk_size=200 * 1024 * 1024):
    seen_words = set()  # Set to track unique words
    lines_to_keep = []  # List to store lines to keep
    line_numbers_to_remove = set()  # Set to track line numbers to remove

    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            current_chunk = []
            current_chunk_size = 0
            line_number = 0

            for line in file:
                current_chunk.append(line)
                current_chunk_size += len(line.encode('utf-8'))  # Calculate size in bytes
                line_number += 1

                # If the current chunk size exceeds the specified chunk size, process it
                if current_chunk_size >= chunk_size:
                    process_chunk(current_chunk, seen_words, lines_to_keep, line_numbers_to_remove)

                    # Reset for the next chunk
                    current_chunk = []
                    current_chunk_size = 0

            # Process any remaining lines in the last chunk
            if current_chunk:
                process_chunk(current_chunk, seen_words, lines_to_keep, line_numbers_to_remove)

        # Write back only the unique lines to the output file
        with open(output_file, 'w', encoding='utf-8') as file:
            for line in lines_to_keep:
                file.write(line)

        log_message(log_file, "Removed duplicate words from the output file.")
    except Exception as e:
        log_message(log_file, f"Error removing duplicates: {e}")

def process_chunk(chunk, seen_words, lines_to_keep, line_numbers_to_remove):
    for line in chunk:
        words = line.split()
        unique_line = True

        for word in words:
            if word in seen_words:
                unique_line = False
                break
            seen_words.add(word)

        if unique_line:
            lines_to_keep.append(line)
        else:
            line_numbers_to_remove.add(chunk.index(line))

# Main function to execute the script
def main(input_folder, output_file, log_file):
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return

    # Log the start of the process
    log_message(log_file, "Starting the file combining process.")

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            content = read_file(file_path, log_file)
            if content:  # Only write if content is not empty
                with open(output_file, 'a', encoding='utf-8') as output:
                    output.write(content + '\n')  # Write content to output file
                log_message(log_file, f"Wrote content from {filename} to output file.")

    # Remove duplicates from the output file
    remove_duplicates_in_chunks(output_file, log_file)

    # Log the completion of the process
    log_message(log_file, "File combining process completed.")

# Example usage
if __name__ == "__main__":
    input_folder = r'SourceCode\Datasets_Filtered\length_8'  # Replace with your input folder path
    output_file = r'SourceCode\Main_Dict\combined_output.txt'  # Replace with your output file path
    log_file = r'SourceCode\Logs\Dict_Combiner_Logs\Dict_combiner.log'  # Replace with your log file path
    main(input_folder, output_file, log_file)