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

# Function to remove duplicate words from the output file
def remove_duplicates(output_file, log_file):
    temp_file = output_file + '.tmp'  # Temporary file to store unique words
    seen_words = set()  # Set to track unique words

    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            for line in file:
                words = line.split()
                for word in words:
                    if word not in seen_words:
                        seen_words.add(word)  # Add unique word to the set
                        with open(temp_file, 'a', encoding='utf-8') as temp:
                            temp.write(word + '\n')  # Write unique word to temp file

        # Replace the original output file with the temporary file
        os.replace(temp_file, output_file)
        log_message(log_file, "Removed duplicate words from the output file.")
    except Exception as e:
        log_message(log_file, f"Error removing duplicates: {e}")

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
    remove_duplicates(output_file, log_file)

    # Log the completion of the process
    log_message(log_file, "File combining process completed.")

# Example usage
if __name__ == "__main__":
    input_folder = r'SourceCode\Datasets_Filtered\length_8'  # Replace with your input folder path
    output_file = r'SourceCode\Main_Dict\combined_output.txt'  # Replace with your output file path
    log_file = r'SourceCode\Logs\Dict_Combiner_Logs\Dict_combiner.log'  # Replace with your log file path
    main(input_folder, output_file, log_file)