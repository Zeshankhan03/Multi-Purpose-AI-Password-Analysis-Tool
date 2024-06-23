import os
from tqdm import tqdm

def create_new_output_file(output_dir, file_counter):
    output_file_path = os.path.join(output_dir, f"{file_counter:08}.txt")
    return open(output_file_path, 'w', encoding='utf-8'), output_file_path

def log_processed_file(log_file, input_file_path):
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"{input_file_path}\n")

def is_file_processed(log_file, input_file_path):
    if not os.path.exists(log_file):
        return False
    with open(log_file, 'r', encoding='utf-8') as log:
        processed_files = log.readlines()
    return input_file_path + "\n" in processed_files

def process_files(input_dir, output_dir, log_file, max_chunk_size=2*1024*1024*1024):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_counter = 1
    buffer = []
    current_size = 0

    for root, _, files in os.walk(input_dir):
        for file_name in files:
            input_file_path = os.path.join(root, file_name)
            if is_file_processed(log_file, input_file_path):
                continue

            input_file_size = os.path.getsize(input_file_path)
            with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as input_file, tqdm(total=input_file_size, unit='B', unit_scale=True, desc=file_name) as pbar:
                while True:
                    chunk = input_file.read(max_chunk_size)
                    if not chunk:
                        break

                    for line in chunk.splitlines():
                        try:
                            buffer.append(line)
                            current_size += len(line) + 1
                            if current_size >= max_chunk_size:
                                write_buffer_to_file(buffer, output_dir, file_counter)
                                file_counter += 1
                                buffer.clear()
                                current_size = 0
                        except Exception as e:
                            print(f"Error processing line: {e}")
                            continue

                    pbar.update(len(chunk))
                    chunk = None  # Free memory after processing

            log_processed_file(log_file, input_file_path)

    if buffer:
        write_buffer_to_file(buffer, output_dir, file_counter)

def write_buffer_to_file(buffer, output_dir, file_counter):
    output_file, output_file_path = create_new_output_file(output_dir, file_counter)
    output_file.write('\n'.join(buffer))
    output_file.close()

if __name__ == "__main__":
    input_dir = "F:\\passdic"
    output_dir = "H:\\PartsOutputs"
    log_file = "DataSpliter_log_file.log"

    process_files(input_dir, output_dir, log_file)
