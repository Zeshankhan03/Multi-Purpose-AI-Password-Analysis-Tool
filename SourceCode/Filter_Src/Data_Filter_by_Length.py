import os
import argparse
import multiprocessing
from tqdm import tqdm
import csv

SUPPORTED_EXTENSIONS = ['.txt', '.dic', '.lst', '.uniq']
PART_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

def filter_words_by_length(chunk, min_length, max_length):
    filtered_words = {length: [] for length in range(min_length, max_length + 1)}
    for word in chunk:
        if isinstance(word, str):  # Ensure word is a string
            word = word.strip()  # Strip leading and trailing spaces
            word_len = len(word)
            if min_length <= word_len <= max_length:
                filtered_words[word_len].append(word)
    return filtered_words

def process_subchunk(subchunk, min_length, max_length):
    return filter_words_by_length(subchunk, min_length, max_length)

def process_part(part_data, min_length, max_length, num_cores):
    subchunk_size = len(part_data) // num_cores
    results = {length: [] for length in range(min_length, max_length + 1)}
    total_lines_read = 0
    try:
        with multiprocessing.Pool(num_cores) as pool:
            subchunks = [part_data[i:i + subchunk_size] for i in range(0, len(part_data), subchunk_size)]
            print(f"Subchunks created: {len(subchunks)}")
            subchunk_results = pool.starmap(process_subchunk, [(subchunk.splitlines(), min_length, max_length) for subchunk in subchunks])
        
        for result in subchunk_results:
            for length, words in result.items():
                results[length].extend(words)
            total_lines_read += sum(len(words) for words in result.values())
    
    except Exception as e:
        print(f"Error processing part: {e}")
    
    return results, total_lines_read

def get_file_parts(filename, part_size):
    file_size = os.path.getsize(filename)
    return [(i, min(part_size, file_size - i)) for i in range(0, file_size, part_size)]

def write_filtered_words(filtered_words, filtered_dir, filename, encoding):
    for length, words in filtered_words.items():
        if words:  # Ensure there are words to write
            output_txt_filename = os.path.join(filtered_dir, f'filtered_words_{length}_{os.path.basename(filename)}.txt')
            with open(output_txt_filename, 'a', encoding=encoding) as output_file:
                print(f"Writing {len(words)} words of length {length} to {output_txt_filename}")
                output_file.write('\n'.join(words) + '\n')

def parallel_filter_words(directory, min_length, max_length, encoding):
    total_input_passwords = 0
    total_output_passwords = 0
    stats = []
    length_stats = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filtered_dir = os.path.join(script_dir, "filtered_passwords")
    stats_dir = os.path.join(script_dir, "stats")

    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    filenames = [
        os.path.join(directory, file) 
        for file in os.listdir(directory) 
        if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS
    ]

    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    for filename in filenames:
        print(f"Loading file: {filename}")
        file_total_input = 0
        file_total_output = 0
        parts = get_file_parts(filename, PART_SIZE)

        for part_start, part_size in tqdm(parts, desc=f"Processing parts of {filename}"):
            try:
                print(f"Loading part {part_start}-{part_start + part_size} into memory")
                with open(filename, 'rb') as file:
                    file.seek(part_start)
                    part_data = file.read(part_size).decode(encoding, errors='ignore')

                print(f"Processing part {part_start}-{part_start + part_size}")
                if len(part_data.strip()) == 0:
                    print(f"Part {part_start}-{part_start + part_size} is empty after reading.")
                    continue

                # Process the part in parallel for each length
                part_filtered_words, lines_read = process_part(part_data, min_length, max_length, num_cores)
                
                print(f"Lines read: {lines_read}")
                file_total_input += lines_read
                file_total_output += sum(len(words) for words in part_filtered_words.values())

                print(f"Filtered words collected, writing to file.")
                # Write filtered words to files
                write_filtered_words(part_filtered_words, filtered_dir, filename, encoding)

                print(f"Removing part {part_start}-{part_start + part_size} from memory")
                del part_data  # Free up memory
                del part_filtered_words

            except Exception as e:
                print(f"Error processing part {part_start}-{part_start + part_size} in file {filename}: {e}")

        total_input_passwords += file_total_input
        total_output_passwords += file_total_output

        stats.append({
            'filename': filename,
            'input_passwords': file_total_input,
            'output_passwords': file_total_output
        })

    with open(os.path.join(stats_dir, 'total_password_count.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['total_input_passwords', 'total_output_passwords'])
        writer.writerow([total_input_passwords, total_output_passwords])

    with open(os.path.join(stats_dir, 'file_password_counts.csv'), 'w', newline='') as csvfile:
        fieldnames = ['filename', 'input_passwords', 'output_passwords']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            writer.writerow(stat)

    with open(os.path.join(stats_dir, 'length_password_counts.csv'), 'w', newline='') as csvfile:
        fieldnames = ['filename', 'length', 'filtered_passwords']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stat in length_stats:
            writer.writerow(stat)

    print(f"Total input passwords: {total_input_passwords}")
    print(f"Total output passwords: {total_output_passwords}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter words by length from multiple large text files.")
    parser.add_argument('--directory', type=str, default='PasswrodsBIG', help='The directory containing the large text files.')
    parser.add_argument('--min_length', type=int, default=5, help='The minimum length of words to filter.')
    parser.add_argument('--max_length', type=int, default=8, help='The maximum length of words to filter.')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding (default: utf-8).')

    args = parser.parse_args()

    parallel_filter_words(args.directory, args.min_length, args.max_length, args.encoding)
