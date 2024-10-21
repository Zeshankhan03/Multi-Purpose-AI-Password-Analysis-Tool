import os
import argparse
import multiprocessing
from tqdm import tqdm
import csv
import psutil
import pickle

SUPPORTED_EXTENSIONS = ['.txt', '.dic','.lst','.uniq']

def filter_words_by_length(chunk, min_length, max_length):
    filtered_words = {length: [] for length in range(min_length, max_length + 1)}
    for word in chunk:
        word = word.strip()  # Strip leading and trailing spaces
        word_len = len(word)
        if min_length <= word_len <= max_length:
            filtered_words[word_len].append(word)
    return filtered_words

def process_chunk(chunk_start, chunk_size, min_length, max_length, filename, subchunk_size, encoding='utf-8'):
    filtered_words = {length: [] for length in range(min_length, max_length + 1)}
    total_lines_read = 0
    try:
        with open(filename, 'r', encoding=encoding, errors='ignore') as file:
            file.seek(chunk_start)
            while chunk_size > 0:
                subchunk = file.read(min(subchunk_size, chunk_size)).splitlines()
                total_lines_read += len(subchunk)
                filtered = filter_words_by_length(subchunk, min_length, max_length)
                for length in filtered:
                    filtered_words[length].extend(filtered[length])
                chunk_size -= subchunk_size
                if not subchunk:
                    break
    except Exception as e:
        print(f"Error processing chunk {chunk_start}-{chunk_start+chunk_size} in file {filename}: {e}")
    return filtered_words, total_lines_read

def get_file_chunks(filename, chunk_size):
    file_size = os.path.getsize(filename)
    return [(i, min(chunk_size, file_size - i)) for i in range(0, file_size, chunk_size)]

def save_progress(progress_file, progress):
    with open(progress_file, 'wb') as file:
        pickle.dump(progress, file)

def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as file:
            return pickle.load(file)
    return {}

def write_filtered_words(filtered_words, filtered_dir, filename, min_length, max_length, encoding):
    for length in range(min_length, max_length + 1):
        output_txt_filename = os.path.join(filtered_dir, f'filtered_words_{length}_{os.path.basename(filename)}.txt')
        with open(output_txt_filename, 'a', encoding=encoding) as output_file:
            output_file.write('\n'.join(filtered_words[length]) + '\n')

def parallel_filter_words(directory, min_length, max_length, encoding, chunk_size, subchunk_size):
    total_input_passwords = 0
    total_output_passwords = 0
    stats = []
    length_stats = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filtered_dir = os.path.join(script_dir, "filtered_passwords")
    stats_dir = os.path.join(script_dir, "stats")
    progress_file = os.path.join(script_dir, "progress.pkl")

    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    filenames = [
        os.path.join(directory, file) 
        for file in os.listdir(directory) 
        if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS
    ]

    progress = load_progress(progress_file)

    for filename in filenames:
        if filename in progress and progress[filename] == "completed":
            print(f"Skipping already processed file: {filename}")
            continue
        
        file_total_input = 0
        file_total_output = 0
        chunks = get_file_chunks(filename, chunk_size)
        pool = multiprocessing.Pool(multiprocessing.cpu_count())

        if filename not in progress:
            progress[filename] = {length: 0 for length in range(min_length, max_length + 1)}

        results = []
        for chunk_index, (chunk_start, chunk_size) in enumerate(tqdm(chunks, desc=f"Processing {filename}")):
            if chunk_index < max(progress[filename].values()):
                continue
            
            result = pool.apply_async(process_chunk, (chunk_start, chunk_size, min_length, max_length, filename, subchunk_size, encoding))
            results.append((result, chunk_index))

        chunk_pbar = tqdm(total=len(results), desc=f"Processing chunks for {filename}")
        for result, chunk_index in results:
            try:
                filtered_words, lines_read = result.get()
                file_total_input += lines_read

                write_filtered_words(filtered_words, filtered_dir, filename, min_length, max_length, encoding)
                file_total_output += sum(len(words) for words in filtered_words.values())

                for length in range(min_length, max_length + 1):
                    progress[filename][length] = chunk_index + 1

                save_progress(progress_file, progress)
                chunk_pbar.update(1)
            except Exception as e:
                print(f"Error processing result from chunk {chunk_index} in file {filename}: {e}")
        chunk_pbar.close()

        pool.close()
        pool.join()

        total_input_passwords += file_total_input
        total_output_passwords += file_total_output

        stats.append({
            'filename': filename,
            'input_passwords': file_total_input,
            'output_passwords': file_total_output
        })

        progress[filename] = "completed"
        save_progress(progress_file, progress)

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
    parser.add_argument('--directory', type=str, default='H:\\OUTPUTTXT', help='The directory containing the large text files.')
    parser.add_argument('--min_length', type=int, default=5, help='The minimum length of words to filter.')
    parser.add_argument('--max_length', type=int, default=8, help='The maximum length of words to filter.')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding (default: utf-8).')
    parser.add_argument('--chunk_size', type=int, default=1024*1024*10, help='Chunk size in bytes (default: 10MB).')
    parser.add_argument('--subchunk_size', type=int, default=1024*1024*5, help='Subchunk size in bytes.')
    
    args = parser.parse_args()

    if args.subchunk_size is None:
        args.subchunk_size = args.chunk_size // 2

    parallel_filter_words(args.directory, args.min_length, args.max_length, args.encoding, args.chunk_size, args.subchunk_size)