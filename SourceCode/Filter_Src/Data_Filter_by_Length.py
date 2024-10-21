import os
import argparse
import multiprocessing
from tqdm import tqdm

def filter_words_by_length(chunk, lengths):
    return [word for word in chunk if len(word) in lengths]

def process_subchunk(subchunk, lengths):
    return filter_words_by_length(subchunk, lengths)

def process_chunk(chunk_start, chunk_size, lengths, filename, subchunk_size=1024*1024, encoding='utf-8'):
    filtered_words = []
    with open(filename, 'r', encoding=encoding, errors='ignore') as file:
        file.seek(chunk_start)
        while chunk_size > 0:
            subchunk = file.read(min(subchunk_size, chunk_size)).splitlines()
            filtered_words.extend(process_subchunk(subchunk, lengths))
            chunk_size -= subchunk_size
            if not subchunk:
                break
    return filtered_words

def get_file_chunks(filename, num_chunks):
    file_size = os.path.getsize(filename)
    chunk_size = file_size // num_chunks
    return [(i * chunk_size, chunk_size) for i in range(num_chunks)]

def parallel_filter_words(directory, word_lengths, num_chunks, subchunk_size, encoding, output_directory):
    pool = multiprocessing.Pool(num_chunks)
    results = []

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'processing_log.txt')
    with open(log_file, 'w') as log:
        for filename in os.listdir(directory):
            if not filename.endswith('.txt'):
                continue
            file_path = os.path.join(directory, filename)
            chunks = get_file_chunks(file_path, num_chunks)

            for length in word_lengths:
                length_dir = os.path.join(output_directory, f'length_{length}')
                if not os.path.exists(length_dir):
                    os.makedirs(length_dir)

                output_filename = os.path.join(length_dir, f'filtered_words_{os.path.splitext(filename)[0]}.txt')
                
                file_results = []
                for chunk_start, chunk_size in tqdm(chunks, desc=f"Processing {filename} for length {length}"):
                    result = pool.apply_async(process_chunk, (chunk_start, chunk_size, [length], file_path, subchunk_size, encoding))
                    file_results.append(result)

                results.append((file_results, output_filename))
                log.write(f"Processed input file: {file_path}, Output file: {output_filename}\n")

    for file_results, output_filename in results:
        with open(output_filename, 'w', encoding=encoding) as output_file:
            for result in file_results:
                filtered_words = result.get()
                if filtered_words:  # Only write if there are filtered words
                    output_file.write('\n'.join(filtered_words) + '\n')

    pool.close()
    pool.join()

    print(f"Filtered words saved in directory: {output_directory}")

def calculate_statistics(directory, word_lengths, output_directory):
    total_input_passwords = 0
    filtered_passwords_count = {length: 0 for length in word_lengths}

    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            total_input_passwords += len(file.readlines())

    for length in word_lengths:
        length_dir = os.path.join(output_directory, f'length_{length}')
        for filename in os.listdir(length_dir):
            if not filename.endswith('.txt'):
                continue
            file_path = os.path.join(length_dir, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                filtered_passwords_count[length] += len(file.readlines())

    print(f"Total number of input passwords: {total_input_passwords}")
    for length, count in filtered_passwords_count.items():
        print(f"Total number of filtered passwords of length {length}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter words by length from text files in a directory.")
    parser.add_argument('--directory', type=str, default='H:\\PartsOutputs', help='The path to the directory containing text files.')
    parser.add_argument('--word_lengths', type=int, nargs='+', help='The lengths of words to filter. Default: [5, 6, 7, 8]')
    parser.add_argument('--num_chunks', type=int, default=10, help='Number of chunks to split each file into for parallel processing.')
    parser.add_argument('--subchunk_size', type=int, default=1024*1024, help='Size of subchunks to process at a time (in bytes).')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding (default: utf-8).')
    parser.add_argument('--output_directory', type=str, default='.', help='The directory to save filtered words.')

    args = parser.parse_args()

    # Set default word lengths if not provided
    if args.word_lengths is None:
        args.word_lengths = [5, 6, 7, 8]

    parallel_filter_words(args.directory, args.word_lengths, args.num_chunks, args.subchunk_size, args.encoding, args.output_directory)
    calculate_statistics(args.directory, args.word_lengths, args.output_directory)

