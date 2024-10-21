import os
import argparse
import multiprocessing
from tqdm import tqdm
import csv
import unittest

def filter_words_by_length(chunk, min_length, max_length):
    return [word for word in chunk if min_length <= len(word) <= max_length]

def process_subchunk(subchunk, min_length, max_length):
    return filter_words_by_length(subchunk, min_length, max_length)

def process_chunk(chunk_start, chunk_size, min_length, max_length, filename, subchunk_size=1024*1024, encoding='utf-8'):
    filtered_words = []
    total_lines_read = 0
    with open(filename, 'r', encoding=encoding, errors='ignore') as file:
        file.seek(chunk_start)
        while chunk_size > 0:
            subchunk = file.read(min(subchunk_size, chunk_size)).splitlines()
            total_lines_read += len(subchunk)
            filtered_words.extend(process_subchunk(subchunk, min_length, max_length))
            chunk_size -= subchunk_size
            if not subchunk:
                break
    return filtered_words, total_lines_read

def get_file_chunks(filename, num_chunks):
    file_size = os.path.getsize(filename)
    chunk_size = file_size // num_chunks
    return [(i * chunk_size, chunk_size) for i in range(num_chunks)]

def parallel_filter_words(filenames, min_length, max_length, num_chunks, subchunk_size, encoding):
    total_input_passwords = 0
    total_output_passwords = 0
    stats = []
    length_stats = []

    filtered_dir = "filtered_passwords"
    stats_dir = "stats"
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    for filename in filenames:
        file_total_input = 0
        file_total_output = 0
        chunks = get_file_chunks(filename, num_chunks)
        pool = multiprocessing.Pool(num_chunks)
        
        for length in range(min_length, max_length + 1):
            filtered_words = []
            results = []
            for chunk_start, chunk_size in tqdm(chunks, desc=f"Processing {filename} for length {length}"):
                result = pool.apply_async(process_chunk, (chunk_start, chunk_size, length, length, filename, subchunk_size, encoding))
                results.append(result)

            length_input_count = 0
            length_output_count = 0
            for result in results:
                filtered, lines_read = result.get()
                length_input_count += lines_read
                length_output_count += len(filtered)
                filtered_words.extend(filtered)

            file_total_input += length_input_count
            file_total_output += length_output_count

            output_txt_filename = os.path.join(filtered_dir, f'filtered_words_{length}_{os.path.basename(filename)}.txt')
            with open(output_txt_filename, 'w', encoding=encoding) as output_file:
                output_file.write('\n'.join(filtered_words) + '\n')

            length_stats.append({
                'filename': filename,
                'length': length,
                'filtered_passwords': length_output_count
            })

        pool.close()
        pool.join()

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
    parser.add_argument('filenames', nargs='+', type=str, help='The paths to the large text files.')
    parser.add_argument('min_length', type=int, help='The minimum length of words to filter.')
    parser.add_argument('max_length', type=int, help='The maximum length of words to filter.')
    parser.add_argument('--num_chunks', type=int, default=20, help='Number of chunks to split each file into for parallel processing.')
    parser.add_argument('--subchunk_size', type=int, default=1024*1024, help='Size of subchunks to process at a time (in bytes).')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding (default: utf-8).')

    args = parser.parse_args()

    parallel_filter_words(args.filenames, args.min_length, args.max_length, args.num_chunks, args.subchunk_size, args.encoding)
