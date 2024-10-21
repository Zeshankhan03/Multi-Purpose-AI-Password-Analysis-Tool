import os
import argparse
import multiprocessing
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

def filter_words_by_length(chunk, min_length, max_length):
    return [word for word in chunk if min_length <= len(word) <= max_length]

def process_subchunk(subchunk, min_length, max_length):
    return filter_words_by_length(subchunk, min_length, max_length)

def process_chunk(chunk_start, chunk_size, min_length, max_length, filename, subchunk_size=1024*1024, encoding='utf-8'):
    filtered_words = []
    output_counts = {length: 0 for length in range(min_length, max_length + 1)}
    with open(filename, 'r', encoding=encoding, errors='ignore') as file:
        file.seek(chunk_start)
        while chunk_size > 0:
            subchunk = file.read(min(subchunk_size, chunk_size)).splitlines()
            for length in range(min_length, max_length + 1):
                filtered_subchunk = process_subchunk(subchunk, length, length)
                output_counts[length] += len(filtered_subchunk)
                filtered_words.extend(filtered_subchunk)
            chunk_size -= subchunk_size
            if not subchunk:
                break
    return filtered_words, output_counts

def get_file_chunks(filename, num_chunks):
    file_size = os.path.getsize(filename)
    chunk_size = file_size // num_chunks
    return [(i * chunk_size, chunk_size) for i in range(num_chunks)]

def parallel_filter_words(filenames, min_length, max_length, num_chunks, subchunk_size, encoding):
    total_input_passwords = 0
    total_output_passwords = 0
    stats = []

    output_counts_fields = [f'output_passwords_length_{length}' for length in range(min_length, max_length + 1)]

    for filename in filenames:
        file_total_input = 0
        file_total_output = 0
        chunks = get_file_chunks(filename, num_chunks)
        pool = multiprocessing.Pool(num_chunks)
        
        for length in range(min_length, max_length + 1):
            output_counts_fields.append(f'output_passwords_length_{length}')
        
        for length in range(min_length, max_length + 1):
            output_counts_fields.append(f'output_passwords_length_{length}')
            output_filename = f'filtered_words_{length}_{os.path.basename(filename)}'
            with open(output_filename, 'w', encoding=encoding) as output_file:
                results = []
                for chunk_start, chunk_size in tqdm(chunks, desc=f"Processing {filename} for length {length}"):
                    result = pool.apply_async(process_chunk, (chunk_start, chunk_size, length, length, filename, subchunk_size, encoding))
                    results.append(result)

                length_input_count = 0
                output_counts = {length: 0 for length in range(min_length, max_length + 1)}
                for result in results:
                    filtered_words, length_output_counts = result.get()
                    length_input_count += sum(length_output_counts.values())
                    for key, value in length_output_counts.items():
                        output_counts[key] += value
                    output_file.write('\n'.join(filtered_words) + '\n')

                file_total_input += length_input_count
                file_total_output += sum(output_counts.values())

                # Generate graph for each length
                plt.bar(['input_passwords', 'output_passwords'], [length_input_count, sum(output_counts.values())])
                plt.xlabel('Type')
                plt.ylabel('Number of Passwords')
                plt.title(f'Password Counts for {filename} (Length {length})')
                plt.tight_layout()
                plt.savefig(f'{filename}_length_{length}_password_counts.png')
                plt.close()

        pool.close()
        pool.join()

        total_input_passwords += file_total_input
        total_output_passwords += file_total_output

        file_stats = {
            'filename': filename,
            'input_passwords': file_total_input,
            'output_passwords': file_total_output,
            **output_counts
        }
        stats.append(file_stats)

    # Output stats to CSV
    with open('password_stats.csv', 'w', newline='') as csvfile:
        fieldnames = ['filename', 'input_passwords', 'output_passwords', *output_counts_fields]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for stat in stats:
            writer.writerow(stat)
    
    # Print total counts
    print(f"Total input passwords: {total_input_passwords}")
    print(f"Total output passwords: {total_output_passwords}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter words by length from multiple large text files.")
    parser.add_argument('filenames', nargs='+', type=str, default='H:\OUTPUTTXT\\18_in_1.lst', help='The paths to the large text files.')
    parser.add_argument('--min_length', type=int, default=5, help='The minimum length of words to filter.')
    parser.add_argument('--max_length', type=int, default=8, help='The maximum length of words to filter.')
    parser.add_argument('--num_chunks', type=int, default=30, help='Number of chunks to split each file into for parallel processing.')
    parser.add_argument('--subchunk_size', type=int, default=1024*1024, help='Size of subchunks to process at a time (in bytes).')
    parser.add_argument('--encoding', type=str, default='utf-8',help='File encoding (default: utf-8).')

    args = parser.parse_args()

    parallel_filter_words(args.filenames, args.min_length, args.max_length, args.num_chunks, args.subchunk_size, args.encoding)
