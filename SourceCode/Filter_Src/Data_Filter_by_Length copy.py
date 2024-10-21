import os
import argparse
import multiprocessing
from tqdm import tqdm

def filter_words_by_length(chunk, length):
    return [word for word in chunk if len(word) == length]

def process_subchunk(subchunk, length):
    return filter_words_by_length(subchunk, length)

def process_chunk(chunk_start, chunk_size, length, filename, subchunk_size=1024*1024, encoding='utf-8'):
    filtered_words = []
    with open(filename, 'r', encoding=encoding, errors='ignore') as file:
        file.seek(chunk_start)
        while chunk_size > 0:
            subchunk = file.read(min(subchunk_size, chunk_size)).splitlines()
            filtered_words.extend(process_subchunk(subchunk, length))
            chunk_size -= subchunk_size
            if not subchunk:
                break
    return filtered_words

def get_file_chunks(filename, num_chunks):
    file_size = os.path.getsize(filename)
    chunk_size = file_size // num_chunks
    return [(i * chunk_size, chunk_size) for i in range(num_chunks)]

def parallel_filter_words(filename, word_length, num_chunks, subchunk_size, encoding):
    chunks = get_file_chunks(filename, num_chunks)
    pool = multiprocessing.Pool(num_chunks)
    results = []

    output_filename = f'filtered_words_{word_length}.txt'
    with open(output_filename, 'w', encoding=encoding) as output_file:
        for chunk_start, chunk_size in tqdm(chunks):
            result = pool.apply_async(process_chunk, (chunk_start, chunk_size, word_length, filename, subchunk_size, encoding))
            results.append(result)

        for result in results:
            filtered_words = result.get()
            output_file.write('\n'.join(filtered_words) + '\n')

    pool.close()
    pool.join()

    print(f"Filtered words saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter words by length from a large text file.")
    parser.add_argument('--filename', type=str, default="H:\\PartsOutputs\\00000001.txt",help='The path to the large text file.')
    parser.add_argument('--word_length', type=int,default=8, help='The length of words to filter.')
    parser.add_argument('--num_chunks', type=int, default=10, help='Number of chunks to split the file into for parallel processing.')
    parser.add_argument('--subchunk_size', type=int, default=1024*1024, help='Size of subchunks to process at a time (in bytes).')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding (default: utf-8).')

    args = parser.parse_args()

    parallel_filter_words(args.filename, args.word_length, args.num_chunks, args.subchunk_size, args.encoding)
