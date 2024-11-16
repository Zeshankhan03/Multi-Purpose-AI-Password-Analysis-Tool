import hashlib
import os
from pathlib import Path

def convert_to_md5(text):
    """Convert a string to MD5 hash"""
    return hashlib.md5(text.encode()).hexdigest()

def process_password_file(input_file, output_file):
    """Process a single password file and convert to MD5"""
    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for line in infile:
                # Remove any whitespace/newlines and convert to MD5
                password = line.strip()
                if password:  # Skip empty lines
                    md5_hash = convert_to_md5(password)
                    # Ensure only the hash is written, no extra characters
                    outfile.write(f"{md5_hash}\n")

def main():
    # Use fixed input directory instead of user input
    input_dir = r"SourceCode\ReworldTest\Plain_Passwords"
    output_dir = r"SourceCode\ReworldTest\MD5_Passwords"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file in the input directory
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    for file_path in input_path.glob('*'):
        if file_path.is_file():
            # Create output filename by replacing 'password' with 'md5'
            output_filename = file_path.name.lower().replace('password', 'md5')
            output_file = output_path / output_filename
            
            print(f"Processing: {file_path.name}")
            process_password_file(file_path, output_file)
            print(f"Created: {output_file.name}")

if __name__ == "__main__":
    try:
        main()
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
