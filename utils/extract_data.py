import os
import tarfile
from tqdm import tqdm
import argparse

def combine_and_extract_files(input_directory, output_directory):
    # Create a dictionary to store file parts
    file_parts = {}

    # Iterate over files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".tar.gz.partaa"):
            base_name = filename[:-11]  # Get the base name of the file
            file_parts[base_name] = []

    # Identify all parts of each file
    for part in os.listdir(input_directory):
        for base_name in file_parts:
            if part.startswith(base_name):
                file_parts[base_name].append(part)

    # Combine and extract each set of parts
    for base_name, parts in file_parts.items():
        print(f"Combining files for: {base_name}")
        full_path = os.path.join(output_directory, base_name + ".tar.gz")

        with open(full_path, 'wb') as outfile:
            for part in tqdm(sorted(parts), desc=f"Combining {base_name}"):
                part_path = os.path.join(input_directory, part)
                with open(part_path, 'rb') as infile:
                    outfile.write(infile.read())
                os.remove(part_path)  # Optionally, remove the part files

        print(f"Extracting: {base_name}")
        with tarfile.open(full_path, 'r:gz') as tar:
            tar.extractall(path=output_directory, members=tqdm(tar, desc=f"Extracting {base_name}"))
        print(f"Extraction complete for {base_name}")

def main():
    parser = argparse.ArgumentParser(description='Combine and extract multipart tar.gz files.')
    parser.add_argument('--input', help='Input directory where the tar.gz parts are located', default='./', required=False)
    parser.add_argument('--output', help='Output directory where to extract the files', required=True)
    
    args = parser.parse_args()

    combine_and_extract_files(args.input, args.output)

if __name__ == "__main__":
    main()