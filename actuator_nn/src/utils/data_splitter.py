import argparse
import os
import random
from itertools import islice

def split_data(input_file, output_dir, train_ratio=0.9, chunk_size=500, number=None):
    """
    Splits a large text file into training and validation sets.
    
    Args:
    input_file (str): Path to the input text file.
    output_dir (str): Directory to save the output files.
    train_ratio (float): Ratio of data to use for training (default: 0.9).
    chunk_size (int): Number of lines to process at once for large files (default: 500).
    number (int): Optional number to append to output file names.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Construct file names based on the 'number' argument
    suffix = f"{number}" if number is not None else ""
    train_file = os.path.join(output_dir, f'train_data{suffix}.txt')
    val_file = os.path.join(output_dir, f'validation_data{suffix}.txt')
    
    # First pass: count total lines
    with open(input_file, 'r') as f:
        total_lines = sum(1 for _ in f)

    train_lines = int((total_lines - 1) * train_ratio)  # Subtract 1 to account for header
    val_lines = total_lines - train_lines - 1  # Subtract 1 for header
    
    with open(input_file, 'r') as infile, \
         open(train_file, 'w') as train_outfile, \
         open(val_file, 'w') as val_outfile:
        
        # Read and write the header to both files
        header = infile.readline()
        train_outfile.write(header)
        val_outfile.write(header)
        
        lines_written = 0
        for chunk in iter(lambda: list(islice(infile, chunk_size)), []):
            lines_written = process_chunk(chunk, train_outfile, val_outfile, train_lines, val_lines, lines_written)

def process_chunk(chunk, train_outfile, val_outfile, train_lines, val_lines, lines_written):
    """
    Process a chunk of lines and write to appropriate output files.
    """
    for line in chunk:
        if lines_written < train_lines:
            train_outfile.write(line)
        else:
            val_outfile.write(line)
        lines_written += 1
    
    return lines_written

def main():
    parser = argparse.ArgumentParser(description="Split a large text file into training and validation sets.")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("output_dir", help="Directory to save the output files")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data to use for training (default: 0.9)")
    parser.add_argument("--chunk_size", type=int, default=500, help="Number of lines to process at once (default: 500)")
    parser.add_argument("--number", type=int, help="Optional number to append to output file names")
    
    args = parser.parse_args()
    
    split_data(args.input_file, args.output_dir, args.train_ratio, args.chunk_size, args.number)
    
    # Construct file names for the print statements
    suffix = f"{args.number}" if args.number is not None else ""
    train_file = f'train_data{suffix}.txt'
    val_file = f'validation_data{suffix}.txt'
    
    print(f"Data split complete. Training data saved to {os.path.join(args.output_dir, train_file)}")
    print(f"Validation data saved to {os.path.join(args.output_dir, val_file)}")

if __name__ == "__main__":
    main()