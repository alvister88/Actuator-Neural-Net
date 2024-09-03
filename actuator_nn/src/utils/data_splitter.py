import argparse
import os
import random
from itertools import islice

def split_data(input_file, output_dir, train_ratio=0.9, chunk_size=3500, number=None, randomize=True):
    """
    Splits a large text file into training and validation sets, preserving chunks of sequential data.
    Args:
    input_file (str): Path to the input text file.
    output_dir (str): Directory to save the output files.
    train_ratio (float): Ratio of data to use for training (default: 0.9).
    chunk_size (int): Number of lines to process at once and preserve as a sequential chunk (default: 3500).
    number (int): Optional number to append to output file names.
    randomize (bool): Whether to randomize the split or use a deterministic approach (default: True).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct file names based on the 'number' argument
    suffix = f"_{number}" if number is not None else ""
    train_file = os.path.join(output_dir, f'train_data{suffix}.txt')
    val_file = os.path.join(output_dir, f'validation_data{suffix}.txt')

    with open(input_file, 'r') as infile, \
         open(train_file, 'w') as train_outfile, \
         open(val_file, 'w') as val_outfile:
        
        # Read and write the header to both files
        header = infile.readline()
        train_outfile.write(header)
        val_outfile.write(header)

        train_count = 0
        val_count = 0
        total_count = 0

        for chunk in iter(lambda: list(islice(infile, chunk_size)), []):
            total_count += len(chunk)
            
            if randomize:
                # Randomized approach
                if random.random() < train_ratio:
                    train_outfile.writelines(chunk)
                    train_count += len(chunk)
                else:
                    val_outfile.writelines(chunk)
                    val_count += len(chunk)
            else:
                # Deterministic approach
                if train_count < int(total_count * train_ratio):
                    train_outfile.writelines(chunk)
                    train_count += len(chunk)
                else:
                    val_outfile.writelines(chunk)
                    val_count += len(chunk)

    return train_count, val_count

def main():
    parser = argparse.ArgumentParser(description="Split a large text file into training and validation sets, preserving sequential chunks.")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("output_dir", help="Directory to save the output files")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data to use for training (default: 0.9)")
    parser.add_argument("--chunk_size", type=int, default=3500, help="Number of lines to process at once and preserve as a sequential chunk (default: 3500)")
    parser.add_argument("--number", type=int, help="Optional number to append to output file names")
    parser.add_argument("--no-randomize", action="store_false", dest="randomize", help="Use deterministic split instead of random (default: randomized)")
    parser.set_defaults(randomize=True)
    args = parser.parse_args()

    # Check if chunk size has been changed
    if args.chunk_size != 3500:
        print("\033[93mWARNING: Chunk size has been changed from the default value of 3500.")
        print("Please ensure you update the train chunk size in your training script accordingly.\033[0m")

    train_count, val_count = split_data(args.input_file, args.output_dir, args.train_ratio, args.chunk_size, args.number, args.randomize)

    # Construct file names for the print statements
    suffix = f"_{args.number}" if args.number is not None else ""
    train_file = f'train_data{suffix}.txt'
    val_file = f'validation_data{suffix}.txt'

    print(f"Data split complete. Training data saved to {os.path.join(args.output_dir, train_file)}")
    print(f"Validation data saved to {os.path.join(args.output_dir, val_file)}")
    print(f"Total lines: Training set: {train_count}, Validation set: {val_count}")
    print(f"Actual split ratio: {train_count / (train_count + val_count):.2f}")
    print(f"Split method: {'Randomized' if args.randomize else 'Deterministic'}")
    print(f"Chunk size used: {args.chunk_size}")

if __name__ == "__main__":
    main()