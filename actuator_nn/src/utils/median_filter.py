import pandas as pd
import numpy as np
from scipy import signal
import argparse

def apply_median_filter(data, window_size):
    return signal.medfilt(data, kernel_size=window_size)

def process_txt(input_file, output_file, column_to_filter, window_size, separator=','):
    # Read the TXT file
    df = pd.read_csv(input_file, sep=separator)

    # Check if the specified column exists
    if column_to_filter not in df.columns:
        raise ValueError(f"Column '{column_to_filter}' not found in the TXT file.")

    # Extract data
    data = df[column_to_filter].values

    # Apply the median filter
    filtered_data = apply_median_filter(data, window_size)

    # Add the filtered data as a new column
    df[f'{column_to_filter}_filtered'] = filtered_data

    # Save the result to a new TXT file
    df.to_csv(output_file, sep=separator, index=False)
    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a median filter to remove spikes in a column in a TXT file.")
    parser.add_argument("input_file", help="Input TXT file path")
    parser.add_argument("output_file", help="Output TXT file path")
    parser.add_argument("column_to_filter", help="Name of the column to apply the filter to")
    parser.add_argument("window_size", type=int, help="Window size for the median filter (must be odd)")
    parser.add_argument("--separator", default=',', help="Column separator in the TXT file (default: comma)")

    args = parser.parse_args()

    if args.window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")

    process_txt(args.input_file, args.output_file, args.column_to_filter, args.window_size, args.separator)