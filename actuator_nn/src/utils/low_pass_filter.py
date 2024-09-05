import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import argparse

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def process_txt(input_file, output_file, column_to_filter, cutoff_freq, sampling_freq, separator=','):
    # Read the TXT file
    df = pd.read_csv(input_file, sep=separator)

    # Check if the specified column exists
    if column_to_filter not in df.columns:
        raise ValueError(f"Column '{column_to_filter}' not found in the TXT file.")

    # Apply the low-pass filter
    filtered_data = apply_lowpass_filter(df[column_to_filter], cutoff_freq, sampling_freq)

    # Replace the original column with the filtered data
    df[column_to_filter] = filtered_data

    # Save the result to a new TXT file
    df.to_csv(output_file, sep=separator, index=False)
    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a low-pass filter to a column in a TXT file.")
    parser.add_argument("input_file", help="Input TXT file path")
    parser.add_argument("output_file", help="Output TXT file path")
    parser.add_argument("column_to_filter", help="Name of the column to apply the filter to")
    parser.add_argument("cutoff_freq", type=float, help="Cutoff frequency for the low-pass filter")
    parser.add_argument("sampling_freq", type=float, help="Sampling frequency of the data")
    parser.add_argument("--separator", default=',', help="Column separator in the TXT file (default: comma)")

    args = parser.parse_args()

    process_txt(args.input_file, args.output_file, args.column_to_filter, args.cutoff_freq, args.sampling_freq, args.separator)