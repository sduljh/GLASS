import pandas as pd
import re
import subprocess
import logging
import concurrent.futures
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)


def process_bam(bam_path, align_path):
    try:
        logging.info(f"Generating alignment info from BAM file: {bam_path}")
        command = f"{grandparent_dir}/C_code/src/get_alignment_info {bam_path} {align_path}"
        subprocess.run(command, shell=True, check=True)
        align_df = pd.read_csv(align_path, sep=r'\s+', engine='python', header=0)
        logging.info(f"Successfully generated {align_path}")
        return align_df
    except Exception as e:
        logging.error(f"Error in process_bam: {e}")
        raise

def calculate_comma_frequency(df):
    
    # Assuming 'junction_info' is the column we need to analyze
    if 'junction_info' not in df.columns:
        print(f"Column 'junction_info' not found in the file {file_path}!")
        return

    # Count the number of hyphens in each row of 'junction_info'
    comma_counts = df['junction_info'].apply(lambda x: x.count('-'))

    # Count how many rows have zero, one, or more than one hyphen
    zero_count = (comma_counts == 0).sum()
    one_count = (comma_counts == 1).sum()
    greater_than_one_count = (comma_counts > 1).sum()

    # Calculate frequencies
    total_count = len(df)
    zero_frequency = zero_count / total_count
    one_frequency = one_count / total_count
    greater_than_one_frequency = greater_than_one_count / total_count

    greater_than_one_frequency_pro = greater_than_one_count / (greater_than_one_count + one_count)

    return zero_frequency, one_frequency, greater_than_one_frequency, greater_than_one_frequency_pro


def main():
    # File paths
    bam_path_1 = '/home/lijiahao/refine/data_transfer/NA12878.bam'
    bam_path_2 = '/home/lijiahao/refine/data_transfer/SRR4235527.bam'
    directory_path1 = os.path.dirname(file_path_1)
    align_path1 = os.path.join(directory_path1, f"{os.path.splitext(os.path.basename(bam_path_1))[0]}.info")
    directory_path2 = os.path.dirname(file_path_2)
    align_path2 = os.path.join(directory_path1, f"{os.path.splitext(os.path.basename(bam_path_2))[0]}.info")


    align_df1 = process_bam(bam_path_1, align_path1)
    align_df2 = process_bam(bam_path_2, align_path2)

    # Calculate the frequencies for the two files
    zero_freq_1, one_freq_1, greater_freq_1, greater_freq_ori_1 = calculate_comma_frequency(align_df1)
    zero_freq_2, one_freq_2, greater_freq_2, greater_freq_ori_2 = calculate_comma_frequency(align_df2)

    # Output the results
    print(f"Statistics for file {file_path_1}:")
    print(
        f"Zero splicing ratio: {zero_freq_1:.4f}, One splicing ratio: {one_freq_1:.4f}, Greater than one splicing ratio: {greater_freq_1:.4f}")

    print(f"Proportion of multiple splicing in spliced reads: {greater_freq_ori_1:.4f}")

    print(f"Statistics for file {file_path_2}:")
    print(
        f"Zero splicing ratio: {zero_freq_2:.4f}, One splicing ratio: {one_freq_2:.4f}, Greater than one splicing ratio: {greater_freq_2:.4f}")
    print(f"Proportion of multiple splicing in spliced reads: {greater_freq_ori_2:.4f}")


if __name__ == "__main__":
    main()
