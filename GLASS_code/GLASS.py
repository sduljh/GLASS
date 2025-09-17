import argparse
import logging
import os
import torch
import psutil
import time  # Importing time module to track runtime
from model import data_test, data_train, BipartiteGCN
from bam_clean import clean

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and Test the model with BAM files")
    
    # Required arguments
    parser.add_argument('test_bam_files', type=str, nargs='+', 
                        help="List of test BAM files (at least one required)")

    # Optional arguments with defaults
    parser.add_argument('--train_junction_gtf_file', type=str, default=None,
                        help="Train junction GTF file (default: None)")
    parser.add_argument('--train_bam_files', type=str, nargs='+', default=None,
                        help="List of train BAM files (default: None)")
    parser.add_argument('--val_bam_file', type=str, default=None,
                        help="Validation BAM file (default: None)")
    parser.add_argument('--val_junction_gtf_file', type=str, default=None,
                        help="Validation junction GTF file (default: None)")
    parser.add_argument('--train_device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help="Training device (default: 'cpu')")
    parser.add_argument('--test_device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help="Testing device (default: 'cpu')")

    # Parse arguments
    args = parser.parse_args()

    # Start time measurement
    start_time = time.time()

    # Track memory usage at the start of the program
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    logging.info(f"Initial memory usage: {initial_memory:.2f} MB")

    # Assign values from parsed arguments
    train_bam_files = args.train_bam_files
    test_bam_files = args.test_bam_files
    train_junction_gtf_file = args.train_junction_gtf_file
    val_junction_gtf_file = args.val_junction_gtf_file
    val_bam_file = args.val_bam_file

    # Set devices for training and testing
    train_device = torch.device(f'cuda' if args.train_device == 'cuda' and torch.cuda.is_available() else 'cpu')
    test_device = torch.device(f'cuda' if args.test_device == 'cuda' and torch.cuda.is_available() else 'cpu')
    
    logging.info(f'Training on device: {train_device}')
    logging.info(f'Testing on device: {test_device}')

    # Train models
    models, weights = data_train(train_junction_gtf_file, train_bam_files, val_bam_file,
                                 val_junction_gtf_file, train_device=train_device)

    for alignment_bam_path in test_bam_files:  # Loop through each BAM file
        try:
            directory_path = os.path.dirname(alignment_bam_path)
            # Add "_filter" suffix to the new BAM file.
            alignment_bam_filter_path = os.path.join(directory_path,
                                                     f"{os.path.splitext(os.path.basename(alignment_bam_path))[0]}_filter.bam")

            # Perform the data testing
            feature_df, predictions_df = data_test(alignment_bam_path, models, weights, test_device=test_device)

            # Clean the data
            clean(feature_df, predictions_df, alignment_bam_path, alignment_bam_filter_path)
        except FileNotFoundError:
            logging.info(f"The file {alignment_bam_path} was not found.")
            continue
        except Exception as e:
            logging.error(f"Error occurred while processing {alignment_bam_path}: {str(e)}")

    # Track memory usage after processing
    final_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    logging.info(f"Final memory usage: {final_memory:.2f} MB")

    # End time measurement
    end_time = time.time()
    runtime = end_time - start_time  # Calculate runtime in seconds
    logging.info(f"Total runtime: {runtime:.2f} seconds")


if __name__ == "__main__":
    main()
