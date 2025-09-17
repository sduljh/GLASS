import pandas as pd
import argparse
import logging
import os
from bam_process import extract_bam_info
from anno_process import process_junction
from align_compute import process_and_match

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)

def junction_compute(df, bam_file, output_csv):
    total_rows = len(df)
    label_1_rows = len(df[df['label'] == 1])
    incorrect_rows = total_rows - label_1_rows
    accuracy = label_1_rows / total_rows if total_rows > 0 else 0
    error_rate = incorrect_rows / total_rows if total_rows > 0 else 0

    # Print metrics to console
    print(f"Total junctions: {total_rows}")
    print(f"Correct junctions: {label_1_rows}")
    print(f"Incorrect junctions: {incorrect_rows}")
    print(f"Junction accuracy: {accuracy:.2%}")
    print(f"Junction error rate: {error_rate:.2%}")

    # Prepare data for CSV
    metrics = {
        'BAM_File': [os.path.basename(bam_file)],  # Extract filename for clarity
        'Total_Junctions': [total_rows],
        'Correct_Junctions': [label_1_rows],
        'Incorrect_Junctions': [incorrect_rows],
        'Junction_Accuracy': [f"{accuracy:.2%}"],
        'Junction_Error_Rate': [f"{error_rate:.2%}"]
    }
    metrics_df = pd.DataFrame(metrics)

    # Append to CSV (create if doesn't exist, append if it does)
    if os.path.exists(output_csv):
        metrics_df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(output_csv, mode='w', header=True, index=False)

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Analyze splicing junctions in BAM files and GTF annotations')
    parser.add_argument('--bam', required=True, help='Input BAM file path')
    parser.add_argument('--gtf', required=True, help='Input GTF annotation file path')
    parser.add_argument('--output_csv', default='junction_metrics.csv', help='Output CSV file path for metrics')
    args = parser.parse_args()
    
    logging.info(f"Starting to process BAM file: {args.bam}")
    align_df = extract_bam_info(args.bam)
    
    logging.info(f"Starting to process GTF annotation: {args.gtf}")
    junction_df = process_junction(args.gtf)
    
    logging.info("Performing alignment analysis")
    df = process_and_match(align_df, mode='train', junction_df=junction_df)
    
    junction_compute(df, args.bam, args.output_csv)
    logging.info(f"Analysis completed, metrics saved to {args.output_csv}")

if __name__ == "__main__":
    main()
