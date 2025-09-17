import pandas as pd
import os
import argparse
from bam_process import extract_bam_info


def calculate_read_frequency(df):
    if 'read_id2' not in df.columns:
        print("Column 'read_id2' not found!")
        return None, None, None
    # Count occurrences per read_id2
    read_counts = df['read_id2'].value_counts().value_counts()
    # Get total unique reads
    total_unique_reads = len(df['read_id2'].unique())
    # Calculate counts for each category
    single_count = read_counts.get(1, 0)
    multi_count = total_unique_reads - single_count
    # Calculate frequencies
    single_freq = single_count / total_unique_reads
    multi_freq = multi_count / total_unique_reads
    multi_freq_pro = multi_count / total_unique_reads
    return 0, single_freq, multi_freq, multi_freq_pro


def process_bam_file(bam_path):
    try:
        align_df = extract_bam_info(bam_path)
        _, single_freq, multi_freq, multi_ratio = calculate_read_frequency(align_df)
        return {
            'file': bam_path,
            'single_splicing_ratio': single_freq,
            'multi_splicing_ratio': multi_freq,
            'multi_ratio_in_spliced': multi_ratio
        }
    except Exception as e:
        print(f"Error processing {bam_path}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Analyze splicing ratios in BAM files.')
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='Input BAM file paths (space separated)')
    parser.add_argument('-o', '--output', help='Output CSV file path (optional)')

    args = parser.parse_args()

    results = []


    for bam_path in args.input:
        if not os.path.exists(bam_path):
            print(f"File not found: {bam_path}")
            continue

        stats = process_bam_file(bam_path)
        if stats:
            results.append(stats)
            print(f"\nStatistics for file {bam_path}:")
            print(f"  Single-splicing ratio: {stats['single_splicing_ratio']:.4f}")
            print(f"  Multi-splicing ratio: {stats['multi_splicing_ratio']:.4f}")
            print(f"  Proportion of multi-splicing in spliced reads: {stats['multi_ratio_in_spliced']:.4f}")


    if args.output and results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()