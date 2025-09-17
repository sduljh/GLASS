import os
import argparse
import pandas as pd
from bam_process import extract_bam_info
from anno_process import process_junction
from align_compute import process_bam, process_and_match, read_label


def main():

    DEFAULT_GTF = "/storage/lijiahao/Index/Human/hg38.ncbiRefSeq.gtf"


    parser = argparse.ArgumentParser(description='Analyze BAM files and compute label distribution.')
    parser.add_argument('-b', '--bam', nargs='+', required=True,
                        help='Input BAM file paths (one or more)')
    parser.add_argument('-g', '--gtf', required=True, help='Input GTF file path')
    parser.add_argument('-o', '--output', help='Output CSV file for summary results')

    args = parser.parse_args()


    results = []

    print("Processing GTF file...")
    junction_df = process_junction(args.gtf)


    for i, bam_path in enumerate(args.bam, 1):
        bam_name = os.path.basename(bam_path)

        print(f"\n{'=' * 50}")
        print(f"Processing BAM file #{i}: {bam_name}")

        if not os.path.exists(bam_path):
            print(f"Error: File not found - {bam_path}")
            continue

        try:
            print("- Extracting alignment information...")
            align_df = extract_bam_info(bam_path)

            print("- Processing and matching junctions...")
            df = process_and_match(align_df, mode='train', junction_df=junction_df)

            print("- Assigning labels to reads...")
            read_df = read_label(df)

            if 'label' in read_df.columns:

                label_counts = read_df['label'].value_counts()
                total_count = len(read_df)
                ratio_0 = round(label_counts.get(0, 0) / total_count * 100, 2)
                ratio_1 = round(label_counts.get(1, 0) / total_count * 100, 2)

                result = {
                    'file': bam_name,
                    'total_reads': total_count,
                    'label_0_count': label_counts.get(0, 0),
                    'label_1_count': label_counts.get(1, 0),
                    'label_0_percent': ratio_0,
                    'label_1_percent': ratio_1,
                    'class_ratio': round(ratio_1 / max(ratio_0, 0.01), 2)
                }
                results.append(result)

                print("\nLabel Distribution Results:")
                print(f"{'Label':<10} | {'Count':<10} | {'Percentage':<10}")
                print(f"{'-' * 33}")
                print(f"{'0':<10} | {result['label_0_count']:<10} | {ratio_0}%")
                print(f"{'1':<10} | {result['label_1_count']:<10} | {ratio_1}%")
                print(f"{'-' * 33}")
                print(f"{'TOTAL':<10} | {total_count:<10} | 100%")
                print(f"\nAdditional Stats:")
                print(f"Positive examples (1): {ratio_1}%")
                print(f"Negative examples (0): {ratio_0}%")
                print(f"Class Ratio (1:0): {result['class_ratio']:.2f}:1")
            else:
                print("Error: 'label' column not found in DataFrame")

            print(f"Completed processing BAM file #{i}")
            print(f"{'=' * 50}\n")

        except Exception as e:
            print(f"Error processing {bam_name}: {str(e)}")
            continue

    if args.output and results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(args.output, index=False)
        print(f"\nSummary results saved to {args.output}")
        print(summary_df)


if __name__ == "__main__":
    main()