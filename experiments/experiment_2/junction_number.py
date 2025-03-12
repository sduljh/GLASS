import pandas as pd
import subprocess
import logging
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
grandparent_dir = os.path.dirname(parent_dir)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_bam(bam_path):
    # Process BAM file
    directory_path = os.path.dirname(bam_path)
    align_path = os.path.join(directory_path,
                              f"{os.path.splitext(os.path.basename(bam_path))[0]}.info")

    command = f"{grandparent_dir}/C_code/src/get_alignment_info {bam_path} {align_path}"
    subprocess.run(command, shell=True, check=True)
    align_df = pd.read_csv(align_path, sep=r'\s+', engine='python', header=0)
    logging.info(f"Successfully generated {align_path}")
    return align_df


def process_and_match(align_df, junction_df):
    # Step 1: Remove duplicate rows
    df = align_df
    df = df.drop_duplicates()

    # Step 2: Filter data
    df = df[(df['strand'].isin(['+', '-'])) & (df['flag'].str.contains('primary', na=False))]

    # Step 3: Add 'read_id2' column and verify
    df['read_id2'] = range(1, len(df) + 1)
    if 'read_id2' in df.columns and all(df['read_id2'] == range(1, len(df) + 1)):
        logging.info("Column 'read_id2' added successfully with correct values.")

    # Process junction information
    processed_rows = []
    for index, row in df.iterrows():
        junction_info = row['junction_info']
        elements = junction_info.strip('()').split(',')

        for element in elements:
            new_row = row.copy()
            new_row['junction_info'] = element.strip()
            processed_rows.append(new_row)

    processed_df = pd.DataFrame(processed_rows)

    # Match and label junctions
    # Create lookup set for fast searching
    junction_set = set(
        zip(junction_df['chromosome'], junction_df['strand'], junction_df['junction_info'])
    )

    # Add label column
    processed_df['label'] = processed_df.apply(
        lambda row: 1 if (row['chromosome'], row['strand'], row['junction_info']) in junction_set else 0, axis=1
    )

    # Check label consistency in groups
    grouped = processed_df.groupby(['chromosome', 'strand', 'junction_info'])

    # Find inconsistent groups
    for name, group in grouped:
        if group['label'].nunique() > 1:
            logging.info(
                f"Found inconsistent group: 'chromosome'={name[0]}, 'strand'={name[1]}, 'junction_info'={name[2]}")
            logging.info("Corresponding rows:")
            logging.info(group)
            break
    else:
        logging.info("All groups have consistent labels.")

    return processed_df


def process_junction(junction_gtf_file):
    # Process GTF file
    directory_path = os.path.dirname(junction_gtf_file)
    junction_input_file = os.path.join(directory_path,
                                       f"{os.path.splitext(os.path.basename(junction_gtf_file))[0]}.info")

    # Generate junction information
    command = f"../CorrectAlignment/get_junction_fromGTF {junction_gtf_file} > {junction_input_file}"
    subprocess.run(command, shell=True, check=True)
    logging.info(f"Successfully generated {junction_input_file}")

    # Load and process junction data
    junction_df = pd.read_csv(junction_input_file, sep=r'\s+', header=None)

    # 1. Remove duplicates
    junction_df = junction_df.drop_duplicates()

    # 2. Create combined junction info
    junction_df[4] = junction_df[2].astype(str) + '-' + junction_df[3].astype(str)

    # 3. Set column names
    junction_df.columns = ['strand', 'chromosome', 'read_start_pos', 'read_end_pos', 'junction_info']

    return junction_df


def align_compute(df):
    # Calculate alignment statistics
    total_rows = len(df)
    label_1_rows = len(df[df['label'] == 1])

    print(f"Total junctions: {total_rows}")
    print(f"Correct junctions: {label_1_rows}")
    print(f"Incorrect junctions: {total_rows - label_1_rows}")
    print(f"Junction accuracy: {label_1_rows / total_rows:.2%}")
    print(f"Junction error rate: {(total_rows - label_1_rows) / total_rows:.2%}")


def main():
    bam_path = f"./alignment.bam"
    junction_gtf_file = f"./annotation.gtf"
    junction_df = process_junction(junction_gtf_file)
    align_df = process_bam(bam_path)
    processed_df = process_and_match(align_df, junction_df)

    align_compute(processed_df)


if __name__ == "__main__":
    main()