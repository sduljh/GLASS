import pandas as pd
import re
import subprocess
import logging
import concurrent.futures
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def score_cigar(cigar):
    # logging.info(f"Scoring CIGAR string: {cigar}")
    # Regular expression pattern for valid CIGAR strings
    cigar_pattern = re.compile(r'^(\d+[MIDNSHP=X])+$')

    if not cigar_pattern.match(cigar):
        logging.warning(f"Invalid CIGAR string: {cigar}")
        return None  # Return None for invalid CIGAR strings

    # Scoring scheme configuration
    match_score = 2  # Increased positive score for matches
    insertion_penalty = -3  # Increased penalty for insertions
    deletion_penalty = -3  # Increased penalty for deletions
    mismatch_penalty = -2  # Penalty for mismatches
    skip_penalty = -1  # Small penalty for skip operations

    score = 0  # Initialize score

    # Parse CIGAR string
    matches = re.findall(r'(\d+)([MIDNSHP=X])', cigar)

    for length, operation in matches:
        length = int(length)
        if operation == 'M' or operation == '=':
            score += match_score * length
        elif operation == 'I':
            score += insertion_penalty * length
        elif operation == 'D':
            score += deletion_penalty * length
        elif operation == 'N':
            score += skip_penalty * length
        elif operation == 'X':
            score += mismatch_penalty * length
        # S and H are typically not scored, but can be included if needed

    # logging.info(f"Calculated CIGAR score: {score}")
    return score


def process_junction(junction_gtf_file, junction_input_file):
    try:
        logging.info(f"Generating junction input file from GTF: {junction_gtf_file}")
        command = f"{script_dir}/C_code/get_junction_fromGTF {junction_gtf_file} > {junction_input_file}"
        subprocess.run(command, shell=True, check=True)
        logging.info(f"Successfully generated {junction_input_file}")

        # Read the file assuming tab-separated values
        junction_df = pd.read_csv(junction_input_file, sep=r'\s+', header=None)

        # 1. Remove duplicate rows
        junction_df = junction_df.drop_duplicates()

        # 2. Create 5th column by concatenating columns 3 and 4
        junction_df[4] = junction_df[2].astype(str) + '-' + junction_df[3].astype(str)

        # 3. Add column names
        junction_df.columns = ['strand', 'chromosome', 'read_start_pos', 'read_end_pos', 'junction_info']

        logging.info("Junction dataframe processed successfully")
        return junction_df
    except Exception as e:
        logging.error(f"Error in process_junction: {e}")
        raise


def process_bam(bam_path, align_path):
    try:
        logging.info(f"Generating alignment info from BAM file: {bam_path}")
        command = f"{script_dir}/C_code/src/get_alignment_info {bam_path} {align_path}"
        subprocess.run(command, shell=True, check=True)
        align_df = pd.read_csv(align_path, sep=r'\s+', engine='python', header=0)

        logging.info(f"Successfully generated {align_path}")
        return align_df
    except Exception as e:
        logging.error(f"Error in process_bam: {e}")
        raise



def process_and_match(align_df, mode='train', junction_df=None):
    try:
        logging.info(f"Processing and matching data in {mode} mode...")
        # Step 1: Remove duplicates
        # df = align_df.drop_duplicates()
        processed_df = align_df
        logging.info(f"Processed {len(processed_df)} rows for junction info")

        if mode == 'train':
            # Step 5: Matching and labeling in parallel
            junction_set = set(
                zip(junction_df['chromosome'], junction_df['strand'], junction_df['junction_info'])
            )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for index, row in processed_df.iterrows():
                    futures.append(executor.submit(apply_label, row, junction_set))

                # Collect the results and apply labels
                for future in concurrent.futures.as_completed(futures):
                    row, label = future.result()
                    processed_df.loc[processed_df.index == row.name, 'label'] = label

            # Check label consistency in groups
            grouped = processed_df.groupby(['chromosome', 'strand', 'junction_info'])

            # Find inconsistent groups
            for name, group in grouped:
                if group['label'].nunique() > 1:
                    logging.warning(
                        f"Inconsistent group found: 'chromosome'={name[0]}, 'strand'={name[1]}, 'junction_info'={name[2]}")
                    logging.warning("Affected rows:")
                    logging.warning(group)
                    break
            else:
                logging.info("All groups have consistent 'label' values.")

        return processed_df

    except Exception as e:
        logging.error(f"Error in process_and_match: {e}")
        raise

def process_junction_row(row):
    """
    Helper function to process a single row and expand its junction info.
    """
    processed_rows = []
    junction_info = row['junction_info']
    elements = junction_info.strip('()').split(',')

    for element in elements:
        new_row = row.copy()
        new_row['junction_info'] = element.strip()
        processed_rows.append(new_row)
    
    return processed_rows

def apply_label(row, junction_set):
    """
    Helper function to apply label based on junction information.
    """
    label = 1 if (row['chromosome'], row['strand'], row['junction_info']) in junction_set else 0
    return row, label


def read_label(df, mode='train'):
    try:

        logging.info("Extracting and processing labels...")
        if mode == 'train':
            # Extract required columns
            read_df = df[['read_id', 'label']]

            # Remove duplicates
            read_df = read_df.drop_duplicates()

            # Set label=1 for read_ids appearing ≥2 times
            read_id_counts = read_df['read_id'].value_counts()
            count_read_id_twice = read_id_counts[read_id_counts >= 2].index.tolist()
            read_df.loc[read_df['read_id'].isin(count_read_id_twice), 'label'] = 1

            # Final deduplication
            read_df = read_df.drop_duplicates()

            # Verify uniqueness
            assert read_df['read_id'].is_unique, "Error: Duplicate read_id values found."

            logging.info(f"Read labels processed with {len(read_df)} unique read_ids")
        if mode == 'test':
            # Extract required columns
            read_df = df[['read_id']]

            # Remove duplicates
            read_df = read_df.drop_duplicates()

            # Verify uniqueness
            assert read_df['read_id'].is_unique, "Error: Duplicate read_id values found."

            logging.info(f"Read labels processed with {len(read_df)} unique read_ids")
        return read_df
    except Exception as e:
        logging.error(f"Error in read_label: {e}")
        raise


def process_feature(df, read_df, mode='train'):
    try:
        logging.info(f"Processing features in {mode} mode...")
        # Filter rows with read-length=0
        df = df[df['read-length'] != 0]

        # Calculate CIGAR scores
        df['CIGAR_score'] = df['CIGAR'].apply(score_cigar)
        df = df.dropna(subset=['CIGAR_score'])

        # Calculate percentage features
        df['mapped-length_percent'] = df['mapped-length'] / df['read-length']
        df['SOFT_CLIP_Left_percent'] = df['SOFT_CLIP_Left'] / df['read-length']
        df['SOFT_CLIP_right_percent'] = df['SOFT_CLIP_right'] / df['read-length']

        # Calculate intron-related features
        df['read_id2_intron_number'] = df.groupby('read_id2')['read_id2'].transform('count')
        df['unique_intron_number'] = df.groupby(['chromosome', 'strand', 'junction_info'])['chromosome'].transform(
            'count')
        df['unique_intron_number_average'] = df.groupby('read_id2')['unique_intron_number'].transform('mean')
        df['unique_intron_number_1'] = df.groupby('read_id2')['unique_intron_number'].transform(
            lambda x: (x == 1).sum())
        df['unique_intron_number_1_average'] = df['unique_intron_number_1'] / df['read_id2_intron_number']

        # Process junction_info
        df = df[df['junction_info'].str.contains(r'^\d+-\d+$', na=False)]
        df[['start', 'end']] = df['junction_info'].str.split('-', expand=True).astype(int)

        # Calculate alignment features
        junction_counts = df.groupby(['start', 'end', 'chromosome', 'strand']).size()
        start_counts = df.groupby(['start', 'chromosome', 'strand']).size()
        df['startalignnumber'] = df.apply(
            lambda row: junction_counts[(row['start'], row['end'], row['chromosome'], row['strand'])] / start_counts[
                (row['start'], row['chromosome'], row['strand'])], axis=1)

        end_counts = df.groupby(['end', 'chromosome', 'strand']).size()
        df['endalignnumber'] = df.apply(
            lambda row: junction_counts[(row['start'], row['end'], row['chromosome'], row['strand'])] / end_counts[
                (row['end'], row['chromosome'], row['strand'])], axis=1)

        read_id2_counts = df['read_id2'].value_counts()
        df['start_align_average'] = df.apply(lambda row: row['startalignnumber'] / read_id2_counts[row['read_id2']],
                                             axis=1)
        df['end_align_average'] = df.apply(lambda row: row['endalignnumber'] / read_id2_counts[row['read_id2']], axis=1)

        # Select final features
        selected_columns = [
            'read_id', 'read_id2', 'read_id2_intron_number', 'mapped-length_percent',
            'SOFT_CLIP_Left_percent', 'SOFT_CLIP_right_percent', 'CIGAR_score',
            'unique_intron_number_average', 'unique_intron_number_1_average',
            'chaining_score', 'start_align_average', 'end_align_average',
            'chromosome', 'strand', 'junction_info'
        ]
        compute_df = df[selected_columns]

        # Filter and process final features
        max_values = compute_df.groupby(['read_id'])['unique_intron_number_average'].transform('max')
        compute_df = compute_df[compute_df['unique_intron_number_average'] == max_values]
        max_id2s = compute_df.groupby(['read_id'])['read_id2'].transform('max')
        compute_df = compute_df[compute_df['read_id2'] == max_id2s]

        feature_columns = [
            'read_id', 'read_id2', 'read_id2_intron_number', 'mapped-length_percent',
            'SOFT_CLIP_Left_percent', 'SOFT_CLIP_right_percent', 'CIGAR_score',
            'unique_intron_number_average', 'unique_intron_number_1_average',
            'chaining_score', 'start_align_average', 'end_align_average'
        ]
        feature_df = compute_df[feature_columns]

        # Handle NaN values
        feature_df.loc[:, 'unique_intron_number_average'] = feature_df['unique_intron_number_average'].fillna(0)
        feature_df = feature_df.drop_duplicates(subset=['read_id', 'unique_intron_number_average'])


        feature_df = feature_df.merge(read_df, on='read_id', how='left')

        logging.info(f"Features processed with {len(feature_df)} rows")
        return compute_df, feature_df
    except Exception as e:
        logging.error(f"Error in process_feature: {e}")
        raise


def process_edge_info(df):
    try:
        logging.info("Processing edge info...")
        # Create composite key for edge information
        df['chromosome_strand_junction'] = (
                df['chromosome'].astype(str) + '-' + df['strand'].astype(str) + '-' + df['junction_info'])
        df.sort_values(by=['read_id2'], inplace=True)

        # Group and generate edge information
        intron_df = df.groupby('chromosome_strand_junction')['read_id2'].apply(list).reset_index()
        intron_df.columns = ['chromosome_strand_junction', 'matched_read_id2s']
        intron_df['matched_read_id2s'] = intron_df['matched_read_id2s'].apply(lambda x: tuple(sorted(x)))
        logging.info("Edge info processed successfully")
        return intron_df
    except Exception as e:
        logging.error(f"Error processing edge info: {e}")
        raise


def train_preprocess(bam_path, align_path, junction_df):
    try:
        logging.info("Starting training preprocessing...")
        align_df = process_bam(bam_path, align_path)
        df = process_and_match(align_df, mode='train', junction_df=junction_df)
        read_df = read_label(df, mode='train')
        compute_df, feature_df = process_feature(df, read_df, mode='train')
        intron_df = process_edge_info(compute_df)
        logging.info("Training preprocessing completed successfully")
        return feature_df, intron_df
    except Exception as e:
        logging.error(f"Error in train_preprocess: {e}")
        raise


def test_preprocess(bam_path, align_path):
    try:
        logging.info("Starting test preprocessing...")
        align_df = process_bam(bam_path, align_path)
        df = process_and_match(align_df, mode='test')
        read_df = read_label(df, mode='test')
        compute_df, feature_df = process_feature(df, read_df, mode='test')
        intron_df = process_edge_info(compute_df)
        logging.info("Test preprocessing completed successfully")
        return feature_df, intron_df
    except Exception as e:
        logging.error(f"Error in test_preprocess: {e}")
        raise