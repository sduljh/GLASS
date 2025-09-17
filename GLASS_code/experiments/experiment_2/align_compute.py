import logging
from bam_process import extract_bam_info


#  Configuration log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_bam(bam_path):
    align_df = extract_bam_info(bam_path)
    logging.info(f"Successfully producing align_info")
    return align_df


def process_and_match(align_df, mode='train', junction_df=None):
    processed_df = align_df

    if mode == 'train':
        # Step 4: Matching and labeling
        junction_set = set(
            zip(junction_df['chromosome'], junction_df['strand'], junction_df['junction_info'])
        )

        # Add label column
        processed_df['label'] = processed_df.apply(
            lambda row: 1 if (row['chromosome'], row['strand'], row['junction_info']) in junction_set else 0, axis=1
        )
        logging.info("Lable Columns Successfully Generated")
    if mode == 'test':
        processed_df['label'] = 1
    return processed_df


def read_label(df):
    read_df = df[['read_id', 'label']] # Step 1: Extract the required columns
    read_df = read_df.drop_duplicates() # Step 2: De duplication operation
    # Step 3: Find the elements that appear twice or more in the read_id and set the label value of the corresponding row to 1
    read_id_counts = read_df['read_id'].value_counts()
    count_read_id_twice = read_id_counts[read_id_counts >= 2].index.tolist()
    read_df.loc[read_df['read_id'].isin(count_read_id_twice), 'label'] = 1
    read_df = read_df.drop_duplicates() # Step 4: Second deduplication operation
    return read_df


def process_feature(df, read_df=None):
    try:
        selected_columns = [
            'read_id',
            'read_id2',
            'read_id2_intron_number',
            'mapped-length_percent',
            'SOFT_CLIP_Left_percent',
            'SOFT_CLIP_right_percent',
            'CIGAR_score',
            'unique_intron_number_average',
            'unique_intron_number_1_average',
            'chaining_score',
            'start_number_1_average',
            'end_number_1_average',
            'start_number_average',
            'end_number_average',
            'start_proportion_average',
            'end_proportion_average',
            'chromosome',
            'strand',
            'junction_info'
        ]
        compute_df = df[selected_columns]

        # Filter and process final features
        max_values = compute_df.groupby(['read_id'])['unique_intron_number_average'].transform('max')
        compute_df = compute_df[compute_df['unique_intron_number_average'] == max_values]
        max_id2s = compute_df.groupby(['read_id'])['read_id2'].transform('max')
        compute_df = compute_df[compute_df['read_id2'] == max_id2s]

        selected_columns2 = [
            'read_id',
            'read_id2',
            'read_id2_intron_number',
            'mapped-length_percent',
            'SOFT_CLIP_Left_percent',
            'SOFT_CLIP_right_percent',
            'CIGAR_score',
            'unique_intron_number_average',
            'unique_intron_number_1_average',
            'chaining_score',
            'start_number_1_average',
            'end_number_1_average',
            'start_number_average',
            'end_number_average',
            'start_proportion_average',
            'end_proportion_average',
        ]
        feature_df = compute_df[selected_columns2]
        feature_df = feature_df.drop_duplicates(subset=['read_id', 'unique_intron_number_average'])

        feature_df = feature_df.merge(read_df, on='read_id', how='left')
        logging.info(f"The row number of the feature_df: {feature_df.shape[0]}")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")

    return compute_df, feature_df

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



def train_preprocess(bam_path, junction_df):
    try:
        logging.info("Starting training preprocessing...")
        align_df = process_bam(bam_path)
        df = process_and_match(align_df, mode='train', junction_df=junction_df)
        read_df = read_label(df)
        compute_df, feature_df = process_feature(df, read_df=read_df)
        intron_df = process_edge_info(compute_df)
        logging.info("Training preprocessing completed successfully")
        return feature_df, intron_df
    except Exception as e:
        logging.error(f"Error in train_preprocess: {e}")
        raise


def test_preprocess(bam_path):
    try:
        logging.info("Starting test preprocessing...")
        align_df = process_bam(bam_path)
        df = process_and_match(align_df, mode='test')
        read_df = read_label(df)
        compute_df, feature_df = process_feature(df, read_df=read_df)
        intron_df = process_edge_info(compute_df)
        logging.info("Test preprocessing completed successfully")
        return feature_df, intron_df
    except Exception as e:
        logging.error(f"Error in test_preprocess: {e}")
        raise

