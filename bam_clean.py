import pandas as pd
import pysam

def label_merge(feature_info_df, predictions_df):

    # Extract the read_id and read_id2 columns
    feature_info_filtered = feature_info_df[['read_id', 'read_id2']]

    # Ensure that the two DataFrames have the same number of rows
    if len(feature_info_filtered) != len(predictions_df):
        raise ValueError("The number of rows in the two files are different, cannot merge by row.")

    # Merge the two DataFrames by row
    merged_df = pd.merge(predictions_df, feature_info_filtered, on='read_id2', how='inner')


    return merged_df





def filter_bam_by_label(bam_file, output_bam_file, merged_df):

    # Remove read_id from the BAM file if its predicted_label is 0.0 according to the info_file.

    df = merged_df

    # Filter out read_ids with predicted_label equal to 0.0
    read_ids_to_remove = set(df.loc[df['Predicted_Label'] == 0.0, 'read_id'])

    # Open the BAM file and filter it
    with pysam.AlignmentFile(bam_file, "rb") as bam_in, pysam.AlignmentFile(output_bam_file, "wb",
                                                                            header=bam_in.header) as bam_out:
        for read in bam_in:
            # If the query_name of the read is not in the removal list, write to the new file
            if read.query_name not in read_ids_to_remove:
                bam_out.write(read)


def clean(feature_info_df, predictions_df, alignment_bam_path, alignment_bam_filter_path):

    merged_df = label_merge(feature_info_df, predictions_df)
    filter_bam_by_label(alignment_bam_path, alignment_bam_filter_path, merged_df)





