import pysam
import pandas as pd
from collections import defaultdict
import numpy as np

# Predefined constants for faster membership testing
REF_CONSUMING_OPS = {0, 2, 3, 7, 8}  # M, D, N, =, X
CLIP_OPS = {4, 5}  # S, H
N_OP = 3

# Scoring constants
match_score = 2  # Match score
insertion_penalty = -3  # Insertion penalty
deletion_penalty = -3  # Deletion penalty
mismatch_penalty = -2  # Mismatch penalty
skip_penalty = -1  # Skip penalty


def parse_cigar(cigar_tuples, reference_start):
    junctions = []
    left_clip = 0
    right_clip = 0
    total_ref_length = 0
    total_N_length = 0
    current_ref_pos = reference_start
    junction_tuples = []

    # Initialize score variable
    score = 0

    # Check left clip
    if cigar_tuples:
        op_type, op_length = cigar_tuples[0]
        if op_type in CLIP_OPS:
            left_clip = op_length

    # Check right clip
    if cigar_tuples:
        op_type, op_length = cigar_tuples[-1]
        if op_type in CLIP_OPS:
            right_clip = op_length

    # Pre-store flag for N existence
    has_N = False

    # Fast local variables for loop
    ref_ops = REF_CONSUMING_OPS
    n_op = N_OP

    # Optimized cigar processing
    for i, cigar_tuple in enumerate(cigar_tuples):
        op_type = cigar_tuple[0]
        op_length = cigar_tuple[1]

        if op_type in ref_ops:
            total_ref_length += op_length

            if op_type == n_op:
                has_N = True
                total_N_length += op_length
                junction_start = current_ref_pos
                junction_end = junction_start + op_length + 1
                junction_tuples.append((junction_start, junction_end))

            current_ref_pos += op_length

        # Apply scoring logic based on CIGAR operations
        if op_type == 0 or op_type == 7:  # 'M' or '=' (Match)
            score += match_score * op_length
        elif op_type == 1:  # 'I' (Insertion)
            score += insertion_penalty * op_length
        elif op_type == 2:  # 'D' (Deletion)
            score += deletion_penalty * op_length
        elif op_type == 3:  # 'N' (Skip)
            score += skip_penalty * op_length
        elif op_type == 8:  # 'X' (Mismatch)
            score += mismatch_penalty * op_length

    mapped_length = total_ref_length - total_N_length

    merged_junction_tuples = []
    if junction_tuples:
        current_start, current_end = junction_tuples[0]

        for next_start, next_end in junction_tuples[1:]:
            if next_start == current_end - 1:
                current_end = next_end
            else:
                merged_junction_tuples.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        merged_junction_tuples.append((current_start, current_end))

    junction_tuples = merged_junction_tuples


    return left_clip, right_clip, mapped_length, junction_tuples, has_N, score


def get_strand(record, has_N):
    if not has_N:
        return '.'

    if record.has_tag('ts'):
        ts_tag = record.get_tag('ts')
        if ts_tag == '+':
            return '-' if record.flag & 16 else '+'
        elif ts_tag == '-':
            return '+' if record.flag & 16 else '-'
        else:
            return ts_tag

    if record.has_tag('XS'):
        xs_tag = record.get_tag('XS')
        if xs_tag == '+':
            return '-' if record.flag & 16 else '+'
        elif xs_tag == '-':
            return '+' if record.flag & 16 else '-'





    return '.'


def get_supplementary(flag):
    return 'supplementary' if flag & 2048 else 'not_supplementary'


def get_flag_type(flag):
    if flag & 256:  # Secondary alignment
        return 'secondary'
    return 'primary'  # Primary alignment

def get_start_end(start_df, end_df, both_df, bam_df):
    merged_start_both = pd.merge(both_df, start_df, on=['start', 'chromosome', 'strand'], how='left')
    merged_start_end_df = pd.merge(merged_start_both, end_df, on=['end', 'chromosome', 'strand'], how='left')
    merged_start_end_df['start_proportion'] = merged_start_end_df['unique_intron_number'] / merged_start_end_df[
        'start_number']
    merged_start_end_df['end_proportion'] = merged_start_end_df['unique_intron_number'] / merged_start_end_df[
        'end_number']
    further_df = pd.merge(bam_df, merged_start_end_df, on=['start', 'end', 'chromosome', 'strand'], how='left')
    further_df['unique_intron_number_average'] = further_df.groupby('read_id2')['unique_intron_number'].transform('mean')

    further_df['start_number_average'] = further_df.groupby('read_id2')['start_number'].transform('mean')
    further_df['end_number_average'] = further_df.groupby('read_id2')['end_number'].transform('mean')
    further_df['start_proportion_average'] = further_df.groupby('read_id2')['start_proportion'].transform('mean')
    further_df['end_proportion_average'] = further_df.groupby('read_id2')['end_proportion'].transform('mean')

    # Convert grouped keys to integer indexes Create, Conditional Boolean (1=Compliant, 0=Noncompliant)
    read_id2_cat = further_df['read_id2'].astype('category')
    codes = read_id2_cat.cat.codes.values

    # Calculate the sum of each group using bincount  Map the results of the calculation back to the original DataFrame
    both_condition = further_df['unique_intron_number'].values == 1
    both_counts = np.bincount(codes, weights=both_condition.astype(int), minlength=len(read_id2_cat.cat.categories))
    further_df['unique_intron_number_1'] = both_counts[codes]
    further_df['unique_intron_number_1_average'] = further_df['unique_intron_number_1'] / further_df['read_id2_intron_number']

    # Calculate the sum of each group using bincount  Map the results of the calculation back to the original DataFrame
    start_condition = further_df['start_number'].values == 1
    start_counts = np.bincount(codes, weights=start_condition.astype(int), minlength=len(read_id2_cat.cat.categories))
    further_df['start_number_1'] = start_counts[codes]
    further_df['start_number_1_average'] = further_df['start_number_1'] / further_df['read_id2_intron_number']

    # Calculate the sum of each group using bincount  Map the results of the calculation back to the original DataFrame
    end_condition = further_df['end_number'].values == 1
    end_counts = np.bincount(codes, weights=end_condition.astype(int), minlength=len(read_id2_cat.cat.categories))
    further_df['end_number_1'] = end_counts[codes]
    further_df['end_number_1_average'] = further_df['end_number_1'] / further_df['read_id2_intron_number']

    return further_df


def extract_bam_info(input_bam):
    records = []  # Store all records in a list for batch writing
    row_count = 1  # Row number tracker
    start_count = defaultdict(int)
    end_count = defaultdict(int)
    both_count = defaultdict(int)
    with pysam.AlignmentFile(input_bam, 'rb') as bam:
        for record in bam:
            if record.is_unmapped or record.cigartuples is None:
                continue

            # Basic info extraction
            read_id = record.query_name
            chrom = record.reference_name
            # CIGAR parsing
            cigar_tuples = record.cigartuples
            left_clip, right_clip, mapped_length, junction_tuples, has_N, cigar_score = parse_cigar(cigar_tuples,
                                                                                              record.reference_start)


            # Determine strand based on presence of N operations
            strand = get_strand(record, has_N)
            start_1based = record.reference_start + 1
            end_1based = record.reference_end
            # Tag extraction
            s1_tag = record.get_tag('s1') if record.has_tag('s1') else 0
            read_length = record.query_length
            # Supplementary flag
            supplementary = get_supplementary(record.flag)
            # Flag type (primary or secondary)
            flag_type = get_flag_type(record.flag)
            # Apply filters: only primary alignments and strands '+' or '-'
            if flag_type != 'primary' or strand not in ['+', '-']:
                continue
            # print(f"read_length: {read_length}, mapped_length: {mapped_length}")

            mapped_length_average = mapped_length / read_length
            right_clip_average = left_clip / read_length
            left_clip_average = left_clip / read_length



            for start, end in junction_tuples:
                junction = f"{start}-{end}"
                # Split junction_info into separate rows
                junction_number = len(junction_tuples)  # Count the number of junctions for this record
                records.append([
                    row_count, read_id, chrom, strand, start, end, junction,
                    s1_tag, flag_type, supplementary, mapped_length_average,
                    left_clip_average, right_clip_average, cigar_score, junction_number
                ])
                # Count for "start" "end" both start and end" table
                start_count[(start, chrom, strand)] += 1
                end_count[(end, chrom, strand)] += 1
                both_count[(start, end, chrom, strand)] += 1
            row_count += 1  # Increment row count

    # Convert to DataFrame and save the tables
    start_df = pd.DataFrame([(start, chrom, strand, count) for (start, chrom, strand), count in start_count.items()],
                            columns=['start', 'chromosome', 'strand', 'start_number'])

    end_df = pd.DataFrame([(end, chrom, strand, count) for (end, chrom, strand), count in end_count.items()],
                          columns=['end', 'chromosome', 'strand', 'end_number'])

    both_df = pd.DataFrame([(start, end, chrom, strand, count) for (start, end, chrom, strand), count in both_count.items()],
                           columns=['start', 'end', 'chromosome', 'strand', 'unique_intron_number'])
    # Use pandas to write all records at once
    bam_df = pd.DataFrame(records, columns=[
        'read_id2', 'read_id', 'chromosome', 'strand', 'start', 'end', 'junction_info',
        'chaining_score', 'flag', 'supplementary?', 'mapped-length_percent',
        'SOFT_CLIP_Left_percent', 'SOFT_CLIP_right_percent', 'CIGAR_score', 'read_id2_intron_number'
    ])
    further_df = get_start_end(start_df, end_df, both_df, bam_df)
    return further_df



