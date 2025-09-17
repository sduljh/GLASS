import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Configure paths
real_exp_path = "/storage/lijiahao/data/sim/transcript_expression.tsv"
output_dir = "/storage/lijiahao/data/sim/"  # Modify to actual output directory

# 1. Read real expression data
real_exp = pd.read_csv(real_exp_path, sep='\t', skiprows=1, names=['transcript_id', 'count'])
real_exp = real_exp[real_exp['transcript_id'].str.startswith('ENST')]
real_dict = real_exp.set_index('transcript_id')['count'].to_dict()

# 2. Define file groups
file_groups = {
    # StringTie group
    "StringTie_default": {
        "tracking": "/storage/lijiahao/data/sim/default_output/sim_default_evaluation.tracking",
        "expression": "/storage/lijiahao/data/sim/default_output/stringtie_transcript_expression.tsv"
    },
    "StringTie_EASTR": {
        "tracking": "/storage/lijiahao/data/sim/EASTR_filtered_output/sim_EASTR_filtered_evaluation.tracking",
        "expression": "/storage/lijiahao/data/sim/EASTR_filtered_output/stringtie_transcript_expression.tsv"
    },
    "StringTie_filter": {
        "tracking": "/storage/lijiahao/data/sim/filter_output/sim_filter_evaluation.tracking",
        "expression": "/storage/lijiahao/data/sim/filter_output/stringtie_transcript_expression.tsv"
    },
    # Bambu group
    "Bambu_default": {
        "tracking": "/storage/lijiahao/data/sim/default_output_bambu/sim_default_evaluation.tracking",
        "expression": "/storage/lijiahao/data/sim/default_output_bambu/CPM_transcript.txt"
    },
    "Bambu_EASTR": {
        "tracking": "/storage/lijiahao/data/sim/EASTR_filtered_output_bambu/sim_EASTR_filtered_evaluation.tracking",
        "expression": "/storage/lijiahao/data/sim/EASTR_filtered_output_bambu/CPM_transcript.txt"
    },
    "Bambu_filter": {
        "tracking": "/storage/lijiahao/data/sim/filter_output_bambu/sim_filter_evaluation.tracking",
        "expression": "/storage/lijiahao/data/sim/filter_output_bambu/CPM_transcript.txt"
    },
    # IsoQuant group
    "IsoQuant_default": {
        "tracking": "/storage/lijiahao/data/sim/default_output_isoquant/sim_default_evaluation.tracking",
        "expression": "/storage/lijiahao/data/sim/default_output_isoquant/OUT/OUT.discovered_transcript_tpm.tsv"
    },
    "IsoQuant_EASTR": {
        "tracking": "/storage/lijiahao/data/sim/EASTR_filtered_output_isoquant/sim_EASTR_filtered_evaluation.tracking",
        "expression": "/storage/lijiahao/data/sim/EASTR_filtered_output_isoquant/OUT/OUT.discovered_transcript_tpm.tsv"
    },
    "IsoQuant_filter": {
        "tracking": "/storage/lijiahao/data/sim/filter_output_isoquant/sim_filter_evaluation.tracking",
        "expression": "/storage/lijiahao/data/sim/filter_output_isoquant/OUT/OUT.discovered_transcript_tpm.tsv"
    }
}

# 3. Process all file groups
results = []
common_transcripts = set(real_dict.keys())

for name, paths in file_groups.items():
    # Parse tracking file
    tracking_data = []
    with open(paths['tracking']) as f:
        for line in f:
            if line.startswith('TCONS'):  # Skip header line
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    enst_id = parts[2].split('|')[-1]
                    software_id = parts[4].split('|')[1]
                    tracking_data.append((enst_id, software_id))

    tracking_df = pd.DataFrame(tracking_data, columns=['enst_id', 'software_id'])
    tracking_df = tracking_df[tracking_df['enst_id'].str.startswith('ENST')]

    # Parse expression file
    if "StringTie" in name:
        exp_df = pd.read_csv(paths['expression'], sep='\t', header=None, names=['software_id', 'tpm', 'fpkm'])
        exp_df = exp_df[['software_id', 'tpm']]
    elif "Bambu" in name:
        exp_df = pd.read_csv(paths['expression'], sep='\t', skiprows=1, header=None,
                             names=['txname', 'geneid', 'cpm'])
        exp_df = exp_df.rename(columns={'txname': 'software_id', 'cpm': 'tpm'})
    elif "IsoQuant" in name:
        exp_df = pd.read_csv(paths['expression'], sep='\t', comment='#', header=None,
                             names=['feature_id', 'tpm'])
        exp_df = exp_df.rename(columns={'feature_id': 'software_id'})

    # Merge data
    merged = pd.merge(tracking_df, exp_df, on='software_id')

    # Update common transcript set
    valid_transcripts = set(merged['enst_id'])
    common_transcripts &= valid_transcripts

    # Temporarily store results
    results.append({
        'name': name,
        'data': merged.set_index('enst_id')['tpm'].to_dict()
    })

# 4. Calculate correlation coefficients
correlation_results = []
for group in results:
    name = group['name']
    exp_dict = group['data']

    # Prepare data vectors
    counts, expressions = [], []
    for tid in common_transcripts:
        if tid in real_dict and tid in exp_dict:
            counts.append(real_dict[tid])
            expressions.append(exp_dict[tid])

    # Filter zeros (needed for log transformation)
    counts = np.array(counts)
    expressions = np.array(expressions)
    valid_idx = (counts > 0) & (expressions > 0)

    # Calculate correlation coefficients
    pearson_log = pearsonr(np.log1p(counts[valid_idx]), np.log1p(expressions[valid_idx]))[0]
    spearman = spearmanr(counts, expressions)[0]

    correlation_results.append({
        'Method': name.split('_')[0],
        'Condition': name.split('_')[1],
        'Pearson(log)': round(pearson_log, 4),
        'Spearman': round(spearman, 4)
    })

# 5. Output results
result_df = pd.DataFrame(correlation_results)
print("\nCorrelation results:")
print(result_df.to_markdown(index=False))

# Save results to CSV
result_df.to_csv(os.path.join(output_dir, "correlation_results.csv"), index=False)
print(f"Results saved to {os.path.join(output_dir, 'correlation_results.csv')}")