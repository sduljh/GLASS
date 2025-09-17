#!/bin/bash
conda activate GLASS_env

OUTPUT_CSV="junction_metrics.csv"
bam_file="alignment.bam"
gtf_file="annotation.gtf"

python Figure1_result_compute.py \
    --bam "$bam_file" \
    --gtf "$gtf_file" \
    --output_csv "$OUTPUT_CSV"

date
