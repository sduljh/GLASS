#!/bin/bash
conda activate GLASS_env
python read_label_percent.py -b sample1.bam sample2.bam -g annotation.gtf -o results.csv