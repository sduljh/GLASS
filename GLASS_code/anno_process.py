import sys
import pandas as pd
from collections import defaultdict

def load_ref(file_path):
    """
    Load reference GTF file and extract junction information.
    Returns a DataFrame containing junctions in the format: strand, chr, start, end
    """
    junctions_list = []
    
    vec_exon = []
    current_chr = ""
    current_strand = ""
    current_tranid = ""
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):  # Skip comment lines
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != "exon":
                continue
                
            chr_name = fields[0]
            label = fields[2]
            exon_l = int(fields[3])
            exon_r = int(fields[4])
            strand = fields[6]
            
            # Extract transcript_id from attributes
            attributes = fields[8].split(';')
            tranid = None
            for attr in attributes:
                attr = attr.strip()
                if attr.startswith('transcript_id'):
                    _, tranid_val = attr.split(' ', 1)
                    tranid = tranid_val.strip('" ')
                    break
            
            if not tranid:
                continue
                
            if not current_tranid:
                # Initialize for first exon
                current_tranid = tranid
                current_chr = chr_name
                current_strand = strand
                vec_exon.extend([exon_l, exon_r])
                
            elif tranid == current_tranid:
                # Same transcript, add exon
                vec_exon.extend([exon_l, exon_r])
                
            else:
                # Process previous transcript
                if len(vec_exon) > 2:
                    vec_sorted = sorted(vec_exon)
                    vec_mid = vec_sorted[1:-1]  # Remove first and last elements
                    
                    for i in range(0, len(vec_mid), 2):
                        start = vec_mid[i]
                        end = vec_mid[i+1]
                        junctions_list.append({
                            'strand': current_strand,
                            'chr': current_chr,
                            'start': start,
                            'end': end
                        })
                
                # Reset for new transcript
                current_tranid = tranid
                current_chr = chr_name
                current_strand = strand
                vec_exon = [exon_l, exon_r]
        
        # Process last transcript
        if len(vec_exon) > 2:
            vec_sorted = sorted(vec_exon)
            vec_mid = vec_sorted[1:-1]
            
            for i in range(0, len(vec_mid), 2):
                start = vec_mid[i]
                end = vec_mid[i+1]
                junctions_list.append({
                    'strand': current_strand,
                    'chr': current_chr,
                    'start': start,
                    'end': end
                })
    
    # Create DataFrame from collected junctions
    return pd.DataFrame(junctions_list)

def process_junction(gtf_file):
    junction_df = load_ref(gtf_file)
    junction_df = junction_df.drop_duplicates()
    junction_df['junction_info'] = junction_df['start'].astype(str) + '-' + junction_df['end'].astype(str)
    junction_df.columns = ['strand', 'chromosome', 'read_start_pos', 'read_end_pos', 'junction_info']
    return junction_df




