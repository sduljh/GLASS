

# GLASS

This tool is designed for training and testing a machine learning model with BAM files using a Bipartite Graph Convolutional Network (BipartiteGCN). It allows for the processing of BAM files for testing and optional training with optional validation.

## Prerequisites

Before running this script, you need to compile the C code that is part of this project. Follow the step to set up your environment:



### Install Python Dependencies
Ensure you have all necessary Python libraries installed. You may need to install dependencies such as `torch`, `psutil`, and others:

```bash
pip install -r requirements.txt
```

## Usage

You can run this script from the command line using Python. The script processes BAM files and applies a trained model for testing, logging memory usage, and calculating runtime.

### Command-line Arguments

#### Required Arguments

- **`test_bam_files`** (required):  
  A list of paths to BAM files you want to test. At least one BAM file must be provided.

  Example:  
  ```bash
  python GLASS.py test_bam_files file1.bam file2.bam
  ```

#### Optional Arguments

- **
- **`--train_bam_files`** (optional):  
  List of paths to BAM files used for training (default: `None`).
- **`--train_junction_gtf_file`** (optional):  
  Path to the training junction GTF file (default: `None`).
- **`--val_bam_file`** (optional):  
  Path to the validation BAM file (default: `None`).
- **`--val_junction_gtf_file`** (optional):  
  Path to the validation junction GTF file (default: `None`).
- **`--train_device`** (optional):  
  Device for training (either `cpu` or `cuda`). Default is `cpu`.
- **`--test_device`** (optional):  
  Device for testing (either `cpu` or `cuda`). Default is `cpu`.

### Example Command

```bash
python GLASS.py test_bam_files test1.bam test2.bam --train_bam_files train1.bam train2.bam --train_junction_gtf_file train.gtf --val_bam_file val.bam --val_junction_gtf_file val.gtf --train_device cuda --test_device cuda
```

This will train and test the model using the specified files and devices.

### Output

- **Logs**: The script logs detailed information such as:
  - Memory usage at the start and end of the program.
  - Total runtime of the script.
  - Any errors or missing files.

- **Filtered BAM Files**: After processing the test BAM files, new filtered BAM files will be created in the same directory as the original BAM files. The new file names will have the `_filter` suffix.

## Functionality

1. **Training**: The model is trained using the training BAM files and the corresponding GTF file. A validation BAM file and GTF file can also be provided for model validation.

2. **Testing**: The provided test BAM files are processed using the trained models, and the results are saved after filtering.

3. **Memory and Runtime Tracking**: The script logs memory usage before and after execution, and also tracks the total runtime.

## Example Output

```bash
2025-02-03 10:00:00 - INFO - Initial memory usage: 200.00 MB
2025-02-03 10:00:01 - INFO - Training on device: cuda
2025-02-03 10:00:01 - INFO - Testing on device: cuda
2025-02-03 10:02:30 - INFO - Final memory usage: 250.00 MB
2025-02-03 10:02:30 - INFO - Total runtime: 150.00 seconds
```

## Error Handling

- If a BAM file is not found, the script will log an error and skip that file.
- If any other error occurs during the processing, an error message will be logged.

## Some processing operations that may be required for GTF files

```bash
grep "exon" original.gtf > new.gtf
```

If your GTF file contains more information besides the exon data, please first use the above command before inputting the GTF file into the script.

```bash
grep -P '\btranscript_id\s+"[^"]+"' original.gtf > fixed.gtf
```

In some cases, you need to use this command to process the GTF file before running gffcompare.



