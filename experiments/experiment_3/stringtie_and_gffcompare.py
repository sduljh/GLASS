import subprocess
import os


# Function to run stringtie and gffcompare commands
def run_stringtie_and_gffcompare(alignment_bam_path, alignment_gtf_path, reference_gtf_path, gffcompare_result_path):
    try:
        # Check if the paths are valid
        if not os.path.isfile(alignment_bam_path):
            raise FileNotFoundError(f"The file {alignment_bam_path} does not exist.")
        if not os.path.isfile(reference_gtf_path):
            raise FileNotFoundError(f"The file {reference_gtf_path} does not exist.")

        # Step 1: Run stringtie to generate the GTF file from the BAM file
        stringtie_cmd = f"stringtie {alignment_bam_path} -L -o {alignment_gtf_path}"
        print(f"Running command: {stringtie_cmd}")
        subprocess.run(stringtie_cmd, shell=True, check=True)

        # Step 2: Run gffcompare to compare the generated GTF with the reference GTF
        gffcompare_cmd = f"gffcompare -r {reference_gtf_path} -o {gffcompare_result_path} {alignment_gtf_path}"
        print(f"Running command: {gffcompare_cmd}")
        subprocess.run(gffcompare_cmd, shell=True, check=True)

        print("Stringtie and Gffcompare have been successfully executed.")

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running commands: {e}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():

    alignment_bam_path = "./alignment.bam"
    alignment_gtf_path = "./output.gtf"
    reference_gtf_path = "./annotation.gtf"
    gffcompare_result_path = "./gffcompare_result"

    run_stringtie_and_gffcompare(alignment_bam_path, alignment_gtf_path, reference_gtf_path, gffcompare_result_path)


if __name__ == "__main__":
    main()
