import os
import sys
import numpy as np

from pathlib import Path
from collections import OrderedDict

# from utils.sequence import *

logger_obj = None


def set_logger(logger):
    global logger_obj
    logger_obj = logger


def pssm_check(pssm_file, config):
    """
    Checks the existence of a PSSM file and provides error details if missing.

    Args:
        pssm_file (str): Path to the PSSM file to be checked.
        config (dict): Configuration dictionary containing PSIBLAST settings.

    Returns:
        bool: True if the PSSM file exists, False otherwise.

    Raises:
        SystemExit: If the PSSM file is missing, with an error message detailing the configuration.
    """
    if os.path.exists(pssm_file):
        return True

    # Prepare error message if PSSM file is missing
    indent = " " * 10
    error_message = [
        f"PSSM file '{pssm_file}' is missing.\n",
        f"{indent}Please check the sequence and PSIBLAST configurations.\n",
        f"{indent}Current configuration:\n",
    ]

    # Include relevant configuration details
    relevant_keys = ["PSIBLASTDIR", "PSIBLASTBASE", "BLAST_NUM_THREADS"]
    for key in relevant_keys:
        if key in config:
            error_message.append(f"{indent}{key}: {config[key]}\n")

    logger_obj.error("".join(error_message))
    sys.exit()


def run_psiblast(config, protein_fasta_file, temporary_files, load_existing=False):
    """
    Runs PSIBLAST to generate a PSSM file for a given protein sequence.

    Args:
        config (dict): Configuration dictionary containing paths and settings.
        protein_fasta_file (str): Path to the input protein FASTA file.
        temporary_files: The list of temporary files generated will be updated.
        load_existing (bool): If True, uses an existing PSSM file if it exists.

    Returns:
        bool: True if the PSSM file was successfully generated or validated, False otherwise.

    Raises:
        SystemExit: If required executables or databases are missing, or if the PSSM generation fails.
    """
    success = False
    pssm_file = f"{protein_fasta_file}.pssm"

    # Check if existing PSSM file should be reused
    if load_existing and os.path.isfile(pssm_file):
        return pssm_check(pssm_file, config)

    # Validate paths for PSIBLAST executable and database
    psiblast_exe = os.path.join(config["PSIBLASTDIR"], "psiblast")
    psiblast_db = f"{config['PSIBLASTBASE']}.fasta"
    errors = []

    if not os.path.isfile(psiblast_exe):
        errors.append(f"PSIBLAST executable '{psiblast_exe}' is missing.")
    if not os.path.isfile(psiblast_db):
        errors.append(f"PSIBLAST database '{psiblast_db}' is missing.")

    # Exit if any required files are missing
    if errors:
        for error in errors:
            logger_obj.error(error)
        sys.exit()

    # Construct and run the PSIBLAST command
    psiblast_command = (
        f"{psiblast_exe} -query {protein_fasta_file} "
        f"-num_threads {config['BLAST_NUM_THREADS']} "
        f"-db {config['PSIBLASTBASE']} "
        f"-num_iterations 3 -out {protein_fasta_file}.out "
        f"-out_ascii_pssm {pssm_file} 2>/dev/null"
    )

    logger_obj.info("running psiblast for PSSM")
    os.system(psiblast_command)
    temporary_files.append(pssm_file)
    temporary_files.append(f"{protein_fasta_file}.out")

    # Verify the generated PSSM file
    success = pssm_check(pssm_file, config)
    if success:
        logger_obj.info("completed psiblast run for PSSM")
    else:
        logger_obj.info("failed psiblast run for PSSM")

    return success


def mutation_position_pssm(sequence_file, sequence_position, wild_aa):
    """
    Extracts PSSM scores for a specific mutation position in a protein sequence.

    Args:
        sequence_file (str): Path to the input sequence file (without extensions).
        sequence_position (int): Position of the mutation in the sequence (1-based index).
        wild_aa (str): Wild-type amino acid at the mutation position (single-letter code).

    Returns:
        dict: A dictionary containing PSSM scores for the specified position,
              where keys are in the format 'row_pssm_<amino_acid>' and values are the scores.

    Raises:
        ValueError: If the wild-type amino acid at the mutation position does not match.
    """
    features = {}
    pssm_aa_index = {}

    with open(f"{sequence_file}.pssm", "r") as pssm_file:
        for line in pssm_file:
            info = line.strip().split()

            # Parse the header amino acids in PSSM
            if len(info) == 40:
                for i in range(20):
                    pssm_aa_index[i + 2] = info[i]

            # Parse the row corresponding to the mutation position
            elif len(info) > 43 and int(info[0]) == int(sequence_position):
                if wild_aa != info[1]:
                    raise ValueError(
                        f"Mismatch at sequence position {sequence_position}: "
                        f"Expected '{wild_aa}', found '{info[1]}'."
                    )
                for i in range(2, 22):
                    features[f"row_pssm_{pssm_aa_index[i]}"] = float(info[i])

    logger_obj.info(
        f"Row PSSM for position [{sequence_position}, {wild_aa}]: {features}"
    )
    return features


def calculate_normalized_mean_pssm(sequence_file):
    """
    Calculate the Normalized Average Position-Specific Scoring Matrix (PSSM)
    features for a given protein sequence.

    Args:
        sequence_file (str): Path to the input sequence file without the ".pssm" extension.

    Returns:
        dict: A dictionary containing normalized mean PSSM values for each amino acid.

    Raises:
        ValueError: If the sequence length is zero, indicating an invalid or empty input file.
    """
    # Initialize data structures
    pssm = {}  # Holds raw PSSM values for each amino acid
    features = {}  # Holds the calculated normalized mean features
    pssm_aa_index = {}  # Maps index positions to amino acids in the PSSM
    sequence_length = 0

    try:
        # Open and process the PSSM file
        with open(f"{sequence_file}.pssm", "r") as file:
            for line in file:
                info = line.strip().split()

                # Identify and parse the header containing amino acid indices
                if len(info) == 40:
                    for i in range(20):
                        pssm_aa_index[i + 2] = info[i]  # Map indices to amino acids
                        pssm[info[i]] = []  # Initialize PSSM list for each amino acid

                # Parse the rows containing PSSM values
                elif len(info) > 43 and info[1].isupper():
                    sequence_length = int(info[0])  # Update sequence length
                    for i in range(2, 22):  # Collect PSSM values for each amino acid
                        pssm[pssm_aa_index[i]].append(info[i])

        # Ensure sequence length is valid
        if sequence_length <= 0:
            raise ValueError(
                "Provided sequence's length is zero. Expected non-zero length."
            )

        # Calculate normalized mean PSSM features
        for _, aa in pssm_aa_index.items():
            raw_values = [float(1) / (1 + np.e ** (-int(v))) for v in pssm[aa]]
            sum_pssm = np.sum(raw_values)
            normalized_mean = float(sum_pssm) / sequence_length
            features[f"norm_mean_pssm_{aa}"] = f"{normalized_mean:.3f}"

        return features

    except FileNotFoundError:
        logger_obj.error(f"File '{sequence_file}.pssm' not found.")
        sys.exit()
    except ValueError as e:
        logger_obj.error(f"{str(e)}")
        sys.exit()


def get_pssm_features(
    config, sequence_file, sequence_position, wt_aa1, temporary_files, use_saved_pssm
):
    """
    Extracts PSSM-based features for a given protein sequence and mutation position.

    Args:
        config (dict): Configuration dictionary containing tool paths and settings.
        sequence_file (str): Path to the input sequence file (without extensions).
        sequence_position (int): Position of the mutation in the sequence (1-based index).
        wt_aa1 (str): Wild-type amino acid at the mutation position (single-letter code).
        temporary_files: The list of temporary files generated will be updated.
        use_saved_pssm (bool): If True, use an existing PSSM file if available.

    Returns:
        dict: A dictionary of PSSM-based features, including normalized mean PSSM values
              and mutation-specific PSSM scores.

    Raises:
        SystemExit: If PSSM generation or validation fails.
    """
    features = {}

    # Check or generate the PSSM file
    if not use_saved_pssm:
        pssm_status = run_psiblast(
            config, sequence_file, temporary_files, use_saved_pssm
        )
    else:
        pssm_status = pssm_check(f"{sequence_file}.pssm", config)

    if pssm_status:
        # Extract normalized mean PSSM features
        features.update(calculate_normalized_mean_pssm(sequence_file))

        # Extract mutation-specific PSSM scores
        features.update(
            mutation_position_pssm(sequence_file, sequence_position, wt_aa1)
        )

    return features
