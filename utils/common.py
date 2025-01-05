#!/usr/bin/env python
# coding: utf-8


import os
import sys
import csv

logger_obj = None


def set_logger(logger):
    global logger_obj
    logger_obj = logger


def cleanup_temporary_files(temporary_log_files, ignore_ext):
    retained_files = []
    for name in temporary_log_files:
        if os.path.isfile(name):
            base_name, extension = os.path.splitext(name)
            if (
                (not base_name.startswith("dssr-"))
                and extension[1:] in ignore_ext
                and not base_name.endswith("_prot")
            ):
                logger_obj.debug(f"keeping file: {name}")
                retained_files.append(name)
            else:
                logger_obj.debug(f"removing file: {name}")
                os.remove(name)
    return retained_files


# Helper Functions
def mutation_ss_label(ss_label):
    """Map secondary structure labels to numeric identifiers."""
    label_map = {"H": "1", "B": "2", "E": "3", "G": "4", "I": "5", "T": "6", "S": "7"}
    return label_map.get(ss_label, "0")


def dict_to_csv(data_dict, filename, fieldnames=None):
    """
    Writes a dictionary to a CSV file.

    Args:
        data_dict (dict): The dictionary containing data to write.
        filename (str): The name of the CSV file to create.
        fieldnames (list, optional):  A list of keys to use as column headers.
                   If not provided, all keys from the dictionary will be used.
    """

    if fieldnames is None:
        fieldnames = list(data_dict.keys())

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(zip(*data_dict.values()))


def list_to_csv(data_list, fieldnames, filename):
    """
    Writes a list of lists to a CSV file.

    Args:
        data_list (list): The list of lists/tuples containing data to write.
        fieldnames (list, str):  A list of keys to use as column headers.
        filename (str): The name of the CSV file to create.
    """

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data_list)


def read_fasta(fasta_file):
    """
    Reads a protein sequence from a FASTA file, validates it, and returns the sequence.

    Args:
        fasta_file (str): Path to the FASTA file containing the protein sequence.

    Returns:
        str: The validated protein sequence in uppercase.

    Raises:
        SystemExit: If the sequence contains invalid characters not representing standard amino acids.
    """
    try:
        # Read and concatenate the sequence lines, skipping headers
        with open(fasta_file, "r") as file:
            sequence = "".join(
                line.strip() for line in file if not line.strip().startswith(">")
            ).upper()

        # Validate the sequence for valid amino acids
        valid_amino_acids = "ARNDCQEGHILKMFPSTWYV"
        for residue in sequence:
            if residue not in valid_amino_acids:
                print(
                    f"ERROR>> Invalid protein sequence. Unknown amino acid '{residue}' in file {fasta_file}."
                )
                sys.exit()

        return sequence

    except FileNotFoundError:
        sys.exit(f"ERROR>> File '{fasta_file}' not found.")
