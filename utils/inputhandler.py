#!/usr/bin/env python
# coding: utf-8


import os
import sys
import argparse

import prody as pdy

from utils.common import *
from utils.sequence import *
from utils.dna import *

# Let's suppress the log messages from ProDy, to keep stdout clear.
pdy.confProDy(verbosity="none")

logger_obj = None


def set_logger(logger):
    global logger_obj
    logger_obj = logger


def is_file(filepath):
    val = os.path.isfile(filepath)
    if not val:
        raise argparse.ArgumentTypeError("`%s` is not a valid filepath" % filepath)
    return filepath


def preprocess_pdb(pdb_file, complex_chains):
    cleaned_pdb_file = ""
    try:
        chains_list = ",".join(list(complex_chains))
        env_exe_dir = os.path.split(sys.executable)[0]
        pdb_selmodel = os.path.join(env_exe_dir, "pdb_selmodel")
        pdb_selchain = os.path.join(env_exe_dir, "pdb_selchain")
        pdb_selaltloc = os.path.join(env_exe_dir, "pdb_selaltloc")

        cleaned_pdb_file = f"{pdb_file[:-4]}_cleaned.pdb"

        os.system(f"{pdb_selmodel} -1 {pdb_file} > {pdb_file[:-4]}_mdl1.pdb")
        os.system(
            f"{pdb_selchain} -{chains_list} {pdb_file[:-4]}_mdl1.pdb | {pdb_selaltloc} > {cleaned_pdb_file}"
        )
        os.remove(f"{pdb_file[:-4]}_mdl1.pdb")
        logger_obj.info(
            f"successfully cleaned {pdb_file} to {cleaned_pdb_file} keeping only chains {complex_chains}"
        )
    except Exception as e:
        logger_obj.error(f"pre-processing pdb failed: {e}")
        sys.exit()
    return cleaned_pdb_file


def validate_mutation_info(struct, mutation_list, polymer="protein"):
    """
    Validate the mutation information against the provided structure.

    Args:
        struct: The structure object to validate against.
        mutation_list (list): List of mutations (chain, wild_type, resid, mutant).
        polymer (str): The type of polymer ("protein" or "dna").

    Returns:
        tuple: (errors, warnings, validated_mutations).
    """
    errors, warns, validated_mutations = [], [], []
    res3to1, bb_atom, validate_alphabet_code = None, "", None

    # Set polymer-specific settings
    if polymer == "dna":
        res3to1 = nuc_res3to1
        bb_atom = "C5'"
        validate_alphabet_code = lambda bp: bp in ("A", "T", "G", "C")
        alphabet = "nucleotide"
        direction = "forward strand"
    elif polymer == "protein":
        res3to1 = aa3to1
        bb_atom = "CA"
        validate_alphabet_code = validate_aa_code
        alphabet = "amino acid"
        direction = ""

    # Validate each mutation
    for chain, wild_type, resid, mutant in mutation_list:
        is_valid = True

        # Check if mutation site exists in the structure
        wt_ca = struct.select(f"chain {chain} and resid {resid} and name {bb_atom}")
        if wt_ca is None:
            warns.append(
                f"Mutation site 'chain {chain} resid {resid}' missing in structure."
            )
            is_valid = False

        # Validate wild-type residue
        elif not validate_alphabet_code(wild_type[0]):
            warns.append(f"Wild-type {direction} {alphabet} '{wild_type}' is invalid.")
            is_valid = False

        elif res3to1.get(wt_ca.getResnames()[0], "X") != wild_type[0]:
            warns.append(
                f"Wild-type {direction} {alphabet} '{wild_type[0]}' "
                f"mismatches with sequence residue '{wt_ca.getResnames()[0]}'."
            )
            is_valid = False

        # Validate mutant residue
        if not validate_alphabet_code(mutant[0]):
            warns.append(f"Mutant {direction} {alphabet} '{mutant}' is invalid.")
            is_valid = False

        # Append valid mutation
        if is_valid:
            validated_mutations.append((chain, wild_type, resid, mutant))

    # Escalate warnings to errors if no valid mutations exist
    if not validated_mutations:
        errors.extend(warns)
        warns = []

    return errors, warns, validated_mutations


def validate_input(args):
    """
    Validate input arguments and extract mutation data.

    Args:
        args: Input arguments.

    Returns:
        dict: Validation results including structure, errors, warnings, and validated mutations.
    """
    results = {
        "cleaned_pdb": None,
        "structure": None,
        "errors": [],
        "warns": [],
        "mt_list": [],
    }

    # Validate PDB file
    if not is_file(args.complex_pdb):
        results["errors"].append("Complex PDB file is missing or invalid.")
        return results

    # Cleanup the input pdb
    results["cleaned_pdb"] = preprocess_pdb(args.complex_pdb, args.complex_chains)

    # Parse PDB file
    results["structure"] = pdy.parsePDB(results["cleaned_pdb"])
    if len(results["structure"].getCoordsets()) > 1:
        results["warns"].append(
            "Multiple models detected in the input PDB file. "
            "Only the first model will be used."
        )
        results["structure"] = pdy.parsePDB(args.complex_pdb, model=1)

    if len(set(results["structure"].getAltlocs())) > 1:
        results["warns"].append(
            "Alternate locations detected for some residues in the PDB file. "
            "Using highest occupancy coordinates for alternate atoms."
        )

    # Parse mutation list
    mutations_list = []
    if args.mutation_list_file and is_file(args.mutation_list_file):
        with open(args.mutation_list_file, "r") as file:
            for line in file:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                info = line.strip().split(" ")
                if len(info) == 4:
                    chain, wild_type, resid, mutant = (
                        info[0],
                        info[1],
                        int(info[2]),
                        info[3],
                    )
                    mutations_list.append((chain, wild_type, resid, mutant))
                else:
                    results["errors"].append(
                        "Invalid format in mutation list file. Each line must contain: "
                        "'chain wild_type resid mutant'."
                    )
    elif all(
        [args.chain, args.wild_type, int(args.resid), args.mutant]
    ):  # Single mutation specified
        mutations_list.append((args.chain, args.wild_type, args.resid, args.mutant))

    if mutations_list:
        # Validate mutations
        errors, warns, validated_mutations = validate_mutation_info(
            results["structure"], mutations_list, polymer=args.model_type
        )
        results["errors"].extend(errors)
        results["warns"].extend(warns)
        results["mt_list"] = validated_mutations
    else:
        results["errors"].append("Mutation information is required but missing.")

    # Output errors and warnings
    if results["errors"]:
        for error_message in results["errors"]:
            logger_obj.error(error_message)
        sys.exit()
    if results["warns"] and not args.verbosity != "none":
        for warn_message in results["warns"]:
            logger_obj.warning(warn_message)

    return results


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def argument_parser(version, enable_jobname, logger):
    set_logger(logger)

    parser = argparse.ArgumentParser(
        prog="sampdi3d.py",
        description=f"""
            R|SAMPDI-3Dv{version} : Predict the free energy change of binding due to 
            point mutation in protein or DNA for protein-DNA binding.
            
            For method details check: Rimal, P.; Paul, S.K.; Panday, S.K.; Alexov, E. 
            Further Development of SAMPDI-3D: A Machine Learning Method for Predicting 
            Binding Free Energy Changes Caused by Mutations in Either Protein or DNA. 
            Genes 2025, 16, 101. https://doi.org/10.3390/genes16010101
        """,
        formatter_class=SmartFormatter,
    )
    parser.add_argument(
        "-i",
        "--complex-pdb",
        help="PDB file of the wild-type protein-DNA complex",
        type=is_file,
        required=True,
    )
    parser.add_argument(
        "-cc",
        "--complex-chains",
        help="""R|Chains of the PDB file that participate in protein-DNA complex.
Following pre-processing steps will be performed on input-pdb:
 * If there are more than one models in input-pdb only first is used.
 * Only complex chains will be extracted from the input PDB file. 
 * If alternate positions found for some residues only highest occupancy 
    position is considered.""",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--model-type",
        help="R|protein: mutation is in protein;\ndna: mutation is in DNA",
        choices={"protein", "dna"},
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output file name for the predictions (default: output.out)",
        default="output.out",
    )
    parser.add_argument(
        "--seq-from-coords",
        dest="derive_seq_from_coords",
        action="store_true",
        help="""R|Derive sequence directly from ATOM records instead of using the SEQRES header. 
            If the header is missing, this mode is enabled automatically. (default: False)
            """,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        help="print verbose messages up to level",
        choices={"none", "warning", "info", "debug"},
        default="warning",
    )
    parser.add_argument(
        "-k",
        "--keep-temp-files",
        help="keep temporary files with specified extension(s)",
        choices={"none", "pdb", "fasta", "pssm", "csv", "seq", "log"},
        default="none",
        nargs="+",
    )
    if enable_jobname:
        parser.add_argument(
            "-j",
            "--jobname",
            help="Provide a jobname to write in first column of output for each mutation",
            required=True,
        )
    group1 = parser.add_argument_group(
        "single point-mutation", "single point-mutation information"
    )
    group1.add_argument(
        "-c",
        "--chain",
        help="""R|chain of the mutation in the input PDB file.
        for mutation in protein provide the mutating chain
        for mutation in DNA provide the chain of the forward strand DNA chain,
        i.e. chain corresponding to first base in wild-type base pair
        """,
    )
    group1.add_argument(
        "-w",
        "--wild-type",
        help="""R|wild-type amino acid's 1 letter code for mutation in protein
        wild-type base-pair (two-letters, one letter code for both bases)""",
    )
    group1.add_argument(
        "-r",
        "--resid",
        type=int,
        help="""R|resid of mutation position for mutation in protein
        position of the base in forward-strand of DNA chain""",
    )
    group1.add_argument(
        "-m",
        "--mutant",
        help="""R|mutant amino acid's 1 letter code for mutation in protein
        mutant-base-pair (two-letters, one letter code for both bases)""",
    )
    group2 = parser.add_argument_group(
        "multiple point-mutations", "multiple point-mutations listing"
    )
    group2.add_argument(
        "-f",
        "--mutation-list-file",
        help="""R|Mutation list file. The file should have one mutation per line,
    where every line has following four information seperated by space.
    protein mutation, each line has a format:
    chain wild-type(1-letter-code) resid mutant(1-letter-code)
    DNA mutation, each line has a format:
    chain wild-type-base-pair resid-base-in-forward-strand mutant-base-pair
    """,
    )
    args = parser.parse_args()
    return args
