#!/usr/bin/env python
# coding: utf-8


import os
import warnings

from time import time

from utils.logger import TinyLogger

logger = TinyLogger(log_to_console=True)

from configuration import configure
from utils.inputhandler import argument_parser, validate_input
from utils.common import set_logger as set_common_logger
from utils.common import cleanup_temporary_files

from predictor.nucleic import predict_mutations_in_dna
from predictor.protein import predict_mutations_in_protein

warnings.simplefilter("ignore")

__author__ = ["Panday, Shailesh Kumar"]
__version__ = 2.0

# Define the base location for third-party tools and dependencies
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

maintainers = ["Panday, Shailesh Kumar; shaileshp51@gmail.com"]


# Initialize configuration
config = configure(__location__)
base_location = __location__

# A list of all temporary files created during execution
temporary_files = [
    "dssr-2ndstrs.bpseq",
    "dssr-2ndstrs.ct",
    "dssr-2ndstrs.dbn",
    "dssr-helices.pdb",
    "dssr-pairs.pdb",
    "dssr-stems.pdb",
    "dssr-torsions.txt",
    "dssr-hairpins.pdb",
    "dssr-stacks.pdb",
]


def main():
    # Whether to enable using a jobname, it helps keep track of jobs
    enable_jobname = True
    # Wheter to dump feature values to a file
    dump_features = False
    # Whether to dump predicted ddG from trained-models also with feature values
    dump_ddg_with_feature = False

    # logs_suffix = f"{int(time() * 1000)}"
    logs_suffix = "logfiles"

    args = argument_parser(__version__, enable_jobname, logger)
    results = validate_input(args)
    job_name = args.jobname if enable_jobname else ""
    temporary_files.append(results["cleaned_pdb"])

    outbase, outext = os.path.splitext(args.output)
    fout = open(args.output, "w")
    fnamefeatures = f"{outbase}_{args.model_type}_features.csv"
    if args.verbosity != "none":
        logger.set_level(args.verbosity)

    set_common_logger(logger)

    preds = []
    if args.model_type == "protein":
        print(
            "Jobname Complexed_chains Mutated_polymer Mutated_chain Wild_aminoacid Position Mutant_aminoacid ddG(kcal/mol) Type",
            file=fout,
        )
        preds = predict_mutations_in_protein(
            results["cleaned_pdb"],
            args.derive_seq_from_coords,
            results["mt_list"],
            config,
            args.verbosity == "none",
            base_location,
            outbase,
            job_name,
            dump_features,
            dump_ddg_with_feature,
            fnamefeatures,
            logger,
            temporary_files,
        )
    elif args.model_type == "dna":
        print(
            "Jobname Complexed_chains Mutated_polymer Forward_chain Wild_basepair Forward_position Mutant_basepair ddG(kcal/mol) Type",
            file=fout,
        )
        preds = predict_mutations_in_dna(
            results["cleaned_pdb"],
            results["mt_list"],
            config,
            base_location,
            job_name,
            dump_features,
            dump_ddg_with_feature,
            fnamefeatures,
            logger,
            temporary_files,
        )
    for mt, pred in zip(results["mt_list"], preds):
        print(
            job_name,
            args.complex_chains,
            args.model_type,
            mt[0],
            mt[1],
            mt[2],
            mt[3],
            pred,
            file=fout,
        )
    fout.close()
    unique_tmp_files = set(temporary_files)
    logger.debug(
        f"following {len(unique_tmp_files)} temporary files generated: "
        + str(unique_tmp_files)
    )
    retained_files = []
    if "none" in args.keep_temp_files:
        retained_files = cleanup_temporary_files(unique_tmp_files, [])
    else:
        retained_files = cleanup_temporary_files(unique_tmp_files, args.keep_temp_files)
    # copy the retained log files to logfiles directory and clean current dir
    os.makedirs(f"{args.output}_{logs_suffix}", mode=0o777, exist_ok=True)
    for fname in retained_files:
        os.system(f"cp {fname} {args.output}_{logs_suffix}/")
        # cleanup current dir
        os.remove(f"{fname}")
    os.system(f"cp {args.output} {args.output}_{logs_suffix}/")
    # os.remove(f"{args.output}")
    print("generated files are stored in: ", f"{args.output}_{logs_suffix}")


if __name__ == "__main__":
    main()
