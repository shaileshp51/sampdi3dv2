#!/usr/bin/env python
# coding: utf-8
import os


def configure(base_location):
    """
    Configures and returns paths to third-party tools, databases, and settings
    required for the application.

    Returns:
        dict: A dictionary containing configuration settings including paths
              to executables, databases, and number of threads.
    """
    # Initialize the configuration dictionary
    config = {}

    # Configure paths for required third-party tools
    config["dssp_exe"] = f"{base_location}/software/dssp"
    config["x3dna_exe"] = f"{base_location}/software/x3dna-dssr"
    config["scwrl4_exe"] = f"{base_location}/software/scwrl4/Scwrl4"

    blast_db_bin_home = os.environ["SAMPDI3Dv2_HOME"]
    # Users can install blast and configure uniref50 in a common path and to use
    # it can export the environemnt variable 'SAMPDI3Dv2_HOME'.
    # Alternatively users can comment line: 22. and provide absolute path to
    # install directory of blast and uniref50 databse in lines 30 and 31, respectively.

    # Configure paths for PSIBLAST tool and associated database
    config["PSIBLASTDIR"] = os.path.join(blast_db_bin_home, "blast/")
    config["PSIBLASTBASE"] = os.path.join(blast_db_bin_home, "uniref50/uniref50")

    # Configure the number of threads to use for BLAST operations
    config["BLAST_NUM_THREADS"] = 4

    return config
