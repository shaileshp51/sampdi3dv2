import os

from utils.common import mutation_ss_label

logger_obj = None


def set_logger(logger):
    global logger_obj
    logger_obj = logger


# BEGIN:: FUNCTIONS USED FOR BOTH PROTEIN AND DNA MUTATIONS
# Constants
basepairs = (
    "AA",
    "AT",
    "AG",
    "AC",
    "TA",
    "TT",
    "TG",
    "TC",
    "GA",
    "GT",
    "GG",
    "GC",
    "CA",
    "CT",
    "CG",
    "CC",
)
nuc_res3to1 = {"DA": "A", "DC": "C", "DG": "G", "DI": "I", "DT": "T"}


# Protein and DNA Mutation Functions
def run_dssp_and_x3dna(pdb_file, polymer, dssp_exe, x3dna_exe, temporary_log_files):
    """
    Perform secondary structure and interaction analysis on the given PDB file.

    Args:
        pdb_file (str): Path to the PDB file.
        polymer (str): Type of polymer ('protein' or 'dna').
        dssp_exe (str): Path to DSSP executable.
        x3dna_exe (str): Path to 3DNA executable.
        temporary_log_files (list[str]): Append temp files created here to it.
    """
    suffix = os.path.basename(pdb_file).rsplit(".", 1)[0]
    os.system(f"{dssp_exe} -i {pdb_file} -o pdb_dssp_{suffix}")
    temporary_log_files.append(f"pdb_dssp_{suffix}")
    logger_obj.info(f"DSSP output generated: pdb_dssp_{suffix}")

    if polymer == "dna":
        os.system(
            f"{x3dna_exe} --more -i={pdb_file} -o=dna_detail_{suffix} >/dev/null 2>&1"
        )
        temporary_log_files.append(f"dna_detail_{suffix}")
        logger_obj.info(f"3DNA output for DNA detail: dna_detail_{suffix}")

    os.system(
        f"{x3dna_exe} snap -i={pdb_file} -o=interaction_detail_{suffix} > num_inter_{suffix} 2>&1"
    )
    temporary_log_files.append(f"interaction_detail_{suffix}")
    temporary_log_files.append(f"num_inter_{suffix}")
    logger_obj.info(f"3DNA interaction output: num_inter_{suffix}")


def get_protein_dna_interactions(pdb_file):
    """
    Extract interaction metrics from the output of 3DNA's SNAP analysis.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        tuple: (nucleotide_contacts, phosphate_hbonds, base_hbonds, base_stacks)
    """
    suffix = os.path.basename(pdb_file).rsplit(".", 1)[0]
    metrics = {
        "nucleotide_contacts": 0,
        "phosphate_hbonds": 0,
        "base_hbonds": 0,
        "base_stacks": 0,
    }

    try:
        with open(f"num_inter_{suffix}") as file:
            for line in file:
                if "total number of nucleotide/amino-acid contacts:" in line:
                    metrics["nucleotide_contacts"] = float(line.split()[-1])
                elif "total number of phosphate/amino-acid H-bonds:" in line:
                    metrics["phosphate_hbonds"] = float(line.split()[-1])
                elif "total number of base/amino-acid H-bonds:" in line:
                    metrics["base_hbonds"] = float(line.split()[-1])
                elif "total number of base/amino-acid stacks:" in line:
                    metrics["base_stacks"] = float(line.split()[-1])
    except FileNotFoundError:
        raise (f"Interaction file not found: num_inter_{suffix}")
    except ValueError:
        raise (f"Error processing interaction file: num_inter_{suffix}")

    logger_obj.info(f"Extracted SNAP metrics for {pdb_file}: {metrics}")
    return tuple(metrics.values())


def get_protein_wt_base_interactions(pdb_file, wt_info):
    """
    Extract protein wild-type base interaction metrics from 3DNA's SNAP output.

    Args:
        pdb_file (str): Path to the PDB file.
        wt_info (tuple): wild-type base info (chain, base(1-letter-code), position)

    Returns:
        tuple: (wt_nucleotide_contacts, wt_phosphate_hbonds, wt_base_hbonds, wt_base_stacks)
    """
    suffix = os.path.basename(pdb_file).rsplit(".", 1)[0]
    metrics = {
        "wt_nucleotide_contacts": (0, "nucleotide/amino-acid interactions", 3),
        "wt_phosphate_hbonds": (0, "phosphate/amino-acid H-bonds", 2),
        "wt_base_hbonds": (0, "base/amino-acid H-bonds", 2),
        "wt_base_stacks": (0, "base/amino-acid stacks", 3),
    }
    metrics_names = list(metrics.keys())
    base_id = f"{wt_info[0]}.D{wt_info[1][0]}{wt_info[2]}"
    # print("base_id:", base_id)
    try:
        with open(f"interaction_detail_{suffix}") as file:
            lines = file.readlines()
            n_lines = len(lines)
            current_line = 0
            while current_line < n_lines:
                line = lines[current_line].strip()
                for metric, metric_value in metrics.items():
                    if line.startswith("List of") and metric_value[1] in line:
                        current_line += 1
                        n_nt_aa_contacts = 0
                        while current_line < n_lines:
                            current_line += 1
                            line = lines[current_line].strip()
                            if not line:
                                break
                            else:
                                interacting_base = line.split()[metric_value[2]]
                                # If this has atom info with base_id, discard atom info
                                if metric_value[2] == 2:
                                    interacting_base = interacting_base[
                                        interacting_base.rindex("@") + 1 :
                                    ]
                                    # print("atom discarded interacting_base", interacting_base)
                                if interacting_base == base_id:
                                    n_nt_aa_contacts += 1
                        metrics[metric] = (n_nt_aa_contacts, metric_value[1])
                        current_line += 1
                        break
                current_line += 1
    except FileNotFoundError:
        raise (f"Interaction file not found: num_inter_{suffix}")
    except ValueError:
        raise (f"Error processing interaction file: num_inter_{suffix}")

    logger_obj.info(f"Extracted SNAP metrics for {pdb_file}: {metrics}")
    return tuple([metrics[m][0] for m in metrics_names])


def mismatch_dna(wild, mutation):
    """Check if the mutation causes a mismatch in DNA base pairing."""
    standard_pairs = {"AT", "TA", "GC", "CG"}
    return 1 if {wild, mutation} - standard_pairs else 0


def pair_type(wild):
    """Determine the type of DNA base pair."""
    if wild in {"AT", "TA"}:
        return 1
    if wild in {"GC", "CG"}:
        return 0
    return 2


def mutation_ss_dna(pdb_file):
    """
    Calculate the ratio of each secondary structure type in protein.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        list: Ratios of each secondary structure type.
    """
    suffix = os.path.basename(pdb_file).rsplit(".", 1)[0]
    ss_counts = [0] * 8  # Initialize counters for 8 structure types

    try:
        with open(f"pdb_dssp_{suffix}") as file:
            in_residue_section = False
            for line in file:
                if "#  RESID" in line:
                    in_residue_section = True
                    continue
                if in_residue_section:
                    ss_counts[int(mutation_ss_label(line[16:17]))] += 1
    except FileNotFoundError:
        raise (f"DSSP file not found: pdb_dssp_{suffix}")

    total = sum(ss_counts)
    ss_ratios = [f"{ss_counts[i] / total:.2f}" for i in [1, 2, 3, 4, 6, 7]]
    logger_obj.info(f"Secondary structure ratios: {ss_ratios}")
    return ss_ratios


def get_dna_structure_features(pdb_file, chain, resid, for_w, rev_w):
    """
    Extract DNA base pair and helical parameters.

    Args:
        pdb_file (str): Path to the PDB file.
        chain (str): Chain ID.
        resid (int): Residue ID.
        for_w (str): Forward nucleotide.
        rev_w (str): Reverse nucleotide.

    Returns:
        list: Base pair, step, and helical parameters.
    """
    suffix = os.path.basename(pdb_file).rsplit(".", 1)[0]
    params = {"bp_pars": [], "bp_steps": [], "bp_helical": []}

    try:
        with open(f"dna_detail_{suffix}") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                # Note: this condition never matches with single stranded DNA due to absence of pairing
                if f"{chain}.D{for_w}{resid}" in line and f".D{rev_w}" in line:
                    if "bp-pars: [" in lines[i + 5]:
                        params["bp_pars"] = extract_parameters(lines[i + 5], 10)
                    elif "step-pars:  [" in lines[i + 2]:
                        params["bp_steps"] = extract_parameters(lines[i + 2], 13)
                        if "heli-pars:  [" in lines[i + 3]:
                            params["bp_helical"] = extract_parameters(lines[i + 3], 13)
                # Keep checking until all three sets are found, once found, stop by breaking the loop
                if (
                    len(params["bp_pars"])
                    and len(params["bp_steps"])
                    and len(params["bp_helical"])
                ):
                    break
            # If DNA is single strand and base-pair features are missing use all zeros
            for prm_name in ["bp_pars", "bp_steps", "bp_helical"]:
                while len(params[prm_name]) < 6:
                    params[prm_name].append(0)

    except FileNotFoundError:
        raise (f"DNA detail file not found: dna_detail_{suffix}")
    except IndexError:
        raise (f"Error processing DNA detail file: dna_detail_{suffix}")

    return params["bp_pars"] + params["bp_steps"] + params["bp_helical"]


def extract_parameters(line, start_column):
    """Helper function to parse numeric parameters from a line."""
    param_values = []
    for num in line.strip()[start_column:-1].split():
        try:
            float(num)
            param_values.append(num)
        except ValueError:
            param_values.append(0)

    # If some of bp_pars, bp_steps, bp_helical are missing use 0 for them.
    while len(param_values) < 6:
        param_values.append(0)

    return param_values


def mutation_type_dna(basepairs, wild, mutation):
    """
    Determines the 1-indexed position of a specific wild-to-mutation pair
    transition within the given DNA basepair sequence.

    Parameters:
        basepairs (tuple): A collection of DNA basepair strings
                           (e.g., "AA", "AT", "AG", "AC", ...).
        wild (str): The wild type basepair (e.g., "AT").
        mutation (str): The mutated basepair (e.g., "TG").

    Returns:
        int: The 1-indexed position of the wild-to-mutation pair transition
             in all possible combinations, or 0 if not found.
    """
    position = 0  # Tracks the position across all valid (wild, mutation) pairs.

    for i in basepairs:
        for j in basepairs:
            if i != j:  # Only consider valid transitions where i != j
                position += 1
                if i == wild and j == mutation:
                    return position  # Return position when match is found.

    return 0  # Return 0 if no valid transition is found.
