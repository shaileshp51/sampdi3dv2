#!/usr/bin/env python
# coding: utf-8


import os
import sys
import prody as pdy

from collections import OrderedDict

from utils.sequence import aa3_map, aa3_aliases, aa3_nonstandard
from utils.sequence import set_logger as set_sequence_logger
from utils.sequence import three_to_one
from utils.common import mutation_ss_label

# Let's suppress the log messages from ProDy, to keep stdout clear.
pdy.confProDy(verbosity="none")

logger_obj = None


def protein_acc_bb_torisons_ss_ratios(pdb_file, chain, position, wild):
    """
    Extract secondary structure and related features for a protein mutation.

    Args:
        pdb_file (str): Path to the PDB file.
        chain (str): Chain identifier.
        position (int): Residue position.
        wild (str): Wild-type residue.

    Returns:
        tuple: Secondary structure label, solvent accessibility, phi, psi, and SS ratios.
    """
    suffix = os.path.basename(pdb_file)[:-4]
    ss_ratio, ss_ratio_list = [], []
    ss, acc, phi, psi = 0, 0, 0, 0

    try:
        with open(f"pdb_dssp_{suffix}") as file:
            for line in file:
                chain_match = line[11:12].strip() == chain
                pos_match = line[5:10].strip() == str(position)

                if chain_match and pos_match:
                    if line[13:14].strip() != wild:
                        sys.exit(
                            f"Wild-type amino acid '{wild}' does not match structure: '{line[13:14].strip()}'."
                            f" at position {position} in {pdb_file}"
                        )
                    ss = mutation_ss_label(line[16:17])
                    acc = float(line[35:38].strip() or 0)
                    phi = float(line[103:109].strip() or 0)
                    psi = float(line[109:115].strip() or 0)
                elif chain_match:
                    ss_ratio.append(mutation_ss_label(line[16:17]))
    except FileNotFoundError:
        logger_obj.error(f"DSSP file not found: pdb_dssp_{suffix}")
        sys.exit()

    # Ignore 0 & 5 th ss as all entries are zero for these two
    ss_ratio_list = [
        f"{ss_ratio.count(i) / len(ss_ratio):.2f}" if len(ss_ratio) > 0 else "0.00"
        for i in map(str, [1, 2, 3, 4, 6, 7])
    ]

    return ss, acc, phi, psi, ss_ratio_list  # Keep only required columns.


def check_scwrl4_log(scwrl4_cmd, logfile):
    if os.path.isfile(logfile):
        logger_obj.debug(
            "************************** SCWRL4 LOG BEGIN *********************"
        )
        logger_obj.debug(f"run: {scwrl4_cmd}")
        logger_obj.debug(
            "".join([ln for ln in open(logfile).readlines() if ln.strip()])
        )
        logger_obj.debug(
            "*************************** SCWRL4 LOG END **********************"
        )
    else:
        logger_obj.error("mutant-protein structure generation failed, aborting now.")
        sys.exit()


def get_protein_coords_sequence(protein_ca_atoms, aa3_nonstandard=aa3_nonstandard):
    """Extract the sequence of a protein based on its backbone coordinates.

    Args:
        protein_atoms: A ProDy AtomGroup object representing the protein.

    Returns:
        dict: (sequence, res_id, warns, errors)
            - sequence: OrderedDict mapping residue IDs to 1-letter codes.
            - non_standard_aa: List of tuples of resnum and chain of non standard residues.
            - warns: List of warnings for non-standard residues.
            - errors: List of errors for unknown residues.
    """
    result = {
        "sequence": OrderedDict(),
        "non_standard_aa": [],
        "warns": [],
        "errors": [],
    }

    for atom in protein_ca_atoms:
        res_id = (atom.getResnum(), atom.getChid())
        resname = atom.getResname()

        if resname in aa3_map:
            result["sequence"][res_id] = three_to_one(resname)
        elif resname in aa3_aliases:
            result["sequence"][res_id] = three_to_one(resname)
        elif resname in aa3_nonstandard:
            std_resname = three_to_one(aa3_nonstandard[resname][0])
            result["sequence"][res_id] = std_resname[0]
            result["non_standard_aa"].append(res_id)
            result["warns"].append(
                f"Non-standard residue {resname} at {res_id} converted to {std_resname}."
            )
        else:
            result["errors"].append(f"Unknown residue {resname} at {res_id}.")
            result["sequence"][res_id] = "X"
            # print(f"Unknown residue {resname} at {res_id}.")

    return result


def create_residue_id_to_index_map(structure_atoms):
    """Generate residue ID to index mapping and vice versa.

    Args:
        structure_atoms: prody AtomGroup object for the structure file.

    Returns:
        tuple: (res_id2index_map, res_index2id_map)
    """
    res_id2index_map = {}
    res_index2id_map = {}
    for a in structure_atoms:
        resi, ch, residx = a.getResnum(), a.getChid(), a.getResindex()
        if (resi, ch) not in res_id2index_map:
            res_id2index_map[(resi, ch)] = (residx + 1, ch)
    res_index2id_map = {v: k for k, v in res_id2index_map.items()}
    return res_id2index_map, res_index2id_map


def renumber_residues(structure_atoms, renumber_map):
    """Renumber residues in a structure using a renumbering map.

    Args:
        structure_atoms: A ProDy AtomGroup object. It will be updated with renumbering
        renumber_map: Dict mapping old residue IDs(key: resnum, chain) to new residue IDs(value: new_resnum, chain).

    Returns:
        None
    """
    for a in structure_atoms:
        old_res_id = (a.getResnum(), a.getChid())
        new_resnum, new_chain = renumber_map[old_res_id]
        a.setResnum(new_resnum)
    return None


def check_backbone_present(structure_atoms):
    """Check the presence of all backbone atoms {N, CA, C, O} for residues.

    Args:
        structure: A ProDy AtomGroup object.

    Returns:
        tuple:
            - backbone_atoms (OrderedDict): A dictionary where keys are residue identifiers
              (residue number and chain_id), and values are:
                - (list): List of backbone atom names found for the residue.

            - backbone_missing (OrderedDict): A subset of `backbone_atoms` containing only the
              residues with incomplete backbone atoms.
    """
    backbone_atoms = {}
    backbone_missing = {}

    hv = structure_atoms.getHierView()

    for res in hv.iterResidues():
        res_id = (res.getResnum(), res.getChid())
        atoms = {atom.getName() for atom in res.iterAtoms()}
        backbone_atoms[res_id] = atoms

        # Determine completeness and populate residues
        missing = {"N", "CA", "C", "O"} - atoms
        if missing:
            backbone_missing[res_id] = missing

    return backbone_atoms, backbone_missing


def generate_mutation_sequence(
    protein_ca_atoms, sequence_file, res_index2id, mt_info=None
):
    """
    Generates a mutated sequence based on the mutation information and the wild-type protein structure.
    It modifies the residue at the specified position to the new amino acid.

    Args:
        protein_ca_atoms (AtomGroup): The wild-type protein CA atoms.
        sequence_file (str): The name of the file to write the sequence.
        res_index2id (dict): The mapping of old residue IDs(key: resindex, chain) to
                             new residue IDs(value: resnum, chain).
        mt_info (tuple): A tuple containing mutation information:
                          (chain_id, position, original_residue, mutated_residue).

    Returns:
        str: The mutated protein sequence.
        list: A list of non standard residues in the protein else empty list
    """
    # Get the protein residue sequence along with non_standard_aa if any
    result_seq = get_protein_coords_sequence(protein_ca_atoms)

    if len(result_seq["errors"]) > 0:
        for error_message in result_seq["errors"]:
            logger_obj.error(error_message)
        sys.exit()
    elif len(result_seq["warns"]) > 0:
        for warn_message in result_seq["warns"]:
            logger_obj.warning(warn_message)
    protein_seq = result_seq["sequence"]
    non_standard_residues = result_seq["non_standard_aa"]
    logger_obj.debug("protein_seq:" + str(protein_seq))

    # Get the wild-type protein sequence
    wt_protein_sequence = [ri[0].lower() for _, ri in protein_seq.items()]
    protein_sequence = list(wt_protein_sequence)

    # Mutate the sequence if mutation details is provided
    if mt_info is not None:
        chain_id, original_residue, position, mutated_residue = mt_info

        # Apply the mutation
        # The position in the mutation info is based on numbering in original pdb structure
        # Find the correct position in renumbered pdb using index2id map
        # if str(res_index2id[(rid[0], chain_id)][0]) == str(position)
        protein_sequence = [
            (
                mutated_residue.upper()
                if str(res_index2id[rid][0]) == str(position) and ri == original_residue
                else ri.lower()
            )
            for rid, ri in protein_seq.items()
        ]

    with open(sequence_file, "w") as seq_out:
        seq_out.write("".join(protein_sequence) + "\n")
        seq_out.close()

    # Return the mutated sequence as a string
    return "".join(protein_sequence), non_standard_residues


def align_protein_to_create_complex(
    protein_common_sel_ref,
    protein_common_sel_model,
    protein_full_model,
    nonprotein_ref,
    complex_file,
):
    """
    Aligns the mutant protein model to the reference protein and creates a complex structure.
    This function applies the best-fit transformation to align the wild-type and mutant proteins
    and generates a complex with the non-protein component.

    Args:
        protein_common_sel_ref (AtomGroup): The reference (wild-type) protein atoms selected for alignment.
        protein_common_sel_model (AtomGroup): The model protein atoms to align to the reference.
        protein_full_model (AtomGroup): The full structure of the mutant protein.
        nonprotein_ref (AtomGroup): Non-protein component (e.g., DNA, ligand) to combine with the aligned protein.
        complex_file: The file path to the generated protein-nonprotein complex PDB file.

    Returns:
        success (bool): Indicates whether the alignment and complex creation were successful.
    """
    success = False
    model_complex = None
    rmsd_max_to_warn = 2.0

    # Ensure the selection of equal-length common residues for alignment
    if len(protein_common_sel_ref) == len(protein_common_sel_model):
        # Apply transformation to align the protein model with the reference
        pdy.calcTransformation(protein_common_sel_model, protein_common_sel_ref).apply(
            protein_full_model
        )

        # Calculate RMSD between aligned models for quality check
        rmsd_ = pdy.calcRMSD(protein_common_sel_model, protein_common_sel_ref)
        if rmsd_ > rmsd_max_to_warn:
            logger_obj.warning(
                f"RMSD between wild-type and aligned model {complex_file}: {rmsd_:0.03f} is greater then {rmsd_max_to_warn},"
                f" check model {complex_file} to ensure correctness."
            )
        else:
            logger_obj.info(
                f"RMSD between wild-type and aligned model {complex_file}: {rmsd_:0.03f}"
            )

        # Create the protein-DNA complex by combining the aligned protein and non-protein components
        model_complex = protein_full_model.copy() + nonprotein_ref.copy()

        # Write the resulting aligned complex to a new PDB file
        # pdy.writePDB(complex_file, model_complex)

        success = True

    else:
        # If protein selections are mismatched, handle by identifying missing atoms
        protein_noh_ref = OrderedDict(
            (tuple(a[0:-1]), a[-1])
            for a in zip(
                protein_common_sel_ref.getChids(),
                protein_common_sel_ref.getResnums(),
                protein_common_sel_ref.getResnames(),
                protein_common_sel_ref.getNames(),
                protein_common_sel_ref.getSerials(),
            )
        )
        protein_noh_model = OrderedDict(
            (tuple(a[0:-1]), a[-1])
            for a in zip(
                protein_common_sel_model.getChids(),
                protein_common_sel_model.getResnums(),
                protein_common_sel_model.getResnames(),
                protein_common_sel_model.getNames(),
                protein_common_sel_model.getSerials(),
            )
        )

        protein_noh_ref_missing = {
            k: v for k, v in protein_noh_model.items() if not k in protein_noh_ref
        }
        protein_noh_model_missing = {
            k: v for k, v in protein_noh_ref.items() if not k in protein_noh_model
        }
        # Print missing atoms for debugging
        serial_missing = " ".join([str(v) for v in protein_noh_ref_missing.values()])
        logger_obj.debug(f"Missing atom serial numbers: {serial_missing}")

        # Select matching atoms for alignment
        protein_model_match = protein_common_sel_model.select(
            f"not serial {serial_missing}"
        )
        pdy.calcTransformation(protein_model_match, protein_common_sel_ref).apply(
            protein_full_model
        )

        rmsd_ = pdy.calcRMSD(protein_model_match, protein_common_sel_ref)
        if rmsd_ > rmsd_max_to_warn:
            logger_obj.warning(
                f"Adjusted RMSD (excluding missing atoms): {rmsd_:0.03f} is greater than {rmsd_max_to_warn},"
                f" check model {complex_file} to ensure correctness."
            )
        else:
            logger_obj.info(
                f"Adjusted RMSD (excluding missing atoms): {rmsd_:0.03f}",
            )

        # Generate the aligned complex and write to a file
        model_complex = protein_full_model.copy() + nonprotein_ref.copy()
        # pdy.writePDB(complex_file, model_complex)

        success = True

    return success, model_complex


def fix_wildtype_complex_structure(pdb_file, scwrl4_exe, mt_info, temporary_log_files):
    result = {"success": False, "wt_complex": "", "wt_protein": ""}
    res_id2index = {}
    res_index2id = {}

    # Read wild-type protein structure
    wt_pdb = pdy.parsePDB(pdb_file)

    # Separate the protein and non-protein components
    wt_protein = wt_pdb.select(f"protein and chain {mt_info[0]}")
    wt_non_protein = wt_pdb.select(f"not (protein and chain {mt_info[0]})")

    # Combine to create the full wild-type complex keeping protein atoms at top
    wt_complex = wt_protein.copy() + wt_non_protein.copy()

    # Renumber residues for consistency in the complex
    res_id2index, res_index2id = create_residue_id_to_index_map(wt_complex)
    renumber_residues(wt_complex, res_id2index)

    # Update the separated protein and non-protein components after residue renumbering
    wt_protein = wt_complex.select(f"protein and chain {mt_info[0]}")
    wt_non_protein = wt_complex.select(f"not (protein and chain {mt_info[0]})")

    # Check for missing backbone atoms and handle them
    _, backbone_missing = check_backbone_present(wt_protein)
    wt_complex_fixed_file = f"{pdb_file[0:-4]}_wt-fixed.pdb"
    wt_protein_fixed_file = f"{pdb_file[0:-4]}_protein.pdb"
    if len(backbone_missing):
        # Update wt_protein and wt_nonprotein after residue renumbering
        wt_protein = wt_complex.select(f"protein and chain {mt_info[0]}")
        wt_non_protein = wt_complex.select(f"not (protein and chain {mt_info[0]})")

        resid_exclude = " ".join([str(k[0]) for k in backbone_missing.keys()])
        resid_exclude = f"not (resid {resid_exclude})"
        wt_protein_clean = wt_protein.select(resid_exclude)
        wt_complex = wt_protein_clean.copy() + wt_non_protein.copy()

        pdy.writePDB(wt_protein_fixed_file, wt_protein_clean)

    else:
        pdy.writePDB(wt_protein_fixed_file, wt_protein)

    renumber_residues(wt_complex, res_index2id)
    pdy.writePDB(wt_complex_fixed_file, wt_complex)
    temporary_log_files.append(wt_protein_fixed_file)
    temporary_log_files.append(wt_complex_fixed_file)
    logger_obj.debug(
        "Cleaned wild-type protein:" + os.path.realpath(wt_protein_fixed_file),
    )
    # result["success"] = True
    # result["wt_complex"] = wt_complex_fixed_file
    # result["wt_protein"] = wt_protein_fixed_file

    # Load the updated pdb where protein comes before anything else
    wt_pdb_fixed = pdy.parsePDB(wt_complex_fixed_file)
    res_id2index, res_index2id = create_residue_id_to_index_map(wt_pdb_fixed)
    renumber_residues(wt_pdb_fixed, res_id2index)

    # Update the protein and nonprotein selection consistining index-renumbered residues
    wt_protein = wt_pdb_fixed.select(f"protein and chain {mt_info[0]}")
    wt_non_protein = wt_pdb_fixed.select(f"not (protein and chain {mt_info[0]})")

    # Clean the wild-type protein structure by replacing non-standard amino acids
    # with their standard counterparts
    ca_atoms = wt_pdb_fixed.select(f"protein and chain {mt_info[0]} and name CA")

    seq_file = f"{pdb_file[0:-4]}_wt-nonstd-to-parent.seq"
    wt_sequence, non_standard_residues = generate_mutation_sequence(
        ca_atoms, seq_file, res_index2id
    )
    logger_obj.debug("wt_sequence:" + str(wt_sequence))
    temporary_log_files.append(seq_file)

    # Identify and replace non-standard residues with their parent standard residue
    if len(non_standard_residues) > 0:
        # Replace non-standard residues in the PDB with standard ones
        nonstd_resids = " ".join(
            [str(i[0]) if i[0] > 0 else f"`{i[0]}`" for i in non_standard_residues]
        )
        nonstd_aa = wt_protein.select("resid " + nonstd_resids)
        for atom_nonstd_residue in nonstd_aa:
            resname = atom_nonstd_residue.getResname()
            resname_std_aa3 = aa3_nonstandard.get(resname, ("UNK", ""))
            atom_nonstd_residue.setResname(resname_std_aa3[0])
            atom_nonstd_residue.setFlag("hetatm", False)

        # Save the cleaned structure
        pdy.writePDB(wt_protein_fixed_file, wt_protein)

        # Run SCWRL4 to fix the protein structure
        scwrl4_cmd = f"{scwrl4_exe} -i {wt_protein_fixed_file} -s {seq_file} -o {seq_file[0:-4]}_prot.pdb -h > {seq_file[0:-4]}_scwrl4.log"
        os.system(scwrl4_cmd)
        check_scwrl4_log(scwrl4_cmd, f"{seq_file[0:-4]}_scwrl4.log")
        temporary_log_files.append(f"{seq_file[0:-4]}_prot.pdb")
        temporary_log_files.append(f"{seq_file[0:-4]}_scwrl4.log")

        os.rename(
            wt_protein_fixed_file,
            f"{pdb_file[0:-4]}_protein-bb-fixed.pdb",
        )
        os.rename(f"{seq_file[0:-4]}_prot.pdb", wt_protein_fixed_file)
        temporary_log_files.append(f"{pdb_file[0:-4]}_protein-bb-fixed.pdb")

        # Update the protein and non-protein parts
        wt_protein_mdl = pdy.parsePDB(wt_protein_fixed_file)
        wt_protein_nostd = wt_protein.select(
            f"backbone and not (resid {nonstd_resids})"
        )
        wt_protein_mdl_nostd = wt_protein_mdl.select(
            f"backbone and not (resid {nonstd_resids})"
        )

        # Align the cleaned wild-type structure with the non-protein components
        # complex_file = f"{pdb_file[0:-4]}_wt-fixed.pdb"
        success, wt_complex = align_protein_to_create_complex(
            wt_protein_nostd,
            wt_protein_mdl_nostd,
            wt_protein_mdl,
            wt_non_protein,
            wt_complex_fixed_file,
        )
        renumber_residues(wt_complex, res_index2id)
        pdy.writePDB(wt_complex_fixed_file, wt_complex)
        result["success"] = success
        result["wt_complex"] = wt_complex_fixed_file
        result["wt_protein"] = wt_protein_fixed_file

    else:
        # No non-standard residues found, simply save the fixed structure
        wt_complex = wt_protein.copy() + wt_non_protein.copy()
        renumber_residues(wt_complex, res_index2id)
        pdy.writePDB(wt_complex_fixed_file, wt_complex)
        result["success"] = True
        result["wt_complex"] = wt_complex_fixed_file
        result["wt_protein"] = wt_protein_fixed_file

    return result


def create_mutant_protein_complex(
    pdb_file, scwrl4_exe, mt_info, mt_index, logger, temporary_log_files
):
    """
    Creates a mutant protein-DNA complex by processing the mutation and adjusting the structure accordingly.
    It includes residue renumbering, fixing non-standard residues, applying SCWRL4 to adjust side chains,
    and generating a final PDB of the complex.

    Args:
        pdb_file (str): Path to the original PDB file.
        scwrl4_exe (str): Path to the SCWRL4 executable for side-chain prediction.
        mt_info (tuple): Mutation information, including chain, position, and mutation type.
        mt_index (int): Index indicating whether wild-type structure needs to fix (zero for fixing).
        logger: The logger object from the main for logging.
        temporary_log_files: A list which tracks temporary files created for cleaning at the end.

    Returns:
        dict: A dictionary containing:
            - success (bool): Whether the mutant complex was successfully created.
            - wt_complex (str): Path to the wild-type complex PDB file.
            - mt_complex (str): Path to the mutant complex PDB file (if mutation applied).
    """
    global logger_obj
    logger_obj = logger
    set_sequence_logger(logger)

    result = {"success": False, "wt_protein": "", "wt_complex": "", "mt_complex": ""}
    res_id2index = {}
    res_index2id = {}

    if not mt_index:
        result_wt_fix = fix_wildtype_complex_structure(
            pdb_file, scwrl4_exe, mt_info, temporary_log_files
        )
        result["success"] = result_wt_fix["success"]
        result["wt_protein"] = result_wt_fix["wt_protein"]
        result["wt_complex"] = result_wt_fix["wt_complex"]

    # If structure is already fixed update the basename of unfixed structure in pdb_file
    if pdb_file.endswith("_wt-fixed.pdb"):
        pdb_file = pdb_file[0:-13] + ".pdb"

    # Parse the fixed wild-type with protein always at top and mutate the structure
    wt_pdb_fixed = pdy.parsePDB(f"{pdb_file[0:-4]}_wt-fixed.pdb")
    if mt_index != 0:
        result["success"] = True
        result["wt_complex"] = f"{pdb_file[0:-4]}_wt-fixed.pdb"
        temporary_log_files.append(f"{pdb_file[0:-4]}_wt-fixed.pdb")

    res_id2index, res_index2id = create_residue_id_to_index_map(wt_pdb_fixed)
    renumber_residues(wt_pdb_fixed, res_id2index)
    logger_obj.debug("ater fixing wt res_index2id:" + str(res_index2id))

    # Select the protein and no_protein AtomGroup based on fixed wild-type structure
    wt_protein = wt_pdb_fixed.select(f"protein and chain {mt_info[0]}")
    wt_non_protein = wt_pdb_fixed.select(f"not (protein and chain {mt_info[0]})")

    # Generate mutation sequence and use SCWRL4 for side-chain prediction
    wt_protein_ca = wt_protein.select("protein and name CA")
    seq_file = f"{pdb_file[0:-4]}_{mt_info[0]}-{mt_info[1]}{mt_info[2]}{mt_info[3]}.seq"
    _, _ = generate_mutation_sequence(wt_protein_ca, seq_file, res_index2id, mt_info)
    temporary_log_files.append(seq_file)

    # Run SCWRL4 for the mutant structure
    scwrl4_cmd = f"{scwrl4_exe} -i {pdb_file[0:-4]}_protein.pdb -s {seq_file} -o {seq_file[0:-4]}_prot.pdb -h > {seq_file[0:-4]}_scwrl4.log"
    os.system(scwrl4_cmd)
    check_scwrl4_log(scwrl4_cmd, f"{seq_file[0:-4]}_scwrl4.log")
    temporary_log_files.append(f"{seq_file[0:-4]}_prot.pdb")
    temporary_log_files.append(f"{seq_file[0:-4]}_scwrl4.log")

    mt_res_index = res_id2index[(mt_info[2], mt_info[0])]
    wt_prot = wt_pdb_fixed.select(
        f"backbone (protein and not (chain {mt_res_index[1]} and resid {mt_res_index[0]}))"
    )
    mt_pdb = pdy.parsePDB(f"{seq_file[0:-4]}_prot.pdb")
    logger_obj.debug(f"{seq_file[0:-4]}_prot.pdb")

    mt_prot = mt_pdb.select(
        f"backbone (protein and not (chain {mt_res_index[1]} and resid {mt_res_index[0]}))"
    )

    # Final alignment of mutant and wild-type proteins
    complex_file_mt = f"{seq_file[0:-4]}.pdb"
    mt_success, mt_complex = align_protein_to_create_complex(
        wt_prot, mt_prot, mt_pdb, wt_non_protein, complex_file_mt
    )
    renumber_residues(mt_complex, res_index2id)
    pdy.writePDB(complex_file_mt, mt_complex)
    result["success"] = result["success"] and mt_success
    result["mt_complex"] = complex_file_mt
    temporary_log_files.append(complex_file_mt)

    return result
