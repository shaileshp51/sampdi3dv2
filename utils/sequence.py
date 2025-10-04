#!/usr/bin/env python
# coding: utf-8

import logging
import numpy as np

from collections import namedtuple
from enum import Enum

import prody as pdy

from utils.common import *

# Let's suppress the log messages from ProDy, to keep stdout clear.
pdy.confProDy(verbosity="none")

logger_obj = None


def set_logger(logger):
    global logger_obj
    logger_obj = logger


# Amino acid property classifications
class AAHydrophobicityClass(Enum):
    """Classification of amino acid hydrophobicity."""

    NEUTRAL = 0
    HYDROPHILIC = 1
    HYDROPHOBIC = 2


class AAChemicalProperty(Enum):
    """Chemical property categories of amino acids."""

    BASIC = 0
    AMIDE = 1
    ACIDIC = 2
    SULFUR = 3
    HYDROXYL = 4
    AROMATIC = 5
    ALIPHATIC = 6


class AASize(Enum):
    """Size categories of amino acids based on molecular volume."""

    VERY_SMALL = 0
    SMALL = 1
    MEDIUM = 2
    LARGE = 3
    VERY_LARGE = 4


class AAHydrogenBonding(Enum):
    """Hydrogen bonding capability of amino acids."""

    NONE = 0
    DONOR = 1
    DONOR_ACCEPTOR = 2
    ACCEPTOR = 3


class AAPolarity(Enum):
    """Polarity categories of amino acids."""

    NONPOLAR = 0
    POLAR_BASIC = 1
    POLAR_NEUTRAL = 2
    POLAR_ACIDIC = 3


# Named tuple to store amino acid properties
AAProperty = namedtuple(
    "AAProperty",
    "name_3_letter, name_1_letter, volume, hydrophobicity_value, "
    "rotamers, chemical_property, size, "
    "polarity, hydrogen_bonding, hydrophobicity_class",
)

# Dictionary to store amino acid properties indexed by 3-letter codes
aa3_map = {}

# Populate amino acid properties
aa3_map["ALA"] = AAProperty(
    name_3_letter="ALA",
    name_1_letter="A",
    volume=88.6,
    hydrophobicity_value=0,
    rotamers=1,
    chemical_property=AAChemicalProperty.ALIPHATIC,
    size=AASize.VERY_SMALL,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.NONE,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHOBIC,
)

aa3_map["CYS"] = AAProperty(
    name_3_letter="CYS",
    name_1_letter="C",
    volume=108.5,
    hydrophobicity_value=0.49,
    rotamers=3,
    chemical_property=AAChemicalProperty.SULFUR,
    size=AASize.SMALL,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.NONE,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHOBIC,
)

aa3_map["ASP"] = AAProperty(
    name_3_letter="ASP",
    name_1_letter="D",
    volume=111.1,
    hydrophobicity_value=2.95,
    rotamers=18,
    chemical_property=AAChemicalProperty.ACIDIC,
    size=AASize.SMALL,
    polarity=AAPolarity.POLAR_ACIDIC,
    hydrogen_bonding=AAHydrogenBonding.ACCEPTOR,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHILIC,
)

aa3_map["GLU"] = AAProperty(
    name_3_letter="GLU",
    name_1_letter="E",
    volume=138.4,
    hydrophobicity_value=1.64,
    rotamers=54,
    chemical_property=AAChemicalProperty.ACIDIC,
    size=AASize.MEDIUM,
    polarity=AAPolarity.POLAR_ACIDIC,
    hydrogen_bonding=AAHydrogenBonding.ACCEPTOR,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHILIC,
)

aa3_map["PHE"] = AAProperty(
    name_3_letter="PHE",
    name_1_letter="F",
    volume=189.9,
    hydrophobicity_value=-2.2,
    rotamers=18,
    chemical_property=AAChemicalProperty.AROMATIC,
    size=AASize.VERY_LARGE,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.NONE,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHOBIC,
)

aa3_map["GLY"] = AAProperty(
    name_3_letter="GLY",
    name_1_letter="G",
    volume=60.1,
    hydrophobicity_value=1.72,
    rotamers=1,
    chemical_property=AAChemicalProperty.ALIPHATIC,
    size=AASize.VERY_SMALL,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.NONE,
    hydrophobicity_class=AAHydrophobicityClass.NEUTRAL,
)

aa3_map["HIS"] = AAProperty(
    name_3_letter="HIS",
    name_1_letter="H",
    volume=153.2,
    hydrophobicity_value=4.76,
    rotamers=36,
    chemical_property=AAChemicalProperty.BASIC,
    size=AASize.MEDIUM,
    polarity=AAPolarity.POLAR_BASIC,
    hydrogen_bonding=AAHydrogenBonding.DONOR_ACCEPTOR,
    hydrophobicity_class=AAHydrophobicityClass.NEUTRAL,
)

aa3_map["ILE"] = AAProperty(
    name_3_letter="ILE",
    name_1_letter="I",
    volume=166.7,
    hydrophobicity_value=-1.56,
    rotamers=9,
    chemical_property=AAChemicalProperty.ALIPHATIC,
    size=AASize.LARGE,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.NONE,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHOBIC,
)

aa3_map["LYS"] = AAProperty(
    name_3_letter="LYS",
    name_1_letter="K",
    volume=168.6,
    hydrophobicity_value=5.39,
    rotamers=81,
    chemical_property=AAChemicalProperty.BASIC,
    size=AASize.LARGE,
    polarity=AAPolarity.POLAR_BASIC,
    hydrogen_bonding=AAHydrogenBonding.DONOR,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHILIC,
)

aa3_map["LEU"] = AAProperty(
    name_3_letter="LEU",
    name_1_letter="L",
    volume=166.7,
    hydrophobicity_value=-1.81,
    rotamers=9,
    chemical_property=AAChemicalProperty.ALIPHATIC,
    size=AASize.LARGE,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.NONE,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHOBIC,
)

aa3_map["MET"] = AAProperty(
    name_3_letter="MET",
    name_1_letter="M",
    volume=162.9,
    hydrophobicity_value=-0.76,
    rotamers=27,
    chemical_property=AAChemicalProperty.SULFUR,
    size=AASize.LARGE,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.NONE,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHOBIC,
)

aa3_map["ASN"] = AAProperty(
    name_3_letter="ASN",
    name_1_letter="N",
    volume=114.1,
    hydrophobicity_value=3.47,
    rotamers=36,
    chemical_property=AAChemicalProperty.AMIDE,
    size=AASize.SMALL,
    polarity=AAPolarity.POLAR_NEUTRAL,
    hydrogen_bonding=AAHydrogenBonding.DONOR_ACCEPTOR,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHILIC,
)

aa3_map["PRO"] = AAProperty(
    name_3_letter="PRO",
    name_1_letter="P",
    volume=112.7,
    hydrophobicity_value=-1.52,
    rotamers=2,
    chemical_property=AAChemicalProperty.ALIPHATIC,
    size=AASize.SMALL,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.NONE,
    hydrophobicity_class=AAHydrophobicityClass.NEUTRAL,
)

aa3_map["GLN"] = AAProperty(
    name_3_letter="GLN",
    name_1_letter="Q",
    volume=143.8,
    hydrophobicity_value=3.01,
    rotamers=108,
    chemical_property=AAChemicalProperty.AMIDE,
    size=AASize.MEDIUM,
    polarity=AAPolarity.POLAR_NEUTRAL,
    hydrogen_bonding=AAHydrogenBonding.DONOR_ACCEPTOR,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHILIC,
)

aa3_map["ARG"] = AAProperty(
    name_3_letter="ARG",
    name_1_letter="R",
    volume=173.4,
    hydrophobicity_value=3.71,
    rotamers=81,
    chemical_property=AAChemicalProperty.BASIC,
    size=AASize.LARGE,
    polarity=AAPolarity.POLAR_BASIC,
    hydrogen_bonding=AAHydrogenBonding.DONOR,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHILIC,
)

aa3_map["SER"] = AAProperty(
    name_3_letter="SER",
    name_1_letter="S",
    volume=89.0,
    hydrophobicity_value=1.83,
    rotamers=3,
    chemical_property=AAChemicalProperty.HYDROXYL,
    size=AASize.VERY_SMALL,
    polarity=AAPolarity.POLAR_NEUTRAL,
    hydrogen_bonding=AAHydrogenBonding.DONOR_ACCEPTOR,
    hydrophobicity_class=AAHydrophobicityClass.NEUTRAL,
)

aa3_map["THR"] = AAProperty(
    name_3_letter="THR",
    name_1_letter="T",
    volume=116.1,
    hydrophobicity_value=1.78,
    rotamers=3,
    chemical_property=AAChemicalProperty.HYDROXYL,
    size=AASize.SMALL,
    polarity=AAPolarity.POLAR_NEUTRAL,
    hydrogen_bonding=AAHydrogenBonding.DONOR_ACCEPTOR,
    hydrophobicity_class=AAHydrophobicityClass.NEUTRAL,
)

aa3_map["VAL"] = AAProperty(
    name_3_letter="VAL",
    name_1_letter="V",
    volume=140.0,
    hydrophobicity_value=-0.78,
    rotamers=3,
    chemical_property=AAChemicalProperty.ALIPHATIC,
    size=AASize.MEDIUM,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.NONE,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHOBIC,
)

aa3_map["TRP"] = AAProperty(
    name_3_letter="TRP",
    name_1_letter="W",
    volume=227.8,
    hydrophobicity_value=-0.38,
    rotamers=36,
    chemical_property=AAChemicalProperty.AROMATIC,
    size=AASize.VERY_LARGE,
    polarity=AAPolarity.NONPOLAR,
    hydrogen_bonding=AAHydrogenBonding.DONOR,
    hydrophobicity_class=AAHydrophobicityClass.HYDROPHOBIC,
)

aa3_map["TYR"] = AAProperty(
    name_3_letter="TYR",
    name_1_letter="Y",
    volume=193.6,
    hydrophobicity_value=-1.09,
    rotamers=18,
    chemical_property=AAChemicalProperty.AROMATIC,
    size=AASize.VERY_LARGE,
    polarity=AAPolarity.POLAR_NEUTRAL,
    hydrogen_bonding=AAHydrogenBonding.DONOR_ACCEPTOR,
    hydrophobicity_class=AAHydrophobicityClass.NEUTRAL,
)

# Create maps for conversions and labels
aa1_map = {aa.name_1_letter: aa for k, aa in aa3_map.items()}
aa3to1 = {aa.name_3_letter: aa.name_1_letter for k, aa in aa1_map.items()}
aa1to3 = {aa.name_1_letter: aa.name_3_letter for k, aa in aa1_map.items()}

aa3_label = {aa: f"{i+1}" for i, aa in enumerate(sorted(aa3to1.keys()))}
aa1_label = {aa3_map[aa].name_1_letter: f"{i}" for aa, i in aa3_label.items()}
aa1_mttype_label = {a: str(i + 1) for i, a in enumerate(aa1_label.keys())}

# Amino acid aliases and non-standard codes
aa3_aliases = {
    "HID": ("HIS", "Histidine protonated at delta-nitrogen: amber"),
    "HIE": ("HIS", "Histidine protonated at epsilon-nitrogen: amber"),
    "HIP": ("HIS", "Histidine protonated at both delta & epsilon-nitrogen: amber"),
    "CYX": ("CYS", "Cysteine involved in disulfide bonds"),
}

"""
aa3_nonstandard: A dictionary containing not-standandard amino acid (AA) and their parent AA.

Source:
    - Data derived from CHARMM-GUI PDB Manipulator: 
      "Various PDB Structural Modifications for Biomolecular Modeling and Simulation"
    - Supplementary Information: Table S2
    - Filter applied: #PDBs > 1000
"""
aa3_nonstandard = {
    "MSE": ("MET", "Selenomethionine"),
    "SEP": ("SER", "Phosphoserine"),
    "TPO": ("THR", "Phosphothreonine"),
    "SAH": ("CYS", "S-Adenosyl-L-homocysteine"),
    "CSO": ("CYS", "S-Hydroxycysteine"),
}

# Combined list of all valid 3-letter codes
aa3_combined = (
    list(aa3_map.keys()) + list(aa3_aliases.keys()) + list(aa3_nonstandard.keys())
)

# Example: To look up properties for alanine
# alanine_props = aa3_map["ALA"]


# aa3_map, aa3_aliases, aa3_nonstandard
def three_to_one(
    three_letter_code,
    aa3_map=aa3_map,
    aa3_aliases=aa3_aliases,
    aa3_nonstandard=aa3_nonstandard,
):
    """
    Convert a three-letter amino acid code to its corresponding one-letter code.

    Parameters:
        three_letter_code (str): The three-letter amino acid code.
        aa3_map (dict): Mapping of standard three-letter codes to amino acid properties.
        aa3_aliases (dict): Aliases mapping for non-standard codes.
        aa3_nonstandard (dict): Mapping of non-standard codes to standard equivalents.

    Returns:
        str: One-letter code for the amino acid. Returns 'X' if not found.
    """
    if three_letter_code in aa3_map:
        return aa3_map[three_letter_code].name_1_letter
    elif three_letter_code in aa3_aliases:
        alias = aa3_aliases[three_letter_code][0]
        return aa3_map[alias].name_1_letter
    elif three_letter_code in aa3_nonstandard:
        alias = aa3_nonstandard[three_letter_code][0]
        return aa3_map[alias].name_1_letter
    else:
        logger_obj.error("Unknown resname:", three_letter_code)
        return "X"


def parse_seqres(lines, seqres):
    """
    Parse SEQRES records from the PDB file header.

    SEQRES records describe the primary sequence of residues for each polymer chain.

    PDB Record Fixed Format:
        COLUMNS  DATA TYPE       FIELD           DEFINITION
        --------------------------------------------------------------
         1 -  6  Record name     "SEQRES"        Record identifier.
         8 - 10  Integer         serNum          Serial number of the record.
        12       Character       chainID         Chain identifier.
        14 - 17  Integer         numRes          Number of residues in chain.
        20 - 70  Residue name    resName         List of residue names.

    Example:
        SEQRES   1 A    8  MET GLU ASN LYS GLY ARG VAL ILE

    Parameters:
        lines (list of str): Lines containing SEQRES records.
        seqres (dict): Dictionary to store parsed sequences indexed by chain ID.

    Returns:
        None: Updates the `seqres` dictionary in-place.
    """
    for line in lines:
        line_stripped = line.strip()
        if len(line_stripped) < 12:
            continue

        sr_no = int(line_stripped[7:10].strip())
        chain = line_stripped[11].strip()
        res_count = int(line_stripped[13:17].strip())
        resnames = seqres.setdefault(chain, []) if sr_no == 1 else seqres.get(chain, [])

        # ensure `i+3 <= len(line_stripped)`, so range for i ends at len-3+1
        for i in range(19, len(line_stripped) - 2, 4):
            if len(resnames) >= res_count:
                break
            resnames.append(line_stripped[i : i + 3].strip())

        seqres[chain] = resnames


def parse_modres(lines, modres):
    """
    Parse MODRES records from the PDB file header.

    MODRES records describe chemically modified residues.

    PDB Record Fixed Format:
        COLUMNS  DATA TYPE       FIELD           DEFINITION
        --------------------------------------------------------------
         1 -  6  Record name     "MODRES"        Record identifier.
         8 - 11  IDcode          idCode          PDB ID code.
        13 - 15  Residue name    resName         Residue name.
        17       Character       chainID         Chain identifier.
        19 - 22  Integer         seqNum          Sequence number.
        23       AChar           iCode           Insertion code.
        25 - 27  Residue name    stdRes          Standard residue name.
        30 - 70  String          comment         Description of modification.

    Example:
        MODRES 3FEN C   502    HEM   HEME  HEME-IRON COMPLEX

    Parameters:
        lines (list of str): Lines containing MODRES records.
        modres (dict): Dictionary to store modified residues indexed by chain ID.

    Returns:
        None: Updates the `modres` dictionary in-place.
    """
    for line in lines:
        if len(line) < 27:
            continue

        res_name = line[12:15].strip()
        chain = line[16].strip()
        seq_num = int(line[18:22].strip())
        i_code = line[22:23].strip()
        std_res_name = line[24:27].strip()
        comment = line[29:].strip()

        modres.setdefault(chain, []).append(
            (seq_num, i_code, res_name, std_res_name, comment)
        )


def parse_remark_465(lines, remark465):
    """
    Parse REMARK 465 records for missing residues.

    REMARK 465 describes residues that are not observed in the structure.

    PDB Record Fixed Format:
        COLUMNS  DATA TYPE       FIELD           DEFINITION
        --------------------------------------------------------------
         1 - 10  String          "REMARK 465"    Record identifier.
        16 - 18  Residue name    resName         Residue name.
        20       Character       chainID         Chain identifier.
        22 - 26  Integer         seqNum          Sequence number.
        27       AChar           iCode           Insertion code.

    Example:
        REMARK 465   HIS A  101

    Parameters:
        lines (list of str): Lines containing REMARK 465 records.
        remark465 (dict): Dictionary to store missing residues indexed by chain ID.

    Returns:
        None: Updates the `remark465` dictionary in-place.
    """
    for line in lines:
        if not line.startswith("REMARK 465") or len(line) < 27:
            continue

        try:
            res_name = line[15:18].strip()
            chain = line[19:20].strip()
            seq_num = int(line[22:26].strip())
            i_code = line[26:27].strip()
            remark465.setdefault(chain, []).append((seq_num, i_code, res_name))
        except ValueError:
            pass


def parse_seqadv(lines, seqadv):
    """
    Parse SEQADV records from the PDB header.

    SEQADV records describe sequence discrepancies between the PDB entry and the reference sequence.

    PDB Record Fixed Format:
        COLUMNS  DATA TYPE       FIELD           DEFINITION
        --------------------------------------------------------------
         1 -  6  Record name     "SEQADV"        Record identifier.
         8 - 10  Residue name    resName         Residue name.
        12       Character       chainID         Chain identifier.
        14 - 17  Integer         seqNum          Sequence number.
        18       AChar           iCode           Insertion code.
        20 - 23  String          database        Database name.
        25 - 31  String          dbAccession     Database accession code.
        33 - 35  Residue name    dbRes           Database residue name.
        37 - 40  Integer         dbSeq           Database sequence number.
        42 - 70  String          conflict        Description of the conflict.

    Example:
        SEQADV  ARG A   41          UNP Q7Z7L4    MET   45          SEQUENCE MISMATCH

    Parameters:
        lines (list of str): Lines containing SEQADV records.
        seqadv (dict): Dictionary to store discrepancies indexed by chain ID and conflict.

    Returns:
        None: Updates the `seqadv` dictionary in-place.
    """
    for line in lines:
        if len(line) < 28:
            continue

        res_name = line[12:15].strip()
        chain = line[16].strip()
        seq_num = int(line[18:22].strip())
        i_code = line[22].strip()
        db = line[24:28].strip()
        db_accn = line[29:38].strip()
        db_res = line[39:42].strip()
        db_seq = line[43:48].strip()
        conflict = line[49:].strip()

        seqadv.setdefault(chain, {}).setdefault(conflict, []).append(
            (seq_num, i_code, res_name, db, db_accn, db_res, db_seq)
        )


def parse_pdb_header(filename):
    """parse REMARK 465/SEQRES/SEQADV/MODRES records from the PDB header

    return a dictionary with lowercase record (spaces deleled) as keys.
    in case pdb file not found None is returned
    """
    lines_remark_465, remark465 = [], {}
    lines_seqadv, seqadv = [], {}
    lines_seqres, seqres = [], {}
    lines_modres, modres = [], {}
    header = {}
    with open(filename, "r") as pdb_file:
        for line in pdb_file:
            if line.startswith("REMARK 465"):
                lines_remark_465.append(line)
            elif line.startswith("SEQRES"):
                lines_seqres.append(line)
            elif line.startswith("SEQADV"):
                lines_seqadv.append(line)
            elif line.startswith("MODRES"):
                lines_modres.append(line)
        if len(lines_remark_465):
            parse_remark_465(lines_remark_465, remark465)
        if len(lines_seqres):
            parse_seqres(lines_seqres, seqres)
        if len(lines_seqadv):
            parse_seqadv(lines_seqadv, seqadv)
        if len(lines_modres):
            parse_modres(lines_modres, modres)
        header["missing"] = remark465
        header["seqres"] = seqres
        header["seqadv"] = seqadv
        header["modres"] = modres
        return header
    return None


def compress_residue_ranges(residue_list):
    """
    Compress a list of residue numbers into ranges.

    This function takes a list of residue numbers, sorts them, and returns a
    compressed string where consecutive residue numbers are represented as ranges.

    Parameters:
    residue_list (list of int): A list of residue numbers to be compressed.

    Returns:
    str: A string representing the compressed residue ranges.
          Example: "1-5, 8, 10-12".
    """
    if not residue_list:
        return ""

    residue_list.sort()  # Ensure the list is sorted
    ranges = []
    start = residue_list[0]
    prev = residue_list[0]

    for i in range(1, len(residue_list)):
        current = residue_list[i]
        if current == prev + 1:  # Consecutive residue
            prev = current
        else:
            # End of a range, append the range or single residue
            if start == prev:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{prev}")
            start = current
            prev = current

    # Append the last range or single residue
    if start == prev:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{prev}")

    return ", ".join(ranges)


def validate_aa_code(aa):
    """
    Validate an amino acid code (1-letter or 3-letter).

    This function checks if the given amino acid code exists in either the 1-letter
    or 3-letter code dictionaries.

    Parameters:
    aa (str): Amino acid code (either 1-letter or 3-letter).

    Returns:
    int: The corresponding code if valid, else 0.
    """
    code = 0
    if len(aa) == 1:
        code = aa1_label.get(aa, 0)  # Check 1-letter code
    elif len(aa) == 3:
        code = aa3_label.get(aa, 0)  # Check 3-letter code
    return code


def net_flexibility(aa1_map, wild, mutant):
    """
    Calculate the difference in rotamers between wild-type and mutant amino acids.

    This function computes the change in rotamers (log ratio) between the wild-type
    and mutant amino acid using the number of rotamers from the provided map. The number
    of rotamers are taken from Dunbrack rotamer library.
    Reference: Shapovalov, M.V. and Dunbrack, R.L., 2011. A smoothed backbone-dependent
                rotamer library for proteins derived from adaptive kernel density estimates
                and regressions. Structure, 19(6), pp.844-858. DOI: 10.1016/j.str.2011.03.019
                SuppInfo: Table S1.

    Parameters:
    aa1_map (dict): A dictionary mapping amino acid 1-letter codes to their properties.
    wild (str): The 1-letter code of the wild-type amino acid.
    mutant (str): The 1-letter code of the mutant amino acid.

    Returns:
    float: The difference in rotamers (log scale).
    """
    wt_flex = aa1_map[wild].rotamers if wild in aa1_map else 0
    mt_flex = aa1_map[mutant].rotamers if mutant in aa1_map else 0
    return round(np.log(int(mt_flex)) - np.log(int(wt_flex)), 3)


def net_volume(aa1_map, wild, mutant):
    """
    Calculate the difference in volume between wild-type and mutant amino acids.

    This function calculates the difference in molecular volume between the wild-type
    and mutant amino acids using their volume properties from the map.

    Parameters:
    aa1_map (dict): A dictionary mapping amino acid 1-letter codes to their properties.
    wild (str): The 1-letter code of the wild-type amino acid.
    mutant (str): The 1-letter code of the mutant amino acid.

    Returns:
    str: The difference in volume formatted to one decimal place.
    """
    wt_volm = aa1_map[wild].volume if wild in aa1_map else 0
    mt_volm = aa1_map[mutant].volume if mutant in aa1_map else 0
    return "{:0.1f}".format(float(mt_volm) - float(wt_volm))


def net_hydrophobicity(aa1_map, wild, mutant):
    """
    Calculate the difference in hydrophobicity between wild-type and mutant amino acids.

    This function calculates the change in hydrophobicity (ddG) between the wild-type
    and mutant amino acids using hydrophobicity values from the provided map.
    The hydrophobicity values used are based on the study:
    - Reference: https://doi.org/10.1073/pnas.1103979108

    Parameters:
    aa1_map (dict): A dictionary mapping amino acid 1-letter codes to their properties.
    wild (str): The 1-letter code of the wild-type amino acid.
    mutant (str): The 1-letter code of the mutant amino acid.

    Returns:
    str: The difference in hydrophobicity formatted to one decimal place.
    """
    wt_hydphob = aa1_map[wild].hydrophobicity_value if wild in aa1_map else 0
    mt_hydphob = aa1_map[mutant].hydrophobicity_value if mutant in aa1_map else 0
    return "{:0.1f}".format(float(mt_hydphob) - float(wt_hydphob))


def mutation_chemical(aa1_map, wild, mutant):
    """
    Encode the difference in chemical properties between wild-type and mutant amino acids.

    This function calculates a label encoding for the chemical property difference
    between the wild-type and mutant amino acids based on their chemical property values.

    Parameters:
    aa1_map (dict): A dictionary mapping amino acid 1-letter codes to their properties.
    wild (str): The 1-letter code of the wild-type amino acid.
    mutant (str): The 1-letter code of the mutant amino acid.

    Returns:
    int: An encoded label representing the chemical property difference.
    """
    wt_chem = aa1_map[wild].chemical_property.value
    mt_chem = aa1_map[mutant].chemical_property.value
    n_chem_props = len(AAChemicalProperty)
    label_chem = wt_chem * n_chem_props + mt_chem
    return label_chem


def mutation_hydrophobicity(aa1_map, wild, mutant):
    """
    Encode the difference in hydrophobicity class between wild-type and mutant amino acids.

    This function calculates a label encoding for the hydrophobicity class difference
    between the wild-type and mutant amino acids.

    Parameters:
    aa1_map (dict): A dictionary mapping amino acid 1-letter codes to their properties.
    wild (str): The 1-letter code of the wild-type amino acid.
    mutant (str): The 1-letter code of the mutant amino acid.

    Returns:
    int: An encoded label representing the hydrophobicity class difference.
    """
    wt_hydrphbcls = aa1_map[wild].hydrophobicity_class.value
    mt_hydrphbcls = aa1_map[mutant].hydrophobicity_class.value
    n_hydrphbcls = len(AAHydrophobicityClass)
    label_indx = wt_hydrphbcls * n_hydrphbcls + mt_hydrphbcls
    return label_indx


def mutation_polarity(aa1_map, wild, mutant):
    """
    Encode the difference in polarity between wild-type and mutant amino acids.

    This function calculates a label encoding for the polarity difference between
    the wild-type and mutant amino acids.

    Parameters:
    aa1_map (dict): A dictionary mapping amino acid 1-letter codes to their properties.
    wild (str): The 1-letter code of the wild-type amino acid.
    mutant (str): The 1-letter code of the mutant amino acid.

    Returns:
    int: An encoded label representing the polarity difference.
    """
    wt_pol = aa1_map[wild].polarity.value
    mt_pol = aa1_map[mutant].polarity.value
    n_polcls = len(AAPolarity)
    label_polarity = wt_pol * n_polcls + mt_pol
    return label_polarity


def mutation_size(aa1_map, wild, mutant):
    """
    Encode the difference in size between wild-type and mutant amino acids.

    This function calculates a label encoding for the size difference between
    the wild-type and mutant amino acids.

    Parameters:
    aa1_map (dict): A dictionary mapping amino acid 1-letter codes to their properties.
    wild (str): The 1-letter code of the wild-type amino acid.
    mutant (str): The 1-letter code of the mutant amino acid.

    Returns:
    int: An encoded label representing the size difference.
    """
    wt_size = aa1_map[wild].size.value
    mt_size = aa1_map[mutant].size.value
    n_size = len(AASize)
    label_size = wt_size * n_size + mt_size
    return label_size


def mutation_hbonds(aa1_map, wild, mutant):
    """
    Encode the difference in hydrogen bonding between wild-type and mutant amino acids.

    This function calculates a label encoding for the hydrogen bonding difference
    between the wild-type and mutant amino acids.

    Parameters:
    aa1_map (dict): A dictionary mapping amino acid 1-letter codes to their properties.
    wild (str): The 1-letter code of the wild-type amino acid.
    mutant (str): The 1-letter code of the mutant amino acid.

    Returns:
    int: An encoded label representing the hydrogen bonding difference.
    """
    wt_hb = aa1_map[wild].hydrogen_bonding.value
    mt_hb = aa1_map[mutant].hydrogen_bonding.value
    n_hbonds = len(AAHydrogenBonding)
    label_hbonds = wt_hb * n_hbonds + mt_hb
    return label_hbonds


def mutation_type(aa1_label, wild, mutant):
    wt_lbl = int(aa1_label.get(wild, 0))
    mt_lbl = int(aa1_label.get(mutant, 0))
    if wild == mutant:
        return (wt_lbl, mt_lbl, 0)
    mutation_lbl = wt_lbl * 20 + mt_lbl + 1
    return mutation_lbl


# def get_protein_polymer_sequence(pdb_file, outfileprefix):
#     """
#     Parse the PDB file to extract the protein sequence while ignoring expression tags.

#     Args:
#         pdb_file (str): Path to the PDB file.
#         outfileprefix (str): Prefix for the output files.

#     Returns:
#         tuple: A tuple containing:
#             - resmap_list (list): A list of residue mappings with detailed information.
#             - fasta_file (str): Path to the generated FASTA file containing the protein sequence.

#     Raises:
#         SystemExit: If no backbone atoms are found in the PDB file.
#     """
#     # Define the residue selection string for proteins
#     protein_selstr = "(resname {})".format(" ".join(aa3_combined))

#     # Parse the PDB structure
#     structure = pdy.parsePDB(pdb_file)
#     bb_atoms = structure.select(protein_selstr + " and name N CA C O")
#     if bb_atoms is None:
#         logger_obj.error(f"File: {pdb_file} - No backbone atoms found.\n")
#         sys.exit()

#     # Initialize the header and chain iterator
#     chains = bb_atoms.getHierView().iterChains()
#     header = parse_pdb_header(pdb_file)

#     for chain in chains:
#         chain_id = chain.getChid()
#         residues = list(chain.iterResidues())
#         structural_residues = {}
#         seqres = {}

#         # Process residues and their backbone atoms
#         for residue in residues:
#             residue_backbone = residue.select("name N CA C O")
#             if residue_backbone is not None:
#                 for atom in residue_backbone:
#                     reskey = (
#                         atom.getResnum(),
#                         atom.getIcode().strip(),
#                         atom.getResname(),
#                     )
#                     if reskey not in structural_residues:
#                         structural_residues[reskey] = [atom.getName()]
#                     else:
#                         structural_residues[reskey].append(atom.getName())

#         # Handle missing residues
#         remark_residues = []
#         if header and "missing" in header and chain_id in header["missing"]:
#             remark_residues = header["missing"][chain_id]

#         logger_obj.debug("structural_residues:" + str(structural_residues))
#         logger_obj.debug("remark_residues:" + str(remark_residues))
#         # Combine residues and sort by residue number
#         all_residues = {
#             (r[0], r[1]): r[2]
#             for r in list(structural_residues.keys()) + remark_residues
#         }
#         logger_obj.debug("all_residues:" + str(all_residues))
#         all_residues = {
#             r[0]: r[1] for r in sorted(all_residues.items(), key=lambda x: x[0][0])
#         }
#         logger_obj.debug(
#             "length, sorted all_residues:"
#             + str(len(all_residues))
#             + " "
#             + str(all_residues)
#         )
#         logger_obj.debug(
#             f"length, header['seqres']['{chain_id}']:"
#             + str(len(header["seqres"][chain_id]))
#             + " "
#             + str(header["seqres"][chain_id])
#         )

#         # Resolve discrepancies with SEQRES and expression tags
#         if (
#             header
#             and "seqres" in header
#             and len(header["seqres"][chain_id]) == len(all_residues)
#         ):
#             seqres[chain_id] = {}
#             for srn, rinfo in zip(header["seqres"][chain_id], all_residues.items()):
#                 seqres[chain_id][rinfo[0]] = srn
#                 if srn != rinfo[1]:
#                     logger_obj.debug("conflict: " + str(rinfo) + " seq: " + str(srn))

#         if header and "seqadv" in header and chain_id in header["seqadv"]:
#             expression_tags = header["seqadv"][chain_id].get("EXPRESSION TAG", [])
#             for tag in expression_tags:
#                 if (tag[0], tag[1]) in all_residues:
#                     del all_residues[(tag[0], tag[1])]
#                     logger_obj.debug(
#                         f"chain: {chain_id}, deleting expression tag: {tag}"
#                     )

#         # Detect gaps in the sequence
#         gaps = []
#         all_residues_keys = list(all_residues.keys())
#         for i in range(1, len(all_residues)):
#             k = all_residues_keys[i]
#             k_prev = all_residues_keys[i - 1]
#             if int(k[0]) - int(k_prev[0]) > 1:
#                 gaps.append(f"{k_prev[0] + 1} to {k[0] - 1}")
#         logger_obj.debug(f"Gaps detected: {gaps}")

#         # Convert residues to sequences
#         sequence_one_letter = [three_to_one(res[1]) for res in all_residues.items()]

#         # Output sequences and gaps
#         logger_obj.debug(
#             f"SEQRES (Chain {chain_id}):\n"
#             + "".join([three_to_one(r) for r in header["seqres"][chain_id]]),
#         )
#         logger_obj.debug(
#             f"ATOMS+MISSING-EXPRESSION_TAG (Chain {chain_id}):\n"
#             + "".join(sequence_one_letter)
#         )

#         # Create residue mapping list
#         resmap_list = [
#             (i + 1, e[0][0], e[0][1], chain_id, e[1], three_to_one(e[1]))
#             for i, e in enumerate(all_residues.items())
#         ]

#         # Save residue mapping to CSV
#         resmap_file = f"{outfileprefix}_chain{chain_id}_resmap.csv"
#         list_to_csv(
#             resmap_list,
#             fieldnames=[
#                 "SeqResID",
#                 "PdbResID",
#                 "PdbICode",
#                 "Chain",
#                 "ResName3",
#                 "ResName1",
#             ],
#             filename=resmap_file,
#         )

#         # Save protein sequence to FASTA file
#         fasta_file = f"{outfileprefix}_chain{chain_id}.fasta"
#         with open(fasta_file, "w") as seq_file:
#             seq_file.write(
#                 f">{outfileprefix}|chain {chain_id}|protein-sequence excluding expression tag\n"
#             )
#             seq_file.write("".join(sequence_one_letter))
#             seq_file.write("\n")
#         logger_obj.info(f"protein sequence:\n{''.join(sequence_one_letter)}")

#     return resmap_list, fasta_file, resmap_file


# ... (rest of the code remains the same)

# def get_protein_polymer_sequence(pdb_file, outfileprefix, use_coord_as_ground_truth=False):
#     """
#     Parse the PDB file to extract the protein sequence while ignoring expression tags.

#     Args:
#         pdb_file (str): Path to the PDB file.
#         outfileprefix (str): Prefix for the output files.
#         use_coord_as_ground_truth (bool): If True, forces the use of the
#                                           coordinate-derived sequence (ATOMS + REMARK 465)
#                                           as the ground truth, ignoring SEQRES/SEQADV.

#     Returns:
#         tuple: A tuple containing:
#             - resmap_list (list): A list of residue mappings with detailed information.
#             - fasta_file (str): Path to the generated FASTA file containing the protein sequence.
#             - resmap_file (str): Path to the generated residue map CSV file.

#     Raises:
#         SystemExit: If no backbone atoms are found in the PDB file.
#     """
#     # Define the residue selection string for proteins
#     protein_selstr = "(resname {})".format(" ".join(aa3_combined))

#     # Parse the PDB structure
#     structure = pdy.parsePDB(pdb_file)
#     bb_atoms = structure.select(protein_selstr + " and name N CA C O")
#     if bb_atoms is None:
#         logger_obj.error(f"File: {pdb_file} - No backbone atoms found.\n")
#         # Use a non-zero exit code for failure, as sys is not imported
#         # sys.exit() 
#         # Since 'sys' is not imported, let's raise a RuntimeError instead of SystemExit 
#         # to stop execution in a function.
#         raise RuntimeError(f"File: {pdb_file} - No backbone atoms found.")

#     # Initialize the header and chain iterator
#     chains = bb_atoms.getHierView().iterChains()
    
#     # 1. Parse the PDB header
#     header = parse_pdb_header(pdb_file)

#     for chain in chains:
#         chain_id = chain.getChid()
#         residues = list(chain.iterResidues())
#         structural_residues = {}
#         # seqres is no longer needed here, but kept for clarity: seqres = {}

#         # Process residues and their backbone atoms (Coordinate Ground Truth)
#         for residue in residues:
#             residue_backbone = residue.select("name N CA C O")
#             if residue_backbone is not None:
#                 for atom in residue_backbone:
#                     # Key: (PdbResID, PdbICode, ResName3)
#                     reskey = (
#                         atom.getResnum(),
#                         atom.getIcode().strip(),
#                         atom.getResname(),
#                     )
#                     # We only care that the residue is present, not necessarily all atoms
#                     if reskey not in structural_residues:
#                         structural_residues[reskey] = [atom.getName()]
#                     else:
#                         structural_residues[reskey].append(atom.getName())

#         # 2. Get missing residues from REMARK 465
#         remark_residues = []
#         if header and "missing" in header and chain_id in header["missing"]:
#             remark_residues = header["missing"][chain_id]

#         logger_obj.debug("structural_residues:" + str(structural_residues))
#         logger_obj.debug("remark_residues:" + str(remark_residues))
        
#         # 3. Combine structural atoms and missing residues (Our *Primary* Sequence List)
#         all_residues_with_icode = {
#             (r[0], r[1]): r[2]
#             for r in list(structural_residues.keys()) + remark_residues
#         }
        
#         # Sort by residue number, removing ICode from the key for a simple sequence map
#         all_residues = {
#             r[0]: r[1] 
#             for r in sorted(all_residues_with_icode.items(), key=lambda x: x[0][0])
#         }

#         # The header must be considered missing or incomplete if:
#         # a) header is None
#         # b) 'seqres' key is missing from the header
#         # c) the chain_id isn't in 'seqres'
#         # d) the user explicitly forces coordinate-as-ground-truth.
        
#         is_header_missing = (
#             header is None 
#             or "seqres" not in header 
#             or chain_id not in header["seqres"]
#         )

#         # 4. Sequence Reconciliation Logic
#         if not is_header_missing and not use_coord_as_ground_truth:
#             # ORIGINAL LOGIC: The PDB contains SEQRES and we want to use it
#             logger_obj.debug("Using SEQRES-based reconciliation.")
            
#             # Resolve discrepancies with SEQRES and expression tags
#             if len(header["seqres"][chain_id]) == len(all_residues):
#                 # Only log conflicts if the lengths match, otherwise the sequence 
#                 # from coordinates is probably better.
#                 seqres_map = {}
#                 for srn, rinfo in zip(header["seqres"][chain_id], all_residues.items()):
#                     seqres_map[rinfo[0]] = srn
#                     if srn != rinfo[1]:
#                         logger_obj.debug("conflict: " + str(rinfo) + " seq: " + str(srn))

#             if "seqadv" in header and chain_id in header["seqadv"]:
#                 # Remove residues marked as 'EXPRESSION TAG' in SEQADV
#                 expression_tags = header["seqadv"][chain_id].get("EXPRESSION TAG", [])
#                 for tag in expression_tags:
#                     # Note: all_residues uses PdbResID as key (int), but this logic 
#                     # requires the (PdbResID, PdbICode) tuple key to be present.
#                     # Since we simplified all_residues to map PdbResID -> ResName3, 
#                     # we must check against the *original* all_residues_with_icode structure
#                     # and then remove from the sorted map.
#                     tag_key = (tag[0], tag[1]) # (PdbResID, PdbICode)
#                     if tag_key in all_residues_with_icode:
#                         # Remove from the dictionary used for the final sequence
#                         del all_residues[tag[0]] # all_residues is keyed by PdbResID (int)
#                         logger_obj.debug(
#                             f"chain: {chain_id}, deleting expression tag: {tag}"
#                         )

#         # ELSE (is_header_missing or use_coord_as_ground_truth is True):
#         # We proceed with the sequence from ATOMS + REMARK 465 as the ground truth
#         # which is already captured in `all_residues`.
#         else:
#             logger_obj.debug("SEQRES/Header is missing/ignored. Using coordinate-derived sequence as ground truth.")
#             # Remove the now-defunct SEQRES-based comparison and tag removal logic.
#             # We assume the coordinate-derived list is correct (as per prompt).
            
#         # Detect gaps in the sequence (still relevant for the final sequence)
#         gaps = []
#         print("all_residues:", all_residues)
#         all_residues_keys = list(all_residues.keys())
#         for i in range(1, len(all_residues)):
#             k = all_residues_keys[i]
#             k_prev = all_residues_keys[i - 1]
#             print(f"Comparing k: {k} with k_prev: {k_prev}")
#             if int(k[0]) - int(k_prev[0]) > 1:
#                 gaps.append(f"{k_prev[0] + 1} to {k[0] - 1}")
#         logger_obj.debug(f"Gaps detected: {gaps}")
        
#         # Convert residues to sequences
#         # all_residues is now a dictionary of {PdbResID: ResName3}
#         # The .items() gives a sorted list of (PdbResID, ResName3) tuples
#         sequence_one_letter = [three_to_one(res[1]) for res in all_residues.items()]

#         # Output sequences (updated to reflect the final sequence source)
#         if not is_header_missing:
#              logger_obj.debug(
#                 f"SEQRES (Chain {chain_id}):\n"
#                 + "".join([three_to_one(r) for r in header["seqres"][chain_id]]),
#             )
#         logger_obj.info(
#             f"Final Sequence (Chain {chain_id}):\n"
#             + "".join(sequence_one_letter)
#         )

#         # Create residue mapping list
#         # Re-derive resmap_list using the keys/values from the final all_residues map.
#         # We need the ICode back, so we use the original `all_residues_with_icode` 
#         # filtered down to the final PdbResIDs in `all_residues`.
#         final_pdb_resids = set(all_residues.keys())
#         final_residue_mapping = [
#             e
#             for e in list(structural_residues.keys()) + remark_residues
#             if e[0] in final_pdb_resids
#         ]
        
#         # Sort and create final resmap_list
#         resmap_list = [
#             (i + 1, e[0], e[1], chain_id, e[2], three_to_one(e[2]))
#             for i, e in enumerate(sorted(final_residue_mapping, key=lambda x: x[0]))
#         ]

#         # Save residue mapping to CSV
#         resmap_file = f"{outfileprefix}_chain{chain_id}_resmap.csv"
#         # The list_to_csv function is assumed to be defined in utils.common
#         # For demonstration, we assume it works with the provided fieldnames.
#         list_to_csv(
#             resmap_list,
#             fieldnames=[
#                 "SeqResID",
#                 "PdbResID",
#                 "PdbICode",
#                 "Chain",
#                 "ResName3",
#                 "ResName1",
#             ],
#             filename=resmap_file,
#         )

#         # Save protein sequence to FASTA file
#         fasta_file = f"{outfileprefix}_chain{chain_id}.fasta"
#         with open(fasta_file, "w") as seq_file:
#             seq_file.write(
#                 f">{outfileprefix}|chain {chain_id}|final-sequence\n"
#             )
#             seq_file.write("".join(sequence_one_letter))
#             seq_file.write("\n")
            
#         return resmap_list, fasta_file, resmap_file
    

def get_protein_polymer_sequence(pdb_file, outfileprefix, use_coord_as_ground_truth=False):
    """
    Parse the PDB file to extract the protein sequence. Prioritizes coordinate
    (ATOM) data as ground truth if header data (SEQRES) is missing or overridden.

    Args:
        pdb_file (str): Path to the PDB file.
        outfileprefix (str): Prefix for the output files.
        use_coord_as_ground_truth (bool): If True, forces the use of the
                                          coordinate-derived sequence (ATOMS + REMARK 465)
                                          as the ground truth, ignoring SEQRES/SEQADV.

    Returns:
        tuple: A tuple containing:
            - resmap_list (list): A list of residue mappings with detailed information.
            - fasta_file (str): Path to the generated FASTA file containing the protein sequence.
            - resmap_file (str): Path to the generated residue map CSV file.

    Raises:
        RuntimeError: If no backbone atoms are found in the PDB file.
    """
    # Define the residue selection string for proteins
    protein_selstr = "(resname {})".format(" ".join(aa3_combined))

    # Parse the PDB structure
    structure = pdy.parsePDB(pdb_file)
    bb_atoms = structure.select(protein_selstr + " and name N CA C O")
    
    if bb_atoms is None:
        if logger_obj:
            logger_obj.error(f"File: {pdb_file} - No backbone atoms found.")
        raise RuntimeError(f"File: {pdb_file} - No backbone atoms found.")

    # 1. Parse the PDB header
    header = parse_pdb_header(pdb_file)

    # Initialize return values outside chain loop (assuming one chain or combining logic later)
    resmap_list = []
    fasta_file = f"{outfileprefix}_chainX.fasta" # Default placeholders
    resmap_file = f"{outfileprefix}_chainX_resmap.csv"

    for chain in bb_atoms.getHierView().iterChains():
        chain_id = chain.getChid()
        residues = list(chain.iterResidues())
        # structural_residues: { (PdbResID, PdbICode, ResName3): [AtomNames] }
        structural_residues = {} 

        # A. Process residues and their backbone atoms (Coordinate Ground Truth)
        for residue in residues:
            residue_backbone = residue.select("name N CA C O")
            if residue_backbone is not None:
                for atom in residue_backbone:
                    # Key: (PdbResID, PdbICode, ResName3)
                    reskey = (
                        atom.getResnum(),
                        atom.getIcode().strip(),
                        atom.getResname(),
                    )
                    if reskey not in structural_residues:
                        structural_residues[reskey] = [atom.getName()]
                    else:
                        structural_residues[reskey].append(atom.getName())

        # B. Get missing residues from REMARK 465
        # remark_residues: list of (PdbResID, PdbICode, ResName3) tuples
        remark_residues = [] 
        if header and "missing" in header and chain_id in header["missing"]:
            remark_residues = header["missing"][chain_id]

        # C. Initial Combined Residues (ATOMS + REMARK 465)
        # This list retains the ICode for the final resmap_list creation.
        initial_combined_residues = list(structural_residues.keys()) + remark_residues
        
        # D. all_residues: {PdbResID (int): ResName3 (str)} used for sequence/gap logic
        all_residues = { 
            r[0]: r[2] for r in sorted(initial_combined_residues, key=lambda x: x[0])
        }
        
        # 2. Sequence Reconciliation Logic
        is_header_missing = (
            header is None 
            or "seqres" not in header 
            or chain_id not in header["seqres"]
        )
        
        # The list of tuples that will define the final sequence and resmap_list
        final_residue_tuples = [] 
        
        if not is_header_missing and not use_coord_as_ground_truth:
            # --- HEADER EXISTS: Run SEQRES Reconciliation & Tag Removal ---
            if logger_obj:
                logger_obj.debug("Using SEQRES-based reconciliation.")
            
            # The all_residues map already has the combined PDB-derived sequence.
            # We now apply SEQADV filtering, which modifies all_residues in place.
            if "seqadv" in header and chain_id in header["seqadv"]:
                expression_tags = header["seqadv"][chain_id].get("EXPRESSION TAG", [])
                for tag in expression_tags:
                    tag_res_id = tag[0] # PdbResID (int)
                    if tag_res_id in all_residues:
                        del all_residues[tag_res_id]
                        if logger_obj:
                            logger_obj.debug(f"chain: {chain_id}, deleting expression tag: {tag}")
            
            # After filtering, rebuild the final list of (PdbResID, ICode, ResName3) tuples 
            # by matching the PdbResID keys remaining in all_residues against the original
            # initial_combined_residues list.
            final_pdb_resids = set(all_residues.keys())
            
            # Reconstruct the list of tuples for the final map
            final_residue_tuples = [
                r for r in initial_combined_residues 
                if r[0] in final_pdb_resids
            ]
        
        else:
            # --- HEADER MISSING/IGNORED: Use Coordinate Ground Truth ---
            if logger_obj:
                logger_obj.debug("SEQRES/Header is missing/ignored. Using coordinate-derived sequence as ground truth.")
            
            # The initial combined list is the final ground truth
            final_residue_tuples = initial_combined_residues


        # Sort the final list
        final_residue_tuples.sort(key=lambda x: x[0])
        
        # 3. Gap Detection (FIXED: Uses integer keys directly)
        gaps = []
        # Get keys from the *final* sequence map to check for gaps
        final_res_ids = [r[0] for r in final_residue_tuples] 
        
        for i in range(1, len(final_res_ids)):
            k = final_res_ids[i]
            k_prev = final_res_ids[i - 1]
            
            # k and k_prev are guaranteed to be PdbResID integers
            if k - k_prev > 1:
                gaps.append(f"{k_prev + 1} to {k - 1}")
        
        if logger_obj:
            logger_obj.debug(f"Gaps detected: {gaps}")

        # 4. Sequence Generation (from the final, filtered list of tuples)
        sequence_one_letter = [
            three_to_one(res_name3) for pdb_res_id, icode, res_name3 in final_residue_tuples
        ]

        # 5. Create final resmap_list (FIXED: Uses final_residue_tuples)
        # resmap_list will be generated correctly even if final_residue_tuples is 
        # based only on structural_residues (i.e., no header).
        resmap_list = [
            (i + 1, e[0], e[1], chain_id, e[2], three_to_one(e[2]))
            for i, e in enumerate(final_residue_tuples)
        ]

        # 6. Output Files
        resmap_file = f"{outfileprefix}_chain{chain_id}_resmap.csv"
        list_to_csv(
            resmap_list,
            fieldnames=[
                "SeqResID", "PdbResID", "PdbICode", "Chain", "ResName3", "ResName1",
            ],
            filename=resmap_file,
        )

        fasta_file = f"{outfileprefix}_chain{chain_id}.fasta"
        with open(fasta_file, "w") as seq_file:
            seq_file.write(
                f">{outfileprefix}|chain {chain_id}|final-sequence\n"
            )
            seq_file.write("".join(sequence_one_letter))
            seq_file.write("\n")
            
        if logger_obj:
            logger_obj.info(f"protein sequence:\n{''.join(sequence_one_letter)}")

    return resmap_list, fasta_file, resmap_file