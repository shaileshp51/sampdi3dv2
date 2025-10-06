#!/usr/bin/env python
# coding: utf-8


import os

import numpy as np
import xgboost as xgb
import pandas as pd
import joblib

from utils.common import *
from utils.sequence import *
from utils.structure import *
from utils.pssm import *
from utils.dna import *
from utils.pssm import set_logger as set_pssm_logger
from utils.dna import set_logger as set_dna_logger

# Let's suppress the log messages from ProDy, to keep stdout clear.
# pdy.confProDy(verbosity="none")
# Let's suppress non-critical XGBoost warnings arising because model is developed
# using an older verion 0.82.
xgb.set_config(verbosity=0)

logger_obj = None


def predict_protein_mutation_ddg(
    df_feature_this_mt, wild_aa, mutation_aa, base_location
):
    """
    Predict the impact of a protein mutation based on its features.

    Args:
        label (list): Feature vector for the mutation.
        wild_aa (str): Wild-type amino acid.
        mutation_aa (str): Mutant amino acid.

    Returns:
        str: Predicted impact of the mutation.
    """
    model_dir = f"{base_location}/trainedmodels/protein_xgboost"
    model_file_list = [
        "model_1_pcc_0.7132.pkl",
        "model_2_pcc_0.7450.pkl",
        "model_3_pcc_0.6790.pkl",
        "model_4_pcc_0.6760.pkl",
        "model_5_pcc_0.6644.pkl",
    ]

    # Categorical feature list (from training code)
    categorical_columns = [
        "mutation_hydrophobicity_label",
        "mutation_polarity_label",
        "mutation_type_label",
        "mutation_size_label",
        "mutation_hbonds_label",
        "mutation_chemical_label",
        "wt_ss_dssp_label",
    ]

    # Identify categorical and numeric features
    existing_categorical_columns = [
        col for col in categorical_columns if col in df_feature_this_mt.columns
    ]
    missing_categorical_columns = [
        col for col in categorical_columns if col not in df_feature_this_mt.columns
    ]

    logger_obj.debug(
        f"Info: Existing categorial columns: {', '.join(existing_categorical_columns)}"
    )

    if missing_categorical_columns:
        raise (
            f"Warning: Missing categorical columns: {', '.join(missing_categorical_columns)}"
        )

    categorical_predictors = (
        df_feature_this_mt[existing_categorical_columns]
        if existing_categorical_columns
        else pd.DataFrame()
    )

    # Identify numeric predictors dynamically
    numeric_columns = [
        col
        for col in df_feature_this_mt.columns
        if col
        not in categorical_columns
        + ["target"]  # Adjust 'target' to match the actual target column name
    ]
    numeric_predictors = df_feature_this_mt[numeric_columns]

    # Combine predictors
    features = pd.concat([categorical_predictors, numeric_predictors], axis=1)

    y_pred = []

    if wild_aa == mutation_aa:
        y_pred = [0.0 for i in range(len(model_file_list))]
    else:
        for model_file in model_file_list:
            required_features = None
            try:
                # Load the model
                model = joblib.load(f"{model_dir}/{model_file}")
                logger_obj.debug(f"Model '{model_file}' loaded successfully.")

                # Ensure the required features match the model's expectations
                try:
                    required_features = model.feature_names_in_
                    logger_obj.debug(
                        f"required_features: {', '.join(required_features)}"
                    )
                except AttributeError:
                    logger_obj.error(
                        f"Error: The model '{model_file}' does not have feature information. Skipping this model."
                    )
                    sys.exit()

                missing_features = [
                    feature
                    for feature in required_features
                    if feature not in features.columns
                ]
                if missing_features:
                    logger_obj.error(
                        f"Error: The following required features are missing for model '{model_file}': {', '.join(missing_features)}"
                    )
                    sys.exit()

                # Filter the data to include only required features
                model_features = features[required_features]

                # Make predictions
                predictions = model.predict(model_features)
                logger_obj.debug(
                    f"Predictions generated successfully for model '{model_file}'."
                )

                # Add predictions to output data with column named after the model\
                y_pred.append(predictions[0])

            except Exception as e:
                logger_obj.error(f"Error processing model '{model_file}': {e}")
                sys.exit()

    y_pred_mean = round(np.mean(y_pred), 2)
    logger_obj.info("[" + ", ".join([str(v) for v in y_pred]) + "] " + str(y_pred_mean))

    stability_impact = ("Neutral", "Destabilizing", "Stabilizing")
    # Neutral if y_pred_mean==0; Destabilizing if y_pred_mean > 0; else Stabilizing
    impact = f"{y_pred_mean:.2f} {stability_impact[int(np.sign(y_pred_mean))]}"

    return impact, tuple(y_pred + [y_pred_mean])


def predict_mutations_in_protein(
    pdb_file,
    derive_seq_from_coords,
    mutations_list,
    config,
    no_warnings,
    base_location,
    outbase,
    job_name,
    dump_features,
    dump_ddg_with_feature,
    features_file,
    logger,
    temporary_log_files,
):
    """
    Predicts mutation effects in a protein using structural and sequence-based features.

    Args:
        pdb_file (str): Path to the PDB file.
        mutations_list (list): List of mutations as (chain, wild_aa, resid, mutation_aa).
        config (dict): Dictionary of configurations.
        no_warnings (bool): Suppress warnings.
        base_location (str): Path to the package's root directory.
        outbase (str): Base path for output files.
        job_name (str): Use this jobname in output as first column if non-empty
        dump_features (bool): Whether to dump the features values to a file.
        dump_ddg_with_feature (bool): Whether to also dump predicted ddG with features.
        features_file (str): Path to save extracted features.
        logger: The logger object from the main for logging.
        temporary_log_files (list[str]): extend the list of temporary with file generated here.

    Returns:
        list: Predicted features for each mutation.
    """
    global logger_obj
    logger_obj = logger

    # Share to global logger to following modules
    set_pssm_logger(logger)
    set_dna_logger(logger)

    gen_pssm_for_first_mutant = True
    # Whether to make ddG prediction
    make_prediction = True

    # initialize configured variables
    dssp_exe = config["dssp_exe"]
    x3dna_exe = config["x3dna_exe"]
    scwrl4_exe = config["scwrl4_exe"]

    # H = alpha-helix
    # B = residue in isolated beta-bridge
    # E = extended strand, participates in beta ladder
    # G = 3-helix (310 helix)
    # I = 5 helix (pi-helix)
    # T = hydrogen bonded turn
    # S = bend

    mutation_info_dtypes = {"polymer": np.str_, "mutation": np.str_}
    prot_feature_dtypes = {
        "mutation_hydrophobicity_label": np.int64,
        "mutation_polarity_label": np.int64,
        "mutation_type_label": np.int64,
        "mutation_size_label": np.int64,
        "mutation_hbonds_label": np.int64,
        "mutation_chemical_label": np.int64,
        "wt_ss_dssp_label": np.int64,
        "net_volume": np.float64,
        "net_hydrophobicity": np.float64,
        "net_flexibility": np.float64,
        "wt_acc_dssp": np.int64,
        "ss_H_ratio_dssp": np.float64,
        "ss_B_ratio_dssp": np.float64,
        "ss_E_ratio_dssp": np.float64,
        "ss_G_ratio_dssp": np.float64,
        "ss_T_ratio_dssp": np.float64,
        "ss_S_ratio_dssp": np.float64,
        "wt_psi_dssp": np.float64,
        "wt_phi_dssp": np.float64,
        "nucleotide_aminoacid_contacts": np.int64,
        "phosphate_aminoacid_hbonds": np.int64,
        "base_aminoacid_hbonds": np.int64,
        "base_aminoacid_stacks": np.int64,
        "delta_acc_dssp": np.int64,
        "delta_nucleotide_aminoacid_contacts": np.int64,
        "delta_phosphate_aminoacid_hbonds": np.int64,
        "delta_base_aminoacid_hbonds": np.int64,
        "delta_base_aminoacid_stacks": np.int64,
    }

    mutation_info_columns = list(mutation_info_dtypes.keys())
    prot_feature_names = list(prot_feature_dtypes.keys())

    pssm_feature_names = None
    pssm_feature_dtypes = None
    only_feature_names = None
    only_feature_dtypes = None

    # all_feature_names includes only_feature_names and mutation_info_columns
    all_feature_names = None
    ddg_columns = None

    protein_resmap_list = None
    protein_resmap_dict = None
    protein_fasta_file = ""

    results = []
    feature_values = []

    # Track mutations sequentially
    pdb_to_mutate = pdb_file

    # Process each mutation
    for mt_index, mt_info in enumerate(mutations_list):
        chain, wild_aa, resid, mutation_aa = mt_info

        try:
            # Convert amino acids to single-letter codes if necessary
            wild_aa = aa3to1(wild_aa) if len(wild_aa) == 3 else wild_aa
            mutation_aa = aa3to1(mutation_aa) if len(mutation_aa) == 3 else mutation_aa

            # Create WT and MT protein structures
            result_create_mt = create_mutant_protein_complex(
                pdb_to_mutate,
                scwrl4_exe=scwrl4_exe,
                mt_info=mt_info,
                mt_index=mt_index,
                logger=logger_obj,
                temporary_log_files=temporary_log_files,
            )

            if not result_create_mt["success"]:
                raise RuntimeError("Failed to generate mutant protein complex")

            pdb_wt = result_create_mt["wt_complex"]
            pdb_mt = result_create_mt["mt_complex"]
            pdb_to_mutate = pdb_wt  # Update for next iteration

            # Initialize residue mapping and secondary structure analysis
            if mt_index == 0:
                (
                    protein_resmap_list,
                    protein_fasta_file,
                    protein_resmap_file,
                ) = get_protein_polymer_sequence(pdb_file, outbase, derive_seq_from_coords)
                temporary_log_files.append(protein_fasta_file)
                temporary_log_files.append(protein_resmap_file)
                protein_resmap_dict = {
                    (e[1], e[3]): (e[0], e[4], e[5]) for e in protein_resmap_list
                }
                run_dssp_and_x3dna(
                    pdb_to_mutate,
                    polymer="protein",
                    dssp_exe=dssp_exe,
                    x3dna_exe=x3dna_exe,
                    temporary_log_files=temporary_log_files,
                )

            # Extract sequence position and features
            seq_mt_position, _, seq_wt_aa1 = protein_resmap_dict[(int(resid), chain)]
            labels = []

            # Compute mutation-specific features
            labels.extend(
                [
                    mutation_hydrophobicity(aa1_map, wild_aa, mutation_aa),
                    mutation_polarity(aa1_map, wild_aa, mutation_aa),
                    mutation_type(aa1_mttype_label, wild_aa, mutation_aa),
                    mutation_size(aa1_map, wild_aa, mutation_aa),
                    mutation_hbonds(aa1_map, wild_aa, mutation_aa),
                    mutation_chemical(aa1_map, wild_aa, mutation_aa),
                ]
            )

            # DSSP features
            (
                wt_ss,
                wt_acc,
                wt_phi,
                wt_psi,
                wt_ss_ratio_list,
            ) = protein_acc_bb_torisons_ss_ratios(pdb_wt, chain, resid, wild_aa)
            labels.append(wt_ss)
            labels.extend(
                [
                    net_volume(aa1_map, wild_aa, mutation_aa),
                    net_hydrophobicity(aa1_map, wild_aa, mutation_aa),
                    net_flexibility(aa1_map, wild_aa, mutation_aa),
                ]
            )
            labels.extend([wt_acc] + wt_ss_ratio_list + [wt_psi, wt_phi])

            # Interaction features
            (
                wt_nt_aa_cntct,
                wt_phosph_aa_hb,
                wt_base_aa_hb,
                wt_base_aa_stack,
            ) = get_protein_dna_interactions(pdb_to_mutate)
            labels.extend(
                [wt_nt_aa_cntct, wt_phosph_aa_hb, wt_base_aa_hb, wt_base_aa_stack]
            )

            # Analyze mutant structure
            run_dssp_and_x3dna(
                pdb_mt,
                polymer="protein",
                dssp_exe=dssp_exe,
                x3dna_exe=x3dna_exe,
                temporary_log_files=temporary_log_files,
            )
            (
                mt_ss,
                mt_acc,
                mt_phi,
                mt_psi,
                mt_ss_ratio_list,
            ) = protein_acc_bb_torisons_ss_ratios(pdb_mt, chain, resid, mutation_aa)
            mt_prot_dna_interaction = get_protein_dna_interactions(pdb_mt)
            delta_features = [
                float(wt_acc) - float(mt_acc),
                wt_nt_aa_cntct
                - mt_prot_dna_interaction[0],  # Delta nucleotide_aminoacid contacts
                wt_phosph_aa_hb
                - mt_prot_dna_interaction[1],  # Delta phosphate_aminoacid H-bonds
                wt_base_aa_hb
                - mt_prot_dna_interaction[2],  # Delta base_amino-acid H-bonds
                wt_base_aa_stack
                - mt_prot_dna_interaction[3],  # Delta base_amino-acid stacks
            ]
            labels.extend(delta_features)

            # Generate sequence-based PSSM features
            if mt_index == 0 and gen_pssm_for_first_mutant:
                pssm_features = get_pssm_features(
                    config,
                    protein_fasta_file,
                    seq_mt_position,
                    seq_wt_aa1,
                    temporary_log_files,
                    use_saved_pssm=False,
                )
            elif mt_index == 0 and not os.path.exists(protein_fasta_file + ".pssm"):
                pssm_features = get_pssm_features(
                    config,
                    protein_fasta_file,
                    seq_mt_position,
                    seq_wt_aa1,
                    temporary_log_files,
                    use_saved_pssm=False,
                )
            elif os.path.exists(protein_fasta_file + ".pssm"):
                pssm_features = get_pssm_features(
                    config,
                    protein_fasta_file,
                    seq_mt_position,
                    seq_wt_aa1,
                    temporary_log_files,
                    use_saved_pssm=True,
                )
            pssm_features["row_mt_odds"] = (
                pssm_features[f"row_pssm_{mutation_aa}"]
                - pssm_features[f"row_pssm_{wild_aa}"]
            )
            if pssm_feature_names is None:
                pssm_feature_names = [
                    f for f in pssm_features.keys() if not f.startswith("row_pssm_")
                ]
                pssm_feature_dtypes = {
                    k: np.float64 if k.startswith("norm_") else np.int64
                    for k in pssm_feature_names
                }
                only_feature_names = prot_feature_names + pssm_feature_names
                only_feature_dtypes = {**prot_feature_dtypes, **pssm_feature_dtypes}
                logger_obj.debug(", ".join([str(v) for v in only_feature_dtypes]))
                if job_name:
                    all_feature_names = (
                        ["jobname"] + mutation_info_columns + only_feature_names
                    )
                else:
                    all_feature_names = mutation_info_columns + only_feature_names

            labels.extend([pssm_features[k] for k in pssm_feature_names])

            df_feature_this_mt = pd.DataFrame.from_dict(
                {
                    k: [
                        only_feature_dtypes[k](float(v)),
                    ]
                    for k, v in zip(only_feature_names, labels)
                }
            )
            logger_obj.debug("df_feature_this_mt:" + str(df_feature_this_mt))
            if make_prediction:
                mt_impact, ddg_predicted = predict_protein_mutation_ddg(
                    df_feature_this_mt, wild_aa, mutation_aa, base_location
                )
                results.append(mt_impact)
                ddg_predicted = [round(v, 2) for v in ddg_predicted]

            features_annot_this_mt = [
                "protein",
                f"{chain}.{wild_aa}{resid}{mutation_aa}",
            ] + labels
            if job_name:
                features_annot_this_mt = [job_name] + features_annot_this_mt

            if dump_ddg_with_feature and make_prediction:
                if ddg_columns is None:
                    ddg_columns = [f"ddG(m{i})" for i in range(1, len(ddg_predicted))]
                    ddg_columns += [f"ddG(m1-{len(ddg_predicted)-1}:avg)"]
                features_annot_this_mt = features_annot_this_mt + list(ddg_predicted)
            feature_values.append(features_annot_this_mt)
        except Exception as e:
            if not no_warnings:
                raise (f"Warning: Error processing mutation {mt_info}: {e}")
            continue

    # Save features to file
    if dump_features and features_file:
        feature_data = np.transpose(np.array(feature_values))
        features_dict = None
        if dump_ddg_with_feature and make_prediction:
            column_names = all_feature_names + ddg_columns
            features_dict = {fn: fv for fn, fv in zip(column_names, feature_data)}
        else:
            features_dict = {fn: fv for fn, fv in zip(all_feature_names, feature_data)}
        dict_to_csv(features_dict, features_file)

    return results
