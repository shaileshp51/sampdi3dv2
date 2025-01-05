#!/usr/bin/env python
# coding: utf-8

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


from utils.common import *
from utils.sequence import *
from utils.structure import *
from utils.dna import *

from utils.pssm import set_logger as set_pssm_logger
from utils.dna import set_logger as set_dna_logger

# Let's suppress the log messages from ProDy, to keep stdout clear.
pdy.confProDy(verbosity="none")
# Let's suppress non-critical XGBoost warnings arising because model is developed
# using an older verion 0.82.
xgb.set_config(verbosity=0)

logger_obj = None


def predict_dna_mutation_ddg(df_feature_this_mt, wild_bp, mutation_bp, base_location):
    """
    Predict the impact of a DNA mutation based on its features.

    Args:
        df_feature_this_mt (pd.DataFrame): Feature dataframe for the mutation.
        wild_bp (str): Wild-type base pair.
        mutation_bp (str): Mutant base pair.
        base_location: base directory of the sampdi3d-v2

    Returns:
        str: Predicted impact of the mutation.
    """
    model_dir = f"{base_location}/trainedmodels/DNA_xgboost"
    model_file_list = [
        "model_1_pcc_0.7941.pkl",
        "model_2_pcc_0.7967.pkl",
        "model_3_pcc_0.8084.pkl",
        "model_4_pcc_0.8451.pkl",
        "model_5_pcc_0.8006.pkl",
    ]

    # Categorical feature list (from training code)\
    categorical_columns = [
        "mutation_type_dna_label",
        "mismatch_dna_label",
        "pair_type_label",
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
    numeric_predictors = df_feature_this_mt[numeric_columns].copy()

    # Combine predictors
    features = pd.concat([categorical_predictors, numeric_predictors], axis=1)

    y_pred = []

    logger_obj.debug("df_feature_this_mt:" + str(df_feature_this_mt.columns))
    logger_obj.debug(
        "processed df:" + str(features.columns) + " " + str(features.values)
    )

    if wild_bp == mutation_bp:
        y_pred = [0.0] * len(model_file_list)
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
                    # print("required_features:", required_features, required_features.dtypes)
                    logger_obj.debug("required_features:" + str(required_features))
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

    stability_impact = ("Neutral", "Destabilizing", "Stabilizing")
    # Neutral if y_pred_mean==0; Destabilizing if y_pred_mean > 0; else Stabilizing
    impact = f"{y_pred_mean:.2f} {stability_impact[int(np.sign(y_pred_mean))]}"

    return impact, tuple(y_pred + [y_pred_mean])


def predict_mutations_in_dna(
    pdbid,
    mutations_list,
    config,
    base_location,
    job_name,
    dump_features,
    dump_ddg_with_feature,
    features_file,
    logger,
    temporary_log_files,
):
    """
    Predict the effects of DNA mutations.

    Args:
        pdbid (str): Identifier for the PDB file.
        mutations_list (list): List of mutations (chain, wild_bp, resid, mutation_bp).
        config (dict): Dictionary of configurations.
        base_location (str): Path to the package's root directory.
        job_name (str): Use this jobname in output as first column if non-empty.
        dump_features (bool): Whether to dump the features values to a file.
        dump_ddg_with_feature (bool): Whether to also dump predicted ddG with features.
        features_file (str): Filename to save mutation features.
        logger: The logger object from the main for logging.
        temporary_log_files (list[str]): extend the list of temporary with file generated here.

    Returns:
        list: Predicted impacts for each mutation.
    """
    global logger_obj
    logger_obj = logger

    # Share the global logger to following modules
    set_pssm_logger(logger)
    set_dna_logger(logger)

    # Whether to make ddG prediction
    make_prediction = True

    # Initialize configured variables
    dssp_exe = config["dssp_exe"]
    x3dna_exe = config["x3dna_exe"]

    results = []
    feature_values = []
    ddg_columns = None
    mutation_info_dtypes = {"polymer": np.str_, "mutation": np.str_}

    # H = alpha-helix
    # B = residue in isolated beta-bridge
    # E = extended strand, participates in beta ladder
    # G = 3-helix (310 helix)
    # I = 5 helix (pi-helix)
    # T = hydrogen bonded turn
    # S = bend

    feature_dtypes = {
        "mutation_type_dna_label": np.int64,
        "mismatch_dna_label": np.int64,
        "pair_type_label": np.int64,
        "ss_H_ratio_dssp": np.float64,
        "ss_B_ratio_dssp": np.float64,
        "ss_E_ratio_dssp": np.float64,
        "ss_G_ratio_dssp": np.float64,
        "ss_T_ratio_dssp": np.float64,
        "ss_S_ratio_dssp": np.float64,
        "total_nt_aa_contacts": np.int64,
        "total_phosphate_aa_hbonds": np.int64,
        "total_base_aa_hbonds": np.int64,
        "total_base_aa_stacks": np.int64,
        "wt_pos_nt_aa_contacts": np.int64,
        "wt_pos_phosphate_aa_hbonds": np.int64,
        "wt_pos_base_aa_hbonds": np.int64,
        "wt_pos_base_aa_stacks": np.int64,
        "bp_par_shear": np.float64,
        "bp_par_stretch": np.float64,
        "bp_par_stagger": np.float64,
        "bp_par_bucklle": np.float64,
        "bp_par_propeller": np.float64,
        "bp_par_opening": np.float64,
        "bp_step_shift": np.float64,
        "bp_step_slide": np.float64,
        "bp_step_rise": np.float64,
        "bp_step_tilt": np.float64,
        "bp_step_roll": np.float64,
        "bp_step_twist": np.float64,
        "heli_par_x_displacement": np.float64,
        "heli_par_y_displacement": np.float64,
        "heli_par_rise": np.float64,
        "heli_par_inclination": np.float64,
        "heli_par_tip": np.float64,
        "heli_par_twist": np.float64,
    }
    mutation_info_columns = list(mutation_info_dtypes.keys())
    only_dna_features = list(feature_dtypes.keys())

    for mt_index, (chain, wild_bp, resid, mutation_bp) in enumerate(mutations_list):
        if mt_index == 0:
            run_dssp_and_x3dna(
                pdbid,
                polymer="dna",
                dssp_exe=dssp_exe,
                x3dna_exe=x3dna_exe,
                temporary_log_files=temporary_log_files,
            )

        labels = [
            mutation_type_dna(basepairs, wild_bp, mutation_bp),
            mismatch_dna(wild_bp, mutation_bp),
            pair_type(wild_bp),
            *mutation_ss_dna(pdb_file=pdbid),
        ]
        labels += get_protein_dna_interactions(pdb_file=pdbid)
        labels += get_protein_wt_base_interactions(
            pdb_file=pdbid, wt_info=(chain, wild_bp, resid)
        )
        labels += get_dna_structure_features(
            pdbid, chain, resid, wild_bp[0], wild_bp[1]
        )

        df_feature_this_mt = pd.DataFrame(
            {k: [feature_dtypes[k](v)] for k, v in zip(only_dna_features, labels)}
        )
        if make_prediction:
            mt_impact, ddg_predicted = predict_dna_mutation_ddg(
                df_feature_this_mt, wild_bp, mutation_bp, base_location
            )
            ddg_predicted = [round(v, 2) for v in ddg_predicted]
            results.append(mt_impact)

        features_annot_this_mt = [
            "dna",
            f"{chain}.{wild_bp}{resid}{mutation_bp}",
        ] + labels
        if job_name:
            features_annot_this_mt = [job_name] + features_annot_this_mt

        if dump_ddg_with_feature and make_prediction:
            if ddg_columns is None:
                ddg_columns = [f"ddG(m{i})" for i in range(1, len(ddg_predicted))]
                ddg_columns += [f"ddG(m1-{len(ddg_predicted)-1}:avg)"]
            features_annot_this_mt = features_annot_this_mt + list(ddg_predicted)
        feature_values.append(features_annot_this_mt)

    if dump_features and features_file:
        dna_features = mutation_info_columns + only_dna_features
        if job_name:
            dna_features = ["jobname"] + dna_features
        feature_data = np.transpose(np.array(feature_values))

        if dump_ddg_with_feature and make_prediction:
            column_names = dna_features + ddg_columns
            features_dict = {fn: fv for fn, fv in zip(column_names, feature_data)}
        else:
            features_dict = {fn: fv for fn, fv in zip(dna_features, feature_data)}
        dict_to_csv(features_dict, features_file)

    return results
