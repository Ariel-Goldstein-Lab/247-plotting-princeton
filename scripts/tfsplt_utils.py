from distutils.command.config import dump_file
import glob
import os
import pandas as pd
import numpy as np
import pickle
from multiprocessing import Pool
from functools import partial
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame
from tqdm import tqdm
from scipy.stats import sem
from scipy.stats import zscore

HEB_SID = [7, 8]
ARAB_SID = [1, 3, 4]
NON_ENGLISH_SID = HEB_SID + ARAB_SID

def read_sig_file(filename, filedir, old_results=False):
    sig_file = pd.read_csv(os.path.join(get_dir(filedir), filename))
    sig_file["sid_electrode"] = (
        sig_file["subject"].astype(str) + "_" + sig_file["electrode"]
    )
    elecs = sig_file["sid_electrode"].tolist()

    if old_results:  # might need to use this for old 625-676 results
        elecs = sig_file["electrode"].tolist()  # no sid name in front

    return set(elecs)


def read_file(file_name, sigelecs, sigelecs_key, load_sid, key, label1, label2):
    elec = os.path.basename(file_name).replace(".csv", "")[:-5]
    if (  # Skip electrodes if they're not part of the sig list
        len(sigelecs)
        and elec not in sigelecs[sigelecs_key]
        # and "whole_brain" not in sigelecs_key
    ):
        return None
    # if 'LGA' not in elec and 'LGB' not in elec: # for 717, only grid
    #     continue
    df = pd.read_csv(file_name, header=None)
    # if df.max(axis=1)[0] < 0.1:
    #     return None
    df.insert(0, "sid", load_sid)
    df.insert(0, "key", key)
    df.insert(0, "electrode", elec)
    df.insert(0, "label1", label1)
    df.insert(0, "label2", label2)

    return df


def read_folder2(
    data,
    fname,
    load_sid="load_sid",
    label="label",
    mode="mode",
    type="all",
):
    files = glob.glob(fname)
    assert len(files) == 1, f"No files or multiple files found"
    df = pd.read_csv(files[0], header=None)
    df = df.dropna(axis=1)
    df.columns = np.arange(-1, 161)
    df = df.rename({-1: "electrode"}, axis=1)
    df.insert(1, "sid", load_sid)
    df.insert(1, "mode", mode)
    df.insert(0, "label", label)
    df.insert(0, "type", type)
    df.electrode = df.sid.astype(str) + "_" + df.electrode
    for i in np.arange(len(df)):
        data.append(df.iloc[[i]])
    return data


def read_folder(
    data,
    fname,
    sigelecs,
    sigelecs_key,
    load_sid="load_sid",
    key="key",
    label1="label1",
    label2="label2",
    parallel=True,
):
    files = glob.glob(fname)
    assert (
        len(files) > 0
    ), f"No results found under {fname}"  # check files exist under format

    if parallel:
        p = Pool(10)
        for result in p.map(
            partial(
                read_file,
                sigelecs=sigelecs,
                sigelecs_key=sigelecs_key,
                load_sid=load_sid,
                key=key,
                label1=label1,
                label2=label2,
            ),
            files,
        ):
            data.append(result)

    else:
        for resultfn in files:
            data.append(
                read_file(
                    resultfn,
                    sigelecs,
                    sigelecs_key,
                    load_sid,
                    key,
                    label1,
                    label2,
                )
            )

    return data


def load_pickle(file, key=None):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    print(f"Loading {file}")
    with open(file, "rb") as fh:
        datum = pickle.load(fh)

    if key:
        df = pd.DataFrame.from_dict(datum[key])
    else:
        df = pd.DataFrame.from_dict(datum)
    return df


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'"""
    add_ext = "" if file_name.endswith(".pkl") else ".pkl"

    file_name = file_name + add_ext

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as fh:
        pickle.dump(item, fh)
    return


def colorFader(c1, c2, mix):
    """Get color in between two colors (based on linear interpolate)

    Args:
        c1: color 1 in hex format
        c2: color 2 in hex format
        mix: percentage between two colors (0 is c1, 1 is c2)

    Returns:
        a color in hex format
    """
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def get_cat_color(num=10):
    """Get categorical colors"""
    if num > 10:
        print("Can't get more than 10 categorical colors")
        return None
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]  # separate colors
    return colors


def get_con_color(colormap, num):
    """Get continuous colors"""
    cmap = plt.cm.get_cmap(colormap)
    colors = [cmap(i / num) for i in range(0, num)]
    return colors


def get_fader_color(c1, c2, num):
    """Get self-defined continuous colors"""
    colors = [colorFader(c1, c2, i / num) for i in range(0, num)]
    return colors

def get_dir(path_ending):
    """Get the path a directory. Used to allow running through Makefile or through script directly (e.g. for debugging)"""
    current_dir = os.getcwd()
    # Check if we're in the main directory (with data/ as a direct subdirectory)
    if os.path.isdir(os.path.join(current_dir, path_ending)):
        return os.path.join(current_dir, path_ending)
    # Check if we're in a script/ subdirectory (need to go one level up)
    elif os.path.basename(current_dir) == "scripts" and os.path.isdir(os.path.join(os.path.dirname(current_dir), path_ending)):
        return os.path.join(os.path.dirname(current_dir), path_ending)
    # If neither condition is met, raise an error
    else:
        raise FileNotFoundError(f"Could not locate the data directory ({path_ending})")

def get_recording_type(sid):
    recording_type = "tfs"
    if sid == 777:
        recording_type = "podcast"
    return recording_type

def load_electrode_names(filter_type):
    """
    Returns all electrodes names for a given filter_type (e.g. 160 for the 160 significant electrodes for podcast)
    """
    electrode_names_csv_path = None
    if filter_type == 160 or filter_type == "160":
        electrode_names_csv_path = "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/useful/podcast_777_160.csv"
    elif filter_type == 50 or filter_type == "50":
        electrode_names_csv_path = "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/useful/podcast_777_glove_elecs.csv"
    elif filter_type == 39 or filter_type == "39":
        electrode_names_csv_path = "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/useful/gemma_scope_sig_elec_alphas--0.7_0.27_30_enc_over_0.1.csv"
    elif filter_type == "IFG":
        electrode_names_csv_path = "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/useful/podcast_ifg.csv"
    elif filter_type == "IFG160":
        electrode_names_csv_path = "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/useful/podcast_160ifg.csv"
    elif filter_type == "STG":
        electrode_names_csv_path = "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/useful/podcast_stg.csv"
    elif filter_type == "STG160":
        electrode_names_csv_path = "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/useful/podcast_160stg.csv"
    if electrode_names_csv_path:
        electrode_names_df = pd.read_csv(electrode_names_csv_path)
        electrode_names_df["full_elec_name"] = electrode_names_df["subject"].astype(str) + "_" + electrode_names_df[
            "electrode"]
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}. Please provide a valid filter_type.")

    return electrode_names_df

def get_processed_coeffs_df_paths(sid, model_name, layer, context, min_alpha, max_alpha, num_alphas, mode, filter_type, kfolds_threshold, emb_mod=None):
    """
    Returns paths to encoding and coeffs files for a given electrode for a specific subject, model, layer, context, alpha range, and mode.
    :param model_name: full model name (e.g., "gemma-scope-2b-pt-res-canonical")
    :param full_elec_name: long electrode name (e.g., "777_LGB2")
    :param mode: "comp" or "prod"
    :return:
    """
    recording_type = get_recording_type(sid)
    model_path = f"{get_dir('data')}/encoding/{recording_type}/tk-{recording_type}-{sid}-{model_name}-lag2k-25-all"

    # Prep general paths
    lasso_path = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}{f'-mod_{emb_mod}' if emb_mod else ''}/{filter_type}_{mode}_k-{kfolds_threshold}_coeffs_df.pkl"
    all_data_path = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}-all_data{f'-mod_{emb_mod}' if emb_mod else ''}/{filter_type}_{mode}_coeffs_df.pkl"
    corr_path = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-corr_coeffs{f'-mod_{emb_mod}' if emb_mod else ''}/{filter_type}_{mode}_coeffs_df.pkl"
    ridge_path = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-regridge-alphas_{min_alpha}_{max_alpha}_{num_alphas}{f'-mod_{emb_mod}' if emb_mod else ''}/{filter_type}_{mode}_coeffs_df.pkl"
    pca_path = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-pca50{f'-mod_{emb_mod}' if emb_mod else ''}/{filter_type}_{mode}_coeffs_df.pkl"
    corr_pca_path = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-pca50-corr_coeffs{f'-mod_{emb_mod}' if emb_mod else ''}/{filter_type}_{mode}_coeffs_df.pkl"

    return lasso_path, all_data_path, corr_path, ridge_path, pca_path, corr_pca_path

def get_electrode_encoding_and_coeffs_paths(sid, model_name, layer, context, min_alpha, max_alpha, num_alphas, full_elec_name, mode, filter_type, emb_mod):
    """
    Returns paths to encoding and coeffs files for a given electrode for a specific subject, model, layer, context, alpha range, and mode.
    :param model_name: full model name (e.g., "gemma-scope-2b-pt-res-canonical")
    :param full_elec_name: long electrode name (e.g., "777_LGB2")
    :param mode: "comp" or "prod"
    :return:
    """
    recording_type = get_recording_type(sid)
    model_path = f"{get_dir('data')}/encoding/{recording_type}/tk-{recording_type}-{sid}-{model_name}-lag2k-25-all"

    # Prep general paths
    lasso_path_template = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}{f'-mod_{emb_mod}' if emb_mod else ''}/{full_elec_name}_{mode}{{ending}}"
    all_data_path_template = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}-all_data{f'-mod_{emb_mod}' if emb_mod else ''}/{full_elec_name}_{mode}{{ending}}"
    corr_path_template = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-corr_coeffs{f'-mod_{emb_mod}' if emb_mod else ''}/{{ending}}"
    corr_pca_path_template = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-pca50-corr_coeffs{f'-mod_{emb_mod}' if emb_mod else ''}/{{ending}}"
    ridge_path_template = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-regridge-alphas_{min_alpha}_{max_alpha}_{num_alphas}{f'-mod_{emb_mod}' if emb_mod else ''}/{full_elec_name}_{mode}{{ending}}"
    pca_path_template = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-pca50{f'-mod_{emb_mod}' if emb_mod else ''}/{full_elec_name}_{mode}{{ending}}"

    # Prep encodings paths
    lasso_enc_path = lasso_path_template.format(ending=".csv")
    all_data_enc_path = all_data_path_template.format(ending=".csv")
    ridge_enc_path = ridge_path_template.format(ending=".csv")
    pca_enc_path = pca_path_template.format(ending=".csv")

    # Prep coeffs paths
    lasso_coeffs_path = lasso_path_template.format(ending="_coeffs.npy")
    all_data_coeffs_path = all_data_path_template.format(ending="_coeffs.npy")
    corr_val_path = corr_path_template.format(ending=f"{full_elec_name}_{mode}_corr.npy")
    corr_pca_val_path = corr_pca_path_template.format(ending=f"{full_elec_name}_{mode}_corr.npy")
    ridge_coeffs_path = ridge_path_template.format(ending="_coeffs.npy")
    pca_coeffs_path = pca_path_template.format(ending="_coeffs.npy")

    #TODO
    if filter_type == "IFG160" or filter_type == "STG160":
        filter_type = "160"

    corr_elec_names_path = corr_path_template.format(ending=f"pvals_combined_names{f'({filter_type})' if filter_type else ''}.pkl")
    corr_pca_elec_names_path = corr_pca_path_template.format(ending=f"pvals_combined_names{f'({filter_type})' if filter_type else ''}.pkl")
    pvals_combined_corrected_path = corr_path_template.format(ending=f"pvals_combined_corrected{f'({filter_type})' if filter_type else ''}.npy")
    corr_pca_pvals_combined_corrected_path = corr_pca_path_template.format(ending=f"pvals_combined_corrected{f'({filter_type})' if filter_type else ''}.npy")

    return (lasso_enc_path, lasso_coeffs_path,
            all_data_enc_path, all_data_coeffs_path,
            corr_val_path, corr_elec_names_path, pvals_combined_corrected_path,
            ridge_enc_path, ridge_coeffs_path,
            pca_enc_path, pca_coeffs_path,
            corr_pca_val_path, corr_pca_elec_names_path, corr_pca_pvals_combined_corrected_path,
            )

def get_elec_locations(subjects_type):
    """
    Returns the locations of the electrodes for a group
        (if sid==777 will return locations of electrodes of all podcast subjects,
        else will return locations of all 24.7 subjects).
    :param subjects_type: can just give sid
    :return:
    """
    if "777" in str(subjects_type):
        elec_locations = pd.read_csv(
            "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/podcast-old/elec_masterlist.csv")
        # elec_locations["full_elec_name"] = elec_locations["subject"].astype(str) + "_" + elec_locations["name"]
    elif "888" in str(subjects_type):
        elec_locations = pd.read_csv(
            "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/paper-whisper/data/888_base_df.csv")
        elec_locations.rename(columns={"sid": "subject", "elec_1": "name"}, inplace=True)
    else:
        elec_locations = pd.read_csv(
            "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/paper-whisper/data/base_df.csv")
        elec_locations.rename(columns={"sid": "subject", "elec_1": "name"}, inplace=True)
    elec_locations["full_elec_name"] = elec_locations["subject"].astype(str) + "_" + elec_locations["name"]
    return elec_locations

def load_electrode_names_and_locations(subjects_type, filter_type):
    """
    Returns a df of all the electrode names and locations for a given filter and subject type (meaning podcast or 24.7).
    :param subjects_type: can just be sid, if == 777, then will return all podcast, else will return all 24.7.
    :param filter_type:
    :return:
    """
    electrode_names_df = load_electrode_names(filter_type)
    electrode_names_df = electrode_names_df[["subject", "electrode", "full_elec_name"]]
    elec_locations = get_elec_locations(subjects_type)
    elec_locations.drop(columns=["subject", "name"], inplace=True)
    merged_df = pd.merge(electrode_names_df, elec_locations, on="full_elec_name", how="left")
    print(merged_df.shape)
    return merged_df

def _get_exploded_united_lasso_and_corr(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                                        kfolds_threshold:int=10, emb_mod=None, query:str=""):
    kfolds_df = prepare_coeffs_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
                                           num_alphas, sid, emb_mod, query=query, df_type="lasso")
    exploded_kfolds_df = kfolds_df.query("num_of_chosen_coeffs > 0").explode(["actual_chosen_coeffs", "chosen_coeffs_val"])

    corr_df = prepare_coeffs_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
                                         num_alphas, sid, emb_mod, query=query, df_type="corr")
    exploded_corr_df = corr_df.query("num_of_chosen_coeffs > 0").explode(["actual_chosen_coeffs", "chosen_coeffs_val"])[
                                             ["full_elec_name", "time_index", "actual_chosen_coeffs", "chosen_coeffs_val"]]

    exploded_df = exploded_kfolds_df.merge(exploded_corr_df, how="inner", suffixes=("_lasso", "_corr"),
                                           on=["full_elec_name", "time_index", "actual_chosen_coeffs"])

    exploded_df['num_of_chosen_coeffs'] = exploded_df.groupby(['full_elec_name', 'time_index'])['actual_chosen_coeffs'].transform('count')
    assert not exploded_df.duplicated(subset=['full_elec_name', 'time_index', 'actual_chosen_coeffs']).any(), "Duplicate combinations found!"

    updated_counts = exploded_df[['full_elec_name', 'time_index', 'num_of_chosen_coeffs']].drop_duplicates()

    # TODO: add values of corr to kfolds_df
    kfolds_df = kfolds_df.drop(columns=['num_of_chosen_coeffs']).merge(
        updated_counts,
        on=['full_elec_name', 'time_index'],
        how='left'
    )
    kfolds_df['num_of_chosen_coeffs'] = kfolds_df['num_of_chosen_coeffs'].fillna(0).astype(int)

    # Same as above, just shows what are the dups:
    # duplicates = df[df.duplicated(subset=['full_elec_name', 'time_index', 'actual_chosen_coeffs'], keep=False)]
    # assert duplicates.empty, f"Found {len(duplicates)} duplicate rows:\n{duplicates}"

    return exploded_df, kfolds_df

def get_coeffs_df(patient, mode, model_info, filter_type, min_alpha, max_alpha, num_alphas, reliable_kfolds_threshold, df_type, emb_mod=None, query=""):
    if df_type == "lasso&corr":
        _, df = _get_exploded_united_lasso_and_corr(patient, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                                                    reliable_kfolds_threshold, emb_mod, query)
    else:
        df = prepare_coeffs_df(filter_type, reliable_kfolds_threshold, max_alpha, min_alpha, mode,
                                      model_info, num_alphas, patient, emb_mod, query=query, df_type=df_type)
    return df

# def get_exploded_df(sid, mode, model_info, filter_type, min_alpha, max_alpha, num_alphas, kfolds_threshold, df_type, query=""):
#     if df_type == "lasso&corr" or df_type == "corr&lasso":
#         exploded_df, _ = _get_exploded_united_kfolds_and_corr(sid, filter_type, model_info, min_alpha, max_alpha,
#                                                               num_alphas, mode,
#                                                               kfolds_threshold, query=query)
#     else:
#         non_zero_df = prepare_coeffs_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
#                                         num_alphas, sid, emb_mod, query=query, df_type=df_type)
#         exploded_df = non_zero_df.query("num_of_chosen_coeffs > 0").explode(["actual_chosen_coeffs", "chosen_coeffs_val"])
#         exploded_df.rename(columns={"chosen_coeffs_val": f"chosen_coeffs_val_{df_type}", }, inplace=True)
#     return exploded_df

def prepare_lang_coeffs_df(sid, kfolds_threshold, query=""):
    import numpy as np
    import pandas as pd

    kfolds_coeffs_path = f"/scratch/gpfs/HASSON/tk6637/princeton/247-encoding/results/podcast/tk-podcast-heb-Mistral-7B-v0.3-lag2k-25-all/tk-200ms-0-lay16-con150-reglasso-alphas_-2_10_100/sub-P{sid}_task-podcast_embed-mean_band-highgamma_nfolds-10_tmax-2_context-150_layer-16__coeffs_df.pkl"

    if os.path.exists(kfolds_coeffs_path):
        print(f"Loading saved lasso")
        kfolds_df = pd.read_pickle(kfolds_coeffs_path)
        kfolds_df = _process_coeff_df(kfolds_df)
        if query:
            kfolds_df = kfolds_df.query(query)
        return kfolds_df

    all_data = pd.read_pickle(
        f"/scratch/gpfs/HASSON/tk6637/princeton/247-encoding/results/podcast/tk-podcast-heb-Mistral-7B-v0.3-lag2k-25-all/tk-200ms-0-lay16-con150-reglasso-alphas_-2_10_100/sub-P{sid}_task-podcast_embed-mean_band-highgamma_nfolds-10_tmax-2_context-150_layer-16_encoding.pkl")
    embedding_size = all_data["coefs"][0].shape[0]
    coeffs_indx = np.arange(embedding_size).astype(str)  # Needed for indexing later

    time_index = range(161)
    time_points = np.linspace(-2, 2, 161)

    kfolds_encoding_raw = np.array(all_data["corrs"]) # shape (folds, elecs, timepoints)
    kfolds_encoding_all = kfolds_encoding_raw.mean(axis=0) # shape (elecs, timepoints)
    # kfolds_std_all = kfolds_encoding_raw.std(axis=0) # shape (elecs, timepoints)
    kfolds_se_all = sem(kfolds_encoding_raw, axis=0) # shape (elecs, timepoints)

    kfolds_coeffs_all = np.array(all_data["coefs"]) # shape (folds, embedding_size, elecs, timepoints)
    # kfolds_non_zero_mask = kfolds_coeffs_all != 0
    # kfold_coeffs_in_folds_all = kfolds_non_zero_mask.sum(axis=0) # shape (embedding_size, elecs, timepoints)

    kfolds_dfs = []
    for elec_indx, elec_name in enumerate(tqdm(all_data["electrodes"])):
        kfolds_encoding = kfolds_encoding_all[elec_indx, :]
        kfolds_se = kfolds_se_all[elec_indx, :]

        kfolds_coeffs = kfolds_coeffs_all[:, :, elec_indx, :]
        kfolds_non_zero_mask = kfolds_coeffs != 0
        kfold_coeffs_in_folds = kfolds_non_zero_mask.sum(axis=0)  # For each timepoint, for each coeff, how many kfolds is it non-zero in
        # kfold_coeffs_in_folds = kfold_coeffs_in_folds_all[:, elec_indx, :]
        reliable_chosen_coeffs = [coeffs_indx[col] for col in (kfold_coeffs_in_folds >= kfolds_threshold).T]
        reliable_coeffs_counts = (kfold_coeffs_in_folds >= kfolds_threshold).sum(axis=0)  # For each timepoint, how many coeffs are non-zero in at least `threshold` kfolds
        kfolds_mean_coeff = np.mean(kfolds_coeffs, axis=0)
        kfolds_coeffs_values = [kfolds_mean_coeff[:, i] for i in range(kfolds_mean_coeff.shape[1])]
        reliable_chosen_coeffs_values = [kfolds_coeffs_values[i][chosen_coeffs.astype(int)] for i, chosen_coeffs in enumerate(reliable_chosen_coeffs)]
        assert np.array_equal([len(coeffs) for coeffs in reliable_chosen_coeffs], reliable_coeffs_counts)  # Making sure the amounts make sense


        current_kfolds_df = pd.DataFrame(data={"full_elec_name": elec_name,
                                               "time_index": time_index,
                                               "time": time_points.round(2),
                                               "num_of_chosen_coeffs": reliable_coeffs_counts,
                                               "actual_chosen_coeffs": reliable_chosen_coeffs,
                                               "all_coeffs_index": [np.arange(len(kfolds_coeffs_values[0])) for _ in range(len(kfolds_coeffs_values))] if kfolds_coeffs_values else None,
                                               "all_coeffs_val": kfolds_coeffs_values,
                                               "chosen_coeffs_val": reliable_chosen_coeffs_values,
                                               "encoding": kfolds_encoding,
                                               "encoding_se": kfolds_se})
        kfolds_dfs.append(current_kfolds_df)

    kfolds_df = pd.concat(kfolds_dfs, ignore_index=True)
    # kfolds_df = kfolds_df.merge(elec_df, on="full_elec_name", how="left")

    kfolds_df.to_pickle(kfolds_coeffs_path)
    kfolds_df.to_csv(kfolds_coeffs_path.replace(".pkl", ".csv"), index=False)

    kfolds_df = _process_coeff_df(kfolds_df)

    if query:
        kfolds_df = kfolds_df.query(query)

    return kfolds_df


    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[sid],
        specs=[[{"secondary_y": True}]],
        shared_xaxes=True
    )

    fig.add_trace(go.Scatter(
        x=time_points,
        y=all_data["corrs"][0][41, :],
        mode='lines',
        name="encoding",
        showlegend=True,
        yaxis="y2"
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=time_points,
        y=coeffs,
        mode='lines',
        name="coeffs",
        showlegend=True,
        yaxis="y2"
    ), row=1, col=1, secondary_y=False)


def prepare_coeffs_df(filter_type, kfolds_threshold: int, max_alpha, min_alpha, mode, model_info, num_alphas, sid, emb_mod=None, query="", df_type="lasso"):
    """
    Returns a df of type df_type, after conducting the query.
    The df has a row for each electrode*time lag. Each row has the number of non_zero/sig coeffs (num_of_chosen_coeffs), the list of them (actual_chosen_coeffs), the encoding value (encoding), and brain area info (brain_area etc.).
    It also has some useful columns such as time_bin, rounded_encoding, and time_bin*brain area.

    In the all_data_df the coeffs are all the non-zero coeffs, and the encoding is on the train.
    In the lasso_df the coeffs are those that appears in over kfolds_threshold of the folds, and the encoding is the usual one.
    In the corr_df the coeffs are those that have a significant correlation with the signal, and the encdoing is taken from the kfolds.
    """
    if sid in NON_ENGLISH_SID: # heb/arabic
        return prepare_lang_coeffs_df(sid, kfolds_threshold, query="")
    return_all_data, return_kfolds, return_corr, return_ridge, return_pca, return_corr_pca = False,False,False,False,False,False

    if df_type == "all_data":
        return_all_data = True
    elif df_type == "lasso":
        return_kfolds = True
    elif df_type == "corr":
        return_corr = True
    elif df_type == "ridge":
        return_ridge = True
    elif df_type == "pca":
        return_pca = True
    elif df_type == "corr_pca" or df_type == "pca_corr":
        return_corr_pca = True
    else:
        raise ValueError("df_type must be 'all_data', 'lasso', 'corr', or 'all_data'")

    all_data_df, lasso_df, corr_df, ridge_df, pca_df, corr_pca_df = get_raw_coeffs_dfs(sid, filter_type, model_info["model_full_name"], model_info["layer"], model_info["context"],
                                                                   min_alpha, max_alpha, num_alphas, mode, kfolds_threshold, emb_mod,
                                                                   return_all_data=return_all_data, return_lasso=return_kfolds, return_corr=return_corr, return_ridge=return_ridge, return_pca=return_pca, return_corr_pca=return_corr_pca)

    if df_type == "all_data":
        df = all_data_df
    elif df_type == "lasso":
        df = lasso_df
    elif df_type == "corr":
        df = corr_df
    elif df_type == "ridge":
        df = ridge_df
    elif df_type == "pca":
        df = pca_df
    elif df_type == "corr_pca" or df_type == "pca_corr":
        df = corr_pca_df

    df = _process_coeff_df(df)

    if query:
        df = df.query(query)

    return df

def get_raw_coeffs_dfs(sid, filter_type, model_name, layer, context, min_alpha, max_alpha, num_alphas, mode, kfolds_threshold=10, emb_mod=None,
                       return_all_data=True, return_lasso=True, return_corr=True, return_ridge=True, return_pca=True, return_corr_pca=True):
    """
    Returns three dfs - all_data_df, lasso_df and corr_df.
    These dfs have a row for each electrode*time lag. Each row has the number of non_zero/sig coeffs (num_of_chosen_coeffs), the list of them (actual_chosen_coeffs), the encoding value (encoding), and brain area info (princeton_class etc.).
    In the all_data_df the coeffs are all the non-zero coeffs, and the encoding is on the train.
    In the lasso_df the coeffs are those that appears in over kfolds_threshold of the folds, and the encoding is the usual one.
    In the corr_df the coeffs are those that have a significant correlation with the signal, and the encdoing is taken from the kfolds.


    :param model_name: full model name (e.g., "gemma-scope-2b-pt-res-canonical")
    :param mode: "comp" or "prod"
    :param return_all_data: should also return elec all_data encoding?
    :param return_lasso: should also return elec kfolds encoding?
    :param kfolds_threshold: how many folds does a coeff need to appear in, in order to be reliable
    :param output_elec_name_prefix: what to add to the electrodes names
    """
    assert return_lasso or return_all_data or return_corr or return_ridge or return_pca or return_corr_pca

    lasso_path, all_data_path, corr_path, ridge_path, pca_path, corr_pca_path = get_processed_coeffs_df_paths(sid, model_name, layer, context, min_alpha, max_alpha, num_alphas, mode, filter_type, kfolds_threshold,emb_mod)

    # Load if already exists:
    all_data_df, lasso_df, corr_df, ridge_df, pca_df, corr_pca_df = None, None, None, None, None, None
    if return_all_data and os.path.exists(all_data_path):
        print(f"Loading saved all data")
        all_data_df = pd.read_pickle(all_data_path)
        return_all_data = False # To not run the code below
    if return_lasso and os.path.exists(lasso_path):
        print(f"Loading saved lasso")
        lasso_df = pd.read_pickle(lasso_path)
        return_lasso = False # To not run the code below
    if return_corr and os.path.exists(corr_path):
        print(f"Loading saved corr")
        corr_df = pd.read_pickle(corr_path)
        return_corr = False # To not run the code below
    if return_ridge and os.path.exists(ridge_path):
        print(f"Loading saved ridge")
        ridge_df = pd.read_pickle(ridge_path)
        return_ridge = False # To not run the code below
    if return_pca and os.path.exists(pca_path):
        print(f"Loading saved pca")
        pca_df = pd.read_pickle(pca_path)
        return_pca = False # To not run the code below
    if return_corr_pca and os.path.exists(corr_pca_path):
        print(f"Loading saved corr-pca")
        corr_pca_df = pd.read_pickle(corr_pca_path)
        return_corr_pca = False # To not run the code below


    # Start Process
    elec_df = load_electrode_names_and_locations(sid, filter_type)
    elecs_names = elec_df["full_elec_name"].to_list()
    if return_corr:
        return_ridge = True
        # Load only once:
        (_, _, _, _, _, corr_elec_names_path, corr_pvals_combined_corrected_path, _, _, _, _, _, _, _) = get_electrode_encoding_and_coeffs_paths(sid, model_name, layer, context,
                                                                                                                min_alpha, max_alpha, num_alphas,
                                                                                                                None, mode, filter_type, emb_mod)
        corr_pvals_combined_corrected = np.load(corr_pvals_combined_corrected_path)
        with open(corr_elec_names_path, 'rb') as f:
            corr_pvals_names = pickle.load(f)
    if return_corr_pca:
        return_ridge = True
        # Load only once:
        (_, _, _, _, _, _, _, _, _, _, _, _, corr_pca_elec_names_path, corr_pca_pvals_combined_corrected_path,) = get_electrode_encoding_and_coeffs_paths(sid, model_name, layer, context,
                                                                                                                min_alpha, max_alpha, num_alphas,
                                                                                                                None, mode, filter_type, emb_mod)
        corr_pca_pvals_combined_corrected = np.load(corr_pca_pvals_combined_corrected_path)
        with open(corr_pca_elec_names_path, 'rb') as f:
            corr_pca_pvals_names = pickle.load(f)

    coeffs_indx = None
    time_index = range(161)
    time_points = np.linspace(-2, 2, 161)

    all_data_dfs = []
    lasso_dfs = []
    corr_dfs = []
    ridge_dfs = []
    pca_dfs = []
    corr_pca_dfs = []

    for full_elec_name in tqdm(elecs_names, desc=f"Getting coeff from electrodes"):
        (lasso_enc_path, lasso_coeffs_path,
         all_data_enc_path, all_data_coeffs_path,
         corr_val_path, _, _,
         ridge_enc_path, ridge_coeffs_path,
         pca_enc_path, pca_coeffs_path,
         corr_pca_val_path, _, _,) = get_electrode_encoding_and_coeffs_paths(sid, model_name, layer, context,
                                                         min_alpha, max_alpha, num_alphas, full_elec_name, mode, filter_type, emb_mod)

        # Prep
        if coeffs_indx is None:
            if return_all_data:
                all_data_coeffs = np.load(all_data_coeffs_path)  # shape (embedding_size, timepoints)
                embedding_size = all_data_coeffs.shape[0]
            else:
                lasso_coeffs = np.load(lasso_coeffs_path)
                embedding_size = lasso_coeffs.shape[1]
            coeffs_indx = np.arange(embedding_size).astype(str)  # Needed for indexing later

        # All data
        all_data_encoding = None
        all_data_coeffs_counts = None
        all_data_chosen_coeffs = None
        all_data_coeffs_values = None
        all_data_chosen_coeffs_values = None
        if return_all_data:
            # encoding:
            all_data_encoding = np.genfromtxt(all_data_enc_path, delimiter=',')

            # coeffs:
            all_data_coeffs = np.load(all_data_coeffs_path) # shape (embedding_size, timepoints)
            all_data_non_zero_mask = all_data_coeffs != 0
            all_data_chosen_coeffs = [coeffs_indx[col] for col in all_data_non_zero_mask.T] # A list of numpy arrays of all the chosen coeffs per timepoint
            all_data_coeffs_counts = all_data_non_zero_mask.sum(axis=0)
            all_data_coeffs_values = [all_data_coeffs[:, i] for i in range(all_data_coeffs.shape[1])]
            all_data_chosen_coeffs_values = [all_data_coeffs_values[i][chosen_coeffs.astype(int)] for i, chosen_coeffs in enumerate(all_data_chosen_coeffs)]
            assert np.array_equal([len(coeffs) for coeffs in all_data_chosen_coeffs], all_data_coeffs_counts) # Making sure the amounts make sense

        # Lasso
        lasso_encoding = None
        lasso_sem = None
        lasso_reliable_coeffs_counts = None
        lasso_reliable_chosen_coeffs = None
        lasso_coeffs_values = None
        lasso_reliable_chosen_coeffs_values = None
        if return_lasso:
            # encoding:
            lasso_encoding_raw = np.genfromtxt(lasso_enc_path, delimiter=',')
            lasso_encoding = lasso_encoding_raw.mean(axis=0)
            # lasso_sem = lasso_encoding_raw.std(axis=0)
            lasso_sem = sem(lasso_encoding_raw, axis=0)

            # coeffs:
            lasso_coeffs = np.load(lasso_coeffs_path)  # shape (folds, embedding_size, timepoints)
            lasso_non_zero_mask = lasso_coeffs != 0
            lasso_coeffs_in_folds = lasso_non_zero_mask.sum(axis=0)  # For each timepoint, for each coeff, how many kfolds is it non-zero in
            lasso_reliable_chosen_coeffs = [coeffs_indx[col] for col in (lasso_coeffs_in_folds>=kfolds_threshold).T]
            lasso_reliable_coeffs_counts = (lasso_coeffs_in_folds >= kfolds_threshold).sum(axis=0)  # For each timepoint, how many coeffs are non-zero in at least `threshold` kfolds
            lasso_mean_coeff = np.mean(lasso_coeffs, axis=0)
            lasso_coeffs_values = [lasso_mean_coeff[:, i] for i in range(lasso_mean_coeff.shape[1])]
            lasso_reliable_chosen_coeffs_values = [lasso_coeffs_values[i][chosen_coeffs.astype(int)] for i, chosen_coeffs in enumerate(lasso_reliable_chosen_coeffs)]
            assert np.array_equal([len(coeffs) for coeffs in lasso_reliable_chosen_coeffs], lasso_reliable_coeffs_counts) # Making sure the amounts make sense

        # Ridge
        ridge_encoding = None
        ridge_sem = None
        ridge_reliable_coeffs_counts = None
        ridge_reliable_chosen_coeffs = None
        ridge_coeffs_values = None
        ridge_reliable_chosen_coeffs_values = None
        if return_ridge:
            # encoding:
            ridge_encoding_raw = np.genfromtxt(ridge_enc_path, delimiter=',')
            ridge_encoding = ridge_encoding_raw.mean(axis=0)
            # ridge_sem = ridge_encoding_raw.std(axis=0)
            ridge_sem = sem(ridge_encoding_raw, axis=0)

            # coeffs:
            ridge_coeffs = np.load(ridge_coeffs_path)  # shape (folds, embedding_size, timepoints)
            ridge_non_zero_mask = ridge_coeffs != 0
            ridge_coeffs_in_folds = ridge_non_zero_mask.sum(axis=0)  # For each timepoint, for each coeff, how many kfolds is it non-zero in
            ridge_reliable_chosen_coeffs = [coeffs_indx[col] for col in (ridge_coeffs_in_folds>=kfolds_threshold).T]
            ridge_reliable_coeffs_counts = (ridge_coeffs_in_folds >= kfolds_threshold).sum(axis=0)  # For each timepoint, how many coeffs are non-zero in at least `threshold` kfolds
            ridge_mean_coeff = np.mean(ridge_coeffs, axis=0)
            ridge_coeffs_values = [ridge_mean_coeff[:, i] for i in range(ridge_mean_coeff.shape[1])]
            ridge_reliable_chosen_coeffs_values = [ridge_coeffs_values[i][chosen_coeffs.astype(int)] for i, chosen_coeffs in enumerate(ridge_reliable_chosen_coeffs)]
            assert np.array_equal([len(coeffs) for coeffs in ridge_reliable_chosen_coeffs], ridge_reliable_coeffs_counts) # Making sure the amounts make sense

        # Corr (only coeffs)
        corr_chosen_coeffs = None
        corr_coeffs_counts = None
        corr_coeffs_values = None
        corr_chosen_coeffs_values = None
        corr_non_corrected_pval_values = None
        if return_corr:
            elec_pvals_corrected = corr_pvals_combined_corrected[:, :, corr_pvals_names.index(full_elec_name)]
            corr_chosen_coeffs = [coeffs_indx[col] for col in (elec_pvals_corrected <= 0.05).T]

            corr_coeffs_counts = np.sum(elec_pvals_corrected <= 0.05, axis=0)  # Absolute counts
            corrs_val = np.load(corr_val_path)
            corr_coeffs_values = list(corrs_val.T)
            corr_chosen_coeffs_values = [corr_coeffs_values[i][chosen_coeffs.astype(int)] for i, chosen_coeffs in enumerate(corr_chosen_coeffs)]
            non_corrected_pval = np.load("pval".join(corr_val_path.rsplit("corr", 1)))
            corr_non_corrected_pval_values = list(non_corrected_pval.T)


        # PCA
        pca_encoding = None
        pca_sem = None
        pca_reliable_coeffs_counts = None
        pca_reliable_chosen_coeffs = None
        pca_coeffs_values = None
        pca_reliable_chosen_coeffs_values = None
        if return_pca:
            pca_coeffs_indx = np.arange(50).astype(str)  # Needed for indexing later
            # encoding:
            pca_encoding_raw = np.genfromtxt(pca_enc_path, delimiter=',')
            pca_encoding = pca_encoding_raw.mean(axis=0)
            # pca_sem = pca_encoding_raw.std(axis=0)
            pca_sem = sem(pca_encoding_raw, axis=0)

            # coeffs:
            pca_coeffs = np.load(pca_coeffs_path)  # shape (folds, embedding_size, timepoints)
            pca_non_zero_mask = pca_coeffs != 0
            pca_coeffs_in_folds = pca_non_zero_mask.sum(axis=0)  # For each timepoint, for each coeff, how many kfolds is it non-zero in
            pca_reliable_chosen_coeffs = [pca_coeffs_indx[col] for col in (pca_coeffs_in_folds>=kfolds_threshold).T]
            pca_reliable_coeffs_counts = (pca_coeffs_in_folds >= kfolds_threshold).sum(axis=0)  # For each timepoint, how many coeffs are non-zero in at least `threshold` kfolds
            pca_mean_coeff = np.mean(pca_coeffs, axis=0)
            pca_coeffs_values = [pca_mean_coeff[:, i] for i in range(pca_mean_coeff.shape[1])]
            pca_reliable_chosen_coeffs_values = [pca_coeffs_values[i][chosen_coeffs.astype(int)] for i, chosen_coeffs in enumerate(pca_reliable_chosen_coeffs)]
            assert np.array_equal([len(coeffs) for coeffs in pca_reliable_chosen_coeffs], pca_reliable_coeffs_counts) # Making sure the amounts make sense


        # Corr PCA (only coeffs)
        corr_pca_chosen_coeffs = None
        corr_pca_coeffs_counts = None
        corr_pca_coeffs_values = None
        corr_pca_chosen_coeffs_values = None
        corr_pca_non_corrected_pval_values = None
        if return_corr_pca:
            pca_coeffs_indx = np.arange(50).astype(str)  # Needed for indexing later
            elec_pvals_corrected = corr_pca_pvals_combined_corrected[:, :, corr_pca_pvals_names.index(full_elec_name)]
            corr_pca_chosen_coeffs = [pca_coeffs_indx[col] for col in (elec_pvals_corrected <= 0.05).T]

            corr_pca_coeffs_counts = np.sum(elec_pvals_corrected <= 0.05, axis=0)  # Absolute counts
            corrs_pca_val = np.load(corr_pca_val_path)
            corr_pca_coeffs_values = list(corrs_pca_val.T)
            corr_pca_chosen_coeffs_values = [corr_pca_coeffs_values[i][chosen_coeffs.astype(int)] for i, chosen_coeffs in enumerate(corr_pca_chosen_coeffs)]
            corr_pca_non_corrected_pval_values = np.load("pval".join(corr_pca_val_path.rsplit("corr", 1)))
            corr_pca_non_corrected_pval_values = list(corr_pca_non_corrected_pval_values.T)

        # Save it all
        current_all_data_df = pd.DataFrame(data={"full_elec_name": full_elec_name,
                                                 "time_index": time_index,
                                                 "time": time_points.round(2),
                                                 "num_of_chosen_coeffs": all_data_coeffs_counts,
                                                 "actual_chosen_coeffs": all_data_chosen_coeffs,
                                                 "all_coeffs_index": [np.arange(len(all_data_coeffs_values[0])) for _ in range(len(all_data_coeffs_values))] if all_data_coeffs_values else None,
                                                 "all_coeffs_val": all_data_coeffs_values,
                                                 "chosen_coeffs_val": all_data_chosen_coeffs_values,
                                                 "encoding": all_data_encoding})
        current_lasso_df = pd.DataFrame(data={"full_elec_name": full_elec_name,
                                               "time_index": time_index,
                                               "time": time_points.round(2),
                                               "num_of_chosen_coeffs": lasso_reliable_coeffs_counts,
                                               "actual_chosen_coeffs": lasso_reliable_chosen_coeffs,
                                               "all_coeffs_index": [np.arange(len(lasso_coeffs_values[0])) for _ in range(len(lasso_coeffs_values))] if lasso_coeffs_values else None,
                                               "all_coeffs_val": lasso_coeffs_values,
                                               "chosen_coeffs_val": lasso_reliable_chosen_coeffs_values,
                                               "encoding": lasso_encoding,
                                               "encoding_se": lasso_sem})
        current_corr_df = pd.DataFrame(data={"full_elec_name": full_elec_name,
                                               "time_index": time_index,
                                               "time": time_points.round(2),
                                               "num_of_chosen_coeffs": corr_coeffs_counts,
                                               "actual_chosen_coeffs": corr_chosen_coeffs,
                                               "all_coeffs_index": [np.arange(len(corr_coeffs_values[0])) for _ in range(len(corr_coeffs_values))] if corr_coeffs_values else None,
                                               "all_coeffs_val": corr_coeffs_values,
                                               "chosen_coeffs_val": corr_chosen_coeffs_values,
                                               "encoding": ridge_encoding,
                                               "encoding_se": ridge_sem,
                                               "corr_non_corrected_pval_values": corr_non_corrected_pval_values})
        current_ridge_df = pd.DataFrame(data={"full_elec_name": full_elec_name,
                                               "time_index": time_index,
                                               "time": time_points.round(2),
                                               "num_of_chosen_coeffs": ridge_reliable_coeffs_counts,
                                               "actual_chosen_coeffs": ridge_reliable_chosen_coeffs,
                                               "all_coeffs_index": [np.arange(len(ridge_coeffs_values[0])) for _ in range(len(ridge_coeffs_values))] if ridge_coeffs_values else None,
                                               "all_coeffs_val": ridge_coeffs_values,
                                               "chosen_coeffs_val": ridge_reliable_chosen_coeffs_values,
                                               "encoding": ridge_encoding,
                                               "encoding_se": ridge_sem})
        current_pca_df = pd.DataFrame(data={"full_elec_name": full_elec_name,
                                               "time_index": time_index,
                                               "time": time_points.round(2),
                                               "num_of_chosen_coeffs": pca_reliable_coeffs_counts,
                                               "actual_chosen_coeffs": pca_reliable_chosen_coeffs,
                                               "all_coeffs_index": [np.arange(len(pca_coeffs_values[0])) for _ in range(len(pca_coeffs_values))] if pca_coeffs_values else None,
                                               "all_coeffs_val": pca_coeffs_values,
                                               "chosen_coeffs_val": pca_reliable_chosen_coeffs_values,
                                               "encoding": pca_encoding,
                                               "encoding_se": pca_sem})
        current_corr_pca_df = pd.DataFrame(data={"full_elec_name": full_elec_name,
                                             "time_index": time_index,
                                             "time": time_points.round(2),
                                             "num_of_chosen_coeffs": corr_pca_coeffs_counts,
                                             "actual_chosen_coeffs": corr_pca_chosen_coeffs,
                                             "all_coeffs_index": [np.arange(len(corr_pca_coeffs_values[0])) for _ in range(
                                                 len(corr_pca_coeffs_values))] if corr_pca_coeffs_values else None,
                                             "all_coeffs_val": corr_pca_coeffs_values,
                                             "chosen_coeffs_val": corr_pca_chosen_coeffs_values,
                                             "encoding": ridge_encoding,
                                             "encoding_se": ridge_sem,
                                             "corr_non_corrected_pval_values": corr_pca_non_corrected_pval_values})

        all_data_dfs.append(current_all_data_df)
        lasso_dfs.append(current_lasso_df)
        corr_dfs.append(current_corr_df)
        ridge_dfs.append(current_ridge_df)
        pca_dfs.append(current_pca_df)
        corr_pca_dfs.append(current_corr_pca_df)

    if return_all_data:
        all_data_df = pd.concat(all_data_dfs, ignore_index=True)
        all_data_df = all_data_df.merge(elec_df, on="full_elec_name", how="left")
        all_data_df.to_pickle(all_data_path)
        all_data_df.to_csv(all_data_path.replace(".pkl", ".csv"), index=False)
    if return_lasso:
        lasso_df = pd.concat(lasso_dfs, ignore_index=True)
        lasso_df = lasso_df.merge(elec_df, on="full_elec_name", how="left")
        lasso_df.to_pickle(lasso_path)
        lasso_df.to_csv(lasso_path.replace(".pkl", ".csv"), index=False)
    if return_corr:
        corr_df = pd.concat(corr_dfs, ignore_index=True)
        corr_df = corr_df.merge(elec_df, on="full_elec_name", how="left")
        corr_df.to_pickle(corr_path)
        corr_df.to_csv(corr_path.replace(".pkl", ".csv"), index=False)
    if return_ridge:
        ridge_df = pd.concat(ridge_dfs, ignore_index=True)
        ridge_df = ridge_df.merge(elec_df, on="full_elec_name", how="left")
        ridge_df.to_pickle(ridge_path)
        ridge_df.to_csv(ridge_path.replace(".pkl", ".csv"), index=False)
    if return_pca:
        pca_df = pd.concat(pca_dfs, ignore_index=True)
        pca_df = pca_df.merge(elec_df, on="full_elec_name", how="left")
        pca_df.to_pickle(pca_path)
        pca_df.to_csv(pca_path.replace(".pkl", ".csv"), index=False)
    if return_corr_pca:
        corr_pca_df = pd.concat(corr_pca_dfs, ignore_index=True)
        corr_pca_df = corr_pca_df.merge(elec_df, on="full_elec_name", how="left")
        corr_pca_df.to_pickle(corr_pca_path)
        corr_pca_df.to_csv(corr_pca_path.replace(".pkl", ".csv"), index=False)


    return all_data_df, lasso_df, corr_df, ridge_df, pca_df, corr_pca_df

def _process_coeff_df(df: DataFrame):
    """
    Adds to a df useful columns such as time_bin, rounded_encoding, and time_bin*brain area.
    """
    df['rounded_encoding'] = df['encoding'].round(1)
    df.loc[df['rounded_encoding'] == -0.0, 'rounded_encoding'] = 0.0
    df['rounded_encoding'] = df['rounded_encoding'].astype(str)

    # large_bins = [-np.inf, -0.6, -0.3, 0, 0.3, 0.6, np.inf]
    # large_labels = ['x<-0.6', '-0.6≤x<-0.3', '-0.3≤x<0', '0≤x<0.3', '0.3≤x<0.6', '0.6≤x']
    large_bins = [-np.inf, -0.8, -0.4, 0, 0.4, 0.8, np.inf]
    small_bins = [-np.inf, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, np.inf]

    large_labels = ['x<-0.8', '-0.8≤x<-0.4', '-0.4≤x<0', '0≤x<0.4', '0.4≤x<0.8', '0.8≤x']
    small_labels = ['x<-1.9', '-1.9≤x<-1.8', '-1.8≤x<-1.7', '-1.7≤x<-1.6', '-1.6≤x<-1.5', '-1.5≤x<-1.4', '-1.4≤x<-1.3', '-1.3≤x<-1.2', '-1.2≤x<-1.1', '-1.1≤x<-1.0', '-1.0≤x<-0.9', '-0.9≤x<-0.8', '-0.8≤x<-0.7', '-0.7≤x<-0.6', '-0.6≤x<-0.5', '-0.5≤x<-0.4', '-0.4≤x<-0.3', '-0.3≤x<-0.2', '-0.2≤x<-0.1', '-0.1≤x<0.0', '0.0≤x<0.1', '0.1≤x<0.2', '0.2≤x<0.3', '0.3≤x<0.4', '0.4≤x<0.5', '0.5≤x<0.6', '0.6≤x<0.7', '0.7≤x<0.8', '0.8≤x<0.9', '0.9≤x<1.0', '1.0≤x<1.1', '1.1≤x<1.2', '1.2≤x<1.3', '1.3≤x<1.4', '1.4≤x<1.5', '1.5≤x<1.6', '1.6≤x<1.7', '1.7≤x<1.8', '1.8≤x<1.9', '1.9≤x<2.0', '2.0≤x']

    df['time_bin'] = pd.cut(df['time'], bins=large_bins, labels=large_labels, right=False).astype(str)
    df['small_time_bin'] = pd.cut(df['time'], bins=small_bins, labels=small_labels, right=False).astype(str)
    if "princeton_class" in df.columns:
        df.rename(columns={'princeton_class': 'brain_area'}, inplace=True)
        df["time_bin_and_brain_area"] = df['brain_area'] + "_" + df['time_bin']

    return df