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
    """Get the path a directory. Used to allow running through Makefile-rafi-orig or through script directly (e.g. for debugging)"""
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
    electrode_names_csv_path = None
    if filter_type == 160 or filter_type == "160":
        electrode_names_csv_path = "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/useful/podcast_160.csv"
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

def get_encoding_and_coeffs_paths(sid, model_name, layer, context, min_alpha, max_alpha, num_alphas, full_elec_name, mode):
    """
    Get paths to encoding and coeffs files for a given subject, model, layer, context, alpha range, electrode, and mode.
    :param model_name: full model name (e.g., "gemma-scope-2b-pt-res-canonical")
    :param full_elec_name: long electrode name (e.g., "777_LGB2")
    :param mode: "comp" or "prod"
    :return:
    """
    recording_type = get_recording_type(sid)
    model_path = f"{get_dir('data')}/encoding/{recording_type}/tk-{recording_type}-{sid}-{model_name}-lag2k-25-all"

    # Prep general paths
    kfolds_path_template = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}/{full_elec_name}_{mode}{{ending}}"
    all_data_path_template = f"{model_path}/tk-200ms-{sid}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}-all_data/{full_elec_name}_{mode}{{ending}}"

    # Prep encodings paths
    kfolds_enc_path = kfolds_path_template.format(ending=".csv")
    all_data_enc_path = all_data_path_template.format(ending=".csv")

    # Prep coeffs paths
    kfolds_coeffs_path = kfolds_path_template.format(ending="_coeffs.npy")
    all_data_coeffs_path = all_data_path_template.format(ending="_coeffs.npy")

    return (kfolds_enc_path, kfolds_coeffs_path,
            all_data_enc_path, all_data_coeffs_path)

def get_elec_locations(sid):
    if "777" in str(sid):
        elec_locations = pd.read_csv(
            "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/sig-elecs/podcast-old/elec_masterlist.csv")
        elec_locations["full_elec_name"] = elec_locations["subject"].astype(str) + "_" + elec_locations["name"]
    else:
        elec_locations = pd.read_csv(
            "/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/data/plotting/paper-whisper/data/base_df.csv")
        elec_locations.rename(columns={"sid": "subject", "elec_1": "name"}, inplace=True)
    elec_locations["full_elec_name"] = elec_locations["subject"].astype(str) + "_" + elec_locations["name"]
    return elec_locations

def load_electrode_names_and_locations(sid, filter_type):
    electrode_names_df = load_electrode_names(filter_type)
    elec_locations = get_elec_locations(sid)
    elec_locations.drop(columns=["subject", "name"], inplace=True)
    merged_df = pd.merge(electrode_names_df, elec_locations, on="full_elec_name", how="left")
    return merged_df

def amount_coeffs_per_timepoint_in_agreement_kfolds(kfolds_coeffs, threshold=8):
    kfolds_non_zero_mask = kfolds_coeffs != 0
    kfold_coeffs_count = kfolds_non_zero_mask[:, :, :].sum(axis=0) # For each timepoint, for each coeff, how many kfolds is it non-zero in
    reliable_coeffs_count  = (kfold_coeffs_count >= threshold).sum(axis=0) # For each timepoint, how many coeffs are non-zero in at least `threshold` kfolds
    return reliable_coeffs_count


