import argparse
import glob
import itertools
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tfsplt_utils import (
    get_cat_color,
    get_fader_color,
    get_con_color,
    read_sig_file,
    get_dir,
    get_elec_locations,
)


# -----------------------------------------------------------------------------
# Argument Parser and Setup
# -----------------------------------------------------------------------------


def arg_parser():
    """Argument Parser

    Args:

    Returns:
        args (namespace): commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--formats", nargs="+", required=True)
    parser.add_argument("--sid", type=int, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--sig-elec-file", nargs="+", default=[])
    parser.add_argument(
        "--sig-elec-file-dir", nargs="?", default="data/plotting/sig-elecs"
    )
    parser.add_argument("--fig-size", nargs="+", type=int, default=[18, 6])
    parser.add_argument("--lags-plot", nargs="+", type=float, required=True)
    parser.add_argument("--lags-show", nargs="+", type=float, required=True)
    parser.add_argument("--x-vals-show", nargs="+", type=float, required=True)
    parser.add_argument("--lag-ticks", nargs="+", type=float, default=[])
    parser.add_argument("--lag-tick-labels", nargs="+", type=int, default=[])
    parser.add_argument(
        "--y-vals-limit", nargs="+", type=float, default=[0, 0.3]
    )
    parser.add_argument("--lc-by", type=str, default=None)
    parser.add_argument("--ls-by", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--split-by", type=str, default=None)
    parser.add_argument("--outfile", type=str, default="results/figures/tfs-encoding.pdf")
    parser.add_argument("--plot-individuals", action="store_true", default=False, help="Plot individual electrode lines in addition to averages")
    args = parser.parse_args()

    return args


def arg_assert(args):
    """Just some sanity checks

    Args:
        args (namespace): commandline arguments

    Returns:
    """
    assert len(args.fig_size) == 2
    assert len(args.formats) == len(
        args.labels
    ), "Need same number of labels as formats"
    assert len(args.lags_show) == len(
        args.x_vals_show
    ), "Need same number of lags values and x values"
    assert all(
        lag in args.lags_plot for lag in args.lags_show
    ), "Lags plot should contain all lags from lags show"
    assert all(
        lag in args.x_vals_show for lag in args.lag_ticks
    ), "X values show should contain all values from lags ticks"
    assert all(
        lag in args.lags_show for lag in args.lag_tick_labels
    ), "Lags show should contain all values from lag tick labels"
    assert len(args.lag_ticks) == len(
        args.lag_tick_labels
    ), "Need same number of lag ticks and lag tick labels"

    if args.split:
        assert args.split_by, "Need split by criteria"
        assert args.split == "horizontal" or args.split == "vertical"
        assert args.split_by == "keys" or args.split_by == "labels"


def get_cmap_smap(args):
    """Add line color and style map for given label key combinations to args
        cmap: dictionary of {line color: (label, key)}
        smap: dictionary of {line style: (label, key)}

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    if len(args.unique_labels) > 10:
        colors = get_con_color("viridis", len(args.unique_labels))
    else:
        colors = get_cat_color()

    styles = ["-", "--", "-.", ":"]
    cmap = {}  # line color map
    smap = {}  # line style map

    if (
        args.lc_by == "labels" and args.ls_by == "keys"
    ):  # line color by labels and line style by keys
        for label, color in zip(args.unique_labels, colors):
            for key, style in zip(args.unique_keys, styles):
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == "keys" and args.ls_by == "labels"
    ):  # line color by keys and line style by labels
        for key, color in zip(args.unique_keys, colors):
            for label, style in zip(args.unique_labels, styles):
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == args.ls_by == "labels"
    ):  # both line color and style by labels
        for label, color, style in zip(args.unique_labels, colors, styles):
            for key in args.unique_keys:
                cmap[(label, key)] = color
                smap[(label, key)] = style
    elif (
        args.lc_by == args.ls_by == "keys"
    ):  # both line color and style by keys
        for key, color, style in zip(args.unique_keys, colors, styles):
            for label in args.unique_labels:
                cmap[(label, key)] = color
                smap[(label, key)] = style
    else:
        raise Exception("Invalid input for arguments lc_by or ls_by")

    args.cmap = cmap
    args.smap = smap

    return args


def get_sigelecs(args):
    """Add significant electrode lists to args
        sigelecs (Dict): Dictionary in the following format
            tuple of (sid,key) : [list of sig elec]

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    sigelecs = {}
    if len(args.sig_elec_file) == 0:
        pass
    elif len(args.sig_elec_file) == len(args.unique_keys):
        for sid in args.sid:
            for fname, key in zip(args.sig_elec_file, args.unique_keys):
                filename = fname % sid
                sigelecs[(sid, key)] = read_sig_file(
                    filename, args.sig_elec_file_dir
                )
    elif len(args.sig_elec_file) == len(args.sid) * len(args.unique_keys):
        sid_key_tup = [x for x in itertools.product(args.sid, args.unique_keys)]
        for fname, sid_key in zip(args.sig_elec_file, sid_key_tup):
            sigelecs[sid_key] = read_sig_file(fname, args.sig_elec_file_dir)
    else:
        raise Exception(
            "Need a significant electrode file for each subject-key combo"
        )

    args.sigelecs = sigelecs
    return args


def set_up_environ(args):
    """Adding necessary plotting information to args

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    args.x_vals_show = [x_val / 1000 for x_val in args.x_vals_show]
    args.lags_show = [lag / 1000 for lag in args.lags_show]
    args.lags_plot = [lag / 1000 for lag in args.lags_plot]
    args.unique_labels = list(dict.fromkeys(args.labels))
    args.unique_keys = list(dict.fromkeys(args.keys))

    args = get_cmap_smap(args)  # get color and style map
    args = get_sigelecs(args)  # get significant electrodes
    arg_assert(args)  # sanity checks

    return args


# -----------------------------------------------------------------------------
# Aggregate and Organize Data
# -----------------------------------------------------------------------------


def get_sid(fmt, args):
    """Get the correct sid given input folder directory format

    Args:
        fmt (string): folder directory format
        args (namespace): commandline arguments

    Returns:
        load_sid (int): correct sid for the format
    """
    load_sid = 0
    for sid in args.sid:
        if str(sid) in fmt:
            load_sid = sid
    assert (
        load_sid != 0
    ), f"Need subject id for format {fmt}"  # check subject id for format is provided
    return load_sid


def aggregate_data(args):
    """Aggregate encoding data

    Args:
        args (namespace): commandline arguments

    Returns:
        df (DataFrame): df with all encoding results
    """
    print("Aggregating data")

    def read_file(fname):
        fname_dir = get_dir(os.path.dirname(fname))
        files = glob.glob(os.path.join(fname_dir, os.path.basename(fname)))
        assert (
            len(files) > 0
        ), f"No results found under {fname}"  # check files exist under format

        for resultfn in files:
            # elec = os.path.basename(resultfn).replace(".csv", "").replace("_lasso", "")[:-5]
            pattern = r'^(.*?)_(.*?)(?=_(?:comp|prod))'
            compiled_pattern = re.compile(pattern)
            match = compiled_pattern.match(os.path.basename(resultfn))
            if match is None:
                raise ValueError(f"Filename {resultfn} does not match expected pattern")
            elec = match[0]
            # Skip electrodes if they're not part of the sig list
            if (
                len(args.sigelecs)
                and elec not in args.sigelecs[(load_sid, key)]
            ):
                continue
            df = pd.read_csv(resultfn, header=None)
            if df.shape[1] != 161 and df.shape[0] == 161:
                df = df.T  # Should DELETE this in the future! Transpose if needed (_lasso coeff before correction).
            df.insert(0, "sid", load_sid)
            df.insert(0, "key", key)
            df.insert(0, "electrode", elec)
            df.insert(0, "label", label)
            data.append(df)

    data = []

    if len(args.labels) / len(args.unique_labels) == len(args.sid):
        for fmt, label in zip(args.formats, args.labels):
            load_sid = get_sid(fmt, args)
            for key in args.keys:
                fname = fmt % key
                read_file(fname)
    else:
        for load_sid in args.sid:
            for fmt, label in zip(args.formats, args.labels):
                for key in args.keys:
                    fname = fmt % (load_sid, key)
                    read_file(fname)

    if not len(data):
        print("No data found")
        exit(1)
    df = pd.concat(data)
    return df


def organize_data(args, df):
    """Modify encoding data, trimming if necessary

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results

    Returns:
        df (DataFrame): df with correct columns (lags)
    """
    df.set_index(["label", "electrode", "key", "sid"], inplace=True)
    assert len(args.lags_plot) == len(
        df.columns
    ), f"args.lags_plot length ({len(args.lags_plot)}) must be the same size as results ({len(df.columns)})"

    if len(args.lags_show) < len(args.lags_plot):  # plot parts of lags
        print("Trimming Data")
        chosen_lag_idx = [
            idx
            for idx, element in enumerate(args.lags_plot)
            if element in args.lags_show
        ]
        df = df.loc[:, chosen_lag_idx]  # chose from lags to show for the plot
        assert len(args.lags_show) == len(
            df.columns
        ), "args.lags_show length must be the same size as trimmed df column number"

    return df


# -----------------------------------------------------------------------------
# Plotting Average and Individual Electrodes
# -----------------------------------------------------------------------------


def get_elecbrain(electrode):
    """Get filepath for small brain plots

    Args:
        electrode: electrode name

    Returns:
        imname: filepath for small brain plot for the given electrode
    """
    # get proper electrode name
    sid_index = electrode.find("_")
    assert sid_index > 1, "wrong format, no sid"
    assert electrode[:sid_index].isdigit(), "wrong format, sid should be int"
    sid = electrode[:sid_index]
    electrode = electrode[(sid_index + 1) :]  # remove sid
    if sid == "7170":
        sid = "717"
    elecdir = f"/projects/HASSON/247/data/elecimg/{sid}/"
    # HACK need to rewrite this based on the csv
    name = electrode.replace("EEG", "").replace("REF", "").replace("\\", "")
    name = name.replace("_", "").replace("GR", "G")
    imname = elecdir + f"thumb_{name}.png"  # + f'{args.sid}_{name}.png'
    return imname


def plot_average(args, df, pdf):
    """Plot average encoding

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct average plot added
    """
    print("Plotting Average")
    fig, ax = plt.subplots(figsize=args.fig_size)
    for map_key, subdf in df.groupby(["label", "key"], axis=0):
        vals = subdf.mean(axis=0)
        err = subdf.sem(axis=0)
        label = "-".join(map_key)
        ax.fill_between(
            args.x_vals_show,
            vals - err,
            vals + err,
            alpha=0.2,
            color=args.cmap[map_key],
        )
        ax.plot(
            args.x_vals_show,
            vals,
            label=f"{label} ({len(subdf)})",
            color=args.cmap[map_key],
            ls=args.smap[map_key],
        )
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
    ax.axhline(0, ls="dashed", alpha=0.3, c="k")
    ax.axvline(0, ls="dashed", alpha=0.3, c="k")
    ax.legend(loc="upper right", frameon=False)
    ax.set(xlabel="Lag (s)", ylabel="Correlation (r)", title="Global average")
    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_args_summary(args, pdf):
    """Plot a summary page with all command line arguments

    Args:
        args (namespace): commandline arguments
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with arguments summary page added
    """
    fig, ax = plt.subplots(figsize=args.fig_size)
    ax.axis('off')  # Turn off axes for text-only page

    # Create title
    title_text = f"Analysis Parameters Summary for sid {args.sid}"
    ax.text(0.5, 0.95, title_text, transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='center')

    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.5, 0.91, f"Generated: {timestamp}", transform=ax.transAxes,
            fontsize=10, ha='center', style='italic')

    # Convert args namespace to dictionary and format
    args_dict = vars(args)

    # Create formatted text for all arguments with better spacing
    y_pos = 0.85
    line_height = 0.035  # Increased line spacing
    x_pos = 0.05
    column_width = 0.45
    max_line_length = 60  # Characters per line

    items = list(args_dict.items())
    mid_point = len(items) // 2

    # First column
    for i, (key, value) in enumerate(items[:mid_point]):
        # Format the value for better readability
        if isinstance(value, list):
            if len(value) > 8:  # For long lists, show first few items
                value_str = f"[{', '.join(map(str, value[:3]))}...] (len={len(value)})"
            else:
                value_str = str(value)
        elif isinstance(value, str) and len(value) > 40:
            value_str = f"{value[:37]}..."
        else:
            value_str = str(value)

        # Wrap long lines
        text_line = f"{key}: {value_str}"
        if len(text_line) > max_line_length:
            # Split long lines
            key_part = f"{key}:"
            value_part = value_str
            ax.text(x_pos, y_pos, key_part, transform=ax.transAxes,
                    fontsize=9, ha='left', fontweight='bold')
            ax.text(x_pos + 0.02, y_pos - line_height * 0.7, value_part, transform=ax.transAxes,
                    fontsize=8, ha='left', fontfamily='monospace')
            y_pos -= line_height * 1.5
        else:
            ax.text(x_pos, y_pos, text_line, transform=ax.transAxes,
                    fontsize=9, ha='left', fontfamily='monospace')
            y_pos -= line_height

    # Second column
    y_pos = 0.85
    x_pos = 0.52

    for i, (key, value) in enumerate(items[mid_point:]):
        # Format the value for better readability
        if isinstance(value, list):
            if len(value) > 8:  # For long lists, show first few items
                value_str = f"[{', '.join(map(str, value[:3]))}...] (len={len(value)})"
            else:
                value_str = str(value)
        elif isinstance(value, str) and len(value) > 40:
            value_str = f"{value[:37]}..."
        else:
            value_str = str(value)

        # Wrap long lines
        text_line = f"{key}: {value_str}"
        if len(text_line) > max_line_length:
            # Split long lines
            key_part = f"{key}:"
            value_part = value_str
            ax.text(x_pos, y_pos, key_part, transform=ax.transAxes,
                    fontsize=9, ha='left', fontweight='bold')
            ax.text(x_pos + 0.02, y_pos - line_height * 0.7, value_part, transform=ax.transAxes,
                    fontsize=8, ha='left', fontfamily='monospace')
            y_pos -= line_height * 1.5
        else:
            ax.text(x_pos, y_pos, text_line, transform=ax.transAxes,
                    fontsize=9, ha='left', fontfamily='monospace')
            y_pos -= line_height

    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_electrodes(args, df, pdf):
    """Plot individual electrode encoding

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct plots added
    """
    print("Plotting Individual Electrodes")

    results = []
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        fig, ax = plt.subplots(figsize=args.fig_size)
        if args.plot_individuals:
            for (label, _, key, _), values in subdf.iterrows():
                map_key = (label, key)
                label = "-".join(map_key)
                ax.plot(
                    args.x_vals_show,
                    values,
                    # label=label,
                    color=args.cmap[map_key],
                    ls=args.smap[map_key],
                    linewidth=0.5, # Thinner line for individual electrodes
                )

        real_sid = int(electrode.split("_",1)[0])  # Extract the real subject ID
        real_electrode = electrode.split("_",1)[1]

        # Dictionary to store peak info for this electrode
        electrode_peaks = {'electrode': electrode, 'sid': sid, 'real_sid': real_sid, 'real_electrode': real_electrode}

        # Calculate and plot averages for each unique key
        for key_group, key_df in subdf.groupby(level=["label", "key"]):
            # Calculate the average across all rows for this key group
            avg_values = key_df.mean(axis=0)
            avg_sem = key_df.sem(axis=0)
            map_key = (key_group[0], key_group[1])
            label = "-".join(map_key) + " (avg)"

            # Find peak value and time
            peak_idx = np.argmax(np.abs(avg_values))  # Index of peak (absolute value)
            peak_value = avg_values.iloc[peak_idx]
            peak_time = args.x_vals_show[peak_idx]

            # Store peak info with column names based on key_group
            key_name = f"{key_group[0]}_{key_group[1]}"
            electrode_peaks[f"{key_name}_peak_value"] = peak_value
            electrode_peaks[f"{key_name}_peak_time"] = peak_time

            # Plot the average with a thicker line
            ax.plot(
                args.x_vals_show,
                avg_values,
                label=f"{label} ({len(key_df)})",
                color=args.cmap[map_key],
                ls=args.smap[map_key],
                linewidth=3,  # Thicker line for average
            )
            ax.fill_between(
                args.x_vals_show,
                avg_values - avg_sem,
                avg_values + avg_sem,
                alpha=0.5,
                color=args.cmap[map_key],
            )

        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_ylim(args.y_vals_limit[0] - 0.05, args.y_vals_limit[1] + 0.05)
        ax.legend(loc="upper left", frameon=False)
        ax.set(
            xlabel="Lag (s)",
            ylabel="Correlation (r)",
            title=f"{sid} {electrode}",
        )
        imname = get_elecbrain(electrode)
        if os.path.isfile(imname):
            arr_image = plt.imread(imname, format="png")
            fig.figimage(
                arr_image,
                fig.bbox.xmax - arr_image.shape[1],
                fig.bbox.ymax - arr_image.shape[0],
                zorder=5,
            )

        # print(f"Plotting {real_sid} {real_electrode}")
        # print(elec_locations[(elec_locations["subject"] == real_sid) & (elec_locations["name"] == real_electrode)])

        add_ele_loc_text(args, fig, real_sid, real_electrode, electrode_peaks)

        # Add this electrode's peak info to results
        results.append(electrode_peaks)

        pdf.savefig(fig)
        plt.close()

    # Create DataFrame from results
    peak_df = pd.DataFrame(results)
    peak_df.to_csv(args.outfile.replace(".pdf", "_peaks.csv"), index=False)

    return pdf


def add_ele_loc_text(args, fig, real_sid, real_electrode, electrode_peaks=None):
    elec_locations = get_elec_locations(args.sid)

    if 777 in args.sid:
        loc = elec_locations[(elec_locations["subject"] == real_sid) & (elec_locations["name"] == real_electrode)][
            ["princeton_class", "NYU_class"]].iloc[0]
        if electrode_peaks is not None:
            electrode_peaks["princeton_class"] = loc["princeton_class"]
            electrode_peaks["NYU_class"] = loc["NYU_class"]
    else:
        loc_df = elec_locations[(elec_locations["sid"] == real_sid) & (elec_locations["elec_1"] == real_electrode)][
            ["NYU_roi", "roi_1", "roi_2"]]
        if len(loc_df) == 0:
            print(f"Warning: No location data found for sid={real_sid}, electrode={real_electrode}")
            loc = {"NYU_roi": "Unknown", "roi_1": "Unknown", "roi_2": "Unknown"}
        else:
            loc = loc_df.iloc[0]
        if electrode_peaks is not None:
            electrode_peaks["NYU_roi"] = loc["NYU_roi"]
            electrode_peaks["roi_1"] = loc["roi_1"]
            electrode_peaks["roi_2"] = loc["roi_2"]
            electrode_peaks["NYU_roi"] = loc["NYU_roi"]
            electrode_peaks["roi_1"] = loc["roi_1"]
            electrode_peaks["roi_2"] = loc["roi_2"]

    fig.text(0.98, 0.02, loc, ha='right', va='bottom', fontsize=10)


def plot_split_args(args):
    """Get correct arguments for the split plots

    Args:
        args (namespace): commandline arguments

    Returns:
        fig_ver_num (int): number of vertical subplots
        fig_hor_num (int): number of horizontal subplots
        plot_split (str): column to split the subplots (split_by)
        line_split (str): column to split the lines
        plot_lists (list): unique values for the split_by column
    """
    # split by key or label
    if args.split_by == "keys":
        fig_num = len(args.unique_keys)
        plot_lists = args.unique_keys
        plot_split = "key"
        line_split = "label"
    else:
        fig_num = len(args.unique_labels)
        plot_lists = args.unique_labels
        plot_split = "label"
        line_split = "key"

    # split horizontally or vertically
    if args.split == "horizontal":
        fig_ver_num = 1
        fig_hor_num = fig_num
    else:
        fig_ver_num = fig_num
        fig_hor_num = 1

    return (fig_ver_num, fig_hor_num, plot_split, line_split, plot_lists)


def plot_average_split(args, df, pdf):
    """Plot average encoding with a split

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct average plot added
    """
    print(f"Plotting Average split {args.split}ly by {args.split_by}")
    fig_ver_num, fig_hor_num, plot_split, line_split, _ = plot_split_args(args)
    fig, axes = plt.subplots(fig_ver_num, fig_hor_num, figsize=args.fig_size)
    for ax, (plot, subdf) in zip(axes, df.groupby(plot_split, axis=0)):
        for line, subsubdf in subdf.groupby(line_split, axis=0):
            vals = subsubdf.mean(axis=0)
            err = subsubdf.sem(axis=0)
            if args.split_by == "keys":
                map_key = (line, plot)
            else:
                map_key = (plot, line)
            ax.fill_between(
                args.x_vals_show,
                vals - err,
                vals + err,
                alpha=0.2,
                color=args.cmap[map_key],
            )
            ax.plot(
                args.x_vals_show,
                vals,
                label=f"{line} ({len(subsubdf)})",
                color=args.cmap[map_key],
                ls=args.smap[map_key],
            )
        if len(args.lag_ticks) != 0:
            ax.set_xticks(args.lag_ticks)
            ax.set_xticklabels(args.lag_tick_labels)
        ax.axhline(0, ls="dashed", alpha=0.3, c="k")
        ax.axvline(0, ls="dashed", alpha=0.3, c="k")
        ax.set_title(f"{plot} global average")
        ax.legend(loc="upper right", frameon=False)
        ax.set(xlabel="Lag (s)", ylabel="Correlation (r)")
    pdf.savefig(fig)
    plt.close()
    return pdf


def plot_electrodes_split(args, df, pdf):
    """Plot individual electrode encoding with a split

    Args:
        args (namespace): commandline arguments
        df (DataFrame): df with all encoding results
        pdf (PDFPage): pdf with plotting results

    Returns:
        pdf (PDFPage): pdf with correct plots added
    """
    print(f"Plotting Individual Elecs split {args.split}ly by {args.split_by}")
    fig_ver_num, fig_hor_num, plot_split, _, plot_lists = plot_split_args(args)
    for (electrode, sid), subdf in df.groupby(["electrode", "sid"], axis=0):
        fig, axes = plt.subplots(
            fig_ver_num, fig_hor_num, figsize=args.fig_size
        )
        for _, (plot, subsubdf) in zip(axes, subdf.groupby(plot_split)):
            ax = axes[plot_lists.index(plot)]
            if args.plot_individuals:
                for row, values in subsubdf.iterrows():
                    if args.split_by == "keys":
                        line = row[0]
                        map_key = (line, plot)
                    else:
                        line = row[2]
                        map_key = (plot, line)
                    ax.plot(
                        args.x_vals_show,
                        values,
                        # label=line,
                        color=args.cmap[map_key],
                        ls=args.smap[map_key],
                        linewidth=0.5,  # Thinner line for individual electrodes
                    )
            for key_group, key_df in subsubdf.groupby(level=["label", "key"]):
                map_key = (key_group[0], key_group[1])
                label = "-".join(map_key) + " (avg)"
                avg_values = key_df.mean(axis=0)
                ax.plot(
                    args.x_vals_show,
                    avg_values,
                    label=f"{label} ({len(key_df)})",
                    color=args.cmap[map_key],
                    # ls=args.smap[map_key],
                    linewidth=3,  # Thicker line for average
                )
            if len(args.lag_ticks) != 0:
                ax.set_xticks(args.lag_ticks)
                ax.set_xticklabels(args.lag_tick_labels)
            ax.axhline(0, ls="dashed", alpha=0.3, c="k")
            ax.axvline(0, ls="dashed", alpha=0.3, c="k")
            ax.legend(loc="upper left", frameon=False)
            ax.set_ylim(
                args.y_vals_limit[0] - 0.05, args.y_vals_limit[1] + 0.05
            )
            ax.set(
                xlabel="Lag (s)",
                ylabel="Correlation (r)",
                title=f"{sid} {electrode} {plot}",
            )
        imname = get_elecbrain(electrode)
        if os.path.isfile(imname):
            arr_image = plt.imread(imname, format="png")
            fig.figimage(
                arr_image,
                fig.bbox.xmax - arr_image.shape[1],
                fig.bbox.ymax - arr_image.shape[0],
                zorder=5,
            )

        real_sid = int(electrode.split("_",1)[0])  # Extract the real subject ID
        real_electrode = electrode.split("_",1)[1]
        add_ele_loc_text(args, fig, real_sid, real_electrode)

        pdf.savefig(fig)
        plt.close()
    return pdf


def main():
    # Argparse
    args = arg_parser()
    args = set_up_environ(args)

    # Aggregate data
    df = aggregate_data(args)
    df = organize_data(args, df)

    # Plotting
    os.makedirs(get_dir(os.path.dirname(args.outfile)), exist_ok=True)
    pdf = PdfPages(args.outfile)

    # Add arguments summary as first page
    pdf = plot_args_summary(args, pdf)

    if len(args.y_vals_limit) == 1:  # automatic y limit
        args.y_vals_limit = [df.min().min(), df.max().max()]
    if args.split:
        pdf = plot_average_split(args, df, pdf)
        pdf = plot_electrodes_split(args, df, pdf)
    else:
        pdf = plot_average(args, df, pdf)
        pdf = plot_electrodes(args, df, pdf)

    pdf.close()
    print("Plotting done, results saved to", args.outfile)


if __name__ == "__main__":
    main()
