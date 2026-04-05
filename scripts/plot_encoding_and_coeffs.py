import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.graph_objs import Figure
from scipy.stats import pearsonr
from scipy.stats import sem
from statsmodels.stats.multitest import multipletests
from tfsplt_utils import get_elec_locations, get_recording_type, load_electrode_names, \
    _get_exploded_united_lasso_and_corr, prepare_coeffs_df, get_coeffs_df, NON_ENGLISH_SID
from plotly.subplots import make_subplots
from overlapping_coeffs import COLOR_PALETTE

AREA_QUERY = "(brain_area == 'STG') | (brain_area == 'IFG')"
TIME_QUERY = "(time_bin == '-0.4≤x<0') | (time_bin == '0≤x<0.4')"
AREA_AND_TIME_QUERY = f"({AREA_QUERY})&({TIME_QUERY})"

NEUTRAL_COLOR = ('168', '101', '201')# ('255','157','0')  # Light orange
STG_COLOR = ('0', '193', '106')
IFG_COLOR = ('255', '77', '160')
# ENCODING_COLORS = ('240', '98', '146')  # Light pink
# ENCODING_COLORS = ('255','110','1')  # Orange
ENCODING_COLORS = ('100', '181', '246')  # Light blue
# COEFF_COLORS = NEUTRAL_COLOR

CORR_AXIS_COLOR = f"rgb({ENCODING_COLORS[0]}, {ENCODING_COLORS[1]}, {ENCODING_COLORS[2]})"
# COEFF_AXIS_COLOR = f"rgb(0, 0, 0)" #f"rgb({COEFF_COLORS[0]}, {COEFF_COLORS[1]}, {COEFF_COLORS[2]})" #f"rgb(0, 0, 0)" #

def get_coeff_color(filter_type, lines_to_plot):
    line_color = NEUTRAL_COLOR
    axis_color = f"rgb(0, 0, 0)"
    if filter_type == "IFG160":
        line_color = IFG_COLOR
    elif filter_type == "STG160":
        line_color = STG_COLOR

    if any("lasso" in line for line in lines_to_plot) or any("corr" in line for line in lines_to_plot):
        axis_color = f"rgb({line_color[0]}, {line_color[1]}, {line_color[2]})"
    return line_color, axis_color

def load_electrode_dfs_by_type(lines_to_plot, patient, filter_type, model_info, min_alpha, max_alpha,
                               amount_of_alphas, mode, reliable_kfolds_threshold, emb_mod, query):
    dfs = {}
    for line_type in lines_to_plot:
        df = get_coeffs_df(patient, mode, model_info, filter_type, min_alpha, max_alpha, amount_of_alphas, reliable_kfolds_threshold, line_type, emb_mod, query=query)
        dfs[line_type] = df
    return dfs

def plot_encoding_and_coeffs_dual_axis(patient, mode, filter_type, min_alpha, max_alpha, amount_of_alphas,
                                       p_threshold, reliable_kfolds_threshold, models_info, models_to_plot, lines_to_plot, computed_corr_config,
                                       save_dir, save_ending, emb_mod, query="", compare_only_encoding=False):
    """
    Plot encoding and absolute coefficient counts on the same plot with dual y-axes.
    compare_only_encoding: Set to True to compare different encodings (e.g. lasso and ridge in gemma and gemma-scope)
                            If True, will use the same color for encoding and coeffs, and a solid line for encoding vs dashed for coeffs. If False, will use different colors for encoding and coeffs based on the line type (lasso, corr, etc.)
    """
    is_english = patient not in NON_ENGLISH_SID
    if computed_corr_config is None:
        computed_corr_config = [("encoding", "lasso"),
                                ("coeffs", "reliable_lasso")]

    assert len(computed_corr_config) == 2
    assert len(computed_corr_config[0]) == len(computed_corr_config[1]) == 2

    if lines_to_plot is None:
        lines_to_plot = ["lasso&corr", "all_data"]  # , "lasso", "ols", "corr"]
    # if "lasso" in to_plot:
    #     to_plot.append("reliable_lasso")  # Always add reliable kfolds if kfolds is plotted
        # to_plot.append("ridge_lasso") # Always add ridge kfolds if kfolds is plotted

    fig = None
    electrode_names_list = None
    for model_name in models_to_plot:
        print(f"Processing model {model_name}")
        model_color = ('74','145','194') if (model_name =="gemma9b" and compare_only_encoding) else ('181', '124', '210') if (model_name == "gemma-scope9b" and compare_only_encoding) else None
        dfs = load_electrode_dfs_by_type(lines_to_plot, patient, filter_type, models_info[model_name], min_alpha, max_alpha,
                                         amount_of_alphas, mode, reliable_kfolds_threshold, emb_mod, query)
        if not fig:
            electrode_names_list = dfs[list(dfs.keys())[0]]["full_elec_name"].unique().tolist()
            amount_of_electrodes = len(electrode_names_list)
            # If filter type is int:
            if isinstance(filter_type, int):
                assert amount_of_electrodes == filter_type, f"Expected {filter_type} electrodes, got {amount_of_electrodes}"
            fig = prep_dual_axis_figs(amount_of_electrodes, electrode_names_list, filter_type, mode, patient, save_ending)
        plotting_info = get_plotting_info(models_info[model_name], reliable_kfolds_threshold, compare_only_encoding)

        # Plot encoding and num_coeffs
        for line_index, line_type in enumerate(lines_to_plot):
            line_df = dfs[line_type]
            for row_idx, elec_name in enumerate(tqdm(electrode_names_list, desc="Processing electrodes"), start=2): # Row count start from 1 + first row reserved for the overall mean plot
                elec_df = line_df[line_df["full_elec_name"] == elec_name].sort_values("time_index")
                # show_legend = True if row_idx == 2 else False

                plot_encoding_dual_axis(row_idx, elec_df, plotting_info[line_type], False, fig, False, model_color=model_color) #True if "lasso" in line_type else False)
                plot_coeffs_dual_axis(row_idx, elec_df, plotting_info[line_type], model_name, False, fig, filter_type, [line_type], False, True, is_english, compare_only_encoding)

            overall_line_df = line_df.groupby("time").agg(
                encoding_se=('encoding', sem),
                encoding=('encoding', 'mean'),
                num_of_chosen_coeffs_se=('num_of_chosen_coeffs', sem),
                num_of_chosen_coeffs=('num_of_chosen_coeffs', 'mean'),
            ).reset_index()
            overall_line_df["all_coeffs_index"] = [line_df["all_coeffs_index"].reset_index(drop=True)[0]] * len(overall_line_df)
            plot_encoding_dual_axis(1, overall_line_df, plotting_info[line_type], True, fig, True, model_color=model_color)
            plot_coeffs_dual_axis(1, overall_line_df, plotting_info[line_type], model_name, True, fig, filter_type, [line_type], True, True, is_english, compare_only_encoding)
            plot_encoding_num_coeffs_scatter(line_df, model_name, line_type)

        # Add correlation annotation between chosen lines
        first_corr_data = _get_corr_df(dfs, computed_corr_config[0], "first_to_corr")
        second_corr_data = _get_corr_df(dfs, computed_corr_config[1], "second_to_corr")
        corr_data = first_corr_data.merge(second_corr_data, on=["full_elec_name", "time_index"], how="outer")
        corr_df = calc_encoding_coeffs_corr(corr_data)

        plot_encoding_coeffs_corr(["general"] + electrode_names_list, corr_df, computed_corr_config, fig)

    print("All models processed. Now customizing layout...")
    customize_encoding_and_coeffs_dual_axis_layout(len(electrode_names_list), fig, filter_type, lines_to_plot, is_english, compare_only_encoding)
    fig.show()
    save_path = os.path.join(save_dir, f"{f'{save_ending}_' if save_ending else ''}encoding_and_coeffs_dual_axis.html")
    fig.write_html(save_path)

    print(f"!!!!!! Plotting complete. HTML file saved as {save_path} !!!!!!")

def _get_corr_df(dfs, corr_config, column_title):
    corr_df = dfs[corr_config[1]][["full_elec_name", "time_index", corr_config[0]]].rename(columns={corr_config[0]: column_title})
    corr_df_general = corr_df.groupby("time_index")[[column_title]].mean().reset_index()
    corr_df_general["full_elec_name"] = "general"
    corr_df = pd.concat([corr_df, corr_df_general], ignore_index=True)
    return corr_df


def get_plotting_info(model_info, reliable_kfolds_threshold, compare_only_encoding=False):
    model_short_name = model_info["model_short_name"]
    plotting_info = {
        "lasso&corr": {
            "enc_name": f"Encoding - Lasso {model_short_name} (r)",
            "coeffs_name": f"#Coeffs - Lasso&Corr {model_short_name} ({reliable_kfolds_threshold} folds & sig corr)",
            "dash": model_info["dash"],
        },
        "lasso": {
            "enc_name": f"Encoding - Lasso {model_short_name} (r)",
            "coeffs_name": f"#Coeffs - Lasso {model_short_name} (nonzero in {reliable_kfolds_threshold} folds)",
            "dash": "dash" if compare_only_encoding else model_info["dash"],
        },
        "ridge": {
            "enc_name": f"Encoding - Ridge {model_short_name} (r)",
            "coeffs_name": f"#Coeffs - Ridge {model_short_name} (nonzero in {reliable_kfolds_threshold} folds)",
            "dash": "solid" if compare_only_encoding else model_info["dash"],
        },
        "corr": {
            "enc_name": f"Encoding - Ridge {model_short_name} (r)",
            "coeffs_name": f"#Coeffs - Sig Corr of {model_short_name} (sig corr)",
            "dash": model_info["dash"],
        },
        "corr_pca": {
            "enc_name": f"Encoding - Ridge {model_short_name} (r)",
            "coeffs_name": f"#Coeffs (after PCA) - Sig Corr of {model_short_name} (sig corr with PCA)",
            "dash": model_info["dash"],
        },
        "pca": {
            "enc_name": f"Encoding - PCA {model_short_name} (r)",
            "coeffs_name": f"#Coeffs - PCA {model_short_name} (nonzero in {reliable_kfolds_threshold} folds)",
            "dash": model_info["dash"],
        },
    }

    # plots_data = {
    #     "lasso": {"name": f"Kfolds {model_short_name}",
    #            "enc_ending": "(r)", "enc_name": f"Kfolds Encoding {model_short_name} (r)",
    #            "coeffs_ending": "(avg nonzero)", "coeffs_name": f"Kfolds Avg Non Zero of {model_short_name}"},
    # "reliable_lasso": {"name": f"Reliable Kfolds {model_short_name}",
    #                     "coeffs_ending": f"(nonzero in {reliable_kfolds_threshold} folds)",
    #                     "coeffs_name": f"Reliable Kfolds Non Zero of {model_short_name}"},  # Only coeffs, no encoding
    # "all_data": {"name": f"All Data {model_short_name}",
    #              "enc_ending": "(r)", "enc_name": f"All Data Encoding {model_short_name} (r)",
    #              "coeffs_ending": "(nonzero)", "coeffs_name": f"All Data Non Zero of {model_short_name}"},
    # "corr": {"name": f"Corr All Data {model_short_name}",
    #          "coeffs_ending": "(sig)", "coeffs_name": f"Significant Correlation All Data of {model_short_name}"}}

    return plotting_info

def prep_dual_axis_figs(amount_of_electrodes: int, electrode_names_list: DataFrame, filter_type: int | str, mode,
                        patient, save_ending) -> Figure:
    subplot_titles = [f"Mean over all electrodes"] # (filter - {filter_type})
    for i in range(amount_of_electrodes):
        if patient in NON_ENGLISH_SID:
            real_sid = patient
            elec_name = electrode_names_list[i]
            real_electrode = electrode_names_list[i]
        else:
            elec_name = electrode_names_list[i]
            real_sid = int(elec_name.split("_", 1)[0])  # Extract the real subject ID
            real_electrode = elec_name.split("_", 1)[1]

        elec_locations = get_elec_locations(patient)

        if patient in NON_ENGLISH_SID:
            loc_text = ""
        else:
            loc = elec_locations[(elec_locations["subject"] == real_sid) & (elec_locations["name"] == real_electrode)][["princeton_class", "NYU_class"]]
            # Plot on the right side of the plot
            loc_text = f"Locations: Princeton - {loc['princeton_class'].values[0]}, NYU - {loc['NYU_class'].values[0]}"

        subplot_titles.append(f"{patient}, {elec_name} ({mode}), <span style='font-size: 80%;'><i>{loc_text}</i></span>")

    max_vertical_spacing = 1 / (amount_of_electrodes - 1) if amount_of_electrodes > 1 else 0
    chosen_vertical_spacing = max_vertical_spacing * 0.25  # Use 25% of maximum

    # Create subplots with secondary y-axis
    fig = make_subplots(
        rows=amount_of_electrodes + 1, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=chosen_vertical_spacing,
        specs=[[{"secondary_y": True}] for _ in range(amount_of_electrodes + 1)],
        shared_xaxes=True
    )

    # Add a title to the entire figure
    fig.update_layout(
        title_text=save_ending,  # Replace with your actual title
        # title_font_size=20,  # Optional: adjust font size
        title_x=0.5,  # Optional: centers the title (0-1 scale)
        title_font_family="Arial"  # Optional: specify font family
    )
    fig.update_xaxes(showticklabels=True)
    return fig


def plot_encoding_coeffs_corr(electrode_names, corr_df, computed_corr_config, fig):
    # Create a mask for non-NaN values
    mask = ~np.isnan(corr_df['pval'])

    # Apply FDR correction only to non-NaN values
    rejected_nonan, pval_corrected_nonnan, _, _ = multipletests(corr_df['pval'][mask], method='fdr_bh')

    # Initialize corrected p-values with NaNs and put the corrected values back in their original positions
    # corr_df["corrected_pval"] = np.nan
    # corr_df.loc[mask, "corrected_pval"] = pval_corrected_nonnan
    #
    # corr_df["rejected"] = np.nan
    # corr_df.loc[mask, "rejected"] = rejected_nonan

    corr_df["corrected_pval"] = pd.NA
    corr_df["corrected_pval"] = corr_df["corrected_pval"].astype("Float64")  # nullable float
    corr_df.loc[mask, "corrected_pval"] = pval_corrected_nonnan

    corr_df["rejected"] = pd.NA
    corr_df["rejected"] = corr_df["rejected"].astype("boolean")  # nullable boolean
    corr_df.loc[mask, "rejected"] = rejected_nonan

    # # Plot histogram of correlations
    # corr_hist_fig = go.Figure(data=[go.Histogram(x=corr_df["corr"], histnorm='percent', marker_color="#A865C9")])
    # text = f"{computed_corr_config[0][1]} ({computed_corr_config[0][0]}) and {computed_corr_config[1][1]} ({computed_corr_config[1][0]})"
    #
    # # Customize layout (optional)
    # corr_hist_fig.update_layout(
    #     title=f"Correlation Histogram of {text}",
    #     yaxis_title="Percentage",
    #     bargap=0.1,
    #     template='simple_white',
    #     xaxis=dict(
    #         title=f"Correlation of {text}",
    #         range=[0, 1]
    #     )
    # )
    #
    # # Calculate mean
    # mean_val = corr_df["corr"].mean()
    #
    # corr_hist_fig.add_shape(
    #     type="line",
    #     x0=mean_val, y0=0,
    #     x1=mean_val, y1=1,
    #     xref="x", yref="paper",
    #     line=dict(color="grey", width=2, dash="dash"),
    #     layer="above"
    # )
    # corr_hist_fig.add_annotation(
    #     x=mean_val,
    #     y=1,
    #     yref="paper",
    #     text=f"Mean: {mean_val:.3f}",
    #     showarrow=False,
    #     yshift=10
    # )
    #
    # # Show the plot
    # corr_hist_fig.show()

    for elec_idx, elec_name in enumerate(electrode_names):
        row_idx = elec_idx + 1
        corr = corr_df[corr_df["full_elec_name"] == elec_name]["corr"].values[0]
        p_value = corr_df[corr_df["full_elec_name"] == elec_name]["corrected_pval"].values[0]
        # print(f"  Correlation for {elec_name}: r = {corr:.3f}, p = {p_value:.8f}")

        # Annotate the subplot for this electrode with the correlation values
        annotation_text = f"r={corr:.2f}, p={p_value:.3f}"
        fig.add_annotation(
            text=annotation_text,
            x=0.02,
            y=0.9,
            xref="x domain" if row_idx == 1 else f"x{row_idx} domain",
            yref="y domain" if row_idx == 1 else f"y{2 * row_idx - 1} domain",
            showarrow=False,
            align="left",
            font=dict(size=18, color="black"),
            bgcolor="rgba(255,255,255,0.6)",
            row=row_idx,
            col=1,
        )


def plot_coeffs_heatmap(patient, mode, filter_type, min_alpha, max_alpha, amount_of_alphas,
                                       p_threshold, reliable_kfolds_threshold, models_info, models_to_plot, lines_to_plot,
                                       save_dir, save_ending, emb_mod, sort_by_first, sort_by_second, query=""):
    is_english = patient not in NON_ENGLISH_SID
    save_dir = os.path.join(save_dir, f"coeffs_heatmaps_sorted_by_{sort_by_first}_then_by_{sort_by_second}{f'_{save_ending}' if save_ending else ''}")

    for model_name in models_to_plot:
        print(f"Processing model {model_name}")
        save_dir_model = os.path.join(save_dir, model_name)
        os.makedirs(save_dir_model, exist_ok=True)
        dfs = load_electrode_dfs_by_type(lines_to_plot, patient, filter_type, models_info[model_name], min_alpha, max_alpha,
                                         amount_of_alphas, mode, reliable_kfolds_threshold, emb_mod, query) # dict of models to dict of line_types to coeff_exploded_df
        electrode_names_list = dfs[list(dfs.keys())[0]]["full_elec_name"].unique().tolist()
        amount_of_electrodes = len(electrode_names_list)
        plotting_info = get_plotting_info(models_info[model_name], reliable_kfolds_threshold)

        for line_index, line_type in enumerate(lines_to_plot):
            line_df = dfs[line_type]
            # fig = make_subplots(rows=amount_of_electrodes * 2,
            #                     cols=1,
            #                     subplot_titles=[elec_name + suffix for elec_name in electrode_names_list for suffix in ["encoding", "heatmap"]],
            #                     # vertical_spacing=0.04
            #                     )
            # # Link x-axes: for each pair, make the heatmap row match the encoding row
            # for i in range(amount_of_electrodes):
            #     encoding_row = i * 2 + 1  # rows 1, 3, 5, ...
            #     heatmap_row = i * 2 + 2  # rows 2, 4, 6, ...
            #
            #     # xaxis1, xaxis3, xaxis5... are the "anchor" axes
            #     anchor = f"x{encoding_row}" if encoding_row > 1 else "x"
            #     heatmap_axis = f"xaxis{heatmap_row}"
            #
            #     fig.layout[heatmap_axis].matches = anchor

            for elec_idx, elec_name in enumerate(tqdm(electrode_names_list, desc="Processing electrodes"), start=1): # Row count start from 1 + first row reserved for the overall mean plot
                # Create subplots with secondary y-axis
                # fig = make_subplots(
                #     rows=2, cols=1,
                #     subplot_titles=["Encoding and Num of Coeffs", "Heatmap"],
                #     # vertical_spacing=chosen_vertical_spacing,
                #     specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                #     shared_xaxes=True
                # )
                fig = make_subplots(
                    rows=2, cols=1,
                    # subplot_titles=["Encoding and Num of Coeffs", "Heatmap"],
                    specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                    shared_xaxes=True,
                    row_heights=[0.15, 0.85],  # encoding strip is slim, heatmap gets the bulk
                    vertical_spacing=0.0  # no gap between the two subplots
                )

                elec_df = line_df[line_df["full_elec_name"] == elec_name].sort_values("time_index")
                plot_encoding_dual_axis(1, elec_df, plotting_info[line_type], True, fig, True)  # True if "lasso" in line_type else False)
                plot_coeffs_dual_axis(1, elec_df, plotting_info[line_type], model_name, True, fig, filter_type, [line_type], False, True, is_english, compare_only_encoding)

                customize_encoding_and_coeffs_dual_axis_layout(0, fig, filter_type, lines_to_plot, is_english, compare_only_encoding, fix_height=False)
                plot_coeffs_single_heatmap(2, elec_df, fig, sort_by_first, sort_by_second)

                # Add a title to the entire figure
                fig.update_layout(
                    title_text=f"{elec_name}_{line_type}_{model_name}_coeffs_heatmaps_sorted_by_{sort_by_first}_then_by_{sort_by_second}",  # Replace with your actual title
                    # title_font_size=20,  # Optional: adjust font size
                    title_x=0.5,  # Optional: centers the title (0-1 scale)
                    title_font_family="Arial"  # Optional: specify font family
                )
                # fig.update_xaxes(showticklabels=True)

                # Keep the encoding y-axis from auto-expanding
                fig.update_yaxes(autorange=True, row=1, col=1)

                if elec_idx == 2:
                    fig.show()
                save_path = os.path.join(save_dir_model, f"{line_type}_{elec_name}_coeffs_heatmap.html")
                fig.write_html(save_path)
            print(f"Finished with {line_type}")

        # for elec_name in dfs[list(dfs.keys())[0]]["full_elec_name"].unique().tolist():
        #     print(f"Plotting coefficients heatmap for electrode {elec_name}")
        #     plot_coeffs_heatmap_single_elec(elec_name, dfs, sort_by, save_dir)

    print(f"!!!!!! Plotting complete. HTML files saved as {save_dir} !!!!!!")
    return

def plot_coeffs_single_heatmap(row_idx, elec_df, fig, sort_by_first, sort_by_second=None):
    exploded_df = elec_df.explode(["all_coeffs_index", "all_coeffs_val"])

    # Create a pivot table: coeffs (rows) x time (columns)
    coeffs_matrix = exploded_df.pivot_table(
        index="all_coeffs_index",
        columns="time",
        values="all_coeffs_val",
        aggfunc='first'
    )

    # Filter to only keep coeffs that are non-zero at some time
    non_zero_mask = (coeffs_matrix != 0).any(axis=1)

    n_coeffs = len(non_zero_mask)
    fig_height = min(900, max(400, n_coeffs * 8))  # ~20 px per coefficient row, minimum 600

    fig.update_layout(
        title_text=save_ending,
        title_x=0.5,
        title_font_family="Arial",
        height=fig_height,
        margin=dict(t=60, b=40, l=80, r=20),  # tight margins help too
    )

    coeffs_matrix_filtered = coeffs_matrix[non_zero_mask]

    # if coeffs_matrix_filtered.empty:
    #     raise ValueError(f"Warning: No non-zero coefficients found for {elec_name} in {df_type}")

    # Get the coefficient indices that remain after filtering
    coeff_indices = coeffs_matrix_filtered.index.values
    z_values = coeffs_matrix_filtered.values
    time_values = coeffs_matrix_filtered.columns.values

    sorted_indices = sort_coeffs(sort_by_first, sort_by_second, z_values, elec_df)

    # Apply sorting
    z_values_sorted = z_values[sorted_indices, :]
    coeff_labels_sorted = coeff_indices[sorted_indices]

    # Create heatmap
    fig.add_trace(go.Heatmap(
        z=(z_values_sorted != 0).astype(int),
        x=time_values,
        y=coeff_labels_sorted.astype(str),
        colorscale='RdBu_r',  # Red-Blue colorscale, good for positive/negative values
        zmid=0,  # Center colorscale at 0
        showscale=True,
        colorbar=dict(
            title="Coeff Value",
            len=0.85,
            y=0.425,         # centers it within the heatmap row (0.85 / 2 ≈ 0.425)
            yanchor="middle",
            thickness=15,
        ),
    ), row=row_idx, col=1)

    # # Update y-axis for this subplot
    # fig.update_yaxes(title=f'{df_type} - Coeff Index', row=row_idx + 1, col=1)
    #
    # # Update layout
    # fig.update_xaxes(title='Time (s)', row=len(dfs) + 1, col=1)
    # fig.update_layout(
    #     height=400 * (len(dfs) + 1),
    #     width=1200,
    #     template='simple_white',
    #     title_text=f'Coefficients Heatmap - {elec_name}',
    #     showlegend=False
    # )
    return


def sort_coeffs(sort_by_first, sort_by_second, z_values, elec_df):
    first_sort = sort_coeffs_helper(sort_by_first, z_values, elec_df)
    if not sort_by_second:
        sort_indices = np.argsort(first_sort)
    else:
        second_sort = sort_coeffs_helper(sort_by_second, z_values, elec_df)
        sort_indices = np.lexsort((second_sort, first_sort))
    return sort_indices


def sort_coeffs_helper(sort_by, z_values, elec_df):
    # Sort the coefficients based on the sort_by parameter
    if sort_by == "sum":
        features_array = -np.abs(z_values).sum(axis=1)  # Sort by sum of absolute values
    elif sort_by == "first_true":
        features_array = features_array_by_first(z_values != 0)
    elif sort_by == "last_true":
        features_array = features_array_by_last(z_values != 0)
    elif sort_by == "first_last_gap":
        first_true = features_array_by_first(z_values != 0)
        last_true = features_array_by_last(z_values != 0)
        features_array = first_true-last_true
    elif sort_by == "early_vs_late":
        n = z_values.shape[1]
        mid = n / 2
        first_true = features_array_by_first(z_values != 0)
        last_true = features_array_by_last(z_values != 0)
        start_early = first_true < mid
        end_early = last_true < mid

        # 0: start early + end early, 1: start early + end late, 2: start late + end late
        features_array = np.where(start_early & end_early, 2,
                            np.where(start_early & ~end_early, 0, 1))
    elif sort_by == "early_vs_late_continues":
        n = z_values.shape[1]
        first_true = features_array_by_first(z_values != 0)
        last_true = features_array_by_last(z_values != 0)
        features_array = first_true + last_true  # low = both early, high = both late, mid = one of each
    elif sort_by == "early_vs_late_count":
        n = z_values.shape[1]
        mid = n // 2
        count_early = (z_values!= 0)[:, :mid].sum(axis=1)  # number of True in first half
        count_late = (z_values!= 0)[:, mid:].sum(axis=1)  # number of True in second half
        total = (z_values!= 0).sum(axis=1) #1
        features_array = (count_late - count_early) / np.where(total > 0, total, 1)  # in [-1, 1]. negative = leans early, positive = leans late
    elif sort_by == "num_of_appear":
        features_array = -(z_values != 0).sum(axis=1)
    elif sort_by == "peak_encoding":
        # Find the time point where encoding is at its peak
        peak_encoding_idx = elec_df['encoding'].argmax()
        features_array = -np.abs(z_values[:, peak_encoding_idx])
    else:
        raise ValueError(f"Unknown sort method: {sort_by}")
    return features_array

def features_array_by_first(z_values):
    features_array = np.argmax(z_values, axis=1)  # Get the index of the first True value in each row
    no_true_mask = ~z_values.any(axis=1)
    features_array[no_true_mask] = z_values.shape[1]
    return features_array


def features_array_by_last(z_values):
    features_array = (z_values.shape[1] - 1) - np.argmax(z_values[:, ::-1], axis=1)
    no_true_mask = ~z_values.any(axis=1)
    features_array[no_true_mask] = -1  # or 0, sorts these to the beginning
    return features_array


def customize_encoding_and_coeffs_layout(amount_of_electrodes, fig):
    # Customize layout
    for idx in range(amount_of_electrodes):
        row_idx = idx + 1
        # print("Customizing layout for row:", row_idx)
        # fig.add_hline(y=0, line=dict(color='lightgrey', width=1), row=row_idx, col=1)
        # fig.add_vline(x=0, line=dict(color='lightgrey', width=1), row=row_idx, col=1)

        # Add horizontal line at y=0
        fig.add_shape(type='line',
                      x0=-2, x1=2,
                      y0=0, y1=0,
                      line=dict(color='black', width=2), row=row_idx, col=1)

        fig.add_shape(type='line',
                      x0=-2, x1=2,
                      y0=0, y1=0,
                      line=dict(color='black', width=2), row=row_idx, col=2)

        fig.add_shape(type='line',
                      x0=-2, x1=2,
                      y0=0, y1=0,
                      line=dict(color='black', width=2), row=row_idx, col=3)

        # Add vertical line at x=0
        fig.add_shape(type='line',
                      x0=0, x1=0,
                      y0=-0.1, y1=0.5,
                      line=dict(color='black', width=2), row=row_idx, col=1)

        fig.add_shape(type='line',
                      x0=0, x1=0,
                      y0=0, y1=1,
                      line=dict(color='black', width=2), row=row_idx, col=2)

        fig.add_shape(type='line',
                      x0=0, x1=0,
                      y0=0, y1=1000,
                      line=dict(color='black', width=2), row=row_idx, col=3)

        # fig.add_annotation(
        #     text=row_titles[idx],
        #     xref="paper", yref="paper",
        #     x=0.5,
        #     y= 1 - (idx / amount_of_electrodes) + (chosen_vertical_spacing / 2), # Position at top of each row
        #     showarrow=False,
        #     font=dict(size=14, color="black"),
        #     xanchor="center"
        # )
    # Show ticks
    fig.update_xaxes(showticklabels=True, col=1)
    fig.update_xaxes(showticklabels=True, col=2)
    fig.update_xaxes(showticklabels=True, col=3)

    fig.update_yaxes(title="Correlation (r)", col=1)
    fig.update_yaxes(title="Proportion of Coefficients", col=2)
    fig.update_yaxes(title="Number of Coefficients", col=3)

    fig.update_layout(height=300 * amount_of_electrodes, width=2100, template='simple_white',
                      legend=dict(
                          title="Encoding Methods",
                          orientation="v",
                          yanchor="top",
                          y=1,
                          xanchor="left",
                          x=1.02  # Position between columns
                      ),
                      legend2=dict(
                          title="Coefficient Counts",
                          orientation="v",
                          yanchor="top",
                          y=1,
                          xanchor="left",
                          x=1.18  # Position to the right of the plot
                      )
                      )
    # fig.update_xaxes(title="Lags (s)", col=1)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)


def prep_paths(elec_name, max_alpha, min_alpha, mode, model_name, models_info, num_alphas, patient, filter_type=None, emb_mod=None):
    model_path = models_info[model_name]['model_path']
    layer = models_info[model_name]['layer']
    context = models_info[model_name]['context']

    file_suffix = f"-mod_{emb_mod}" if emb_mod else ""

    # Prep general paths
    kfolds_path_template = f"{model_path}/tk-200ms-{patient}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}{file_suffix}/{elec_name}_{mode}{{ending}}"
    kfolds_train_path_template = f"{model_path}/tk-200ms-{patient}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}{file_suffix}_train/{elec_name}_{mode}{{ending}}"
    all_data_path_template = f"{model_path}/tk-200ms-{patient}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}-all_data{file_suffix}/{elec_name}_{mode}{{ending}}"
    sig_coeffs_path_template = f"{model_path}/tk-200ms-{patient}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}-sig_coeffs{file_suffix}/{elec_name}_{mode}{{ending}}"
    corr_path_template = f"{model_path}/tk-200ms-{patient}-lay{layer}-con{context}-corr_coeffs"

    # Prep encodings paths
    kfolds_lasso_enc_path = kfolds_path_template.format(ending=".csv")
    kfolds_train_lasso_enc_path = kfolds_train_path_template.format(ending=".csv")
    all_data_enc_path = all_data_path_template.format(ending=".csv")
    lasso_enc_path = sig_coeffs_path_template.format(ending="_lasso.csv")
    ols_enc_path = sig_coeffs_path_template.format(ending="_ols.csv")

    kfolds_ridge_enc_path = f"/scratch/gpfs/HASSON/tk6637/princeton/247-encoding/results/podcast/tk-podcast-777-gemma-2-2b-lag2k-25-all/tk-200ms-777-lay13-con32-regridge-alphas_-2_30_40/{elec_name}_{mode}.csv"

    # Prep coeffs paths
    kfolds_lasso_coeffs_path = kfolds_path_template.format(ending="_coeffs.npy")
    kfolds_train_lasso_coeffs_path = kfolds_train_path_template.format(ending="_coeffs.npy")
    all_data_coeffs_path = all_data_path_template.format(ending="_coeffs.npy")
    lasso_coeffs_path = sig_coeffs_path_template.format(ending="_coeffs_lasso.npy")
    ols_coeffs_path = sig_coeffs_path_template.format(ending="_ols.pkl")
    pvals_names_path = f"{corr_path_template}/pvals_combined_names{f'({filter_type})' if filter_type else ''}.pkl"
    pvals_combined_corrected_path = f"{corr_path_template}/pvals_combined_corrected{f'({filter_type})' if filter_type else ''}.npy"

    return (kfolds_lasso_enc_path, kfolds_lasso_coeffs_path,
            kfolds_train_lasso_enc_path, kfolds_train_lasso_coeffs_path,
            all_data_enc_path, all_data_coeffs_path,
            lasso_enc_path, lasso_coeffs_path,
            ols_enc_path, ols_coeffs_path,
            pvals_names_path, pvals_combined_corrected_path,)
    # kfolds_ridge_enc_path)

def plot_encoding_dual_axis(row_idx, elec_df, plots_info, show_legend, fig, plot_std=False, model_color=None, dual_axis=True):
    # color = models_info[model_name]['colors'][0] #[plots_data[line_type]["color_index"]]
    color = ENCODING_COLORS if model_color is None else model_color

    enc_name = plots_info["enc_name"]
    dash = plots_info["dash"]

    if plot_std:
        y_upper = elec_df['encoding'] + elec_df["encoding_se"]
        y_lower = elec_df['encoding'] - elec_df["encoding_se"]

        # Add standard deviation to kfolds
        fig.add_trace(go.Scatter(
            x=np.concatenate([elec_df['time'], elec_df['time'][::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=f"rgba({','.join(color)},0.2)",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=enc_name,
            name='±1 SD',
            yaxis="y2" if dual_axis else "y1"
        ), row=row_idx, col=1, secondary_y=True if dual_axis else False)

    elif enc_name == "ols":
        elec_df['encoding'] = np.sqrt(elec_df['encoding'])  # OLS original encoding is r^2

    # Add the main line
    fig.add_trace(go.Scatter(
        x=elec_df['time'],
        y=elec_df['encoding'],
        mode='lines',
        name=enc_name,
        line=dict(color=f"rgb({','.join(color)})", width=2, dash=dash),
        legendgroup=enc_name,
        showlegend=show_legend,
        yaxis="y2" if dual_axis else "y1"
    ), row=row_idx, col=1, secondary_y=True if dual_axis else False)

def calc_encoding_coeffs_corr(corr_data):

    # Calculate Pearson correlation
    # corr, p_value = pearsonr(corr_data[0], corr_data[1])
    def calc_corr(group):
        corr, pval = pearsonr(group['first_to_corr'], group['second_to_corr'])
        return pd.Series({'corr': corr, 'pval': pval})

    corr_df = corr_data.groupby('full_elec_name').apply(calc_corr, include_groups=False).reset_index()
    return corr_df

def plot_encoding_num_coeffs_scatter(df, model_name, line_type):
    import plotly.express as px
    from scipy.stats import pearsonr
    # df = dfs["corr"]

    # Calculate correlation and p-value
    corr, p_value = pearsonr(df['encoding'], df['num_of_chosen_coeffs'])

    # Create scatter plot with trendline
    fig = px.scatter(
        df,
        x='encoding',
        y='num_of_chosen_coeffs',
        trendline='ols',
        trendline_color_override='black',
        title=f'Encoding vs. Num of Coefficients for {model_name} type {line_type}',
        labels={'encoding': 'Encoding', 'num_of_chosen_coeffs': 'Num of Coefficients'},
    )

    # Set fixed figure size
    fig.update_layout(
        width=500,
        height=500,
        template='simple_white',
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        xaxis_tickfont_size=18,
        yaxis_tickfont_size=18,
    )

    # Make points smaller
    fig.update_traces(
        marker=dict(
            size=3,
            color='rgb(100, 181, 246)'
        ),
        selector=dict(mode='markers')
    )

    # Add annotation
    fig.add_annotation(
        text=f"r={corr:.2f}, p={p_value:.3f}",
        x=0.02,
        y=0.98,
        xref="x domain",
        yref="y domain",
        showarrow=False,
        align="left",
        font=dict(size=18, color="black"),
        bgcolor="rgba(255,255,255,0.6)"
    )

    y_max = df['num_of_chosen_coeffs'].max() * 1.1  # Add 10% padding
    fig.update_yaxes(range=[-5, y_max])

    fig.show()

def plot_coeffs_dual_axis(row_idx, elec_df, plots_info, model_name, show_legend, fig, filter_type, lines_to_plot, plot_std=False, plot_percentage=False, is_english=True, compare_only_encoding=False):

    # color = models_info[model_name]['colors'][1]# [plots_data[line_type]["color_index"]]
    line_color, axis_color = get_coeff_color(filter_type, lines_to_plot)
    line_color = (ENCODING_COLORS if model_name == "gemma9b" else NEUTRAL_COLOR) if compare_only_encoding else line_color

    coeffs_name = plots_info["coeffs_name"]
    dash = plots_info["dash"]
    embedding_size = len(elec_df["all_coeffs_index"].reset_index(drop=True)[0])
    mean_num_coeffs = elec_df['num_of_chosen_coeffs']

    if plot_percentage:
        mean_num_coeffs = mean_num_coeffs / embedding_size

    if plot_std:
        std_num_coeffs = elec_df["num_of_chosen_coeffs_se"] if not plot_percentage else elec_df["num_of_chosen_coeffs_se"]/embedding_size

        y_upper = mean_num_coeffs + std_num_coeffs
        y_lower = mean_num_coeffs - std_num_coeffs

        # Add standard deviation
        fig.add_trace(go.Scatter(
            x=np.concatenate([elec_df['time'], elec_df['time'][::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=f"rgba({','.join(line_color)},0.2)",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=coeffs_name,
            name='±1 SD',
            yaxis="y1"
        ), row=row_idx, col=1, secondary_y=False)

    # if line_type == "lasso":
    #     dash = "longdash"
    #     coeffs = np.load(coeffs_path)
    #     non_zero_counts = np.count_nonzero(coeffs, axis=1)
    #     coeffs_mean = non_zero_counts.mean(axis=0)
    #     coeffs_std = non_zero_counts.std(axis=0)
    #
    #     y_upper = coeffs_mean + coeffs_std
    #     y_lower = coeffs_mean - coeffs_std
    #
    #     fig.add_trace(go.Scatter(
    #         x=np.concatenate([time, time[::-1]]),
    #         y=np.concatenate([y_upper, y_lower[::-1]]),
    #         fill='toself',
    #         fillcolor=f"rgba({','.join(color)},0.2)",
    #         line=dict(color='rgba(255,255,255,0)'),
    #         hoverinfo="skip",
    #         showlegend=False,
    #         legendgroup=coeffs_name,
    #         name='±1 SD',
    #         yaxis="y2"
    #     ), row=row_idx, col=1, secondary_y=True)
    #
    #     num_of_coeffs = coeffs_mean

    fig.add_trace(go.Scatter(
        x=elec_df['time'],
        y=mean_num_coeffs,
        mode='lines',
        name=coeffs_name,
        legendgroup=coeffs_name,
        line=dict(color=f"rgb({','.join(line_color)})", width=2, dash=dash),
        showlegend=show_legend,
        yaxis="y1"
    ), row=row_idx, col=1, secondary_y=False)
    # Ratio of 1*5.75
    # if plot_percentage:
    #     if row_idx == 1:
    #         if ("gemma-scope" in model_name):
    #             if "kfold" in coeffs_name.lower():
    #                 range = [-0.0005, 0.0055]
    #             else:
    #                 range = [-0.0017, 0.01]
    #         elif ("gemma" in model_name) or ("arb" in model_name) or ("rand" in model_name):
    #             if ("kfold" in coeffs_name.lower()):
    #                 range = [-0.0018, 0.02]
    #             else:
    #                 range = [-0.07, 0.4]
    #         elif ("mistral" in model_name):
    #             if "kfold" in coeffs_name.lower():
    #                 range = [-0.0012, 0.0125]
    #             else:
    #                 range = [-0.07, 0.4]
    #         elif ("gpt2" in model_name):
    #             range = [-0.020, 0.225]
    #         else:
    #             range = [-0.07, 0.4]
    #     else:
    #         if ("mistral" in model_name):
    #             if "kfold" in coeffs_name.lower():
    #                 if is_english:
    #                     range = [-0.007, 0.04]
    #                 else:
    #                     range = [-0.0017, 0.01]
    #             else:
    #                 range = [-0.07, 0.4]
    #         else:
    #             range = [-0.174, 1]
    # elif "glove" in model_name:
    #     range = [-9, 45]
    # elif "gemma-scope" in model_name:
    #     range = [-140, 700]
    # elif "lasso" in coeffs_name.lower() and "corr" in coeffs_name.lower():
    #     range = [-26, 150]
    # elif "kfold" in coeffs_name.lower():
    #     range = [-27.8, 160]
    # elif "corr" in coeffs_name:
    #     range = [-435, 2500]  # For corr
    # else:
    #     # range = [-25, 125]
    #     range = [-26, 130]
    # if (row_idx != 1) or (plot_percentage and row_idx == 1):
    #     fig.update_yaxes(range=range, row=row_idx, col=1, secondary_y=False)

# def add_loc_to_plot(row_idx, fig, elec_name):
#     real_sid = int(elec_name.split("_", 1)[0])  # Extract the real subject ID
#     real_electrode = elec_name.split("_", 1)[1]
#
#     elec_locations = get_elec_locations(sid)
#     loc = elec_locations[(elec_locations["subject"] == real_sid) & (elec_locations["name"] == real_electrode)][
#         ["princeton_class", "NYU_class"]]
#
#     # Plot on the right side of the plot
#     annotation_text = f"Elec: {elec_name}<br>Princeton: {loc['princeton_class'].values[0]}<br>NYU: {loc['NYU_class'].values[0]}"
#     fig.add_annotation(
#         x=2.1,  # Position to the right of the plot
#         y=0.5,  # Center vertically
#         xref=f"x{row_idx}",
#         yref=f"y{row_idx}",
#         text=annotation_text,
#         showarrow=False,
#         font=dict(size=12, color="black"),
#         align="left",
#         bordercolor="black",
#         borderwidth=1,
#         borderpad=4,
#         bgcolor="white",
#         opacity=0.8
#     )


def customize_encoding_and_coeffs_dual_axis_layout(amount_of_electrodes, fig, filter_type, lines_to_plot, is_english=True, compare_only_encoding=False, fix_height=True):
    # Add horizontal line at y=0 for encoding (spans full x-axis)
    fig.add_hline(y=0,
                  line=dict(color='black', width=2),
                  row=1, col=1)

    # Add vertical line at x=0 (spans full y-axis)
    # fig.add_vline(x=0,
    #               line=dict(color='black', width=2),
    #               row=1, col=1)
    fig.add_shape(type='line',
                  x0=0, x1=0,
                  # y0=-0.025, y1=0.275,
                  y0=-0.02, y1=0.215,
                  line=dict(color='black', width=2), row=1, col=1, secondary_y=True)

    for row_idx in range(2, amount_of_electrodes + 1):
        # row_idx = idx + 1

        # Add horizontal line at y=0 for encoding
        fig.add_shape(type='line',
                      x0=-2, x1=2,
                      y0=0, y1=0,
                      line=dict(color='black', width=2), row=row_idx, col=1)

        # Add vertical line at x=0
        if is_english:
            fig.add_shape(type='line',
                          x0=0, x1=0,
                          y0=-0.1, y1=0.575,
                          line=dict(color='black', width=2), row=row_idx, col=1, secondary_y=True)
        else:
            fig.add_shape(type='line',
                          x0=0, x1=0,
                          y0=-0.02, y1=0.215,
                          line=dict(color='black', width=2), row=row_idx, col=1, secondary_y=True)

    # # Set axis labels and properties
    _, coeffs_axis_color = get_coeff_color(filter_type, lines_to_plot)
    coeffs_axis_color = f"rgb(0, 0, 0)" if compare_only_encoding else coeffs_axis_color
    enc_axis_color = f"rgb(0, 0, 0)" if compare_only_encoding else CORR_AXIS_COLOR

    for i in range(1, amount_of_electrodes + 2):
        fig.update_yaxes(title="Encoding (r)",
                         title_font=dict(color=enc_axis_color, size=20),
                         tickfont=dict(color=enc_axis_color, size=18),
                         linecolor=enc_axis_color,
                         row=i, col=1, secondary_y=True)
        fig.update_yaxes(title="Proportion of Coefficients",#"Number of Coeffs",
                         title_font=dict(color=coeffs_axis_color, size=20),
                         tickfont=dict(color=coeffs_axis_color, size=18),
                         linecolor=coeffs_axis_color,
                         row=i, col=1, secondary_y=False)

    fig.update_xaxes(title="Time (s)",
                 title_font=dict(size=20),
                 tickfont=dict(size=18),
                     title_standoff=11)#, showgrid=True)
    # fig.update_yaxes(showgrid=True, secondary_y=False)
    # fig.update_yaxes(showgrid=True, secondary_y=True)

    layout_kwargs = dict(
        width=1200,
        template='simple_white',
        legend=dict(
            title="Methods",
            orientation="v",
            yanchor="top",
            y=1,#0.95,
            xanchor="left",
            x=1.02
        ),
        hoverlabel=dict(
            namelength=-1  # Set to -1 to show the full name
        )
    )
        hoverlabel=dict(namelength=-1)  # Set to -1 to show the full name
    )
    if fix_height:
        layout_kwargs['height'] = 300 * amount_of_electrodes

    fig.update_layout(**layout_kwargs)
    # fig.update_yaxes(tickformat='.4f')


def add_models_path(models_info, sid):
    recording_type = get_recording_type(sid)
    for model_name in models_info.keys():
        model_path = f"../data/encoding/{recording_type}/tk-{recording_type}-{sid}-{model_name}-lag2k-25-all"
        models_info[model_name]['model_path'] = model_path
    return models_info


if __name__ == '__main__':
    # Choose parameters:
    models_info = {
        "glove": {"model_full_name": "glove50", "layer": 0, "context": 1, "embedding_size": 50,"model_short_name":"glove", "dash": "solid", "colors": [('187','222','251'),('100','181','246'),('33','150','243'),('25','118','210'),('13','71','161')]},
        "gemma2b": {"model_full_name": "gemma-2-2b", "layer": 13, "context": 32, "embedding_size": 2304, "model_short_name": "gemma2b", "dash": "solid", "colors": [('255', '158', '191'), ('158', '210', '255')]},
        # [('200','230','201'),('129','199','132'),('76','175','80'),('56','142','60'),('27','94','32')]},
        "gemma9b": {"model_full_name": "gemma-2-9b", "layer": 21, "context": 32, "embedding_size": 3584, "model_short_name":"gemma9b", "dash": "solid", "colors": [('255', '158', '191'), ('158', '210', '255')]},#[('200','230','201'),('129','199','132'),('76','175','80'),('56','142','60'),('27','94','32')]},
        "arb": {"model_full_name": "gemma-2-9b", "layer": 21, "context": 32, "embedding_size": 3584, "model_short_name":"gemma9b", "dash": "solid", "colors": [('255', '158', '191'), ('158', '210', '255')]},#[('200','230','201'),('129','199','132'),('76','175','80'),('56','142','60'),('27','94','32')]},
        "gpt2": {"model_full_name": "gpt2-xl", "layer": 24, "context": 32, "embedding_size": 1600, "model_short_name": "gpt2", "dash": "solid",
                    "colors": [('218', '204', '233'), ('181', '153', '211'), ('144', '102', '189'),
                               ('122', '58', '192'), ('70', '0', '145')]},
        "gemma-scope2b": {"model_full_name": "gemma-scope-2b-pt-res-canonical", "layer": 13, "context": 32, "embedding_size": 16384,
                                            "model_short_name": "gemma-scope2b", "dash": "solid",
                                            "colors": [('190', '48', '96'), ('50', '131', '196')]},
                                            # "colors": [('248', '187', '208'), ('240', '98', '146'), ('233', '30', '99'),
                                            #            ('194', '24', '91'), ('136', '14', '79')]},
        "gemma-scope9b": {"model_full_name": "gemma-scope-9b-pt-res-canonical", "layer": 21, "context": 32, "embedding_size": 16384,
                                            "model_short_name": "gemma-scope9b", "dash": "solid",
                                            "colors": [('190', '48', '96'), ('50', '131', '196')]},
        "gemma-scope9b-mlp": {"model_full_name": "gemma-scope-9b-pt-res-canonical", "layer": 21, "context": 32, "embedding_size": 16384,
                              "model_short_name": "gemma-scope9b-mlp", "dash": "solid",
                              "colors": [('190', '48', '96'), ('50', '131', '196')]},
                                            # "colors": [('248', '187', '208'), ('240', '98', '146'), ('233', '30', '99'),
                                            #            ('194', '24', '91'), ('136', '14', '79')]},
        "llama8b": {"model_full_name": "Meta-Llama-3.1-8B", "layer": 16, "context": 32, "embedding_size": 4096,
                                            "model_short_name": "llama8b", "dash": "solid",
                                            "colors": [('255', '158', '191'), ('158', '210', '255')]},#[('187','222','251'),('100','181','246'),('33','150','243'),('25','118','210'),('13','71','161')]},
        "mistral7b": {"model_full_name": "Mistral-7B-v0.3", "layer": 16, "context": 32, "embedding_size": 4096, "model_short_name": "mistral7b", "dash": "solid", "colors": [('255', '158', '191'), ('158', '210', '255')]},
        "symbolic-lang": {"model_full_name": "symbolic-lang", "layer": 0, "context": 1, "model_short_name": "symbolic-lang", "dash": "solid", "colors": [('255', '158', '191'), ('158', '210', '255')]},
    }

    patient = 777#777 7, 8
    mode = "comp"
    filter_type = "160"  # Options: None, 39, 50, 160, "IFG", "IFG160", "STG", "STG160"

    min_alpha = -2  # -0.7
    max_alpha = 10  # 1.27
    amount_of_alphas = 100  # 30

    p_threshold = 0.05
    reliable_kfolds_threshold = 10

    emb_mod = None  # "shift-emb" / "arb"
    models_to_plot = ["gemma9b", "gemma-scope9b"]#, "llama8b"] #arb
    lines_to_plot = ["ridge", "lasso"] # Options: "lasso&corr", "lasso", "reliable_lasso", "all_data", "lasso", "ols", "corr", "corr_pca" or None
    computed_corr_config = [("num_of_chosen_coeffs", "lasso"),
                            ("encoding", "lasso")]
    query = ""


    if "arb" in models_to_plot:
        emb_mod = "arb"
    computed_corr_str = '_'.join([f'{k}_{v}' for k, v in computed_corr_config])
    # encoding_corr_line_type="kfold
    # coeffs_corr_line_type="corr"

    models_str = "_".join(models_to_plot)
    lines_to_plot_str = "_".join(lines_to_plot)
    save_dir = "../results/figures/coeffs_analysis"
    sort_by_first = "early_vs_late_count"  # Options: "sum", "first_then_last_then_sum", "first_true", "last_true", "neuron_index", "raster", "num_of_appear"
    sort_by_second = "first_last_gap"

    save_ending = f"{filter_type}_filter_models-{models_str}_lines-{lines_to_plot_str}_kfoldsthresh-{reliable_kfolds_threshold}_alphas-{min_alpha}_{max_alpha}_{amount_of_alphas}{f'_{emb_mod}' if emb_mod else ''}_corr-{computed_corr_str}"  # Optional, can be empty string if not needed
    print(save_ending)
    compare_only_encoding = False
    plot_encoding_and_coeffs_dual_axis(patient, mode, filter_type, min_alpha, max_alpha, amount_of_alphas, p_threshold, reliable_kfolds_threshold,
                                       models_info, models_to_plot, lines_to_plot, computed_corr_config, save_dir, save_ending, emb_mod, query=query, compare_only_encoding=compare_only_encoding)

    save_ending = f"{filter_type}_filter_kfoldsthresh-{reliable_kfolds_threshold}_alphas-{min_alpha}_{max_alpha}_{amount_of_alphas}{f'_{emb_mod}' if emb_mod else ''}_corr-{computed_corr_str}"  # Optional, can be empty string if not needed
    print(save_ending)
    plot_coeffs_heatmap(patient, mode, filter_type, min_alpha, max_alpha, amount_of_alphas, p_threshold, reliable_kfolds_threshold, models_info, models_to_plot, lines_to_plot,
                        save_dir, save_ending, emb_mod, sort_by_first, sort_by_second, query=query)