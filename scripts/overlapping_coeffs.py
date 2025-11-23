import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from upsetplot import from_contents, UpSet
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact, chi2_contingency
import requests
from typing import Optional, Dict, Any
import json
import time
from matplotlib_venn import venn2, venn3
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests
import plotly.express as px
from itertools import combinations
import statsmodels.api as sm
import statsmodels.formula.api as smf

from tfsplt_utils import load_electrode_names_and_locations, get_non_zero_coeffs_old, get_non_zero_coeffs, get_coeffs, _process_coeff_df


AREA_QUERY = "(princeton_class == 'STG') | (princeton_class == 'IFG')"
TIME_QUERY = "(time_bin == '-0.4≤x<0') | (time_bin == '0≤x<0.4')"

ROUNDED_ENCODING = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
ORDERED_PRINCETON_CLASS = ['aMTG', 'MFG', 'pmtg', 'other', 'premotor', 'TP', 'AG', 'IFG', 'parietal', 'precentral', 'postcg', 'STG'] # Sorted by mean num_of_coeffs
# "ORDERED_PRINCETON_CLASS": ['IFG', 'STG', 'AG', 'TP', 'precentral', 'pmtg', 'parietal', 'other', 'MFG', 'aMTG', 'premotor', 'postcg'], # Sorted by amount of data
TIME_BIN = ['x<-0.8', '-0.8≤x<-0.4', '-0.4≤x<0', '0≤x<0.4', '0.4≤x<0.8', '0.8≤x']
# TIME_BIN = ['x<-0.6', '-0.6≤x<-0.3', '-0.3≤x<0', '0≤x<0.3', '0.3≤x<0.6', '0.6≤x']

time_bin_and_princeton_class_by_area = [area +"_" + time for area in ORDERED_PRINCETON_CLASS for time in TIME_BIN]
time_bin_and_princeton_class_by_time = [area +"_" + time for time in TIME_BIN for area in ORDERED_PRINCETON_CLASS]

ORDERS = {"rounded_encoding": ROUNDED_ENCODING,
          "time_bin": TIME_BIN,
          "princeton_class": ORDERED_PRINCETON_CLASS,
          "time_bin_and_princeton_class": time_bin_and_princeton_class_by_area
          }
# 'x<-0.8':'#0077b6', '-0.8≤x<-0.4':'#0096c7', '-0.4≤x<0':'#00b4d8', '0≤x<0.4':'#ff9e00', '0.4≤x<0.8':'#ff9100', '0.8≤x':'#ff8500'
COLOR_PALETTE = {"IFG": '#ff4da0', "STG": '#00c16a', #IFG-pink, STG-green
                 # 'x<-0.8':'#ff8500', '-0.8≤x<-0.4':'#ff9100', '-0.4≤x<0':'#ff9e00', '0≤x<0.4':'#00b4d8', '0.4≤x<0.8':'#0096c7', '0.8≤x':'#0077b6',
                 '-0.4≤x<0':'#ff9d00', '0≤x<0.4':'#ff6e01', #Dark then light
                 'IFG_-0.4≤x<0':"#ff7fbc", 'IFG_0≤x<0.4':"#f72585", 'STG_-0.4≤x<0':"#26cc80", 'STG_0≤x<0.4':"#008e4d",
                 0: "#ff8500", 0.1: "#ff9100", 0.2: "#ff9e00", 0.3: "#00b4d8", 0.4: "#0096c7", 0.5: "#0077b6",}
               # 'precentral': "#C6E7FF", 'premotor': "#C6E7FF", 'MFG': "#C6E7FF", 'postcg': "#C6E7FF", 'aMTG': "#C6E7FF", 'TP': "#C6E7FF", 'AG': "#C6E7FF", 'parietal': "#C6E7FF", 'pmtg': "#C6E7FF", 'other': "#C6E7FF"}
                # B4F8C8, #FBE7C6


def plot_coeff_kfold_agreement_over_time(kfold_coeffs_count, all_data_coeffs_counts):
    """
    Plots
    :param kfold_coeffs_count:
    :param all_data_coeffs_counts:
    :return:
    """
    time_points = np.linspace(-2, 2, 161)
    plot_coeff_kfold_agreement_over_time_histogram(kfold_coeffs_count, time_points)
    plot_coeff_kfold_agreement_over_time_lineplot(kfold_coeffs_count, all_data_coeffs_counts, time_points)


def plot_coeff_kfold_agreement_over_time_lineplot(kfold_coeffs_count, all_data_coeffs_counts, time_points):
    fig = go.Figure()

    # Add multiple traces to the same figure
    fig.add_trace(go.Scatter(
        x=time_points,
        y=(kfold_coeffs_count >= 10).sum(axis=0),
        mode='lines',
        name='>=10',
    ))

    fig.add_trace(go.Scatter(
        x=time_points,
        y=(kfold_coeffs_count >= 9).sum(axis=0),
        mode='lines',
        name='>=9',
    ))

    fig.add_trace(go.Scatter(
        x=time_points,
        y=(kfold_coeffs_count >= 8).sum(axis=0),
        mode='lines',
        name='>=8',
    ))

    fig.add_trace(go.Scatter(
        x=time_points,
        y=all_data_coeffs_counts,
        mode='lines',
        name='all_data_non_zero',
    ))

    # Customize layout
    fig.update_layout(
        title='Multiple Lines Plot',
        xaxis_title='time (s)',
        yaxis_title='#Coeffs',
        legend_title='Functions',
        template='plotly_white'
    )

    # Show the plot
    fig.show()


def plot_coeff_kfold_agreement_over_time_histogram(data, time_points):
    n_plots = data.shape[1]
    # Create subplot titles if not provided
    titles = [f'time: {time_points[i]:.3f}<br>idx: {i}' for i in range(n_plots)]
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=n_plots,
        subplot_titles=titles,
        # horizontal_spacing=0.05  # Adjust spacing between subplots
    )
    # Add a histogram for each dataset
    for i in range(n_plots):
        current_data = data[:, i]
        # Count occurrences of each value
        values, counts = np.unique(current_data, return_counts=True)
        # Filter out zero
        non_zero_mask = values != 0
        values = values[non_zero_mask]
        counts = counts[non_zero_mask]
        # Add vertical histogram to the appropriate subplot
        fig.add_trace(
            go.Bar(
                y=values,
                x=counts,
                orientation='h',
                width=0.8,
                marker_color='royalblue',
                showlegend=False
            ),
            row=1, col=i + 1
        )
        # Update y-axis for each subplot to ensure consistent display
        fig.update_yaxes(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 10.5],
            title_text="Value" if i == 0 else None,  # Only add y-title to first subplot
            row=1, col=i + 1
        )
        # Update x-axis for frequency
        fig.update_xaxes(
            title_text="Frequency",  # Only add x-title to first subplot
            range=[0, 200],
            row=1, col=i + 1
        )
    # Update overall layout
    fig.update_layout(
        title_text='Comparison of Value Frequencies (Excluding Zero)',
        height=500,
        width=250 * n_plots,  # Adjust width based on number of subplots
        plot_bgcolor='rgba(245,245,245,1)'
    )
    # Show the plot
    fig.show()


def overlap_by_area(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, interest_areas=None,
                    kfolds_threshold=10, start_time_idx=0, end_time_idx=161):
    elecs_df = load_electrode_names_and_locations(sid, filter_type)
    areas = elecs_df["princeton_class"].unique()

    all_data_all_elec = {}
    kfolds_all_elecs = {}

    for area in areas:
        if pd.isna(area) or area == "other":
            continue
        if interest_areas and area not in interest_areas:
            continue
        print("Processing area:", area)
        areas_electrodes = elecs_df[elecs_df["princeton_class"] == area]["full_elec_name"].to_list()
        all_data_chosen_coeffs_dict, reliable_chosen_coeffs_dict = get_non_zero_coeffs_old(sid, areas_electrodes,
                                                                                           model_info[
                                                                                               "model_full_name"],
                                                                                           model_info["layer"],
                                                                                           model_info["context"],
                                                                                           min_alpha, max_alpha,
                                                                                           num_alphas, mode,
                                                                                           kfolds_threshold,
                                                                                           output_elec_name_prefix=area + "_")
        for elec in all_data_chosen_coeffs_dict.keys():
            all_data_chosen_coeffs = all_data_chosen_coeffs_dict[elec][1]
            all_data_all_elec[elec] = list(
                {item for sublist in all_data_chosen_coeffs[start_time_idx:end_time_idx] for item in
                 sublist})  # Take union of coeffs in the selected time range
            reliable_chosen_coeffs = reliable_chosen_coeffs_dict[elec][1]
            kfolds_all_elecs[elec] = list(
                {item for sublist in reliable_chosen_coeffs[start_time_idx:end_time_idx] for item in
                 sublist})  # Take union of coeffs in the selected time range
            kfolds_all_elecs[f"---------{area}---------"] = []  # To create a gap between areas in the plot
    # Remove electrodes with empty entries
    kfolds_all_elecs = {k: v for k, v in kfolds_all_elecs.items() if (v or "---" in k)}
    content = from_contents(kfolds_all_elecs)
    upset = UpSet(content, subset_size='count', show_counts=True, sort_categories_by='input', min_degree=2,
                  min_subset_size=4, sort_by='cardinality')
    upset.plot()
    plt.show()

    print("Done!")


def plot_x_vs_num_of_coeffs(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, x="rounded_encoding",  y="num_of_coeffs", hue=None, kfolds_threshold=10, query="",
                            violin_annot=True):
    """
    All electrodes, all times
    :param sid:
    :param filter_type:
    :param model_info:
    :param min_alpha:
    :param max_alpha:
    :param num_alphas:
    :param mode:
    :param interest_areas:
    :param kfolds_threshold:
    :return:
    """
    kfolds_df = prepare_non_zero_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info, num_alphas, sid, query=query)
    _plot_x_vs_num_coeffs(kfolds_df, max_alpha, min_alpha, model_info, num_alphas, x=x, hue=hue, filter_name=query,  y=y,
                          kfolds_threshold=kfolds_threshold, violin_annot=violin_annot)
    # from scipy import stats
    # # Calculate correlation
    # correlation = kfolds_df['encoding'].corr(kfolds_df['num_of_coeffs'])
    #
    # # Calculate linear regression for trendline
    # slope, intercept, r_value, p_value, std_err = stats.linregress(kfolds_df['encoding'], kfolds_df['num_of_coeffs'])
    # line_x = np.array([kfolds_df['encoding'].min(), kfolds_df['encoding'].max()])
    # line_y = slope * line_x + intercept
    #
    # # Create figure
    # fig = go.Figure()
    #
    # # Add scatter plot first (so it's in the back)
    # fig.add_trace(go.Scatter(
    #     x=kfolds_df['encoding'],
    #     y=kfolds_df['num_of_coeffs'],
    #     mode='markers',
    #     marker=dict(
    #         size=2,
    #         color='#636EFA',  # Plotly Express default blue color
    #     ),
    #     name='Data Points'
    # ))
    #
    # # Add correlation line second (so it's in the front)
    # fig.add_trace(go.Scatter(
    #     x=line_x,
    #     y=line_y,
    #     mode='lines',
    #     name=f'Trend Line (r={correlation:.3f})',
    #     line=dict(color='red', width=2)
    # ))
    #
    # # Customize layout
    # fig.update_layout(
    #     title=f'Encoding vs Number of Coefficients<br>Correlation: {correlation:.3f}',
    #     xaxis_title='Encoding',
    #     yaxis_title='Number of Coefficients',
    #     hovermode='closest',
    #     showlegend=True,
    #     template='simple_white',
    # )
    #
    # fig.show()

    nb_model = smf.negativebinomial(f'num_of_coeffs ~ C({x}) + encoding', data=kfolds_df).fit()
    print(nb_model.summary())

    print("Done!")


def prepare_non_zero_df(filter_type, kfolds_threshold: int, max_alpha, min_alpha, mode, model_info, num_alphas, sid, query="", df_type="kfolds"):
    return_all_data_coeffs, return_kfolds_coeffs, return_corr_coeffs, return_all_data_encoding, return_kfolds_encoding = False,False,False,False,False

    if df_type == "all_data":
        return_all_data_coeffs = True
        return_all_data_encoding = True
    elif df_type == "kfolds":
        return_kfolds_coeffs=True
        return_kfolds_encoding=True
    elif df_type == "corr":
        return_corr_coeffs = True

    all_data_df, kfolds_df, corr_df = get_non_zero_coeffs(sid, filter_type, model_info["model_full_name"], model_info["layer"], model_info["context"],
                                                min_alpha, max_alpha, num_alphas, mode, kfolds_threshold,
                                                return_all_data_coeffs=return_all_data_coeffs, return_kfolds_coeffs=return_kfolds_coeffs, return_corr_coeffs=return_corr_coeffs,
                                                return_all_data_encoding=return_all_data_encoding, return_kfolds_encoding=return_kfolds_encoding)

    if df_type == "all_data":
        df = all_data_df
    elif df_type == "kfolds":
        df = kfolds_df
    elif df_type == "corr":
        df = corr_df
    else:
        raise ValueError("df_type must be 'all_data', 'kfolds', 'corr', or 'all_data'")

    df = _process_coeff_df(df)

    if query:
        df = df.query(query)

    return df

def _plot_x_vs_num_coeffs(df: DataFrame, max_alpha, min_alpha, model_info, num_alphas, x="rounded_encoding", y="num_of_coeffs", hue=None, filter_name=None, kfolds_threshold=8, violin_annot=True):
    if df[x].unique().size > 20: # Long x axis
        fig = plt.figure(figsize=(15, 15))
    else:
        fig = plt.figure(figsize=(7, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2])  # 3 rows, 1 column with the top plot larger

    ax1 = plt.subplot(gs[0])
    if hue:
        sns.boxplot(data=df, x=x, y=y, hue=hue, order=get_order(df, x), palette=COLOR_PALETTE,
                    hue_order=get_order(df, hue), ax=ax1)

        pairs = [((item1, pair[0]), (item1, pair[1]))
                 for item1 in get_order(df, x)
                 for pair in zip(get_order(df, hue), get_order(df, hue)[1:])]
        if hue == "time_bin_and_princeton_class":
            pairs = []
            all_time_bin_and_princeton_class = set(df["time_bin_and_princeton_class"])
            ordered_times_bins = get_order(df, "time_bin")
            ordered_princeton_class = get_order(df, "princeton_class")
            # For each hue value
            for category in get_order(df, x):
                # Connect consecutive classes with same time
                for time in ordered_times_bins:
                    for i in range(len(ordered_princeton_class) - 1):
                        area1 = ordered_princeton_class[i]
                        area2 = ordered_princeton_class[i + 1]
                        time_bin_and_princeton_class1 = f"{area1}_{time}"
                        time_bin_and_princeton_class2 = f"{area2}_{time}"
                        if time_bin_and_princeton_class1 in all_time_bin_and_princeton_class and time_bin_and_princeton_class2 in all_time_bin_and_princeton_class:
                            pair = ((category, time_bin_and_princeton_class1), (category, time_bin_and_princeton_class2))
                            pairs.append(pair)

                # Connect consecutive times with same class
                for area in ordered_princeton_class:
                    for i in range(len(ordered_times_bins) - 1):
                        time1 = ordered_times_bins[i]
                        time2 = ordered_times_bins[i + 1]
                        time_bin_and_princeton_class1 = f"{area}_{time1}"
                        time_bin_and_princeton_class2 = f"{area}_{time2}"
                        if time_bin_and_princeton_class1 in all_time_bin_and_princeton_class and time_bin_and_princeton_class2 in all_time_bin_and_princeton_class:
                            pair = ((category, time_bin_and_princeton_class1), (category, time_bin_and_princeton_class2))
                            pairs.append(pair)
        # if hue == "princeton_class":
        #     pairs.extend([((item1, pair[0]), (item1, pair[1]))
        #              for item1 in get_order(df, x)
        #              for pair in zip(["STG","IFG"], ["STG","IFG"][1:])])
        # if hue == "time_bin":
        #     pairs.extend([((item1, pair[0]), (item1, pair[1]))
        #              for item1 in get_order(df, x)
        #              for pair in zip(["-0.4≤x<0","0.4≤x<0.8"], ["-0.4≤x<0","0.4≤x<0.8"][1:])])
        annotator = Annotator(ax1, pairs, data=df, x=x, y=y, hue=hue, order=get_order(df, x), hue_order=get_order(df, hue))
        annotator.configure(test='Mann-Whitney', comparisons_correction="fdr_bh", text_format='star', loc='inside', verbose=2) #text_format='full'
        # annotator.configure(test='t-test_welch', comparisons_correction="fdr_bh", text_format='star', loc='inside', verbose=2) #text_format='full'
        annotator.apply_and_annotate()

    else:
        sns.boxplot(data=df, x=x, y=y, ax=ax1, order=get_order(df, x), palette=COLOR_PALETTE)
        pairs = list(zip(get_order(df, x), get_order(df, x)[1:]))
        if x=="princeton_class" and ("STG","IFG") not in pairs and ("IFG","STG") not in pairs:
            pairs.append(("STG","IFG"))
        # if x=="time_bin":
        #     pairs.append(("-0.4≤x<0","0.4≤x<0.8"))
        if x=="time_bin_and_princeton_class":
            pairs.remove(("IFG_0≤x<0.4", "STG_-0.4≤x<0"))
            pairs.append(("STG_-0.4≤x<0", "IFG_-0.4≤x<0"))
            pairs.append(("STG_0≤x<0.4", "IFG_0≤x<0.4"))
        annotator = Annotator(ax1, pairs, data=df, x=x, y=y, order=get_order(df, x))
        annotator.configure(test='Mann-Whitney', comparisons_correction="fdr_bh", text_format='star', loc='inside', verbose=2)
        # annotator.configure(test='t-test_welch', comparisons_correction="fdr_bh", text_format='star', loc='inside', verbose=2)
        annotator.apply_and_annotate()


    ax1.set_title("Number of coefficients used for each encoding", fontsize=14)
    # ax1.set_xticklabels([])
    customize_time_xaxis(ax1, x)

    # ax2 = plt.subplot(gs[1])
    # if hue:
    #     sns.pointplot(data=df, x=x, y=y, errorbar="sd", hue=hue, ax=ax2)
    # else:
    #     sns.pointplot(data=df, x=x, y=y, errorbar="sd", ax=ax2)
    # ax2.set_title("Mean number of coefficients used for each encoding", fontsize=14)
    # ax2.set_ylabel("Mean number of coefficients", fontsize=12)
    # # ax2.set_xticklabels([])
    # customize_time_xaxis(ax2, x)

    ax2 = plt.subplot(gs[1])
    if hue:
        sns.countplot(data=df, x=x, hue=hue, order=get_order(df, x), hue_order=get_order(df, hue), ax=ax2, palette=COLOR_PALETTE)
    else:
        sns.countplot(data=df, x=x, order=get_order(df, x), ax=ax2, palette=COLOR_PALETTE,)
    ax2.set_title("Count of items in each encoding group", fontsize=12)
    # ax2.set_xlabel(x, fontsize=12)
    customize_time_xaxis(ax2, x)

    # New violin plot for the "encoding" column distribution
    ax3 = plt.subplot(gs[2])
    df["relative_encoding"] = df["rounded_encoding"] - df["encoding"]
    if hue:
        sns.violinplot(data=df, x=x, y="relative_encoding", hue=hue,
                       order=get_order(df, x), palette=COLOR_PALETTE,
                       hue_order=get_order(df, hue), ax=ax3, split=True,
                       inner="quartile", density_norm='width') #linewidth=1
        # Combination: Box plot + Strip plot
        # sns.boxplot(data=df, x=x, y="relative_encoding", hue=hue,
        #             order=get_order(df, x), palette=COLOR_PALETTE,
        #             hue_order=get_order(df, hue), ax=ax3,
        #             width=0.6, linewidth=1.5)
        #
        # sns.stripplot(data=df, x=x, y="relative_encoding", hue=hue,
        #               order=get_order(df, x), dodge=True,
        #               hue_order=get_order(df, hue), ax=ax3,
        #               size=2, alpha=0.4, color='black')
        #
        # # Remove duplicate legend entries
        # handles, labels = ax3.get_legend_handles_labels()
        # ax3.legend(handles[:len(df[hue].unique())],
        #            labels[:len(df[hue].unique())])
        if violin_annot:
            annotator = Annotator(ax3, pairs, data=df, x=x, y=y, hue=hue, order=get_order(df, x), hue_order=get_order(df, hue))
            annotator.configure(test='Mann-Whitney', comparisons_correction="fdr_bh", text_format='star', loc='outside', verbose=2) #text_format='full'
            annotator.apply_and_annotate()

        ylim = ax3.get_ylim()
        y_range = ylim[1] - ylim[0]
        ax3.set_ylim(-0.06, ylim[1] + 0.15 * y_range)  # Add 15% space at top
    else:
        sns.violinplot(data=df, x=x, y="relative_encoding", ax=ax3,
                       order=get_order(df, x), palette=COLOR_PALETTE,
                       inner="quartile", density_norm='width') #linewidth=1
        # annotator = Annotator(ax1, pairs, data=df, x=x, y=y, order=get_order(df, x))
        # annotator.configure(test='Mann-Whitney', comparisons_correction="fdr_bh", text_format='star', loc='inside', verbose=2)
        # # annotator.configure(test='t-test_welch', comparisons_correction="fdr_bh", text_format='star', loc='inside', verbose=2)
        # annotator.apply_and_annotate()
        ax3.set_ylim(-0.06, 0.06)  # Set y-axis limits as requested
    ax3.set_title("Distribution of encoding values", fontsize=14)
    customize_time_xaxis(ax3, x)

    plt.suptitle(
        f"Model: {model_info['model_short_name']} (α=({min_alpha},{max_alpha},{num_alphas}), kfolds threshold={kfolds_threshold})"
        + (f"\nfilter={filter_name}" if filter_name else ""),
        # + f"\nover all times, subjects, and brain areas",
        fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout leaving space for suptitle
    plt.show()


def get_order(df: DataFrame, x: str) -> list[Any]:
    return [item for item in ORDERS[x] if item in set(df[x])]


def customize_time_xaxis(ax, x, n=20):
    if not (x == "time" or x == "time_index"):
        return
    # Set sparse labels if needed
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_visible(idx % n == 0)

    # Get tick positions and labels
    tick_positions = ax.get_xticks()
    ax.axvline(x=tick_positions[80], color='black', linewidth=0.5, alpha=0.7)

def coeffs_venn(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                col_to_compare, group1, group2, kfolds_threshold=10, query=""):
    """
    Finds which coeffs appear in which
    :param sid:
    :param filter_type:
    :param model_info:
    :param min_alpha:
    :param max_alpha:
    :param num_alphas:
    :param mode:
    :param col_to_compare:
    :param kfolds_threshold:
    :return:
    """
    first_group_all_coeffs, first_group_name, second_group_all_coeffs, second_group_name = _get_groups_distinct_coeffs(
        col_to_compare, filter_type, group1, group2, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
        num_alphas, query, sid)
    set_colors = (COLOR_PALETTE[first_group_name], COLOR_PALETTE[second_group_name])

    venn2((first_group_all_coeffs, second_group_all_coeffs), (first_group_name, second_group_name), set_colors=set_colors)
    plt.title("Coefficients for each group"
              + (f"\nModel: {model_info['model_short_name']} (α range: {min_alpha} to {max_alpha}, {num_alphas} values)")
              + (f"\nfilter={query}" if query else ""))
    plt.show()

    # venn3((first_group_sig_coeffs, second_group_sig_coeffs, all_coeffs), (first_group_name, second_group_name, "all_coeffs"))
    plt.title("Distinct Statistically Significant Coefficients for each group"
              + (f"\nModel: {model_info['model_short_name']} (α range: {min_alpha} to {max_alpha}, {num_alphas} values)")
              + (f"\nfilter={query}" if query else ""))
    plt.show()

    print("done!")

def amount_distinct_coeffs(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                           x="rounded_encoding", hue=None, kfolds_threshold=10, query=""):
    kfolds_df = prepare_non_zero_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
                                    num_alphas, sid, query=query)
    exploded_df = kfolds_df.query("num_of_coeffs > 0").explode("actual_coeffs")

    if hue:
        distinct_coeffs_df = exploded_df[["actual_coeffs", x, hue]].drop_duplicates()
        grouped_counts = distinct_coeffs_df.groupby([x, hue]).size().reset_index(name='distinct_coeffs')
    else:
        distinct_coeffs_df = exploded_df[["actual_coeffs", x]].drop_duplicates()
        grouped_counts = distinct_coeffs_df.groupby(x).size().reset_index(name='distinct_coeffs')

    fig = plt.figure(figsize=(7, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3 rows, 1 column with the top plot larger

    ax1 = plt.subplot(gs[0])
    if hue:
        sns.barplot(x=x, y='distinct_coeffs', hue=hue, data=grouped_counts, ax=ax1, palette=COLOR_PALETTE, order=get_order(grouped_counts, x), hue_order=get_order(grouped_counts, hue))

    else:
        sns.barplot(x=x, y='distinct_coeffs', data=grouped_counts, ax=ax1, color='royalblue',
                    order=get_order(grouped_counts, x))


    ax1.set_title("Number of Distinct Coeffs", fontsize=14)
    # ax1.set_xticklabels([])
    customize_time_xaxis(ax1, x)

    ax3 = plt.subplot(gs[1])
    if hue:
        sns.countplot(data=exploded_df, x=x, hue=hue, order=get_order(exploded_df, x), hue_order=get_order(exploded_df, hue), ax=ax3, palette=COLOR_PALETTE)
    else:
        sns.countplot(data=exploded_df, x=x, order=get_order(exploded_df, x), ax=ax3, color='royalblue')
    ax3.set_title("Count of items in each encoding group", fontsize=12)
    # ax3.set_xlabel(x, fontsize=12)
    customize_time_xaxis(ax3, x)

    plt.suptitle(
        f"Model: {model_info['model_short_name']} (α=({min_alpha},{max_alpha},{num_alphas}), kfolds threshold={kfolds_threshold})"
        + (f"\nfilter={query}" if query else ""),
        # + f"\nover all times, subjects, and brain areas",
        fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout leaving space for suptitle
    plt.show()


def _get_groups_distinct_coeffs(col_to_compare, filter_type, group1, group2, kfolds_threshold: int, max_alpha,
                                min_alpha, mode, model_info, num_alphas, query: str, sid) -> tuple[set[Any], str, set[Any], str]:
    if query:
        query = "(" + query + f") & (({col_to_compare} == '{group1}') | ({col_to_compare} == '{group2}'))"
    else:
        query = f"(({col_to_compare} == '{group1}') | ({col_to_compare} == '{group2}'))"
    kfolds_df = prepare_non_zero_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
                                    num_alphas, sid, query=query)
    exploded_df = kfolds_df.query("num_of_coeffs > 0").explode("actual_coeffs")
    first_group_name, second_group_name = sorted([group1, group2])

    first_group_all_coeffs = set(
        exploded_df.query(f"{col_to_compare} == '{first_group_name}'")["actual_coeffs"].to_list())
    second_group_all_coeffs = set(
        exploded_df.query(f"{col_to_compare} == '{second_group_name}'")["actual_coeffs"].to_list())
    return first_group_all_coeffs, first_group_name, second_group_all_coeffs, second_group_name


def prepare_kfolds_coeffs_df(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, query="") -> DataFrame:
    _, kfolds_df = get_coeffs(sid, filter_type, model_info["model_full_name"], model_info["layer"], model_info["context"], min_alpha, max_alpha, num_alphas, mode,
                                        return_all_data_coeffs=False, return_all_data_encoding=False)

    kfolds_df = _process_coeff_df(kfolds_df)

    if query:
        kfolds_df = kfolds_df.query(query)

    return kfolds_df

def run_all_boxplots(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, kfolds_threshold):

    # x,hue,query:
    all_options = [
        # {"x": "rounded_encoding", "hue": None, "query": ""},
        # {"x": "rounded_encoding", "hue": None, "query": "princeton_class == 'IFG'"},
        # {"x": "rounded_encoding", "hue": None, "query": "princeton_class == 'STG'"},
        # {"x": "rounded_encoding", "hue": None, "query": "time_bin == '-0.4≤x<0'"},
        # {"x": "rounded_encoding", "hue": None, "query": "time_bin == '0≤x<0.4'"},
        # {"x": "princeton_class", "hue": None, "query": ""},
        # {"x": "princeton_class", "hue":None, "query": AREA_QUERY},
        # {"x": "rounded_encoding", "hue": "princeton_class", "query": AREA_QUERY},
        # {"x": "time_bin", "hue": None, "query": ""},
        # {"x": "time_bin", "hue": None, "query": TIME_QUERY},
        # {"x": "rounded_encoding", "hue": "time_bin", "query": TIME_QUERY},
        # {"x": "time_bin_and_princeton_class", "hue": None, "query": f"({TIME_QUERY})&({AREA_QUERY})"},
        # {"x": "rounded_encoding", "hue": "time_bin_and_princeton_class", "query": f"({TIME_QUERY})&({AREA_QUERY})", "violin_annot": False},
        # {"x": "rounded_encoding", "hue": "time_bin_and_princeton_class", "query": "({TIME_QUERY})&((princeton_class == 'IFG'))"},
        # {"x": "rounded_encoding", "hue": "time_bin_and_princeton_class", "query": "({TIME_QUERY})&((princeton_class == 'STG'))"},
        # {"x": "rounded_encoding", "hue": "time_bin_and_princeton_class", "query": f"((time_bin == '-0.4≤x<0'))&({AREA_QUERY})"},
        # {"x": "rounded_encoding", "hue": "time_bin_and_princeton_class", "query": f"((time_bin == '0≤x<0.4'))&({AREA_QUERY})"},
        {"x": "princeton_class", "y":"encoding", "hue": None, "query": AREA_QUERY},
        {"x": "princeton_class", "y":"num_of_coeffs", "hue": None, "query": AREA_QUERY},
        {"x": "time_bin", "y":"encoding", "hue": None, "query": TIME_QUERY},
        {"x": "time_bin", "y":"num_of_coeffs", "hue": None, "query": TIME_QUERY},
        {"x": "time_bin_and_princeton_class", "y": "encoding", "hue": None, "query": f"({TIME_QUERY})&({AREA_QUERY})"},
        {"x": "time_bin_and_princeton_class", "y": "num_of_coeffs", "hue": None, "query": f"({TIME_QUERY})&({AREA_QUERY})"},
    ]

    for curr in all_options:
        print(curr)
        violin_annot = curr["violin_annot"] if ("violin_annot" in curr.keys()) else True
        y = curr["y"] if ("y" in curr.keys()) else "num_of_coeffs"
        plot_x_vs_num_of_coeffs(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                            x=curr["x"], y=y, hue=curr["hue"], kfolds_threshold=kfolds_threshold, query=curr["query"], violin_annot=violin_annot)
        print()


def coeffs_statistics(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, col_to_compare, group1, group2,
                      query="", kfolds_threshold=10):
    # TODO: written very non efficient, fix!
    non_zero_df = prepare_non_zero_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
                                      num_alphas, sid, query=query)
    exploded_non_zero_df = non_zero_df.query("num_of_coeffs > 0").explode("actual_coeffs").rename(columns={'actual_coeffs': 'coeff_indx'}).reset_index(drop=True)

    all_coeffs_df = prepare_kfolds_coeffs_df(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, query=query).rename(columns={'coeffs': 'coeffs_mean_value'})
    # all_coeffs_df['coeff_indx'] = all_coeffs_df['coeffs_mean_value'].apply(lambda x: list(range(len(x))))
    all_coeffs_df['coeff_indx'] = all_coeffs_df['coeffs_mean_value'].apply(lambda x: [str(i) for i in range(len(x))])
    exploded_all_coeffs_df = all_coeffs_df.explode(['coeff_indx', 'coeffs_mean_value']).reset_index(drop=True)

    coeffs = exploded_non_zero_df.merge(exploded_all_coeffs_df, how='left')

    first_group_all_coeffs, first_group_name, second_group_all_coeffs, second_group_name = _get_groups_distinct_coeffs(
        col_to_compare, filter_type, group1, group2, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
        num_alphas, query, sid)
    coeffs[f"is_in_{first_group_name}"] = coeffs["coeff_indx"].isin(first_group_all_coeffs)
    coeffs[f"is_in_{second_group_name}"] = coeffs["coeff_indx"].isin(second_group_all_coeffs)

    MEAN_COLORS = ('240', '98', '146')  # Light pink (similar to CORR_COLORS in example)
    COUNT_COLORS = ('100', '181', '246')  # Light blue (similar to COEFF_COLORS in example)

    MEAN_AXIS_COLOR = f"rgb({MEAN_COLORS[0]}, {MEAN_COLORS[1]}, {MEAN_COLORS[2]})"
    COUNT_AXIS_COLOR = f"rgb({COUNT_COLORS[0]}, {COUNT_COLORS[1]}, {COUNT_COLORS[2]})"


    def apply_filter(df, first_group_name, second_group_name, filter_type):
        first_col = f"is_in_{first_group_name}"
        second_col = f"is_in_{second_group_name}"

        if filter_type == f"Only_in_{first_group_name}":
            return df[df[first_col] & (~df[second_col])]
        elif filter_type == f"Only_in_{second_group_name}":
            return df[(~df[first_col]) & df[second_col]]
        elif filter_type == "In_both":
            return df[df[first_col] & df[second_col]]
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    group_types = [
        f"Only_in_{first_group_name}",
        f"Only_in_{second_group_name}",
        "In_both"
    ]

    # Create subplots with specs for secondary y-axis
    fig = make_subplots(
        rows=2,
        cols=3,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=[f"{group_type} - Mean Values and SD" for group_type in group_types]+ [f"{group_type} - Count by Index" for group_type in group_types],
        row_heights=[0.7, 0.3]
    )

    all_strongest_coeffs = []
    for group_idx, group_type in enumerate(group_types):
        curr_coeffs = apply_filter(coeffs, first_group_name, second_group_name, group_type)

        # Calculate statistics
        stats = curr_coeffs.groupby('coeff_indx').agg(
            mean_value=('coeffs_mean_value', 'mean'),
            std_value=('coeffs_mean_value', 'std'),
            count=('coeffs_mean_value', 'count')
        ).reset_index()
        stats['abs_mean_value'] = stats['mean_value'].abs()

        # Sort by ID (or you can sort by another criterion)
        # stats = stats.sort_values('coeff_indx')
        stats = stats.sort_values('mean_value', ascending=False)
        stats = stats.sort_values('count', ascending=False)

        # Add standard deviation area
        fig.add_trace(
            go.Bar(
                x=stats['coeff_indx'],
                y=stats['mean_value'],
                error_y=dict(
                    type='data',
                    array=stats['std_value'],
                    visible=True,
                    thickness=0.5,  # Makes the error bar line thinner (default is 2)
                    width=2, # Controls the width of the horizontal caps (default is 6)
                ),
                name="Mean ± SD",
                marker_color=MEAN_AXIS_COLOR,
                showlegend=True if group_idx == 0 else False,
            ),
            row=1, col=group_idx+1
        )

        # Add count bar chart
        fig.add_trace(
            go.Bar(
                x=stats['coeff_indx'],
                y=stats['count'],
                name="Count",
                marker_color=COUNT_AXIS_COLOR,
                showlegend=True if group_idx == 0 else False,
            ),
            row=2, col=group_idx+1
        )

        # # Add horizontal line at y=0 for mean values
        # fig.add_shape(
        #     type='line',
        #     x0=stats['coeff_indx'].iloc[0],
        #     x1=stats['coeff_indx'].iloc[-1],
        #     y0=0, y1=0,
        #     line=dict(color='black', width=1),
        #     row=1, col=group+1
        # )

        # Display the top 10 as a formatted list
        print(f"\n********** {group_type} **********")

        top_5_mean = stats.sort_values('abs_mean_value', ascending=False).head(5)
        print("Top 5 Indexes by Mean Value:")
        for i, (_, row) in enumerate(top_5_mean.iterrows(), 1):
            print(f"{i}. Index: {row['coeff_indx']} - Mean Value: {row['mean_value']:.3f} (Count: {row['count']})")
        print()

        top_5_count = stats.sort_values('count', ascending=False).head(5)
        # Display the top 5 as a formatted list
        print("Top 5 Indexes by Count:")
        for i, (_, row) in enumerate(top_5_count.iterrows(), 1):
            print(f"{i}. Index: {row['coeff_indx']} - Mean Value: {row['mean_value']:.3f} (Count: {row['count']})")

        strongest_coeffs = set(top_5_mean + top_5_count)
        all_strongest_coeffs.append(strongest_coeffs)

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    # Update layout
    fig.update_layout(
        height=700,
        width=1200,
        showlegend=True,
        template='plotly_white',
        title_text="Statistics by coeff indx",
    )
    fig.show()
    a=0

def pca_coeffs(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, query=""):
    """
    Does PCA analysis for the transformation vectors themselves (the trained linear vectors).
    """
    kfolds_df = prepare_kfolds_coeffs_df(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, query)
    vectors_array = np.vstack(kfolds_df['coeffs'].values)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(vectors_array)

    # To determine how many components explain e.g. 95% of variance
    pca_full = PCA().fit(scaled_data)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Number of components needed for 95% variance: {n_components}")

    # Plot the explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'Explained Variance vs. Number of Components, model: {model_info["model_short_name"]}')
    plt.grid(True)
    plt.show()

    pca2 = PCA(n_components=2)
    principal_components2 = pca2.fit_transform(scaled_data)
    # 3. Create DataFrame with PCA results and categories
    pca_df2 = pd.DataFrame(data=principal_components2, columns=['PC1', 'PC2'])

    pca3 = PCA(n_components=3)
    principal_components3 = pca3.fit_transform(scaled_data)
    # 3. Create DataFrame with PCA results and categories
    pca_df3 = pd.DataFrame(data=principal_components3, columns=['PC1', 'PC2', 'PC3'])

    # 2D+3D Discrete Categories
    discrete_categories_names = ["princeton_class", "time_bin", "rounded_encoding"]
    for category_name in discrete_categories_names:
        # 2D:
        pca_df2['Category'] = kfolds_df[category_name]  # Add the category column
        # 4. Plot with colors by category
        plt.figure(figsize=(10, 8))
        # Get unique categories for coloring
        unique_categories = pca_df2['Category'].unique()
        # Plot each category with a different color
        for category in unique_categories:
            category_data = pca_df2[pca_df2['Category'] == category]
            plt.scatter(
                category_data['PC1'],
                category_data['PC2'],
                label=category,
                alpha=0.7
            )
        plt.title(f'model: {model_name}, query: {query}')
        plt.xlabel(f'Principal Component 1 ({pca2.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca2.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend(title=category_name)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


        # 3D:
        pca_df3['Category'] = kfolds_df[category_name]  # Add your categories
        # Optional: Add hover text if you have IDs or additional info
        pca_df3['hover_text'] = [f"ID: {i}, Category: {cat}" for i, cat in enumerate(pca_df3['Category'])]
        # 4. Create interactive 3D plot with Plotly Express
        fig = px.scatter_3d(
            pca_df3,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Category',
            hover_data=['hover_text'],  # Optional
            opacity=0.7,
            title=f'model: {model_name}, query: {query}',
            labels={
                'PC1': f'PC1 ({pca3.explained_variance_ratio_[0]:.2%})',
                'PC2': f'PC2 ({pca3.explained_variance_ratio_[1]:.2%})',
                'PC3': f'PC3 ({pca3.explained_variance_ratio_[2]:.2%})'
            }
        )
        # 5. Update layout for better appearance
        fig.update_layout(
            legend_title_text=category_name,
            scene=dict(
                xaxis_title=f'PC1 ({pca3.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'PC2 ({pca3.explained_variance_ratio_[1]:.2%})',
                zaxis_title=f'PC3 ({pca3.explained_variance_ratio_[2]:.2%})'
            )
        )
        # 6. Show the plot
        fig.show()
        # To save the interactive plot as HTML
        # fig.write_html("pca_3d_interactive.html")

    continues_categories_names = ["time", "encoding"]
    for category_name in continues_categories_names:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_df2['PC1'], pca_df2['PC2'],
                              c=kfolds_df[category_name],  # Numeric values for color
                              cmap='viridis',  # Color map
                              alpha=0.7)
        plt.colorbar(scatter, label=category_name)
        plt.title(f'model: {model_name}, query: {query}')
        plt.xlabel(f'Principal Component 1 ({pca2.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca2.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend(title=category_name)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    models_info = {"gemma2b": {"layer": 13, "context": 32, "embedding_size": 2304,
                             "model_short_name": "gemma2b", "model_full_name": "gemma-2-2b"},
                   "gemma9b": {"layer": 13, "context": 32, "embedding_size": 3584,
                               "model_short_name": "gemma9b", "model_full_name": "gemma-2-9b"},
                   "gpt2": {"layer": 24, "context": 32, "embedding_size": 1600,
                             "model_short_name": "gpt2", "model_full_name": "gpt2-xl"},
                   "mistral": {"layer": 16, "context": 32, "embedding_size": 4096,
                            "model_short_name": "mistral", "model_full_name": "Mistral-7B-v0.3"},
                   "llama": {"layer": 16, "context": 32, "embedding_size": 4096,
                             "model_short_name": "llama", "model_full_name": "Meta-Llama-3.1-8B"},
                   "glove": {"layer": 0, "context": 1, "embedding_size": 50,
                             "model_short_name": "glove", "model_full_name": "glove50"},
                   "gemma-scope": {"layer": 13, "context": 32, "embedding_size": 16384,
                                   "model_short_name": "gemma-scope",
                                   "model_full_name": "gemma-scope-2b-pt-res-canonical"},
                   }
    time_points = np.linspace(-2, 2, 161)
    time_to_index = {t: i for i, t in enumerate(np.round(time_points, 3))}

    sid = 777
    filter_type = '160'
    model_name = 'gemma'
    min_alpha = -2
    max_alpha = 10
    num_alphas = 100
    mode = 'comp'
    kfolds_threshold = 8

    # # Create BoxPlots (num coeffs)
    x="princeton_class" #rounded_encoding, princeton_class, time_bin
    y="num_of_coeffs" #num_of_coeffs, encoding
    hue = None #None, rounded_encoding, princeton_class, time_bin
    query = AREA_QUERY#TIME_QUERY, AREA_QUERY
    violin_annot = True

    # plot_x_vs_num_of_coeffs(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
    #                         x=x, y=y, hue=hue, kfolds_threshold=kfolds_threshold, query=query, violin_annot=violin_annot)
    run_all_boxplots(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,kfolds_threshold)
    # amount_distinct_coeffs(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
    #                        x=x, hue=hue, kfolds_threshold=kfolds_threshold, query=query)

    # # Venn Diagrams (which coeffs)
    # query = "rounded_encoding==0.5"  # "(time_bin == '-0.4≤x<0') | (time_bin == '0≤x<0.4') | (time_bin == '0.4≤x<0.8')" #"(princeton_class == 'STG') | (princeton_class == 'IFG')"
    # col_to_compare = "time_bin"#"princeton_class"
    # group1 = '-0.4≤x<0'#"STG"
    # group2 = '0≤x<0.4'#"IFG"
    #
    # for encoding in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']:
    #     query = f"rounded_encoding=={encoding}"
    #     coeffs_venn(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
    #                       col_to_compare=col_to_compare, group1=group1, group2=group2, kfolds_threshold=kfolds_threshold, query=query)

    query = "rounded_encoding==0.4"
    # pca_coeffs(sid, filter_type, models_info[model_name]["model_full_name"], models_info[model_name]["layer"],
    #            models_info[model_name]["context"], min_alpha, max_alpha, num_alphas, mode, query)
    coeffs_statistics(sid, filter_type, models_info[model_name]["model_full_name"], models_info[model_name]["layer"],
               models_info[model_name]["context"], min_alpha, max_alpha, num_alphas, mode, query)

    # overlap_by_area(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode, interest_areas, kfolds_threshold, start_time_idx, end_time_idx)
