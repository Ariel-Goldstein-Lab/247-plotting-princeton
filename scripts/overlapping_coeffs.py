import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
import plotly.express as px
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from upsetplot import from_contents, UpSet
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import matplotlib.colors as mc
from scipy.stats import fisher_exact, chi2_contingency
import requests
from typing import Optional, Dict, Any
import json
import time
import gc
from matplotlib_venn import venn2, venn3
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from tqdm import tqdm
from tfsplt_utils import get_coeffs_df, prepare_coeffs_df, _get_exploded_united_lasso_and_corr
from sklearn.metrics.pairwise import cosine_similarity

SHOW_FIGS = True

AREA_QUERY = "(brain_area == 'STG') | (brain_area == 'IFG')"
TIME_QUERY = "(time_bin == '-0.4≤x<0') | (time_bin == '0≤x<0.4')"
AREA_AND_TIME_QUERY = f"({AREA_QUERY})&({TIME_QUERY})"

ROUNDED_ENCODING = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
ORDERED_PRINCETON_CLASS = ['aMTG', 'MFG', 'pmtg', 'other', 'premotor', 'TP', 'AG', 'IFG', 'parietal', 'precentral', 'postcg', 'STG'] # Sorted by mean num_of_chosen_coeffs
# "ORDERED_PRINCETON_CLASS": ['IFG', 'STG', 'AG', 'TP', 'precentral', 'pmtg', 'parietal', 'other', 'MFG', 'aMTG', 'premotor', 'postcg'], # Sorted by amount of data
TIME_BIN = ['x<-0.8', '-0.8≤x<-0.4', '-0.4≤x<0', '0≤x<0.4', '0.4≤x<0.8', '0.8≤x']
# TIME_BIN = ['x<-0.6', '-0.6≤x<-0.3', '-0.3≤x<0', '0≤x<0.3', '0.3≤x<0.6', '0.6≤x']

time_bin_and_brain_area_by_area = [area + "_" + time for area in ORDERED_PRINCETON_CLASS for time in TIME_BIN]
time_bin_and_brain_area_by_time = [area + "_" + time for time in TIME_BIN for area in ORDERED_PRINCETON_CLASS]

ORDERS = {"rounded_encoding": ROUNDED_ENCODING,
          "time_bin": TIME_BIN,
          "brain_area": ORDERED_PRINCETON_CLASS,
          "time_bin_and_brain_area": time_bin_and_brain_area_by_area
          }
# 'x<-0.8':'#0077b6', '-0.8≤x<-0.4':'#0096c7', '-0.4≤x<0':'#00b4d8', '0≤x<0.4':'#ff9e00', '0.4≤x<0.8':'#ff9100', '0.8≤x':'#ff8500'
COLOR_PALETTE = {"IFG": '#ff4da0', "STG": '#00c16a', #IFG-pink, STG-green
                 "precentral": "#C6E7FF",
                 # 'x<-0.8':'#ff8500', '-0.8≤x<-0.4':'#ff9100', '-0.4≤x<0':'#ff9e00', '0≤x<0.4':'#00b4d8', '0.4≤x<0.8':'#0096c7', '0.8≤x':'#0077b6',
                 '-0.4≤x<0':'#1CC4F8', '0≤x<0.4':'#ff6e01', #Dark then light orange
                 # '-0.4≤x<0':'#ff9d00', '0≤x<0.4':'#64B5F6', #Dark then light
                 # '-0.4≤x<0':'#D1D1D1', '0≤x<0.4':'#A6A6A6', #Dark then light grey
                 'IFG_-0.4≤x<0':"#ff7fbc", 'IFG_0≤x<0.4':"#f72585", 'STG_-0.4≤x<0':"#26cc80", 'STG_0≤x<0.4':"#008e4d",
                 0: "#ff8500", 0.1: "#ff9100", 0.2: "#ff9e00", 0.3: "#00b4d8", 0.4: "#0096c7", 0.5: "#0077b6",
                 '0': "#ff8500", '0.1': "#ff9100", '0.2': "#ff9e00", '0.3': "#00b4d8", '0.4': "#0096c7", '0.5': "#0077b6",}
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

def plot_x_vs_num_of_coeffs(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, x="rounded_encoding", hue=None, kfolds_threshold=10, query="",
                            violin_annot=True, df_type="lasso&corr", sig_test="t-test"):
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
    kfolds_df = get_coeffs_df(sid, mode, model_info, filter_type, min_alpha, max_alpha, num_alphas, kfolds_threshold, df_type, query=query)

    _plot_x_vs_num_coeffs(kfolds_df, max_alpha, min_alpha, model_info, num_alphas, x=x, hue=hue, filter_name=query,
                          kfolds_threshold=kfolds_threshold, sig_test=sig_test) #, violin_annot=violin_annot
    # from scipy import stats
    # # Calculate correlation
    # correlation = kfolds_df['encoding'].corr(kfolds_df['num_of_chosen_coeffs'])
    #
    # # Calculate linear regression for trendline
    # slope, intercept, r_value, p_value, std_err = stats.linregress(kfolds_df['encoding'], kfolds_df['num_of_chosen_coeffs'])
    # line_x = np.array([kfolds_df['encoding'].min(), kfolds_df['encoding'].max()])
    # line_y = slope * line_x + intercept
    #
    # # Create figure
    # fig = go.Figure()
    #
    # # Add scatter plot first (so it's in the back)
    # fig.add_trace(go.Scatter(
    #     x=kfolds_df['encoding'],
    #     y=kfolds_df['num_of_chosen_coeffs'],
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
    kfolds_df[x] = pd.Categorical(
        kfolds_df[x],
        categories=kfolds_df[x].unique()
    )

    nb_model = smf.negativebinomial(f'num_of_chosen_coeffs ~ C({x}) + encoding', data=kfolds_df).fit()
    print(nb_model.summary())

    print("Done!")


def _plot_x_vs_num_coeffs(df: DataFrame, max_alpha, min_alpha, model_info, num_alphas, x="rounded_encoding", hue=None, filter_name=None, kfolds_threshold=10, sig_test="t-test"):
    if df[x].unique().size > 20: # Long x axis
        fig = plt.figure(figsize=(15, 15))
    else:
        fig = plt.figure(figsize=(7, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 2])  # 3 rows, 1 column with the top plot larger

    pairs = get_statistical_pairs(x, hue, df)
    # _plot_dist(x=x, y="encoding", hue=hue, pairs=pairs, df=df, ax=plt.subplot(gs[0]), sig_test=sig_test)
    _plot_dist(x=x, y="num_of_chosen_coeffs", hue=hue, pairs=pairs, df=df, ax=plt.subplot(gs[0]), sig_test=sig_test)

    # Plot Amounts
    ax2 = plt.subplot(gs[1])
    if hue:
        sns.countplot(data=df, x=x, hue=hue, order=get_order(df, x), hue_order=get_order(df, hue), ax=ax2, palette=COLOR_PALETTE)
    else:
        sns.countplot(data=df, x=x, order=get_order(df, x), ax=ax2, palette=COLOR_PALETTE,)
    ax2.set_title("Count of items in each encoding group", fontsize=12)
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
        # if violin_annot:
        annotator = Annotator(ax3, pairs, data=df, x=x, y=y, hue=hue, order=get_order(df, x), hue_order=get_order(df, hue))
        annotator.configure(test=sig_test, comparisons_correction="fdr_bh", text_format='star', loc='outside', verbose=2) #text_format='full'
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
        + (f"\nfilter={filter_name}" if filter_name else "")
        + (f"\nsig test={sig_test}"),
        # + f"\nover all times, subjects, and brain areas",
        fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout leaving space for suptitle
    plt.show()


def get_statistical_pairs(x: str, hue, df: DataFrame):
    if hue:
        pairs = [((item1, pair[0]), (item1, pair[1]))
                 for item1 in get_order(df, x)
                 for pair in zip(get_order(df, hue), get_order(df, hue)[1:])]
        if hue == "time_bin_and_brain_area":
            pairs = []
            all_time_bin_and_brain_area = set(df["time_bin_and_brain_area"])
            ordered_times_bins = get_order(df, "time_bin")
            ordered_brain_area = get_order(df, "brain_area")
            # For each hue value
            for category in get_order(df, x):
                # Connect consecutive classes with same time
                for time in ordered_times_bins:
                    for i in range(len(ordered_brain_area) - 1):
                        area1 = ordered_brain_area[i]
                        area2 = ordered_brain_area[i + 1]
                        time_bin_and_brain_area1 = f"{area1}_{time}"
                        time_bin_and_brain_area2 = f"{area2}_{time}"
                        if time_bin_and_brain_area1 in all_time_bin_and_brain_area and time_bin_and_brain_area2 in all_time_bin_and_brain_area:
                            pair = ((category, time_bin_and_brain_area1),
                                    (category, time_bin_and_brain_area2))
                            pairs.append(pair)

                # Connect consecutive times with same class
                for area in ordered_brain_area:
                    for i in range(len(ordered_times_bins) - 1):
                        time1 = ordered_times_bins[i]
                        time2 = ordered_times_bins[i + 1]
                        time_bin_and_brain_area1 = f"{area}_{time1}"
                        time_bin_and_brain_area2 = f"{area}_{time2}"
                        if time_bin_and_brain_area1 in all_time_bin_and_brain_area and time_bin_and_brain_area2 in all_time_bin_and_brain_area:
                            pair = ((category, time_bin_and_brain_area1),
                                    (category, time_bin_and_brain_area2))
                            pairs.append(pair)
        # if hue == "brain_area":
        #     pairs.extend([((item1, pair[0]), (item1, pair[1]))
        #              for item1 in get_order(df, x)
        #              for pair in zip(["STG","IFG"], ["STG","IFG"][1:])])
        # if hue == "time_bin":
        #     pairs.extend([((item1, pair[0]), (item1, pair[1]))
        #              for item1 in get_order(df, x)
        #              for pair in zip(["-0.4≤x<0","0.4≤x<0.8"], ["-0.4≤x<0","0.4≤x<0.8"][1:])])

    else:
        pairs = list(zip(get_order(df, x), get_order(df, x)[1:]))
        if x == "brain_area" and ("STG", "IFG") not in pairs and ("IFG", "STG") not in pairs:
            pairs.append(("STG", "IFG"))
        # if x=="time_bin":
        #     pairs.append(("-0.4≤x<0","0.4≤x<0.8"))
        if x == "time_bin_and_brain_area":
            pairs.remove(("IFG_0≤x<0.4", "STG_-0.4≤x<0"))
            pairs.append(("STG_-0.4≤x<0", "IFG_-0.4≤x<0"))
            pairs.append(("STG_0≤x<0.4", "IFG_0≤x<0.4"))
    return pairs


def _plot_dist(x: str, y:str, hue, pairs, df: DataFrame, ax, sig_test="t-test"):
    if hue:
        sns.boxplot(data=df, x=x, y=y, hue=hue, order=get_order(df, x), palette=COLOR_PALETTE,
                    hue_order=get_order(df, hue), ax=ax)
        # sns.violinplot(data=df, x=x, y=y, hue=hue, order=get_order(df, x), palette=COLOR_PALETTE, hue_order=get_order(df, hue), ax=ax, split=True, inner="quartile") #linewidth=1, density_norm='width'
        annotator = Annotator(ax, pairs, data=df, x=x, y=y, hue=hue, order=get_order(df, x),
                              hue_order=get_order(df, hue))
        annotator.configure(test=sig_test, comparisons_correction="fdr_bh", text_format='star', loc='outside',
                            verbose=2)  # text_format='full'
        # annotator.configure(test='t-test_welch', comparisons_correction="fdr_bh", text_format='star', loc='inside', verbose=2) #text_format='full'
        annotator.apply_and_annotate()
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        ax.set_ylim(ylim[0], ylim[1] + 0.15 * y_range)  # Add 15% space at top

    else:
        sns.boxplot(data=df, x=x, y=y, ax=ax, order=get_order(df, x), palette=COLOR_PALETTE)
        # sns.violinplot(data=df, x=x, y=y, ax=ax, order=get_order(df, x), palette=COLOR_PALETTE, inner="quartile") #linewidth=1, density_norm='width'
        annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=get_order(df, x))
        annotator.configure(test=sig_test, comparisons_correction="fdr_bh", text_format='star', loc='outside',
                            verbose=2)
        # annotator.configure(test='t-test_welch', comparisons_correction="fdr_bh", text_format='star', loc='inside', verbose=2)
        annotator.apply_and_annotate()
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        ax.set_ylim(ylim[0], ylim[1] + 0.15 * y_range)  # Add 15% space at top
    ax.set_title(f"{y} for each {x}", fontsize=14)


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
                    col_to_compare, group1, group2, kfolds_threshold=10, emb_mod=None, query="", df_type="lasso&corr",
                text_shift_left=0.2, text_shift_right=0.2):
    def create_venn_diagram(group1_only, overlap, group2_only,
                            label1="Group A", label2="Group B",
                            title="Venn Diagram",
                            color1="red", color2="blue", overlap_color="purple",
                            text1=None, text2=None, text_overlap=None,
                            font_size=14,
                            text_shift_right=0.0,
                            text_shift_left=0.0,
                            edge_color="darkgrey",
                            edge_width=1.0):
        """
        Create a 2-circle Venn diagram with custom colors and text.

        Parameters:
        -----------
        group1_only : int
            Number of elements only in group 1 (not in group 2)
        overlap : int
            Number of elements in both groups
        group2_only : int
            Number of elements only in group 2 (not in group 1)
        label1 : str
            Label for the first circle
        label2 : str
            Label for the second circle
        title : str
            Title for the diagram
        color1 : str
            Color for the first circle (group 1 only area)
        color2 : str
            Color for the second circle (group 2 only area)
        overlap_color : str
            Color for the overlap area (both groups)
        text1 : str or None
            Custom text for group 1 only area. If None, shows the number. Use "" to hide.
        text2 : str or None
            Custom text for group 2 only area. If None, shows the number. Use "" to hide.
        text_overlap : str or None
            Custom text for overlap area. If None, shows the number. Use "" to hide.
        font_size : int or float
            Font size for the text in the circles (default: 14)
        text_outside : bool
            If True, position text labels outside the circles (default: False)
        edge_color : str or None
            Color for the circle outlines. Use None for no outline (default: "darkgrey")
        edge_width : float
            Width of the circle outlines in points (default: 1.0)
        """
        # Create the Venn diagram
        # venn2 takes a tuple of (Ab, aB, AB) where:
        # Ab = only in A
        # aB = only in B
        # AB = in both A and B
        v = venn2(subsets=(group1_only, group2_only, overlap),
                  set_labels=(label1, label2))

        # Customize colors
        # v.get_patch_by_id('10') is the part only in set A (group 1 only)
        # v.get_patch_by_id('01') is the part only in set B (group 2 only)
        # v.get_patch_by_id('11') is the intersection (overlap)

        if v.get_patch_by_id('10'):
            v.get_patch_by_id('10').set_color(color1)
            v.get_patch_by_id('10').set_alpha(0.6)
            if edge_color:
                v.get_patch_by_id('10').set_edgecolor(edge_color)
                v.get_patch_by_id('10').set_linewidth(edge_width)

        if v.get_patch_by_id('01'):
            v.get_patch_by_id('01').set_color(color2)
            v.get_patch_by_id('01').set_alpha(0.6)
            if edge_color:
                v.get_patch_by_id('01').set_edgecolor(edge_color)
                v.get_patch_by_id('01').set_linewidth(edge_width)

        if v.get_patch_by_id('11'):
            v.get_patch_by_id('11').set_color(overlap_color)
            v.get_patch_by_id('11').set_alpha(0.6)
            if edge_color:
                v.get_patch_by_id('11').set_edgecolor(edge_color)
                v.get_patch_by_id('11').set_linewidth(edge_width)

        # Customize text labels
        # If custom text is provided, replace the default numbers
        if text1 is not None and v.get_label_by_id('10'):
            v.get_label_by_id('10').set_text(text1)

        if text2 is not None and v.get_label_by_id('01'):
            v.get_label_by_id('01').set_text(text2)

        if text_overlap is not None and v.get_label_by_id('11'):
            v.get_label_by_id('11').set_text(text_overlap)

        # Set font size for all labels
        if v.get_label_by_id('10'):
            v.get_label_by_id('10').set_fontsize(font_size)

        if v.get_label_by_id('01'):
            v.get_label_by_id('01').set_fontsize(font_size)

        if v.get_label_by_id('11'):
            v.get_label_by_id('11').set_fontsize(font_size)

        # Move text outside if requested
        # Get current positions and adjust them
        if v.get_label_by_id('10'):
            pos = v.get_label_by_id('10').get_position()
            # Move left circle text further left
            v.get_label_by_id('10').set_position((pos[0] - text_shift_left, pos[1]))

        if v.get_label_by_id('01'):
            pos = v.get_label_by_id('01').get_position()
            # Move right circle text further right
            v.get_label_by_id('01').set_position((pos[0] + text_shift_right, pos[1]))

        # if v.get_label_by_id('11'):
        #     pos = v.get_label_by_id('11').get_position()
        #     # Move overlap text down
        #     v.get_label_by_id('11').set_position((pos[0], pos[1] - 0.3))

        plt.title(title)
        plt.show()

    def create_venn_from_totals(total_group1, total_group2, overlap,
                                label1="Group A", label2="Group B",
                                title="Venn Diagram",
                                color1="red", color2="blue", overlap_color="purple",
                                text1=None, text2=None, text_overlap=None,
                                font_size=14,
                                text_outside=False,
                                edge_color="darkgrey",
                                edge_width=1.0):
        """
        Create a 2-circle Venn diagram from total group sizes with custom colors and text.

        Parameters:
        -----------
        total_group1 : int
            Total number of elements in group 1
        total_group2 : int
            Total number of elements in group 2
        overlap : int
            Number of elements in both groups
        label1 : str
            Label for the first circle
        label2 : str
            Label for the second circle
        title : str
            Title for the diagram
        color1 : str
            Color for the first circle (group 1 only area)
        color2 : str
            Color for the second circle (group 2 only area)
        overlap_color : str
            Color for the overlap area (both groups)
        text1 : str or None
            Custom text for group 1 only area. If None, shows the number. Use "" to hide.
        text2 : str or None
            Custom text for group 2 only area. If None, shows the number. Use "" to hide.
        text_overlap : str or None
            Custom text for overlap area. If None, shows the number. Use "" to hide.
        font_size : int or float
            Font size for the text in the circles (default: 14)
        text_outside : bool
            If True, position text labels outside the circles (default: False)
        edge_color : str or None
            Color for the circle outlines. Use None for no outline (default: "darkgrey")
        edge_width : float
            Width of the circle outlines in points (default: 1.0)
        """
        group1_only = total_group1 - overlap
        group2_only = total_group2 - overlap

        create_venn_diagram(group1_only, overlap, group2_only, label1, label2, title,
                            color1, color2, overlap_color, text1, text2, text_overlap,
                            font_size, text_outside, edge_color, edge_width)

    exploded_df, first_group_all_coeffs, first_group_name, second_group_all_coeffs, second_group_name = _get_groups_distinct_coeffs(
        col_to_compare, filter_type, group1, group2, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
        num_alphas, emb_mod, query, sid, coeff_type=df_type)
    first_group_all_coeffs = {int(coeff) for coeff in first_group_all_coeffs}
    second_group_all_coeffs = {int(coeff) for coeff in second_group_all_coeffs}

    group1_unique_amount = len(first_group_all_coeffs - second_group_all_coeffs)
    overlap_amount = len(second_group_all_coeffs & first_group_all_coeffs)
    group2_unique_amount = len(second_group_all_coeffs - first_group_all_coeffs)

    print(f"\nOnly in {first_group_name} ({group1_unique_amount}): {first_group_all_coeffs - second_group_all_coeffs}")
    print(f"\nOnly in {second_group_name} ({group2_unique_amount}): {second_group_all_coeffs - first_group_all_coeffs}")
    print(f"\nIn both groups ({overlap_amount}): {second_group_all_coeffs & first_group_all_coeffs}")

    divide_by = len(second_group_all_coeffs | first_group_all_coeffs)#6064 if model_info['model_short_name'] == "gemma-scope9b" else model_info['embedding_size']
    group1_only = round((group1_unique_amount / divide_by) * 100, 1) * 10
    overlap = round((overlap_amount / divide_by) * 100, 1) * 10
    group2_only = round((group2_unique_amount / divide_by) * 100, 1) * 10
    create_venn_diagram(
        label1=first_group_name, #"" if col_to_compare == "brain_area" else "before" if col_to_compare == "time" else "",
        label2=second_group_name, #"STG" if col_to_compare == "brain_area" else "after" if col_to_compare == "time" else "",
        color1=COLOR_PALETTE[first_group_name], #"#FFB7D9" if col_to_compare == "area" else "#FFC599" if col_to_compare == "time" else "",
        color2=COLOR_PALETTE[second_group_name], #"#99E6C3" if col_to_compare == "area" else "#9EE6FC" if col_to_compare == "time" else "",
        overlap_color="#E0DADF",
        group1_only=group1_only,
        overlap=overlap,
        group2_only=group2_only,
        title=    f"Coefficients for each group, type of coeffs - {df_type}"
                  + (f"\nModel: {model_info['model_short_name']} (k={kfolds_threshold},α range: {min_alpha} to {max_alpha}, {num_alphas} values)")
                  + (f"\nfilter={query}" if query else ""),
        text1=f"{group1_only / 10}%",
        text2=f"{group2_only / 10}%",
        text_overlap=f"{overlap / 10}%",
        font_size=25,
        text_shift_left=text_shift_left,
        text_shift_right=text_shift_right,
        edge_color="black",
        edge_width=0.8,
    )

def coeffs_venn_old(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                col_to_compare, group1, group2, kfolds_threshold=10, emb_mod=None, query="", df_type="lasso&corr"):
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
    exploded_df, first_group_all_coeffs, first_group_name, second_group_all_coeffs, second_group_name = _get_groups_distinct_coeffs(
        col_to_compare, filter_type, group1, group2, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
        num_alphas, emb_mod, query, sid, coeff_type=df_type)
    first_group_all_coeffs = {int(coeff) for coeff in first_group_all_coeffs}
    second_group_all_coeffs = {int(coeff) for coeff in second_group_all_coeffs}

    print(f"\nOnly in {first_group_name} ({len(first_group_all_coeffs - second_group_all_coeffs)}): {first_group_all_coeffs - second_group_all_coeffs}")
    print(f"\nOnly in {second_group_name} ({len(second_group_all_coeffs - first_group_all_coeffs)}): {second_group_all_coeffs - first_group_all_coeffs}")
    print(f"\nIn both groups ({len(second_group_all_coeffs & first_group_all_coeffs)}): {second_group_all_coeffs & first_group_all_coeffs}")

    set_colors = (COLOR_PALETTE[first_group_name], COLOR_PALETTE[second_group_name])
    venn2((first_group_all_coeffs, second_group_all_coeffs), (first_group_name, second_group_name), set_colors=set_colors)
    plt.title(f"Coefficients for each group, type of coeffs - {df_type}"
              + (f"\nModel: {model_info['model_short_name']} (α range: {min_alpha} to {max_alpha}, {num_alphas} values)")
              + (f"\nfilter={query}" if query else ""))
    plt.show()

    # venn3((first_group_sig_coeffs, second_group_sig_coeffs, all_coeffs), (first_group_name, second_group_name, "all_coeffs"))
    # plt.title("Distinct Statistically Significant Coefficients for each group"
    #           + (f"\nModel: {model_info['model_short_name']} (α range: {min_alpha} to {max_alpha}, {num_alphas} values)")
    #           + (f"\nfilter={query}" if query else ""))
    # plt.show()

    print("done!")

def amount_distinct_coeffs(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                           x="rounded_encoding", hue=None, kfolds_threshold=10, query=""):
    kfolds_df = prepare_coeffs_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
                                  num_alphas, sid, query=query)
    exploded_df = kfolds_df.query("num_of_chosen_coeffs > 0").explode("actual_chosen_coeffs")

    if hue:
        distinct_coeffs_df = exploded_df[["actual_chosen_coeffs", x, hue]].drop_duplicates()
        grouped_counts = distinct_coeffs_df.groupby([x, hue]).size().reset_index(name='distinct_coeffs')
    else:
        distinct_coeffs_df = exploded_df[["actual_chosen_coeffs", x]].drop_duplicates()
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
                                min_alpha, mode, model_info, num_alphas, emb_mod, query: str, sid, coeff_type="lasso&corr"):
    query = get_group_query(col_to_compare, group1, group2, query)

    if coeff_type == "lasso&corr" or coeff_type == "corr&lasso":
        exploded_df, _ = _get_exploded_united_lasso_and_corr(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                                                             kfolds_threshold, emb_mod=emb_mod, query=query)
    else:
        coeff_df = prepare_coeffs_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
                                        num_alphas, sid, emb_mod=emb_mod, query=query, df_type=coeff_type)
        exploded_df = coeff_df.query("num_of_chosen_coeffs > 0").explode(["actual_chosen_coeffs", "chosen_coeffs_val"])

        # important_args = {"model_info": model_info, "emb_mod": emb_mod, "query": query, "coeff_type": coeff_type, "filter_type": filter_type}
        calc_and_plot_prop("frequency", exploded_df, coeff_df, coeff_type, model_info, col_to_compare, group1, group2, axis_func=calc_freq_for_axis_value)
        # calc_and_plot_prop("frequency within chosen", exploded_df, coeff_df, coeff_type, model_info, axis_func=calc_freq_within_chosen_for_axis_value)
        # calc_and_plot_prop("abs_mean", exploded_df, coeff_df, coeff_type, model_info, col_to_compare, group1, group2, axis_func=calc_abs_mean_for_axis_value)
        # calc_and_plot_prop("abs_mean", exploded_df, coeff_df, coeff_type, model_info, col_to_compare, group1, group2, axis_func=calc_abs_mean_for_axis_value)
        # calc_and_plot_prop("mean", exploded_df, coeff_df, coeff_type, model_info, col_to_compare, group1, group2, axis_func=calc_mean_for_axis_value)
        exploded_df.rename(columns={"chosen_coeffs_val": f"chosen_coeffs_val_{coeff_type}",}, inplace=True)

    first_group_name, second_group_name = sorted([group1, group2])

    first_group_all_coeffs = set(
        exploded_df.query(f"{col_to_compare} == '{first_group_name}'")["actual_chosen_coeffs"].to_list())
    second_group_all_coeffs = set(
        exploded_df.query(f"{col_to_compare} == '{second_group_name}'")["actual_chosen_coeffs"].to_list())

    if coeff_type == "corr" or coeff_type == "ridge": # Only done on corr/ridge because they have a continues value while lasso is either chosen or not
        first_group_all_coeffs, second_group_all_coeffs, _= run_group_permutation_test(col_to_compare, coeff_df, first_group_all_coeffs, second_group_all_coeffs, first_group_name,
                                                                                        second_group_name, n_permutations=5000)
    run_general_permutation_test(col_to_compare, coeff_df, first_group_all_coeffs, second_group_all_coeffs,
                                 first_group_name, second_group_name, model_info, n_permutations=5000)

    exploded_df[f"is_in_{first_group_name}"] = exploded_df["actual_chosen_coeffs"].isin(first_group_all_coeffs)
    exploded_df[f"is_in_{second_group_name}"] = exploded_df["actual_chosen_coeffs"].isin(second_group_all_coeffs)
    exploded_df['group'] = exploded_df['actual_chosen_coeffs'].apply(lambda x:
                                                        'both' if x in first_group_all_coeffs and x in second_group_all_coeffs
                                                        else first_group_name if x in first_group_all_coeffs
                                                        else second_group_name if x in second_group_all_coeffs
                                                        else None
                                                        )
    # assert exploded_df['group'].notna().all(), "Found None/NaN values in 'group' column" # turned off because it is None when it was removed from the groups in the permutation test

    return exploded_df, first_group_all_coeffs, first_group_name, second_group_all_coeffs, second_group_name

def calc_abs_mean_for_axis_value(exploded_df, non_zero_df, coeff_type, model_name, col_to_compare, group1, group2):
    del exploded_df
    gc.collect()

    all_exp = non_zero_df.explode(["all_coeffs_index", "all_coeffs_val"])

    del non_zero_df
    gc.collect()

    # The direction doesn't matter, just the magnitude, so take absolute values
    all_exp["all_coeffs_val"] = all_exp["all_coeffs_val"].abs()

    stats_df = all_exp.groupby(['all_coeffs_index', col_to_compare])['all_coeffs_val'].agg(
        mean='mean',
        se=stats.sem
    ).reset_index()

    del all_exp
    gc.collect()

    save_prefix = f'/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/results/maybe_paper/scatter_plots/type_{coeff_type}/for_mean/by_{col_to_compare}/model_{model_name}_stats'
    Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(save_prefix+".csv", index=False)
    stats_df.to_pickle(save_prefix+".pkl")

    print(f"Abs Mean stats saved to {save_prefix}")

    # Pivot to get brain areas as columns for mean
    pivot_mean = stats_df.pivot(index='all_coeffs_index',
                                columns=col_to_compare,
                                values='mean')  # .fillna(0)
    pivot_mean = pivot_mean.reset_index()

    # Pivot to get brain areas as columns for SE
    pivot_se = stats_df.pivot(index='all_coeffs_index',
                              columns=col_to_compare,
                              values='se')#.fillna(0)
    pivot_se = pivot_se.reset_index()

    # Rename columns to distinguish mean and SE
    pivot_mean = pivot_mean.rename(columns={group1: f'{group1}_mean', group2: f'{group2}_mean'})
    pivot_se = pivot_se.rename(columns={group1: f'{group1}_se', group2: f'{group2}_se'})

    # Merge mean and SE DataFrames
    result_df = pivot_mean.merge(pivot_se[['all_coeffs_index', f'{group1}_se', f'{group2}_se']],
                                 on='all_coeffs_index')

    epsilon = 1e-9
    # if coeff_type == "corr":
    #     score_text = r'$\frac{||x| - |y||}{|x| + |y| + \epsilon}$'
    #     result_df["score"] = np.abs(result_df[f'{group2}_mean'] - result_df[f'{group1}_mean']) / (
    #             result_df[f'{group2}_mean'] + result_df[f'{group1}_mean'] + epsilon)
    # else:
    score_text = r'$\frac{||x| - |y||}{|x| + |y| + \epsilon}$'
    result_df["score"] = np.abs(result_df[f'{group2}_mean'] - result_df[f'{group1}_mean']) / (
            result_df[f'{group2}_mean'] + result_df[f'{group1}_mean'] + epsilon)  # np.sqrt(2) *

    # Get axis names
    col_names = [f'{group1}_mean', f'{group2}_mean']
    axis_names = [f'{group1} Abs Mean', f'{group2} Abs Mean']
    coeff_name_col = "all_coeffs_index"

    return result_df, col_names, axis_names, coeff_name_col, score_text


def calc_abs_mean_for_chosen_for_axis_value(exploded_df, non_zero_df, coeff_type, model_name, col_to_compare, group1, group2):
    del exploded_df
    gc.collect()

    all_exp = non_zero_df.explode(["all_coeffs_index", "all_coeffs_val"])

    del non_zero_df
    gc.collect()

    # The direction doesn't matter, just the magnitude, so take absolute values
    all_exp["all_coeffs_val"] = all_exp["all_coeffs_val"].abs()

    stats_df = all_exp.groupby(['all_coeffs_index', col_to_compare])['all_coeffs_val'].agg(
        mean='mean',
        se=stats.sem
    ).reset_index()

    del all_exp
    gc.collect()

    save_prefix = f'/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/results/maybe_paper/scatter_plots/type_{coeff_type}/for_mean/by_{col_to_compare}/model_{model_name}_stats'
    Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(save_prefix+".csv", index=False)
    stats_df.to_pickle(save_prefix+".pkl")

    print(f"Abs Mean stats saved to {save_prefix}")

    # Pivot to get brain areas as columns for mean
    pivot_mean = stats_df.pivot(index='all_coeffs_index',
                                columns=col_to_compare,
                                values='mean')  # .fillna(0)
    pivot_mean = pivot_mean.reset_index()

    # Pivot to get brain areas as columns for SE
    pivot_se = stats_df.pivot(index='all_coeffs_index',
                              columns=col_to_compare,
                              values='se')#.fillna(0)
    pivot_se = pivot_se.reset_index()

    # Rename columns to distinguish mean and SE
    pivot_mean = pivot_mean.rename(columns={group1: f'{group1}_mean', group2: f'{group2}_mean'})
    pivot_se = pivot_se.rename(columns={group1: f'{group1}_se', group2: f'{group2}_se'})

    # Merge mean and SE DataFrames
    result_df = pivot_mean.merge(pivot_se[['all_coeffs_index', f'{group1}_se', f'{group2}_se']],
                                 on='all_coeffs_index')

    epsilon = 1e-9
    # if coeff_type == "corr":
    #     score_text = r'$\frac{||x| - |y||}{|x| + |y| + \epsilon}$'
    #     result_df["score"] = np.abs(result_df[f'{group2}_mean'] - result_df[f'{group1}_mean']) / (
    #             result_df[f'{group2}_mean'] + result_df[f'{group1}_mean'] + epsilon)
    # else:
    score_text = r'$\frac{||x| - |y||}{|x| + |y| + \epsilon}$'
    result_df["score"] = np.abs(result_df[f'{group2}_mean'] - result_df[f'{group1}_mean']) / (
            result_df[f'{group2}_mean'] + result_df[f'{group1}_mean'] + epsilon)  # np.sqrt(2) *

    # Get axis names
    col_names = [f'{group1}_mean', f'{group2}_mean']
    axis_names = [f'{group1} Abs Mean', f'{group2} Abs Mean']
    coeff_name_col = "all_coeffs_index"

    return result_df, col_names, axis_names, coeff_name_col, score_text

def calc_mean_for_axis_value(exploded_df, non_zero_df, coeff_type, model_name, col_to_compare, group1, group2):
    del exploded_df
    gc.collect()

    all_exp = non_zero_df.explode(["all_coeffs_index", "all_coeffs_val"])

    del non_zero_df
    gc.collect()

    stats_df = all_exp.groupby(['all_coeffs_index', col_to_compare])['all_coeffs_val'].agg(
        mean='mean',
        se=stats.sem
    ).reset_index()

    del all_exp
    gc.collect()

    save_prefix = f'/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/results/maybe_paper/scatter_plots/type_{coeff_type}/for_mean/by_{col_to_compare}/model_{model_name}_stats'
    Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(save_prefix+".csv", index=False)
    stats_df.to_pickle(save_prefix+".pkl")

    print(f"Mean stats saved to {save_prefix}")

    # Pivot to get brain areas as columns for mean
    pivot_mean = stats_df.pivot(index='all_coeffs_index',
                                columns=col_to_compare,
                                values='mean')  # .fillna(0)
    pivot_mean = pivot_mean.reset_index()

    # Pivot to get brain areas as columns for SE
    pivot_se = stats_df.pivot(index='all_coeffs_index',
                              columns=col_to_compare,
                              values='se')#.fillna(0)
    pivot_se = pivot_se.reset_index()

    # Rename columns to distinguish mean and SE
    pivot_mean = pivot_mean.rename(columns={group1: f'{group1}_mean', group2: f'{group2}_mean'})
    pivot_se = pivot_se.rename(columns={group1: f'{group1}_se', group2: f'{group2}_se'})

    # Merge mean and SE DataFrames
    result_df = pivot_mean.merge(pivot_se[['all_coeffs_index', f'{group1}_se', f'{group2}_se']],
                                 on='all_coeffs_index')

    epsilon = 1e-9
    # if coeff_type == "corr":
    #     score_text = r'$\frac{||x| - |y||}{|x| + |y| + \epsilon}$'
    #     result_df["score"] = np.abs(result_df[f'{group2}_mean'] - result_df[f'{group1}_mean']) / (
    #             result_df[f'{group2}_mean'] + result_df[f'{group1}_mean'] + epsilon)
    # else:
    score_text = r'$\frac{|x - y|}{x + y + \epsilon}$'
    # result_df["score"] = np.abs(result_df[f'{group2}_mean'] - result_df[f'{group1}_mean']) / (
    #         result_df[f'{group2}_mean'] + result_df[f'{group1}_mean'] + epsilon)  # np.sqrt(2) *

    result_df["score"] = 1

    # Get axis names
    col_names = [f'{group1}_mean', f'{group2}_mean']
    axis_names = [f'{group1} Mean', f'{group2} Mean']
    coeff_name_col = "all_coeffs_index"

    return result_df, col_names, axis_names, coeff_name_col, score_text

def calc_freq_for_axis_value(exploded_df, non_zero_df, coeff_type, model_name, col_to_compare, group1, group2):
    score_text = r'$\frac{|x - y|}{x + y + \epsilon}$' #\sqrt{2} \cdot

    counts = exploded_df.groupby(['actual_chosen_coeffs', col_to_compare]).size().reset_index(name='count') # long df with columns - 'actual_chosen_coeffs', 'brain_area'/'time_bin' etc., 'count'
    # Pivot to get brain areas as columns
    pivot_counts = counts.pivot(index='actual_chosen_coeffs',
                                columns=col_to_compare,
                                values='count').fillna(0) # wide df with columns - 'IFG' and 'STG' or 'before_onset' and 'after_onset' etc.

    # Reset index to make actual_chosen_coeffs a column
    pivot_counts = pivot_counts.reset_index()
    pivot_counts.columns.name = None

    result = non_zero_df.groupby(col_to_compare)[["full_elec_name"]].nunique() # Used for normalizing - counts how many unique electrodes per brain area/time_bin

    pivot_counts[group1] = pivot_counts[group1] / (result.loc[group1, "full_elec_name"] * 161)
    pivot_counts[group2] = pivot_counts[group2] / (result.loc[group2, "full_elec_name"] * 161)

    epsilon = 1e-9
    pivot_counts["score"] = np.abs(pivot_counts[group2] - pivot_counts[group1]) / (pivot_counts[group2] + pivot_counts[group1] + epsilon) #np.sqrt(2) *
    # pivot_counts["score"] = np.abs(np.log(np.clip(pivot_counts[group1], epsilon, 1.0)) - np.log(np.clip(pivot_counts[group2], epsilon, 1.0)))

    # Get the two brain area names
    col_names = [group1, group2]#[col for col in pivot_counts.columns if col != 'actual_chosen_coeffs']
    axis_names = [f'Frequency in {col}' for col in col_names]

    coeff_name_col = "actual_chosen_coeffs"

    return pivot_counts, col_names, axis_names, coeff_name_col, score_text

def calc_freq_within_chosen_for_axis_value(exploded_df, non_zero_df, coeff_type, model_name, col_to_compare, group1, group2):
    score_text = r'$\frac{|x - y|}{x + y + \epsilon}$' #\sqrt{2} \cdot

    counts = exploded_df.groupby(['actual_chosen_coeffs', col_to_compare]).size().reset_index(name='count')

    # Pivot to get brain areas as columns
    pivot_counts = counts.pivot(index='actual_chosen_coeffs',
                                columns=col_to_compare,
                                values='count').fillna(0)

    # Reset index to make actual_chosen_coeffs a column
    pivot_counts = pivot_counts.reset_index()
    pivot_counts.columns.name = None

    pivot_counts[group1] = pivot_counts[group1] / sum(pivot_counts[group1])
    pivot_counts[group2] = pivot_counts[group2] / sum(pivot_counts[group2])

    epsilon = 1e-9
    pivot_counts["score"] = np.abs(pivot_counts[group2] - pivot_counts[group1]) / (pivot_counts[group2] + pivot_counts[group1] + epsilon) #np.sqrt(2) *
    # pivot_counts["score"] = np.abs(np.log(np.clip(pivot_counts[group1], epsilon, 1.0)) - np.log(np.clip(pivot_counts[group2], epsilon, 1.0)))

    # Get the two brain area names
    col_names = [group1, group2]#[col for col in pivot_counts.columns if col != 'actual_chosen_coeffs']
    axis_names = [f'Frequency in {col} from all chosen coeffs' for col in col_names]

    coeff_name_col = "actual_chosen_coeffs"

    return pivot_counts, col_names, axis_names, coeff_name_col, score_text


def plot_prop_scatter_plot_and_hist(df, calc_name, col_names, axis_names, coeff_type, model_info, score_text, coeff_name_col):
    # Create subplot with 2 rows, different heights
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],  # Top plot taller
        vertical_spacing=0.12
    )

    # --- Scatter plot (top) ---
    scatter = px.scatter(
        df,
        x=col_names[0],
        y=col_names[1],
        color='score',
        hover_name=coeff_name_col,
        hover_data={col_names[0]: True, col_names[1]: True, 'score': ':.3f'},
        color_continuous_scale='sunsetdark',
    )

    for trace in scatter.data:
        fig.add_trace(trace, row=1, col=1)

    # # --- Histogram (bottom) ---
    # hist = px.histogram(
    #     df,
    #     x="score",
    #     histnorm='probability',
    #     color_discrete_sequence=['#F9A175']
    # )
    #
    # for trace in hist.data:
    #     trace.xbins = dict(size=0.02)
    #     trace.marker.line = dict(color='white', width=0.5)
    #     trace.opacity = 0.85
    #     fig.add_trace(trace, row=2, col=1)

    # --- Compute histogram manually ---
    bin_size = 0.02
    bins = np.arange(0, 1 + bin_size, bin_size)
    counts, bin_edges = np.histogram(df['score'], bins=bins)
    probabilities = counts / counts.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Get colors from the same colorscale
    colors = px.colors.sample_colorscale('sunsetdark', bin_centers)  # bin_centers are 0-1, perfect for colorscale

    # --- Histogram (bottom) - use go.Bar instead ---
    hist_trace = go.Bar(
        x=bin_centers,
        y=probabilities,
        width=bin_size,
        marker=dict(
            color=colors,
            line=dict(color='white', width=0.5)
        ),
        opacity=0.85,
        showlegend=False
    )
    fig.add_trace(hist_trace, row=2, col=1)

    # --- Layout ---
    max_value = max(df[col_names[0]].max(), df[col_names[1]].max())

    fig.update_layout(
        width=700,
        height=800,
        template='plotly_white',
        title=f'Coefficient {calc_name} by {col_to_compare} ({coeff_type}, {model_info["model_short_name"]})',
        coloraxis=dict(colorscale='sunsetdark',
                       cmin=0,
                       cmax=1,
                       colorbar=dict(title='',
                                     len=0.6,  # Length as fraction of plot height (matches top plot)
                                     y=0.75,  # Vertical position (0=bottom, 1=top)
                                     yanchor='middle')),
        hovermode='closest',
    )

    # Top plot axes
    fig.update_xaxes(range=[0, max_value], title=axis_names[0], title_font_size=18, tickfont_size=14, row=1, col=1)
    fig.update_yaxes(range=[0, max_value], title=axis_names[1], title_font_size=18, tickfont_size=14, row=1, col=1)

    # Bottom plot axes
    fig.update_xaxes(range=[0, 1], title='Score', tickformat='.2f', title_font_size=18, tickfont_size=14, row=2, col=1)
    fig.update_yaxes(title='Probability', title_font_size=18, tickfont_size=14, row=2, col=1)

    # Add annotation for colorbar
    fig.add_annotation(
        x=1.1, y=1.1,
        xref='paper', yref='paper',
        text=score_text,
        showarrow=False,
        font=dict(size=18)
    )
    save_prefix = f'/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/results/maybe_paper/scatter_plots/type_{coeff_type}/for_{calc_name}/by_{col_to_compare}/model_{model_info["model_short_name"]}_scatterplot'
    Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_prefix+'.html')
    fig.write_image(save_prefix+'.pdf')
    fig.write_image(save_prefix+'.png')

    if SHOW_FIGS:
        fig.show()

def calc_and_plot_prop(calc_name, exploded_df, non_zero_df, coeff_type, model_info, col_to_compare, group1, group2, axis_func=calc_freq_for_axis_value):
    # Group by both actual_chosen_coeffs and brain_area, then count occurrences
    score_df, col_names, axis_names, coeff_name_col, score_text = axis_func(exploded_df, non_zero_df, coeff_type, model_info["model_short_name"], col_to_compare, group1, group2)
    score_df['score'] = pd.to_numeric(score_df['score'])
    score_df.to_pickle(f"/scratch/gpfs/HASSON/tk6637/princeton/temp_data/paper/violin_plot/{coeff_type}_{model_info['model_short_name']}_{col_to_compare}_{calc_name}_.pkl")

    # plot_score_contour()
    plot_prop_scatter_plot_and_hist(score_df, calc_name, col_names, axis_names, coeff_type, model_info, score_text, coeff_name_col)

    group_com_text = f"{col_names[0]}>{col_names[1]}"
    score_df[group_com_text] = score_df[col_names[0]] > score_df[col_names[1]]
    print(f"\n{'*'*8} Everything for {calc_name} {'*'*8}")
    print(f"\nWhere: {group_com_text}")
    print(score_df[score_df[group_com_text] == True].sort_values(by="score", ascending=False)[:10])
    print(f"\nWhere not: {group_com_text}")
    print(score_df[score_df[group_com_text] == False].sort_values(by="score", ascending=False)[:10])
    print("\nWhere they are equal")
    print(score_df.sort_values(by="score", ascending=True)[:10])

def run_general_permutation_test(col_to_compare, coeff_df, first_group_all_coeffs, second_group_all_coeffs, first_group_name, second_group_name, model_info, n_permutations=5000):
    """
    Pipeline-level permutation test for the Jaccard index J = |A ∩ B| / |A ∪ B|.

    Shuffles condition labels n_permutations times and recomputes J each time to build a
    null distribution. Shuffling is done at the appropriate unit level:
      - brain_area: shuffle at electrode level (full_elec_name), preserving within-electrode
        coefficient correlation structure.
      - time_bin: shuffle at time_bin level, preserving within-time-bin structure across
        electrodes.

    Addresses reviewer concern: "Without a condition-label permutation test, the statistical
    significance of J cannot be evaluated."
    """
    # Observed J
    union = first_group_all_coeffs | second_group_all_coeffs
    observed_J = len(first_group_all_coeffs & second_group_all_coeffs) / len(union)
    print(f"Observed J = {observed_J:.4f}")

    # Determine the unit at which to shuffle labels
    if col_to_compare == "brain_area":
        unit_col = "full_elec_name"
    elif col_to_compare == "time_bin":
        unit_col = "time_index"
    else:
        raise ValueError(f"Unsupported col_to_compare: {col_to_compare}. Add a unit_col mapping for it.")

    unit_labels = (
        coeff_df[[unit_col, col_to_compare]]
        .drop_duplicates()
        .set_index(unit_col)[col_to_compare]
    )
    units = unit_labels.index.values
    labels = unit_labels.values

    null_J = np.zeros(n_permutations)
    for i in tqdm(range(n_permutations), desc="Jaccard permutation test"):
        shuffled_labels = np.random.permutation(labels)
        label_map = dict(zip(units, shuffled_labels))
        perm_df = coeff_df.copy()
        perm_df[col_to_compare] = perm_df[unit_col].map(label_map)
        perm_exploded = perm_df.query("num_of_chosen_coeffs > 0").explode(["actual_chosen_coeffs", "chosen_coeffs_val"])
        g1 = set(perm_exploded.query(f"{col_to_compare} == '{first_group_name}'")["actual_chosen_coeffs"])
        g2 = set(perm_exploded.query(f"{col_to_compare} == '{second_group_name}'")["actual_chosen_coeffs"])
        perm_union = g1 | g2
        null_J[i] = len(g1 & g2) / len(perm_union)

    # One-tailed p-value: how often does the null produce J <= observed
    # (tests whether the observed overlap is lower than what chance predicts, i.e. groups are more distinct than chance)
    p_value = np.mean(null_J <= observed_J)

    print(f"Null J: mean = {null_J.mean():.4f}, std = {null_J.std():.4f}")
    print(f"P-value (two-tailed): {p_value:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(null_J, bins=50, alpha=0.7, edgecolor='black', label='Null distribution')
    plt.axvline(observed_J, color='red', linestyle='--', linewidth=2,
                label=f'Observed J = {observed_J:.4f}')
    plt.axvline(null_J.mean(), color='blue', linestyle=':', linewidth=1.5,
                label=f'Null mean = {null_J.mean():.4f}')
    plt.xlabel('Jaccard index (J)')
    plt.ylabel('Frequency')
    plt.title(
        f'Pipeline permutation test: null J distribution\n'
        f'{first_group_name} vs {second_group_name} | {model_info["model_short_name"]} | '
        f'p = {p_value:.4f} (n={n_permutations})'
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    return observed_J, null_J, p_value

def run_group_permutation_test(col_to_compare, coeff_df, first_group_all_coeffs, second_group_all_coeffs, first_group_name, second_group_name, n_permutations=5000):
    """
    For each coefficient that is in only one of the groups, runs a permutation test to see if the difference in means between the two groups is significant. If not, removes it from the group.
    :return: Updated groups, and a dict with all the results
    """
    # only_in_first = first_group_all_coeffs - second_group_all_coeffs
    # only_in_second = second_group_all_coeffs - first_group_all_coeffs
    unique_for_single_group = first_group_all_coeffs ^ second_group_all_coeffs
    coeff_df_cp = coeff_df.copy()

    results = {}
    removed_items_first_group = set()
    removed_items_second_group = set()

    for coeff in tqdm(unique_for_single_group, desc=f"Running permutation test"):
        coeff = int(coeff)
        # Get value of the coefficient for each time point and electrode in the original df
        coeff_df_cp['val'] = coeff_df_cp['all_coeffs_val'].apply(lambda x: x[coeff])

        first_group_size = (coeff_df_cp[col_to_compare] == first_group_name).sum()
        second_group_size = (coeff_df_cp[col_to_compare] == second_group_name).sum()

        first_group_mean = coeff_df_cp[coeff_df_cp[col_to_compare] == first_group_name]['val'].mean()
        second_group_mean = coeff_df_cp[coeff_df_cp[col_to_compare] == second_group_name]['val'].mean()
        observed_diff = first_group_mean - second_group_mean

        group_mask = coeff_df_cp[col_to_compare].isin([first_group_name, second_group_name])
        all_values = coeff_df_cp.loc[group_mask, 'val'].values

        null_distribution = np.zeros(n_permutations)

        for i in range(n_permutations):
            permuted_values = np.random.permutation(all_values)
            # Split according to original group sizes
            perm_first_group = permuted_values[:first_group_size]
            perm_second_group = permuted_values[first_group_size:first_group_size + second_group_size]
            # Calculate difference in means
            null_distribution[i] = perm_first_group.mean() - perm_second_group.mean()

        # Calculate one-sided p-value: test whether coeff is higher in the group that selected it
        if str(coeff) in first_group_all_coeffs and str(coeff) in second_group_all_coeffs:
            raise ValueError("Problem! Not supposed to run on elements that are in both groups")
        elif str(coeff) in first_group_all_coeffs:
            p_value = np.mean(null_distribution >= observed_diff)
        elif str(coeff) in second_group_all_coeffs:
            p_value = np.mean(null_distribution <= observed_diff)
        else:
            raise ValueError("Problem! Coeff should be in at least one of the groups")

        results[coeff] = {
            'null_distribution': null_distribution,
            'observed_difference': observed_diff,
            'p_value': p_value
        }

    # FDR correction across all tested coefficients
    coeffs_list = list(results.keys())
    p_values = [results[c]['p_value'] for c in coeffs_list]
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    for coeff, is_significant, p_corr in zip(coeffs_list, rejected, p_corrected):
        results[coeff]['p_value_corrected'] = p_corr
        if not is_significant:
            if str(coeff) in first_group_all_coeffs:
                first_group_all_coeffs.remove(str(coeff))
                removed_items_first_group.add(coeff)
            elif str(coeff) in second_group_all_coeffs:
                second_group_all_coeffs.remove(str(coeff))
                removed_items_second_group.add(coeff)

    print(f"Removed items from {first_group_name} group: {len(removed_items_first_group)}, items: {removed_items_first_group}")
    print(f"Removed items from {second_group_name} group: {len(removed_items_second_group)}, items: {removed_items_second_group}")

    if len(list(results.keys())) > 0:
    # Example plot
        example_idx = list(results.keys())[0]
        null_dist = results[example_idx]['null_distribution']
        observed = results[example_idx]['observed_difference']

        plt.figure(figsize=(10, 6))
        plt.hist(null_dist, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(observed, color='red', linestyle='--', linewidth=2, label='Observed difference')
        plt.xlabel('Difference in means')
        plt.ylabel('Frequency')
        plt.title(f'Null distribution for all_coeffs_index = {example_idx}')
        plt.legend()
        plt.show()

        print(f"corrected_p-value: {results[example_idx]['p_value_corrected']:.4f}")

    return first_group_all_coeffs, second_group_all_coeffs, results




def get_group_query(col_to_compare, group1, group2, query: str) -> str:
    if query:
        query = "(" + query + f") & (({col_to_compare} == '{group1}') | ({col_to_compare} == '{group2}'))"
    else:
        query = f"(({col_to_compare} == '{group1}') | ({col_to_compare} == '{group2}'))"
    return query

def run_all_boxplots(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, kfolds_threshold, df_type="lasso&corr", sig_test="t-test"):

    # x,hue,query:
    all_options = [
        # {"x": "rounded_encoding", "hue": None, "query": ""},
        # {"x": "rounded_encoding", "hue": None, "query": "brain_area == 'IFG'"},
        # {"x": "rounded_encoding", "hue": None, "query": "brain_area == 'STG'"},
        # {"x": "rounded_encoding", "hue": None, "query": "time_bin == '-0.4≤x<0'"},
        # {"x": "rounded_encoding", "hue": None, "query": "time_bin == '0≤x<0.4'"},
        # {"x": "brain_area", "hue": None, "query": ""},
        # {"x": "brain_area", "hue":None, "query": AREA_QUERY},
        # {"x": "rounded_encoding", "hue": "brain_area", "query": AREA_QUERY},
        # {"x": "time_bin", "hue": None, "query": ""},
        # {"x": "time_bin", "hue": None, "query": TIME_QUERY},
        # {"x": "rounded_encoding", "hue": "time_bin", "query": TIME_QUERY},
        # {"x": "time_bin_and_brain_area", "hue": None, "query": f"({TIME_QUERY})&({AREA_QUERY})"},
        # {"x": "rounded_encoding", "hue": "time_bin_and_brain_area", "query": f"({TIME_QUERY})&({AREA_QUERY})", "violin_annot": False},
        # {"x": "rounded_encoding", "hue": "time_bin_and_brain_area", "query": "({TIME_QUERY})&((brain_area == 'IFG'))"},
        # {"x": "rounded_encoding", "hue": "time_bin_and_brain_area", "query": "({TIME_QUERY})&((brain_area == 'STG'))"},
        # {"x": "rounded_encoding", "hue": "time_bin_and_brain_area", "query": f"((time_bin == '-0.4≤x<0'))&({AREA_QUERY})"},
        # {"x": "rounded_encoding", "hue": "time_bin_and_brain_area", "query": f"((time_bin == '0≤x<0.4'))&({AREA_QUERY})"},
        {"x": "brain_area", "hue": None, "query": AREA_QUERY},
        {"x": "time_bin", "hue": None, "query": TIME_QUERY},
        {"x": "time_bin_and_brain_area", "hue": None, "query": f"({TIME_QUERY})&({AREA_QUERY})"},
    ]

    for curr in all_options:
        print(curr)
        violin_annot = curr["violin_annot"] if ("violin_annot" in curr.keys()) else True
        plot_x_vs_num_of_coeffs(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                            x=curr["x"], hue=curr["hue"], kfolds_threshold=kfolds_threshold, query=curr["query"], violin_annot=violin_annot, df_type=df_type, sig_test=sig_test)
        print()

def coeffs_values(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, col_to_compare, group1, group2,
                      query="", kfolds_threshold=10, df_type="lasso&corr", top=5):
    print("straring coeffs_values")

    exploded_df, first_group_all_coeffs, first_group_name, second_group_all_coeffs, second_group_name = _get_groups_distinct_coeffs(
        col_to_compare, filter_type, group1, group2, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
        num_alphas, query, sid, coeff_type=df_type)

    # Only keep necessary columns in exploded_df to reduce memory
    chosen_coeffs_val_cols = [col for col in exploded_df.columns if col.startswith("chosen_coeffs_val")]
    exploded_df = exploded_df[["full_elec_name", "time_index", "actual_chosen_coeffs", "group", f"is_in_{first_group_name}", f"is_in_{second_group_name}"] + chosen_coeffs_val_cols]

    print(f"starting calc for count")
    max_columns = ['chosen_coeff_count']
    mean_columns = []
    col_actions = {'chosen_coeff_count': ('actual_chosen_coeffs', 'count')}
    if "lasso" in df_type:
        col_actions['mean_val_lasso'] = ('chosen_coeffs_val_lasso', 'mean')
        col_actions['std_val_lasso'] = ('chosen_coeffs_val_lasso', 'std')
        mean_columns.append('mean_val_kfolds')
    if "corr" in df_type:
        col_actions['mean_val_corr'] = ('chosen_coeffs_val_corr', 'mean')
        col_actions['std_val_corr'] = ('chosen_coeffs_val_corr', 'std')
        mean_columns.append('mean_val_corr')
    if "data" in df_type:
        col_actions['mean_val_data'] = ('chosen_coeffs_val_data', 'mean')
        col_actions['std_val_data'] = ('chosen_coeffs_val_data', 'std')
        mean_columns.append('mean_val_data')

    stats = exploded_df.groupby(["group", 'actual_chosen_coeffs']).agg(**col_actions).reset_index()

    for col_name in mean_columns:
        stats[f'abs_{col_name}'] = stats[col_name].abs().astype(float)
        max_columns.append(f'abs_{col_name}')

    # top_coeffs = {}
    # for col in max_columns:
    #     top_coeffs[col] = (stats.groupby('group')
    #                        .apply(lambda x: x.nlargest(5, col)['actual_chosen_coeffs'].astype(int).tolist(),
    #                               include_groups=False)
    #                        .to_dict())
    top_coeffs = {}
    for group in stats['group'].unique():
        print(f"{group}")
        top_coeffs[group] = {}
        group_data = stats[stats['group'] == group]
        for col in max_columns:
            top_coeffs[group][col] = group_data.nlargest(top, col)['actual_chosen_coeffs'].astype(int).tolist()
            print(f"\t{col}: {top_coeffs[group][col]}")

    top_coeffs_df = pd.DataFrame(top_coeffs)
    top_coeffs_df.to_pickle(f'/scratch/gpfs/HASSON/tk6637/princeton/247-encoding/results/podcast/'
                            f'tk-podcast-{sid}-{model_info["model_full_name"]}-lag2k-25-all/'
                            f'tk-200ms-777-lay{model_info["layer"]}-con{model_info["context"]}-alphas_{min_alpha}_{max_alpha}_{num_alphas}-group1_{group1}-group2_{group2}-top_{top}-df_type_{df_type}-kfolds_threshold_{kfolds_threshold}-to_interp.pkl')

    print(top_coeffs_df)
    return top_coeffs_df


# def coeffs_statistics(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, col_to_compare, group1, group2,
#                       query="", kfolds_threshold=10, df_type="lasso&corr"):
#     # TODO: written very non efficient, fix!
#     non_zero_df = prepare_coeffs_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
#                                     num_alphas, sid, query=query, df_type=df_type)
#     exploded_non_zero_df = non_zero_df.query("num_of_chosen_coeffs > 0").explode("actual_chosen_coeffs").rename(columns={'actual_chosen_coeffs': 'coeff_indx'}).reset_index(drop=True)
#
#     all_coeffs_df = prepare_kfolds_coeffs_df(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, query=query).rename(columns={'coeffs': 'coeffs_mean_value'})
#     # all_coeffs_df['coeff_indx'] = all_coeffs_df['coeffs_mean_value'].apply(lambda x: list(range(len(x))))
#     all_coeffs_df['coeff_indx'] = all_coeffs_df['coeffs_mean_value'].apply(lambda x: [str(i) for i in range(len(x))])
#     exploded_all_coeffs_df = all_coeffs_df.explode(['coeff_indx', 'coeffs_mean_value']).reset_index(drop=True)
#
#     coeffs = exploded_non_zero_df.merge(exploded_all_coeffs_df, how='left')
#
#     exploded_df, first_group_all_coeffs, first_group_name, second_group_all_coeffs, second_group_name = _get_groups_distinct_coeffs(
#         col_to_compare, filter_type, group1, group2, kfolds_threshold, max_alpha, min_alpha, mode, model_info,
#         num_alphas, query, sid, df_type=df_type)
#     coeffs[f"is_in_{first_group_name}"] = coeffs["coeff_indx"].isin(first_group_all_coeffs)
#     coeffs[f"is_in_{second_group_name}"] = coeffs["coeff_indx"].isin(second_group_all_coeffs)
#
#     MEAN_COLORS = ('240', '98', '146')  # Light pink (similar to CORR_COLORS in example)
#     COUNT_COLORS = ('100', '181', '246')  # Light blue (similar to COEFF_COLORS in example)
#
#     MEAN_AXIS_COLOR = f"rgb({MEAN_COLORS[0]}, {MEAN_COLORS[1]}, {MEAN_COLORS[2]})"
#     COUNT_AXIS_COLOR = f"rgb({COUNT_COLORS[0]}, {COUNT_COLORS[1]}, {COUNT_COLORS[2]})"
#
#
#     def apply_filter(df, first_group_name, second_group_name, filter_type):
#         first_col = f"is_in_{first_group_name}"
#         second_col = f"is_in_{second_group_name}"
#
#         if filter_type == f"Only_in_{first_group_name}":
#             return df[df[first_col] & (~df[second_col])]
#         elif filter_type == f"Only_in_{second_group_name}":
#             return df[(~df[first_col]) & df[second_col]]
#         elif filter_type == "In_both":
#             return df[df[first_col] & df[second_col]]
#         else:
#             raise ValueError(f"Unknown filter type: {filter_type}")
#
#     group_types = [
#         f"Only_in_{first_group_name}",
#         f"Only_in_{second_group_name}",
#         "In_both"
#     ]
#
#     # Create subplots with specs for secondary y-axis
#     fig = make_subplots(
#         rows=2,
#         cols=3,
#         shared_xaxes=True,
#         vertical_spacing=0.15,
#         subplot_titles=[f"{group_type} - Mean Values and SD" for group_type in group_types]+ [f"{group_type} - Count by Index" for group_type in group_types],
#         row_heights=[0.7, 0.3]
#     )
#
#     all_strongest_coeffs = []
#     for group_idx, group_type in enumerate(group_types):
#         curr_coeffs = apply_filter(coeffs, first_group_name, second_group_name, group_type)
#
#         # Calculate statistics
#         stats = curr_coeffs.groupby('coeff_indx').agg(
#             mean_value=('coeffs_mean_value', 'mean'),
#             std_value=('coeffs_mean_value', 'std'),
#             count=('coeffs_mean_value', 'count')
#         ).reset_index()
#         stats['abs_mean_value'] = stats['mean_value'].abs()
#
#         # Sort by ID (or you can sort by another criterion)
#         # stats = stats.sort_values('coeff_indx')
#         stats = stats.sort_values('mean_value', ascending=False)
#         stats = stats.sort_values('count', ascending=False)
#
#         # Add standard deviation area
#         fig.add_trace(
#             go.Bar(
#                 x=stats['coeff_indx'],
#                 y=stats['mean_value'],
#                 error_y=dict(
#                     type='data',
#                     array=stats['std_value'],
#                     visible=True,
#                     thickness=0.5,  # Makes the error bar line thinner (default is 2)
#                     width=2, # Controls the width of the horizontal caps (default is 6)
#                 ),
#                 name="Mean ± SD",
#                 marker_color=MEAN_AXIS_COLOR,
#                 showlegend=True if group_idx == 0 else False,
#             ),
#             row=1, col=group_idx+1
#         )
#
#         # Add count bar chart
#         fig.add_trace(
#             go.Bar(
#                 x=stats['coeff_indx'],
#                 y=stats['count'],
#                 name="Count",
#                 marker_color=COUNT_AXIS_COLOR,
#                 showlegend=True if group_idx == 0 else False,
#             ),
#             row=2, col=group_idx+1
#         )
#
#         # # Add horizontal line at y=0 for mean values
#         # fig.add_shape(
#         #     type='line',
#         #     x0=stats['coeff_indx'].iloc[0],
#         #     x1=stats['coeff_indx'].iloc[-1],
#         #     y0=0, y1=0,
#         #     line=dict(color='black', width=1),
#         #     row=1, col=group+1
#         # )
#
#         # Display the top 10 as a formatted list
#         print(f"\n********** {group_type} **********")
#
#         top_5_mean = stats.sort_values('abs_mean_value', ascending=False).head(5)
#         print("Top 5 Indexes by Mean Value:")
#         for i, (_, row) in enumerate(top_5_mean.iterrows(), 1):
#             print(f"{i}. Index: {row['coeff_indx']} - Mean Value: {row['mean_value']:.3f} (Count: {row['count']})")
#         print()
#
#         top_5_count = stats.sort_values('count', ascending=False).head(5)
#         # Display the top 5 as a formatted list
#         print("Top 5 Indexes by Count:")
#         for i, (_, row) in enumerate(top_5_count.iterrows(), 1):
#             print(f"{i}. Index: {row['coeff_indx']} - Mean Value: {row['mean_value']:.3f} (Count: {row['count']})")
#
#         strongest_coeffs = set(top_5_mean + top_5_count)
#         all_strongest_coeffs.append(strongest_coeffs)
#
#     fig.update_yaxes(title_text="Value", row=1, col=1)
#     fig.update_yaxes(title_text="Count", row=2, col=1)
#     # Update layout
#     fig.update_layout(
#         height=700,
#         width=1200,
#         showlegend=True,
#         template='plotly_white',
#         title_text="Statistics by coeff indx",
#     )
#     fig.show()
#     a=0

def plot_encoding_num_coeffs_scatterplot_all_data(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                                                  x="rounded_encoding", kfolds_threshold=10, query="", df_type="lasso&corr"):


    kfolds_df = get_coeffs_df(sid, mode, model_info, filter_type, min_alpha, max_alpha, num_alphas, kfolds_threshold, df_type, query=query)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get unique categories from column x
    categories = kfolds_df[x].unique()

    # Starting y position for annotations
    annotation_y = 0.97
    category_data = kfolds_df
    color = '#A865C9'
    ax.scatter(
        category_data['encoding'],
        category_data['num_of_chosen_coeffs'],
        c=color,
        alpha=0.7,
        s=5
    )
    corr, p_value = pearsonr(category_data['encoding'], category_data['num_of_chosen_coeffs'])
    x_vals = np.array([category_data['encoding'].min(), category_data['encoding'].max()])
    z = np.polyfit(category_data['encoding'], category_data['num_of_chosen_coeffs'], 1)
    p = np.poly1d(z)
    ax.plot(x_vals, p(x_vals), color='black', linestyle='--', linewidth=2, alpha=0.8)

    # Add annotation for this category
    annotation_text = f"r={corr:.3f}, p={p_value:.4f}"
    ax.text(
        0.03, annotation_y,
        annotation_text,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor=color, linewidth=1.5),
        color=color
    )

    ax.set_xlabel('Encoding', fontsize=12)
    ax.set_ylabel('Number of Chosen Coefficients', fontsize=12)
    ax.set_title(f'Scatter Plot by {x}', fontsize=14)
    ax.legend(title=x, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def darken_color(color, factor=0.7):
    # Convert string/hex to RGB, multiply by factor (0.7 = 30% darker), and ensure valid range
    rgb = mc.to_rgb(color)
    return tuple([max(0, min(1, val * factor)) for val in rgb])

def plot_score_contour():
    """
    Create a scatter plot with background colored by score function.

    Parameters:
    - pivot_counts: DataFrame with columns [group1, group2, 'score', ...]
    - group1, group2: column names for the two groups
    """
    # Create a mesh grid for the background
    x_range = np.linspace(0, 1, 100)
    y_range = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate score for each point in the grid
    epsilon = 1e-9
    Z = np.abs(X - Y) / (X + Y + epsilon)  # This is just an example score function

    # Create figure
    fig = go.Figure()

    # Add contour/heatmap for background
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale='sunsetdark',  # or 'RdYlBu', 'Plasma', etc.
        showscale=True,
        colorbar=dict(
            title="Score",
            # titleside="right"
            title_font=dict(size=20),  # Colorbar title size
            tickfont=dict(size=18)  # Colorbar tick labels size
        ),
        contours=dict(
            showlabels=True,
            labelfont=dict(size=18, color='white')
        ),
        hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>Score: %{z:.3f}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f'Frequency Comparison: group1 vs group2',
        xaxis_title='$\\huge \\nu_{m, C_1}$',
        yaxis_title='$\\huge \\nu_{m, C_2}$',
        width=700,
        height=700,
        xaxis=dict(  # range=[0, 0.5],
            title_font=dict(size=50),  # Axis title size
            tickfont=dict(size=18)  # Axis labels size
        ),
        yaxis=dict(  # range=[0, 0.5],
            title_font=dict(size=50),  # Axis title size
            tickfont=dict(size=18)  # Axis labels size
        ),
        showlegend=False
    )

    # Top plot axes
    # fig.update_xaxes(range=[-0.05, 0.5])
    # fig.update_yaxes(range=[-0.05, 0.5])

    # Add diagonal reference line
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="red", width=2, dash="dash"),
    )

    fig.show()

def plot_encoding_num_coeffs_scatterplot(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, category_by="brain_area", kfolds_threshold=10, query="",
                                         df_type="lasso&corr"):
    kfolds_df = get_coeffs_df(sid, mode, model_info, filter_type, min_alpha, max_alpha, num_alphas, kfolds_threshold, df_type, query=query)


    # Statistical test:
    # MODEL 1: Interaction (Testing if slopes differ - Your original idea)
    # The '*' operator automatically includes main effects AND the interaction
    model_interaction = smf.ols(f"num_of_chosen_coeffs ~ encoding * C({category_by})", data=kfolds_df).fit()
    print(model_interaction.summary())
    # Look at the row 'encoding:C(x)'. If P < 0.05, the slopes are significantly different.

    print("--- Model 1: Main Effects (Parallel Slopes) ---")
    print(f"Formula: num_of_chosen_coeffs ~ C({category_by}) + encoding")
    # Note: Added maxiter and method to handle convergence warnings common in high-collinearity data
    model_main = smf.negativebinomial(
        f"num_of_chosen_coeffs ~ C({category_by}) + encoding",
        data=kfolds_df
    ).fit(maxiter=1000)

    print(model_main.summary())

    # ==========================================
    # 4. STATISTICAL MODELING (Interaction Effects)
    # ==========================================
    print("\n--- Model 2: Interaction Effects (Different Slopes) ---")
    print(f"Formula: num_of_chosen_coeffs ~ C({category_by}) * encoding")
    print("Testing if the relationship between encoding and count differs by group.")

    model_interaction = smf.negativebinomial(
        f"num_of_chosen_coeffs ~ C({category_by}) * encoding",
        data=kfolds_df
    ).fit(maxiter=1000)

    print(model_interaction.summary())

    # ==========================================
    # 5. INTERPRETING THE INTERACTION
    # ==========================================
    print("\n--- Interaction Interpretation ---")
    interaction_pvalues = model_interaction.pvalues[model_interaction.pvalues.index.str.contains(':')]
    significant_interactions = interaction_pvalues[interaction_pvalues < 0.05]

    if len(significant_interactions) > 0:
        print("SIGNIFICANT INTERACTIONS FOUND:")
        print(significant_interactions)
        print("\nConclusion: The slopes ARE significantly different. The effect of encoding depends on the group.")
        print("You should report the Interaction Model results.")
    else:
        print("NO significant interactions found (P > 0.05).")
        print("Conclusion: The slopes are statistically parallel (differences are likely noise).")
        print("The difference you saw (0.8 vs 0.755) is not significant. Stick to the Main Effects model.")

    # ==========================================
    # 6. EFFECT SIZE & DIRECTION (From Main Model)
    # ==========================================
    print("\n--- Effect Sizes (Incidence Rate Ratios - Main Model) ---")
    params = model_main.params
    conf = model_main.conf_int()
    conf['IRR'] = params
    conf.columns = ['Lower CI', 'Upper CI', 'IRR']
    conf = np.exp(conf)

    print(conf[['IRR', 'Lower CI', 'Upper CI']])

    # ==========================================
    # 7. PAIRWISE COMPARISONS
    # ==========================================
    print("\n--- Pairwise Comparisons (Marginal Means - Main Model) ---")
    pairwise = model_main.t_test_pairwise(f"C({category_by})")
    results_df = pairwise.result_frame

    # Check which p-value column exists
    p_col = 'P>|z|' if 'P>|z|' in results_df.columns else 'P>|t|'

    results_df['Significance'] = results_df[p_col].apply(lambda p: 'Significant' if p < 0.05 else 'Not Sig')
    results_df['Rate_Ratio_Diff'] = np.exp(results_df['coef'])

    display_cols = ['coef', 'Rate_Ratio_Diff', p_col, 'Significance']
    print(results_df[display_cols])


    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get unique categories from column x
    categories = kfolds_df[category_by].unique()

    # Starting y position for annotations
    annotation_y = 0.98

    # Plot each category with its corresponding color
    for category in categories:
        if pd.isna(category):
            mask = kfolds_df[category_by].isna()
        # Filter data for this category
        else:
            mask = kfolds_df[category_by] == category
        category_data = kfolds_df[mask]

        # Get color from the palette
        color = COLOR_PALETTE.get(category, 'gray')  # Default to gray if category not in palette

        # Plot scatter for this category
        ax.scatter(
            category_data['encoding'],
            category_data['num_of_chosen_coeffs'],
            c=color,
            label=category,
            alpha=0.5,
            s=5,
            zorder=1,
        )

        # Calculate correlation
        corr, p_value = pearsonr(category_data['encoding'], category_data['num_of_chosen_coeffs'])

        # Plot correlation line
        x_vals = np.array([category_data['encoding'].min(), category_data['encoding'].max()])
        z = np.polyfit(category_data['encoding'], category_data['num_of_chosen_coeffs'], 1)
        p = np.poly1d(z)
        ax.plot(x_vals, p(x_vals), color=darken_color(color, factor=0.75), linestyle='--', linewidth=2, alpha=1.0, zorder=2)

        # Add annotation for this category
        annotation_text = f"{category}: r={corr:.3f}, p={p_value:.4f}"
        ax.text(
            0.02, annotation_y,
            annotation_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor=color, linewidth=1.5),
            color=color
        )
        annotation_y -= 0.06  # Move down for next annotation

    # Add labels and legend
    ax.set_xlabel('Encoding', fontsize=12)
    ax.set_ylabel('Number of Chosen Coefficients', fontsize=12)
    ax.set_title(f'Scatter Plot by {category_by} for {model_info["model_short_name"]}', fontsize=14)
    ax.legend(title=category_by, bbox_to_anchor=(1.05, 1), loc='upper left')
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return

def pca_coeffs(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
               query:str="", plot_3d:bool=False, plot_continues:bool=False, df_type:str="lasso"):
    """
    Does PCA analysis for the transformation vectors themselves (the trained linear vectors).
    """
    coeff_df = prepare_coeffs_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info, num_alphas, sid, query=query, df_type=df_type)
    # coeff_df = prepare_kfolds_coeffs_df(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, query)
    vectors_array = np.vstack(coeff_df['all_coeffs_val'].values)

    mask = ~np.isnan(vectors_array).any(axis=0) # Needed for corr
    vectors_array = vectors_array[:, mask]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(vectors_array)


    # To determine how many components explain e.g. 95% of variance
    print("starting PCA")
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
    discrete_categories_names = ["brain_area", "time_bin", "time", "time_bin_and_brain_area", "rounded_encoding"]
    for category_name in discrete_categories_names:
        # 2D:
        pca_df2['Category'] = coeff_df[category_name]  # Add the category column
        # 4. Plot with colors by category using Plotly
        fig = px.scatter(
            pca_df2,
            x='PC1',
            y='PC2',
            color='Category',
            title=f'{category_name}, model: {model_name}, df_type: {df_type}<br>query: {query}',
            labels={
                'PC1': f'Principal Component 1 ({pca2.explained_variance_ratio_[0]:.2%} variance)',
                'PC2': f'Principal Component 2 ({pca2.explained_variance_ratio_[1]:.2%} variance)',
                'Category': category_name
            },
            opacity=0.7
        )

        fig.update_layout(
            # width=1000,
            # height=800,
            template='plotly_white',
            legend_title_text=category_name
        )

        save_prefix = f'/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/results/maybe_paper/pca/type_{df_type}/by_{category_name}/model_{model_name}_pca'
        Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(f'{save_prefix}.html')
        fig.write_image(f'{save_prefix}.pdf')
        fig.write_image(f'{save_prefix}.png')

        if SHOW_FIGS:
            fig.show()
        if plot_3d:
            # 3D:
            pca_df3['Category'] = coeff_df[category_name]  # Add your categories
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
            if SHOW_FIGS:
                fig.show()
            # To save the interactive plot as HTML
            # fig.write_html("pca_3d_interactive.html")

    if plot_continues:
        continues_categories_names = ["time", "encoding"]
        for category_name in continues_categories_names:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_df2['PC1'], pca_df2['PC2'],
                                  c=coeff_df[category_name],  # Numeric values for color
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

def plot_violin_of_scatter_score():
    GEMMA_COLOR = (86, 152, 198)
    SAE_COLOR = (181, 124, 210)
    OPACITY = 0.5

    corr_gemma9b_brain_area_frequency = pd.read_pickle(
        "/scratch/gpfs/HASSON/tk6637/princeton/temp_data/paper/violin_plot/corr_gemma9b_brain_area_frequency_.pkl")
    corr_scope_brain_area_frequency = pd.read_pickle(
        "/scratch/gpfs/HASSON/tk6637/princeton/temp_data/paper/violin_plot/corr_gemma-scope9b_brain_area_frequency_.pkl")
    lasso_gemma9b_brain_area_frequency = pd.read_pickle(
        "/scratch/gpfs/HASSON/tk6637/princeton/temp_data/paper/violin_plot/lasso_gemma9b_brain_area_frequency_.pkl")
    lasso_scope_brain_area_frequency = pd.read_pickle(
        "/scratch/gpfs/HASSON/tk6637/princeton/temp_data/paper/violin_plot/lasso_gemma-scope9b_brain_area_frequency_.pkl")

    corr_gemma9b_time_bin_frequency = pd.read_pickle(
        "/scratch/gpfs/HASSON/tk6637/princeton/temp_data/paper/violin_plot/corr_gemma9b_time_bin_frequency_.pkl")
    corr_scope_time_bin_frequency = pd.read_pickle(
        "/scratch/gpfs/HASSON/tk6637/princeton/temp_data/paper/violin_plot/corr_gemma-scope9b_time_bin_frequency_.pkl")
    lasso_gemma9b_time_bin_frequency = pd.read_pickle(
        "/scratch/gpfs/HASSON/tk6637/princeton/temp_data/paper/violin_plot/lasso_gemma9b_time_bin_frequency_.pkl")
    lasso_scope_time_bin_frequency = pd.read_pickle(
        "/scratch/gpfs/HASSON/tk6637/princeton/temp_data/paper/violin_plot/lasso_gemma-scope9b_time_bin_frequency_.pkl")

    area_dfs = [
        (corr_gemma9b_brain_area_frequency, 'corr_gemma9b_brain_area_frequency'),
        (corr_scope_brain_area_frequency, 'corr_scope_brain_area_frequency'),
        (lasso_gemma9b_brain_area_frequency, 'lasso_gemma9b_brain_area_frequency'),
        (lasso_scope_brain_area_frequency, 'lasso_scope_brain_area_frequency')
    ]

    time_dfs = [
        (corr_gemma9b_time_bin_frequency, 'corr_gemma9b_time_bin_frequency'),
        (corr_scope_time_bin_frequency, 'corr_scope_time_bin_frequency'),
        (lasso_gemma9b_time_bin_frequency, 'lasso_gemma9b_time_bin_frequency'),
        (lasso_scope_time_bin_frequency, 'lasso_scope_time_bin_frequency')
    ]

    line_colors = [
        f'rgb({GEMMA_COLOR[0]},{GEMMA_COLOR[1]},{GEMMA_COLOR[2]})',  # 1
        f'rgb({SAE_COLOR[0]},{SAE_COLOR[1]},{SAE_COLOR[2]})',  # 2
        f'rgb({GEMMA_COLOR[0]},{GEMMA_COLOR[1]},{GEMMA_COLOR[2]})',  # 3
        f'rgb({SAE_COLOR[0]},{SAE_COLOR[1]},{SAE_COLOR[2]})'  # 4
    ]
    fill_colors = [
        f'rgba({GEMMA_COLOR[0]},{GEMMA_COLOR[1]},{GEMMA_COLOR[2]},{OPACITY})',
        f'rgba({SAE_COLOR[0]},{SAE_COLOR[1]},{SAE_COLOR[2]},{OPACITY})',
        f'rgba({GEMMA_COLOR[0]},{GEMMA_COLOR[1]},{GEMMA_COLOR[2]},{OPACITY})',
        f'rgba({SAE_COLOR[0]},{SAE_COLOR[1]},{SAE_COLOR[2]},{OPACITY})'
    ]

    fig = go.Figure()
    for i, (df, name) in enumerate(area_dfs):
        fig.add_trace(go.Violin(
            y=df['score'],
            name=name,
            box_visible=True,  # Box plot inside
            fillcolor=fill_colors[i],
            line=dict(color=line_colors[i]),
            box=dict(line=dict(color=line_colors[i]))
            # meanline_visible=True  # Shows mean line
            # points='outliers',          # Show outlier points (or 'all' for all points)
            # fillcolor=color,
            # opacity=0.7,
            # line=dict(color=color, width=1.5)
        ))

    fig.update_layout(
        title='Score Distribution for brain_area',
        height=700,
        width=1150,
        yaxis=dict(
            title=dict(
                text='Distance',
                font=dict(size=20)
            ),
            tickmode='array',
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            tickfont=dict(size=18)  # Adjust tick label size
        ),
        template='simple_white',
    )
    fig.show()

    fig = go.Figure()
    for i, (df, name) in enumerate(time_dfs):
        fig.add_trace(go.Violin(
            y=df['score'],
            name=name,
            box_visible=True,  # Box plot inside
            fillcolor=fill_colors[i],
            line=dict(color=line_colors[i]),
            box=dict(line=dict(color=line_colors[i]))
            # meanline_visible=True  # Shows mean line
            # points='outliers',          # Show outlier points (or 'all' for all points)
            # fillcolor=color,
            # opacity=0.7,
            # line=dict(color=color, width=1.5)
        ))

    fig.update_layout(
        title='Score Distribution for time_bin',
        height=700,
        width=1150,
        yaxis=dict(
            title=dict(
                text='Distance',
                font=dict(size=20)
            ),
            tickmode='array',
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            tickfont=dict(size=18)  # Adjust tick label size
        ),
        template='simple_white',
    )
    fig.show()


    def calculate_bootstrap_stats(dfs_list, list_name="Group", n_resamples=9999):
        results = []

        for df_obj, name in dfs_list:
            # 1. Safe Data Extraction
            if isinstance(df_obj, pd.DataFrame):
                data = df_obj['score'].dropna().to_numpy()
            elif isinstance(df_obj, pd.Series):
                data = df_obj.dropna().to_numpy()
            else:
                data = np.array(df_obj)

            n = len(data)

            if n < 2:
                results.append({
                    "Dataset": name, "Mean": np.nan,
                    "Boot_CI_Lower": np.nan, "Boot_CI_Upper": np.nan, "T_p_value": np.nan
                })
                continue

            # 2. Bootstrap 95% Confidence Interval (BCa method)
            res = stats.bootstrap(
                (data,),
                np.mean,
                confidence_level=0.95,
                n_resamples=n_resamples,
                method='BCa'  # Bias-Corrected and accelerated
            )

            # 3. Standard T-Test P-Value (for reference)
            t_stat, p_val = stats.ttest_1samp(data, 0, alternative='greater')

            results.append({
                "Dataset": name,
                "N": n,
                "Mean": np.mean(data),
                "SD": data.std(),
                "Boot_CI_Lower": res.confidence_interval.low,
                "Boot_CI_Upper": res.confidence_interval.high,
                "T_statistic": t_stat,
                "T_p_value": p_val
            })

        return pd.DataFrame(results)

    # --- Execute & Print ---

    # Calculate
    stats_area = calculate_bootstrap_stats(area_dfs, "Brain Area")
    stats_time = calculate_bootstrap_stats(time_dfs, "Time Bin")

    # Print Brain Area Stats
    print("\n--- Brain Area Statistics ---")
    print(stats_area.to_string(float_format=lambda x: "{:.4f}".format(x)))

    # Print Time Bin Stats
    print("\n--- Time Bin Statistics ---")
    print(stats_time.to_string(float_format=lambda x: "{:.4f}".format(x)))


    ###### Comparing between groups
    # Define colors
    GEMMA_COLOR = (86 / 255, 152 / 255, 198 / 255)
    SAE_COLOR = (181 / 255, 124 / 255, 210 / 255)

    brain_area_data = []
    for df, name in [
        (corr_gemma9b_brain_area_frequency, 'Corr-Gemma'),
        (corr_scope_brain_area_frequency, 'Corr-SAE'),
        (lasso_gemma9b_brain_area_frequency, 'Lasso-Gemma'),
        (lasso_scope_brain_area_frequency, 'Lasso-SAE')
    ]:
        temp_df = df.copy()
        temp_df['group'] = name
        brain_area_data.append(temp_df)

    brain_area_combined = pd.concat(brain_area_data, ignore_index=True)

    # Prepare data for time_bin plot
    time_bin_data = []
    for df, name in [
        (corr_gemma9b_time_bin_frequency, 'Corr-Gemma'),
        (corr_scope_time_bin_frequency, 'Corr-SAE'),
        (lasso_gemma9b_time_bin_frequency, 'Lasso-Gemma'),
        (lasso_scope_time_bin_frequency, 'Lasso-SAE')
    ]:
        temp_df = df.copy()
        temp_df['group'] = name
        time_bin_data.append(temp_df)

    time_bin_combined = pd.concat(time_bin_data, ignore_index=True)

    # Define color palette
    palette = {
        'Corr-Gemma': GEMMA_COLOR,
        'Corr-SAE': SAE_COLOR,
        'Lasso-Gemma': GEMMA_COLOR,
        'Lasso-SAE': SAE_COLOR
    }

    # ============================================================================
    # CONFIGURE YOUR DIRECTIONAL HYPOTHESES HERE
    # ============================================================================
    # For each pair, specify:
    # - 'greater': Test if first group > second group (one-tailed)
    # - 'less': Test if first group < second group (one-tailed)
    # - 'two-sided': Test if groups are different (two-tailed, default)
    #
    # Example: If you expect Corr-Gemma to have HIGHER scores than Corr-SAE:
    #   ('Corr-Gemma', 'Corr-SAE'): 'greater'
    # ============================================================================

    hypothesis_config = {
        ('Corr-SAE', 'Corr-Gemma'): 'greater',  # Change to 'greater' or 'less' if you have a hypothesis
        ('Lasso-Gemma', 'Corr-Gemma'): 'greater',
        ('Lasso-Gemma', 'Corr-SAE'): 'greater',
        ('Lasso-SAE', 'Lasso-Gemma'): 'greater',
        ('Lasso-SAE', 'Corr-SAE'): 'greater',
    }

    # ============================================================================

    pairs = list(hypothesis_config.keys())

    def welch_t_test_with_alternative(group1, group2, alternative='two-sided'):
        """
        Perform Welch's t-test with specified alternative hypothesis

        Parameters:
        -----------
        group1, group2 : array-like
            Data for the two groups
        alternative : str
            'two-sided': group1 != group2
            'greater': group1 > group2 (one-tailed)
            'less': group1 < group2 (one-tailed)

        Returns:
        --------
        t_stat, p_value
        """
        t_stat, p_value_two_sided = stats.ttest_ind(group1, group2, equal_var=False)

        if alternative == 'two-sided':
            return t_stat, p_value_two_sided
        elif alternative == 'greater':
            # One-tailed: is group1 > group2?
            # If t_stat > 0 (group1 mean > group2 mean), p = p_two_sided / 2
            # If t_stat < 0, p = 1 - p_two_sided / 2
            if t_stat > 0:
                return t_stat, p_value_two_sided / 2
            else:
                return t_stat, 1 - p_value_two_sided / 2
        elif alternative == 'less':
            # One-tailed: is group1 < group2?
            # If t_stat < 0 (group1 mean < group2 mean), p = p_two_sided / 2
            # If t_stat > 0, p = 1 - p_two_sided / 2
            if t_stat < 0:
                return t_stat, p_value_two_sided / 2
            else:
                return t_stat, 1 - p_value_two_sided / 2
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Custom test function for statannotations
    def custom_welch_test(group1, group2, **kwargs):
        """Wrapper for statannotations that uses configured hypothesis direction"""
        # This is a simplified version - statannotations doesn't fully support one-tailed
        # So we'll use two-sided for the plot and calculate one-tailed manually for stats
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        return None, p_value

    # Create figure for brain_area
    fig, ax = plt.subplots(figsize=(11.5, 7))
    sns.boxplot(data=brain_area_combined, x='group', y='score',
                palette=palette, ax=ax, width=0.6)

    # Add statistical annotations
    annotator = Annotator(ax, pairs, data=brain_area_combined, x='group', y='score')
    annotator.configure(test='t-test_welch', text_format='star',
                        comparisons_correction='bonferroni', loc='inside')
    annotator.apply_and_annotate()

    ax.set_title('Score Distribution for Brain Area', fontsize=16, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=20)
    ax.set_xlabel('')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=14, rotation=45)
    plt.tight_layout()
    # plt.savefig('/mnt/user-data/outputs/brain_area_boxplot_onetailed.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create figure for time_bin
    fig, ax = plt.subplots(figsize=(11.5, 7))
    sns.boxplot(data=time_bin_combined, x='group', y='score',
                palette=palette, ax=ax, width=0.6)

    # Add statistical annotations
    annotator = Annotator(ax, pairs, data=time_bin_combined, x='group', y='score')
    annotator.configure(test='t-test_welch', text_format='star',
                        comparisons_correction='bonferroni', loc='inside')
    annotator.apply_and_annotate()

    ax.set_title('Score Distribution for Time Bin', fontsize=16, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=20)
    ax.set_xlabel('')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=14, rotation=45)
    plt.tight_layout()
    # plt.savefig('/mnt/user-data/outputs/time_bin_boxplot_onetailed.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed statistics WITH DIRECTIONAL TESTS
    print("=" * 100)
    print("BRAIN AREA - Welch's t-test Results with Directional Hypotheses (Bonferroni corrected)")
    print("=" * 100)
    for pair in pairs:
        group1_data = brain_area_combined[brain_area_combined['group'] == pair[0]]['score']
        group2_data = brain_area_combined[brain_area_combined['group'] == pair[1]]['score']

        alternative = hypothesis_config[pair]
        t_stat, p_val = welch_t_test_with_alternative(group1_data, group2_data, alternative)
        p_val_corrected = min(p_val * len(pairs), 1.0)  # Bonferroni correction

        print(f"\n{pair[0]} vs {pair[1]} (Hypothesis: {alternative}):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value (uncorrected): {p_val:.6f}")
        print(f"  p-value (Bonferroni): {p_val_corrected:.6f}")
        if p_val_corrected < 0.001:
            sig = '***'
        elif p_val_corrected < 0.01:
            sig = '**'
        elif p_val_corrected < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        print(f"  Significance: {sig}")
        print(f"  Mean {pair[0]}: {group1_data.mean():.4f} (SD: {group1_data.std():.4f})")
        print(f"  Mean {pair[1]}: {group2_data.mean():.4f} (SD: {group2_data.std():.4f})")
        print(f"  Mean difference: {group1_data.mean() - group2_data.mean():.4f}")

        if alternative == 'greater':
            if p_val_corrected < 0.05:
                print(f"  → {pair[0]} is significantly GREATER than {pair[1]}")
            else:
                print(f"  → No evidence that {pair[0]} is greater than {pair[1]}")
        elif alternative == 'less':
            if p_val_corrected < 0.05:
                print(f"  → {pair[0]} is significantly LESS than {pair[1]}")
            else:
                print(f"  → No evidence that {pair[0]} is less than {pair[1]}")

    print("\n" + "=" * 100)
    print("TIME BIN - Welch's t-test Results with Directional Hypotheses (Bonferroni corrected)")
    print("=" * 100)
    for pair in pairs:
        group1_data = time_bin_combined[time_bin_combined['group'] == pair[0]]['score']
        group2_data = time_bin_combined[time_bin_combined['group'] == pair[1]]['score']

        alternative = hypothesis_config[pair]
        t_stat, p_val = welch_t_test_with_alternative(group1_data, group2_data, alternative)
        p_val_corrected = min(p_val * len(pairs), 1.0)  # Bonferroni correction

        print(f"\n{pair[0]} vs {pair[1]} (Hypothesis: {alternative}):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value (uncorrected): {p_val:.6f}")
        print(f"  p-value (Bonferroni): {p_val_corrected:.6f}")
        if p_val_corrected < 0.001:
            sig = '***'
        elif p_val_corrected < 0.01:
            sig = '**'
        elif p_val_corrected < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        print(f"  Significance: {sig}")
        print(f"  Mean {pair[0]}: {group1_data.mean():.4f} (SD: {group1_data.std():.4f})")
        print(f"  Mean {pair[1]}: {group2_data.mean():.4f} (SD: {group2_data.std():.4f})")
        print(f"  Mean difference: {group1_data.mean() - group2_data.mean():.4f}")

        if alternative == 'greater':
            if p_val_corrected < 0.05:
                print(f"  → {pair[0]} is significantly GREATER than {pair[1]}")
            else:
                print(f"  → No evidence that {pair[0]} is greater than {pair[1]}")
        elif alternative == 'less':
            if p_val_corrected < 0.05:
                print(f"  → {pair[0]} is significantly LESS than {pair[1]}")
            else:
                print(f"  → No evidence that {pair[0]} is less than {pair[1]}")

    print("\n" + "=" * 100)
    print("\nHYPOTHESIS CONFIGURATION:")
    print("=" * 100)
    for pair, alt in hypothesis_config.items():
        print(f"{pair[0]} vs {pair[1]}: {alt}")
    print("\nNOTE: One-tailed tests have more statistical power to detect effects in the expected direction,")
    print("but they cannot detect effects in the opposite direction. Use them only when you have a")
    print("strong a priori hypothesis about the direction of the effect.")
    print("=" * 100)


if __name__ == '__main__':
    print("Starting overlapping coeffs")
    models_info = {"gemma2b": {"layer": 13, "context": 32, "embedding_size": 2304,
                             "model_short_name": "gemma2b", "model_full_name": "gemma-2-2b"},
                   "gemma9b": {"layer": 21, "context": 32, "embedding_size": 3584,
                               "model_short_name": "gemma9b", "model_full_name": "gemma-2-9b"},
                   "gpt2": {"layer": 24, "context": 32, "embedding_size": 1600,
                             "model_short_name": "gpt2", "model_full_name": "gpt2-xl"},
                   "mistral7b": {"layer": 16, "context": 32, "embedding_size": 4096,
                            "model_short_name": "mistral7b", "model_full_name": "Mistral-7B-v0.3"},
                   "llama8B": {"layer": 16, "context": 32, "embedding_size": 4096,
                             "model_short_name": "llama8B", "model_full_name": "Meta-Llama-3.1-8B"},
                   "glove": {"layer": 0, "context": 1, "embedding_size": 50,
                             "model_short_name": "glove", "model_full_name": "glove50"},
                   "gemma-scope2b": {"layer": 13, "context": 32, "embedding_size": 16384,
                                   "model_short_name": "gemma-scope2b",
                                   "model_full_name": "gemma-scope-2b-pt-res-canonical"},
                   "gemma-scope9b": {"layer": 21, "context": 32, "embedding_size": 16384,
                                     "model_short_name": "gemma-scope9b",
                                     "model_full_name": "gemma-scope-9b-pt-res-canonical"},
                   "gemma-scope9b-mlp": {"layer": 21, "context": 32, "embedding_size": 16384,
                                     "model_short_name": "gemma-scope-mlp",
                                     "model_full_name": "gemma-scope-9b-pt-mlp-canonical"},
                   "symbolic-lang": {"layer": 0, "context": 1, #"embedding_size": 16384,
                                         "model_short_name": "symbolic-lang",
                                         "model_full_name": "symbolic-lang"},
                   "arb": {"model_full_name": "gemma-2-9b", "layer": 21, "context": 32, "embedding_size": 3584,
                           "model_short_name": "gemma9b", "dash": "solid",
                           "colors": [('255', '158', '191'), ('158', '210', '255')]},
                   # [('200','230','201'),('129','199','132'),('76','175','80'),('56','142','60'),('27','94','32')]},
                   # [('200','230','201'),('129','199','132'),('76','175','80'),('56','142','60'),('27','94','32')]},
                   }
    time_points = np.linspace(-2, 2, 161)
    time_to_index = {t: i for i, t in enumerate(np.round(time_points, 3))}

    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", type=int, default=777)
    parser.add_argument("--model-name", type=str, default="gemma9b", choices=models_info.keys())
    parser.add_argument("--df-type",nargs='+', default=["corr"], choices=["lasso&corr", "lasso", "corr" ,"ridge", "pca", "corr_pca"],help="One or more dataframe types")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--col-to-compare", nargs='+',  default=["brain_area"], choices=["brain_area", "rounded_encoding", "time_bin", "time_bin_and_brain_area"], help="One or more columns to compare")
    parser.add_argument("--run-venn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-pca", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-sim", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-violin", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--control", action=argparse.BooleanOptionalAction, default=False,  help="Whether to run control analyses in the venn diagrams")
    parser.add_argument("--filter-type", type=str, default="160", help="Filter type for the data", choices=["160", "160IFG", "160STG"])
    args = parser.parse_args()

    run_venn = args.run_venn
    run_pca = args.run_pca
    run_sim = args.run_sim
    run_violin = args.run_violin
    control = args.control
    df_types = args.df_type #"corr"  # lasso&corr, kfolds, corr
    cols_to_compare = args.col_to_compare #"time_bin"#"time_bin"#"brain_area"

    sid = args.sid
    filter_type = args.filter_type
    model_name = args.model_name #'gemma-scope9b'
    emb_mod = None#"arb"#"arb"  # "shift-emb" / "arb"
    min_alpha = -2
    max_alpha = 10
    num_alphas = 100
    mode = 'comp'
    kfolds_threshold = 10


    # # Create BoxPlots (num coeffs)
    # x="rounded_encoding"#"rounded_encoding" #rounded_encoding, brain_area, time_bin. time_bin_and_brain_area
    # y="num_of_chosen_coeffs" #num_of_chosen_coeffs, encoding
    # hue = "brain_area"#"brain_area" #None, rounded_encoding, brain_area, time_bin. time_bin_and_brain_area
    query = ""#"" #args.query #TIME_QUERY, AREA_QUERY, AREA_AND_TIME_QUERY
    # df_type = args.df_type
    sig_test = "Mann-Whitney"#"t-test_welch"
    # violin_annot = True

    top = 10

    for df_type in df_types:
        for col_to_compare in cols_to_compare:
            print("\n\n")
            print("*"*50)
            print(f"Generating plot for {df_type} and {col_to_compare}")

            # plot_encoding_num_coeffs_scatterplot(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
            #                         category_by=hue, kfolds_threshold=kfolds_threshold, query=query, df_type=df_type)
            # plot_x_vs_num_of_coeffs(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
            #                         x=x, hue=hue, kfolds_threshold=kfolds_threshold, query=query, df_type=df_type, sig_test=sig_test)
            # run_all_boxplots(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode, kfolds_threshold, df_type, sig_test)
            # amount_distinct_coeffs(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
            #                        x=x, hue=hue, kfolds_threshold=kfolds_threshold, query=query)

            # # Venn Diagrams (which coeffs)
            # query = "rounded_encoding==0.4"  # TIME_QUERY, AREA_QUERY
            # col_to_compare = args.col_to_compare
            # df_type = "corr" #lasso&corr, kfolds, corr
            # query = ""#""#f"(rounded_encoding=='{group1}')|(rounded_encoding=='{group2}')" # TIME_QUERY, AREA_QUERY

            if col_to_compare == "time_bin":
                group1 = "-0.4≤x<0"#"STG" #"-0.4≤x<0"
                group2 = "0≤x<0.4"#"IFG"#"0≤x<0.4"
            elif col_to_compare == "brain_area":
                if control:
                    group1 = "IFG" # or STG
                    group2 = "precentral" #or postcg
                else:
                    group1 = "STG"
                    group2 = "IFG"
            else:
                raise ValueError("col_to_compare must be either 'time_bin' or 'brain_area'")
            # for encoding in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']:
            #     query = f"rounded_encoding=={encoding}"
            #     coeffs_venn(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
            #                       col_to_compare=col_to_compare, group1=group1, group2=group2, kfolds_threshold=kfolds_threshold, query=query)
            if run_venn:
                coeffs_venn(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
                            col_to_compare=col_to_compare, group1=group1, group2=group2, kfolds_threshold=kfolds_threshold, emb_mod=emb_mod, query=query, df_type=df_type)

            # coeffs_values(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode, col_to_compare=col_to_compare, group1=group1, group2=group2, query=query, kfolds_threshold=kfolds_threshold, df_type=df_type,  top=top)

            # coeffs_statistics(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode, col_to_compare=col_to_compare, group1=group1, group2=group2, query=query, kfolds_threshold=kfolds_threshold, df_type=df_type)
        if run_pca:
            pca_coeffs(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode, query=query, df_type=df_type)

        if run_sim:
            plot_similarity_matrix(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode, query=query, df_type=df_type)
            # overlap_by_area(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode, interest_areas, kfolds_threshold, start_time_idx, end_time_idx)

        if run_violin:
            plot_violin_of_scatter_score()
    print("Done with overlapping coeffs")
