import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from upsetplot import from_contents, UpSet
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact
import requests
from typing import Optional, Dict, Any
import json
import time
from statannot import add_stat_annotation
from statsmodels.stats.multitest import multipletests

from tfsplt_utils import load_electrode_names_and_locations, get_non_zero_coeffs_old, get_non_zero_coeffs

BASE_NEURONPEDIA_URL = "https://www.neuronpedia.org/api/feature"

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


def plot_x_vs_num_of_coeffs(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, x="rounded_encoding", kfolds_threshold=10, query=""):
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
    kfolds_df = prepare_kfolds_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info, num_alphas, sid)

    _plot_x_vs_num_coeffs(kfolds_df, max_alpha, min_alpha, model_info, num_alphas, x=x)
    _plot_x_vs_num_coeffs(kfolds_df, max_alpha, min_alpha, model_info, num_alphas, x=x, hue="princeton_class")
    _plot_x_vs_num_coeffs(kfolds_df, max_alpha, min_alpha, model_info, num_alphas, x=x, hue="time_bin")
    if query:
        kfolds_df_filtered = kfolds_df.query(query)
        # kfolds_df_filtered = kfolds_df[(kfolds_df["princeton_class"] == "STG") | (kfolds_df["princeton_class"] == "IFG")]
        _plot_x_vs_num_coeffs(kfolds_df_filtered, max_alpha, min_alpha, model_info, num_alphas, x=x, hue="princeton_class")

    print("Done!")


def prepare_kfolds_df(filter_type, kfolds_threshold: int, max_alpha, min_alpha, mode, model_info, num_alphas, sid) -> DataFrame:
    all_data_df, kfolds_df = get_non_zero_coeffs(sid, filter_type, model_info["model_full_name"], model_info["layer"],
                                                 model_info["context"], min_alpha, max_alpha, num_alphas,
                                                 mode, kfolds_threshold)

    kfolds_df['rounded_encoding'] = kfolds_df['encoding'].round(1)
    kfolds_df.loc[kfolds_df['rounded_encoding'] == -0.0, 'rounded_encoding'] = 0.0

    bins = [-np.inf, -0.6, -0.3, 0, 0.3, 0.6, np.inf]
    labels = ['x<-0.6', '-0.6≤x<-0.3', '-0.3≤x<0', '0≤x<0.3', '0≤x<0.6', '0.6≤x']
    kfolds_df['time_bin'] = pd.cut(kfolds_df['time'], bins=bins, labels=labels, right=False)

    return kfolds_df

def _plot_x_vs_num_coeffs(df: DataFrame, max_alpha, min_alpha, model_info, num_alphas, x="rounded_encoding", y="num_of_coeffs", hue=None, filter_name=None, ):
    if df[x].unique().size > 20: # Long x axis
        fig = plt.figure(figsize=(15, 15))
    else:
        fig = plt.figure(figsize=(7, 15))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])  # 3 rows, 1 column with the top plot larger

    ax1 = plt.subplot(gs[0])
    if hue:
        sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax1)
    else:
        sns.boxplot(data=df, x=x, y=y, ax=ax1)

    ax1.set_title("Number of coefficients used for each encoding", fontsize=14)
    # ax1.set_xticklabels([])
    customize_time_xaxis(ax1, x)

    ax2 = plt.subplot(gs[1])
    if hue:
        sns.pointplot(data=df, x=x, y=y, errorbar="sd", hue=hue, ax=ax2)
    else:
        sns.pointplot(data=df, x=x, y=y, errorbar="sd", ax=ax2)
    ax2.set_title("Mean number of coefficients used for each encoding", fontsize=14)
    ax2.set_ylabel("Mean number of coefficients", fontsize=12)
    # ax2.set_xticklabels([])
    customize_time_xaxis(ax2, x)

    ax3 = plt.subplot(gs[2])
    if hue:
        sns.countplot(data=df, x=x, hue=hue, ax=ax3)
    else:
        sns.countplot(data=df, x=x, ax=ax3)
    ax3.set_title("Count of items in each encoding group", fontsize=12)
    # ax3.set_xlabel(x, fontsize=12)
    customize_time_xaxis(ax3, x)

    plt.suptitle(
        f"Model: {model_info['model_short_name']} (α range: {min_alpha} to {max_alpha}, {num_alphas} values)"
        + (f"\nfilter={filter_name}" if filter_name is not None else "")
        + f"\nover all times, subjects, and brain areas",
        fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout leaving space for suptitle
    plt.show()


def customize_time_xaxis(ax, x, n=20):
    if not (x == "time" or x == "time_index"):
        return
    # Set sparse labels if needed
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_visible(idx % n == 0)

    # Get tick positions and labels
    tick_positions = ax.get_xticks()
    ax.axvline(x=tick_positions[80], color='black', linewidth=0.5, alpha=0.7)

def which_coeffs_by_x(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode,
                      col_to_compare="princeton_class", kfolds_threshold=10):
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
    kfolds_df = prepare_kfolds_df(filter_type, kfolds_threshold, max_alpha, min_alpha, mode, model_info, num_alphas, sid)
    # exploded_df = kfolds_df.query("num_of_coeffs > 0")[
    #     ["full_elec_name", "time", "num_of_coeffs", "encoding", "actual_coeffs", col_to_check]].explode("actual_coeffs")    exploded_df = kfolds_df.query("num_of_coeffs > 0")[

    exploded_df = kfolds_df.query("num_of_coeffs > 0").explode("actual_coeffs")
    num_occurrences = exploded_df[[col_to_compare, 'actual_coeffs']].value_counts().reset_index(name='num_occurrences')
    kfolds_df['elec_time_combo'] = kfolds_df['full_elec_name'].astype(str) + '_' + kfolds_df['time'].astype(str)
    max_possible_occurrences = kfolds_df.groupby(col_to_compare)['elec_time_combo'].nunique().reset_index(name='max_possible_occurrences')

    # TODO: merge differently so if the coeff appears 0 times in some areas it will still get a row (with num_occurrences=0)
    full_contingency_table = num_occurrences.merge(max_possible_occurrences, on=col_to_compare, how="left")
    full_contingency_table["num_didnt_occur"] = full_contingency_table["max_possible_occurrences"] - full_contingency_table["num_occurrences"]
    full_contingency_table["percentage_occurrences"] = full_contingency_table["num_occurrences"] / full_contingency_table["max_possible_occurrences"]
    final_summary_df = full_contingency_table.pivot(index='actual_coeffs', columns=col_to_compare,values='percentage_occurrences').reset_index()

    # TODO: from here specific for IFG and STG
    filtered_contingency_table = full_contingency_table.query("(princeton_class == 'STG') | (princeton_class == 'IFG')")

    results = []
    for coeff in sorted(filtered_contingency_table["actual_coeffs"].unique()):
        curr_table = filtered_contingency_table.query(f"actual_coeffs == '{coeff}'").sort_values(by='princeton_class')[["num_occurrences","num_didnt_occur"]]
        curr_array = np.array(curr_table)
        # TODO: FIX!
        if curr_array.shape == (2, 2):
            oddsratio, p_value = fisher_exact(curr_array, alternative='two-sided')
            # print(f"coeff: {coeff}", f"oddsratio: {oddsratio}", f"p_value: {p_value}")
            results.append({
                'coefficient': coeff,
                'oddsratio': oddsratio,
                'p_value': p_value
            })
    results_df = pd.DataFrame(results)
    results_df["reject"], results_df["pvals_corrected"], _, _ = multipletests(results_df["p_value"], method='fdr_bh')
    final_summary_df = final_summary_df.merge(results_df, left_on="actual_coeffs", right_on="coefficient", how="outer",
                                              indicator=True)
    interst = final_summary_df.query("_merge=='both'")[["actual_coeffs", "IFG", "STG", "oddsratio", "pvals_corrected", "reject"]]
    interst['IFG>STG'] = interst["IFG"] > interst["STG"]
    # interst = interst[interst["reject"]]

    a=0

def add_feature_explanation(      model_id: str,
      layer: str,
      index: int,
      explanation_type: str,
      explanation_model_name: str) -> Optional[Dict[str, Any]]:
    """
    Generate a new explanation using the specified parameters.

    Args:
        model_id: Model identifier (e.g., 'gemma-2-2b')
        layer: Layer identifier (e.g., '13-gemmascope-res-16k')
        index: Feature index
        explanation_type: Type of explanation (e.g., 'oai_attention-head')
        explanation_model_name: Name of the explanation model (e.g., 'claude-3-5-haiku-20241022')

    Returns:
        Dictionary containing the response data, or None if request fails
    """
    # Construct the URL for the generate endpoint
    url = "https://www.neuronpedia.org/api/explanation/generate"

    # Prepare the payload
    payload = {
        "modelId": model_id,
        "layer": layer,
        "index": index,
        "explanationType": explanation_type,
        "explanationModelName": explanation_model_name
    }

    try:
        # Make POST request with JSON payload
        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        # Add rate limiting
        time.sleep(self.rate_limit_delay)

        return response.json()

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error for generating explanation: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed for generating explanation: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error for generating explanation: {e}")
        return None



if __name__ == '__main__':
    models_info = {"gemma-scope": {"layer": 13, "context": 32, "embedding_size": 16384,
                                   "model_short_name": "gemma-scope",
                                   "model_full_name": "gemma-scope-2b-pt-res-canonical"},
                   "gemma": {"layer": 13, "context": 32, "embedding_size": 2304,
                             "model_short_name": "gemma", "model_full_name": "gemma-2-2b"},
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

    interest_areas = ["IFG", "STG"]

    start_time = 0
    start_time_idx = time_to_index[start_time]
    end_time = 0.3
    end_time_idx = time_to_index[end_time]  # start_time_idx+1

    x="princeton_class"
    query = "(princeton_class == 'STG') | (princeton_class == 'IFG')"

    # overlap_by_area(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode, interest_areas, kfolds_threshold, start_time_idx, end_time_idx)
    plot_x_vs_num_of_coeffs(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
                            x=x, kfolds_threshold=kfolds_threshold, query=query)
    plot_x_vs_num_of_coeffs(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
                            kfolds_threshold=kfolds_threshold, query=query)
    # which_coeffs_by_x(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode,
    #                   col_to_compare="princeton_class", kfolds_threshold=kfolds_threshold)
