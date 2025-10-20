import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from upsetplot import from_contents, UpSet
from plotly.subplots import make_subplots

from tfsplt_utils import load_electrode_names_and_locations, get_encoding_and_coeffs_paths

def get_non_zero_coeffs(sid, elecs_names, model_info, min_alpha, max_alpha, num_alphas, mode,
                        kfolds_threshold=10, start_time_idx=0, end_time_idx=161, elec_name_suffix=""):
    coeffs_indx = np.arange(model_info["embedding_size"]).astype(str)

    all_data_chosen_coeffs_dict = {}
    kfolds_chosed_coeffs_dict = {}
    for full_elec_name in elecs_names:
        final_elec_name = elec_name_suffix+full_elec_name
        (kfolds_enc_path, kfold_coeffs_path,
         all_data_enc_path, all_data_coeffs_path) = get_encoding_and_coeffs_paths(sid, model_info["model_full_name"], model_info["layer"], model_info["context"],
                                                                                  min_alpha, max_alpha, num_alphas, full_elec_name, mode)
        # All data
        all_data_coeffs = np.load(all_data_coeffs_path) # shape (embedding_size, timepoints)
        all_data_non_zero_mask = all_data_coeffs != 0
        all_data_coeffs_counts = all_data_non_zero_mask.sum(axis=0)
        all_data_chosed_coeffs = [coeffs_indx[col] for col in all_data_non_zero_mask.T]

        assert np.array_equal([len(coeffs) for coeffs in all_data_chosed_coeffs], all_data_coeffs_counts)

        all_data_chosen_coeffs_dict[final_elec_name] = list({item for sublist in all_data_chosed_coeffs[start_time_idx:end_time_idx] for item in sublist})  # Take union of coeffs in the selected time range


        # Kfolds
        kfolds_coeffs = np.load(kfold_coeffs_path)  # shape (folds, embedding_size, timepoints)
        kfolds_non_zero_mask = kfolds_coeffs != 0
        kfold_coeffs_in_folds = kfolds_non_zero_mask[:, :, :].sum(axis=0)  # For each timepoint, for each coeff, how many kfolds is it non-zero in
        reliable_coeffs_counts = (kfold_coeffs_in_folds >= kfolds_threshold).sum(axis=0)  # For each timepoint, how many coeffs are non-zero in at least `threshold` kfolds
        kfolds_chosen_coeffs = [coeffs_indx[col] for col in (kfold_coeffs_in_folds>=kfolds_threshold).T]

        assert np.array_equal([len(coeffs) for coeffs in kfolds_chosen_coeffs], reliable_coeffs_counts)

        kfolds_chosed_coeffs_dict[final_elec_name] = list({item for sublist in kfolds_chosen_coeffs[start_time_idx:end_time_idx] for item in sublist}) # Take union of coeffs in the selected time range
        # plot_coeff_kfold_agreement_over_time(kfold_coeffs_in_folds, all_data_coeffs_counts)

    return all_data_chosen_coeffs_dict, kfolds_chosed_coeffs_dict
    

def plot_coeff_kfold_agreement_over_time(kfold_coeffs_count, all_data_coeffs_counts):
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

def overlap_by_area(sid, filter_type, model_info, min_alpha, max_alpha, num_alphas, mode, interest_areas=None, kfolds_threshold=10, start_time_idx=0, end_time_idx=161):
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
        all_data_chosen_coeffs_dict, kfolds_chosen_coeffs_dict = get_non_zero_coeffs(sid, areas_electrodes, model_info, min_alpha, max_alpha, num_alphas, mode, kfolds_threshold, start_time_idx, end_time_idx, elec_name_suffix=area+"_")
        all_data_all_elec.update(all_data_chosen_coeffs_dict)
        kfolds_all_elecs.update(kfolds_chosen_coeffs_dict)
        kfolds_all_elecs.update({f"---------{area}---------": []})  # To create a gap between areas in the plot
    # Remove empty entries (the area separators)
    kfolds_all_elecs = {k: v for k, v in kfolds_all_elecs.items() if (v or "---" in k)}
    content = from_contents(kfolds_all_elecs)
    upset = UpSet(content, subset_size='count', show_counts=True,  sort_categories_by='input', min_degree=2, min_subset_size=4, sort_by='cardinality')
    upset.plot()
    plt.show()

    print("Done!")


if __name__ == '__main__':
    models_info = {"gemma-scope": {"layer": 13, "context": 32, "embedding_size": 16384,
                                   "model_short_name": "gemma-scope", "model_full_name": "gemma-scope-2b-pt-res-canonical"},}
    time_points = np.linspace(-2, 2, 161)
    time_to_index = {t: i for i, t in enumerate(np.round(time_points, 3))}

    sid = 777
    filter_type = '160'
    model_name = 'gemma-scope'
    min_alpha = -2
    max_alpha = 10
    num_alphas = 100
    mode = 'comp'
    kfolds_threshold = 8

    interest_areas = ["IFG", "STG"]

    start_time = 0
    start_time_idx = time_to_index[start_time]
    end_time = 0.3
    end_time_idx = time_to_index[end_time] # start_time_idx+1


    overlap_by_area(sid, filter_type, models_info[model_name], min_alpha, max_alpha, num_alphas, mode, interest_areas, kfolds_threshold, start_time_idx, end_time_idx)

