import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_encoding_and_coeffs_lines(patient, mode, models_info, filter_type, min_alpha, max_alpha, num_alphas, p_threshold, save_dir):
    """
    Main function to plot the encoding and coefficients comparison.
    """
    models_info, electrode_names_df, recording_type = load_general_files(models_info, filter_type, patient)
    # electrode_names_df = electrode_names_df[:3]  # For testing, remove this line for all electrodes

    amount_of_electrodes = len(electrode_names_df["full_elec_name"])
    assert amount_of_electrodes == filter_type, f"Expected {filter_type} electrodes, got {amount_of_electrodes}"


    subplot_titles = [title for i in range(amount_of_electrodes) for title in [f"Encoding - {patient}, {electrode_names_df['full_elec_name'][i]} ({mode})",
                                                                               f"Coeffs (Relative)",
                                                                               f"Coeffs - {patient}, {electrode_names_df['full_elec_name'][i]} ({mode})"]]

    max_vertical_spacing = 1 / (amount_of_electrodes - 1)
    chosen_vertical_spacing = max_vertical_spacing * 0.25  # Use 25% of maximum

    # Create plots:
    fig = make_subplots(rows=amount_of_electrodes, cols=3,
                        subplot_titles=subplot_titles,
                        # row_titles=row_titles,
                        vertical_spacing=chosen_vertical_spacing,
                        # horizontal_spacing=0.3,
                        shared_xaxes=True)

    for row_idx, elec_name in enumerate(electrode_names_df["full_elec_name"]):
        print(f"Processing electrode {elec_name} ({row_idx + 1}/{amount_of_electrodes})")
        for model_name in models_info.keys():
            plot_single_elec_single_model(row_idx+1, elec_name, model_name, models_info, patient, mode, min_alpha, max_alpha, num_alphas, fig, p_threshold=p_threshold)


    print("All models processed. Now customizing layout...")
    customize_encoding_and_coeffs_layout(amount_of_electrodes, fig)
    fig.show()
    save_path = os.path.join(save_dir, "encoding_and_coeffs_compare.html")
    fig.write_html(save_path)

    print(f"!!!!!! Plotting complete. HTML file saved as {save_path} !!!!!!")

def plot_coeffs_heatmap(patient, mode, models_info, filter_type, min_alpha, max_alpha, num_alphas, p_threshold, sort_by, save_dir):
    models_info, electrode_names_df, recording_type = load_general_files(models_info, filter_type, patient)
    save_dir = os.path.join(save_dir, f"coeffs_heatmaps_sorted_by_{sort_by}")
    os.makedirs(save_dir, exist_ok=True)

    for elec_name in electrode_names_df["full_elec_name"]:
        print(f"Plotting coefficients heatmap for electrode {elec_name}")
        plot_coeffs_heatmap_single_elec(elec_name, max_alpha, min_alpha, mode, models_info, num_alphas, patient, sort_by, save_dir)

    print(f"!!!!!! Plotting complete. HTML file saved as {save_dir} !!!!!!")
    return


def plot_coeffs_heatmap_single_elec(elec_name, max_alpha, min_alpha, mode, models_info, num_alphas, patient, sort_by, save_dir):
    subplot_titles = [f"<b>Encoding of {patient}, {elec_name} ({mode})</b>"] + [
        f"<b>Heatmap Non-Zero (Lasso) Coeffs of {models_info[model_name]['model_short_name']} - {patient}, {elec_name} ({mode})</b>"
        for model_name in models_info.keys()]

    # TODO: Different coeffs - for now only lasso

    fig = make_subplots(rows=len(models_info) + 1,
                        cols=1,
                        subplot_titles=subplot_titles,
                        vertical_spacing=0.04,
                        shared_xaxes=True,
                        )
    for row_idx, model_name in enumerate(models_info.keys()):
        # print(f"Processing model {model_name} for electrode {elec_name} ({row_idx + 1}/{len(models_info)})")
        overall_used_coeffs = plot_heatmap_single_elec_single_model(row_idx + 1, 1, elec_name, fig, max_alpha, min_alpha, mode, model_name, models_info, num_alphas, patient, sort_by)
        models_info[model_name]['overall_used_coeffs'] = overall_used_coeffs

    # print("All models processed. Now customizing layout...")
    customize_coeffs_heatmap_layout(fig, models_info)
    # fig.show()
    save_path = os.path.join(save_dir, f"{elec_name}_lasso.html")
    fig.write_html(save_path)
    # print(f"!!!!!! Plotting complete. HTML file saved as {save_path} !!!!!!")


def customize_coeffs_heatmap_layout(fig, models_info):

    # Customize encoding
    fig.update_yaxes(title='Correlation (r)', showgrid=True, col=1, row=1)
    fig.add_shape(type='line',
                  x0=-2, x1=2,
                  y0=0, y1=0,
                  line=dict(color='black', width=2), row=1, col=1)
    fig.add_shape(type='line',
                  x0=0, x1=0,
                  y0=-0.1, y1=0.8,
                  line=dict(color='black', width=2), row=1, col=1)
    fig.update_yaxes(title="Correlation (r)", col=1, row=1)


    for i, model_name in enumerate(models_info.keys()):
        overall_used_coeffs = models_info[model_name]['overall_used_coeffs']
        embedding_size = models_info[model_name]['embedding_size']
        subtitle = f"<br><i>Overall Used Coeffs: {overall_used_coeffs} / {embedding_size} ({overall_used_coeffs/embedding_size:.2%})</i>"
        fig.layout.annotations[i+1].text += subtitle

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='#e8e8e8'),
        name='False',
        showlegend=True,
        legend="legend2"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='black'),  # , symbol='square'),
        name='True',
        showlegend=True,
        legend="legend2"
    ))
    fig.update_xaxes(showticklabels=True, col=1)

    fig.update_layout(height=1050 * (len(models_info) +1), # + 1 for the encoding row
                      width=2000, template='simple_white',
                      legend=dict(
                          title="Encoding Methods",
                          orientation="v",
                          yanchor="top",
                          y=1,
                          xanchor="left",
                          x=1.02,
                          bgcolor="rgba(255,255,255,0.8)",
                          bordercolor="gray",
                          borderwidth=1
                      ),
                      legend2=dict(
                          title="Coeff Non-Zero",
                          orientation="v",
                          yanchor="top",
                          y=0.755,
                          xanchor="left",
                          x=1.02,
                          bgcolor="rgba(255,255,255,0.8)",
                          bordercolor="gray",
                          borderwidth=1
                      )

                      # margin=dict(l=50, r=50, t=50, b=50),
                      )

    for i in range(2, len(models_info) + 2):
        fig.update_yaxes(title='Coeff Index', showgrid=False, col=1, row=i)
    fig.update_xaxes(title='Time', showgrid=False, tickmode='linear', dtick=0.5, col=1)


def plot_heatmap_single_elec_single_model(row_idx, col_idx, elec_name, fig, max_alpha, min_alpha, mode, model_name, models_info, num_alphas, patient, sort_by):
    model_short_name = models_info[model_name]['model_short_name']

    (kfolds_lasso_enc_path, kfolds_lasso_coeffs_path,
     kfolds_train_lasso_enc_path, kfolds_train_lasso_coeffs_path,
     lasso_enc_path, lasso_coeffs_path,
     ols_enc_path, ols_coeffs_path,
     pvals_names_path, pvals_combined_corrected_path) = prep_paths(elec_name, max_alpha, min_alpha, mode, model_name,
                                                                   models_info, num_alphas, patient)

    plots_data = {"kfolds": {'color_index': 1, "name": f"Kfolds Lasso {model_short_name}",
                             "enc_ending":"(r)", "enc_name": f"Kfolds Lasso Encoding {model_short_name} (r)", "enc_path": kfolds_lasso_enc_path},
                  "lasso": {'color_index': 2, "name": f"Lasso All Data {model_short_name}",
                            "enc_ending": "(r)", "enc_name": f"Lasso All Data Encoding {model_short_name} (r)", "enc_path": lasso_enc_path,}
                  }

    for line_type in plots_data.keys():
        plot_encoding(1, 1, model_name, models_info, line_type, plots_data, True, fig)

    coeffs = np.load(lasso_coeffs_path)

    # Create a True-False matrix of coefficients according to if they are non zero
    is_coeffs_nonzero = coeffs != 0
    non_zero_coeffs_row_indx = np.where(np.any(is_coeffs_nonzero, axis=1))[0]  # Get indices of non-zero rows

    x_values = np.linspace(-2, 2, is_coeffs_nonzero.shape[1])
    row_labels = non_zero_coeffs_row_indx
    z_values = is_coeffs_nonzero[non_zero_coeffs_row_indx, :]

    if sort_by == "sum":
        features_array = -z_values.sum(axis=1) # Sort by the number of non-zero coefficients in each row
        sort_indices = np.argsort(features_array)
    elif sort_by == "first_true":
        # features_array = np.array([
        #     np.argmax(row) if np.any(row) else len(row)
        #     for row in z_values
        # ])
        features_array = features_array_by_first(z_values)
        sort_indices = np.argsort(features_array)
    elif sort_by == "last_true":
        # features_array = np.array([
        #     np.where(row)[0][-1] if np.any(row) else -1
        #     for row in z_values
        # ])
        features_array = features_array_by_last(z_values)
        sort_indices = np.argsort(features_array)
    elif sort_by == "first_then_last_then_sum":
        first_features_array = features_array_by_first(z_values)
        last_features_array = features_array_by_last(z_values)
        sum_features_array = -z_values.sum(axis=1)  # Sort by the number of non-zero coefficients in each row
        sort_indices = np.lexsort((sum_features_array, last_features_array, first_features_array))
    elif sort_by == "neuron_index":
        sort_indices = np.arange(z_values.shape[0])
    else:
        raise ValueError(f"Unknown sort method: {sort_by}")

    z_values_sorted = z_values[sort_indices, :]
    row_labels_sorted = row_labels[sort_indices]

    fig.add_trace(go.Heatmap(
        z=z_values_sorted.astype(int),
        x=x_values,
        y=row_labels_sorted.astype(str),
        colorscale=[[0, '#e8e8e8'], [1, 'black']],
        showscale=False,
        # colorbar=dict(
        #         title="Coeff Non-Zero",
        #         tickvals=[0, 1],
        #         ticktext=['False', 'True'],
        #         tickmode='array',  # Use explicit tick values
        #         dtick=1,  # Set tick spacing to 1
        #         len=0.3,  # Make colorbar shorter
        #         thickness=15,  # Make it thinner
        # ),
    ), col=col_idx, row=row_idx+1)


    # Create heatmap
    # fig.add_trace(go.Heatmap(
    #     z=coeffs_matrix,
    #     colorscale='Viridis',
    #     colorbar=dict(title='Coefficient Value'),
    #     name=model_short_name,
    #     showlegend=True,
    #     hoverongaps=False
    # ), row=row_idx + 1, col=1)
    return len(non_zero_coeffs_row_indx)


def features_array_by_first(z_values):
    features_array = np.argmax(z_values, axis=1)  # Get the index of the first True value in each row
    no_true_mask = ~z_values.any(axis=1)
    features_array[no_true_mask] = z_values.shape[1]
    return features_array


def features_array_by_last(z_values):
    features_array = np.argmax(z_values[:, ::-1], axis=1)
    no_true_mask = ~z_values.any(axis=1)
    features_array[no_true_mask] = z_values.shape[1]
    return features_array


def customize_encoding_and_coeffs_layout(amount_of_electrodes, fig):
    # Customize layout
    for idx in range(amount_of_electrodes):
        row_idx = idx + 1
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
                      y0=-0.1, y1=0.8,
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
    fig.update_yaxes(title="Fraction of Coeffs", col=2)
    fig.update_yaxes(title="Number of Coeffs", col=3)

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


def plot_single_elec_single_model(row_idx, elec_name, model_name, models_info, patient, mode, min_alpha, max_alpha, num_alphas, fig, p_threshold=0.05):
    show_legend = True if row_idx == 1 else False
    model_short_name = models_info[model_name]['model_short_name']

    (kfolds_lasso_enc_path, kfolds_lasso_coeffs_path,
     kfolds_train_lasso_enc_path, kfolds_train_lasso_coeffs_path,
     lasso_enc_path, lasso_coeffs_path,
     ols_enc_path, ols_coeffs_path,
     pvals_names_path, pvals_combined_corrected_path) = prep_paths(elec_name, max_alpha, min_alpha, mode, model_name, models_info, num_alphas, patient)

    plots_data = {"kfolds": {"dash": "solid", 'color_index':1, "name": f"Kfolds Lasso {model_short_name}",
                             "enc_ending":"(r)", "enc_name": f"Kfolds Lasso Encoding {model_short_name} (r)", "enc_path": kfolds_lasso_enc_path,
                             "coeffs_ending":"(-)", "coeffs_name": f"Significant in All Lasso Kfolds of {model_short_name}", "coeffs_path": kfolds_lasso_coeffs_path},
                  # "kflods_train": {"dashdot": "dash", 'color_index':0, "name": f"Train Kfolds Lasso {model_short_name}",
                  #                  "enc_ending":"(r)", "enc_name": f"Train Kfolds Lasso Encoding {model_short_name} (r)", "enc_path": kfolds_train_lasso_enc_path,
                  #                  "coeffs_ending":"(-)", "coeffs_name": f"Significant in All Train Lasso Kfolds of {model_short_name}", "coeffs_path": kfolds_train_lasso_coeffs_path},
                  "lasso": {"dash": "longdash", 'color_index':2, "name": f"Lasso All Data {model_short_name}",
                            "enc_ending":"(r)", "enc_name": f"Lasso All Data Encoding {model_short_name} (r)", "enc_path": lasso_enc_path,
                            "coeffs_ending":"(nonzero)", "coeffs_name": f"Lasso All Data Non Zero of {model_short_name}", "coeffs_path": lasso_coeffs_path},
                  "ols": {"dash": "dash", 'color_index':3, "name": f"OLS All Data {model_short_name}",
                          "enc_ending":"(|R|)", "enc_name": f"OLS All Data Encoding {model_short_name} (|R|)", "enc_path": ols_enc_path,
                          "coeffs_ending":"(sig)", "coeffs_name": f"Significant OLS All Data of {model_short_name}", "coeffs_path": ols_coeffs_path},
                  "corr": {"dash": "dot", 'color_index':4, "name": f"Corr All Data {model_short_name}",
                           "enc_ending":"(-)", "enc_name": f"Corr {model_short_name}", "enc_path": None,
                           "coeffs_ending":"(sig)", "coeffs_name": f"Significant Correlation All Data of {model_short_name}", "coeffs_path": pvals_combined_corrected_path, "coeffs_names_path": pvals_names_path}} # + longdashdot

    for line_type in plots_data.keys():
        plot_encoding(row_idx, 1, model_name, models_info, line_type, plots_data, show_legend, fig)
        plot_coeffs(row_idx, 2, model_name, models_info, line_type, plots_data, show_legend, fig, elec_name, type="relative", p_threshold=p_threshold)
        plot_coeffs(row_idx, 3, model_name, models_info, line_type, plots_data, show_legend, fig, elec_name, type="absolute", p_threshold=p_threshold)


def prep_paths(elec_name, max_alpha, min_alpha, mode, model_name, models_info, num_alphas, patient):
    model_path = models_info[model_name]['model_path']
    layer = models_info[model_name]['layer']
    context = models_info[model_name]['context']

    # Prep general paths
    kfolds_path_template = f"{model_path}/tk-200ms-{patient}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}/{elec_name}_{mode}{{ending}}"
    kfolds_train_path_template = f"{model_path}/tk-200ms-{patient}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}_train/{elec_name}_{mode}{{ending}}"
    sig_coeffs_path_template = f"{model_path}/tk-200ms-{patient}-lay{layer}-con{context}-reglasso-alphas_{min_alpha}_{max_alpha}_{num_alphas}-sig_coeffs/{elec_name}_{mode}{{ending}}"
    corr_path_template = f"{model_path}/tk-200ms-{patient}-lay{layer}-con{context}-corr_coeffs"

    # Prep encodings paths
    kfolds_lasso_enc_path = kfolds_path_template.format(ending=".csv")
    kfolds_train_lasso_enc_path = kfolds_train_path_template.format(ending=".csv")
    lasso_enc_path = sig_coeffs_path_template.format(ending="_lasso.csv")
    ols_enc_path = sig_coeffs_path_template.format(ending="_ols.csv")

    # Prep coeffs paths
    kfolds_lasso_coeffs_path = kfolds_path_template.format(ending="_coeffs.npy")
    kfolds_train_lasso_coeffs_path = kfolds_train_path_template.format(ending="_coeffs.npy")
    lasso_coeffs_path = sig_coeffs_path_template.format(ending="_coeffs_lasso.npy")
    ols_coeffs_path = sig_coeffs_path_template.format(ending="_ols.pkl")
    pvals_names_path = f"{corr_path_template}/pvals_combined_names{f'({filter_type})' if filter_type else ''}.pkl"
    pvals_combined_corrected_path = f"{corr_path_template}/pvals_combined_corrected{f'({filter_type})' if filter_type else ''}.npy"

    return (kfolds_lasso_enc_path, kfolds_lasso_coeffs_path,
            kfolds_train_lasso_enc_path, kfolds_train_lasso_coeffs_path,
            lasso_enc_path, lasso_coeffs_path,
            ols_enc_path, ols_coeffs_path,
            pvals_names_path, pvals_combined_corrected_path)


def plot_coeffs(row_idx, col_idx, model_name, models_info, line_type, plots_data, show_legend, fig, elec_name, type="relative", p_threshold=0.05):

    if "coeffs_path" not in plots_data[line_type] or plots_data[line_type]['coeffs_path'] is None:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name='',  # Empty name creates blank line
            line=dict(color='rgba(0,0,0,0)'),  # Transparent line
            showlegend=type == "relative",
            legend="legend2"
        ))
        return


    color = models_info[model_name]['colors'][plots_data[line_type]["color_index"]]
    dash = "solid" #plots_data[line_type]["dash"]
    coeffs_name = plots_data[line_type]['coeffs_name']
    coeffs_path = plots_data[line_type]["coeffs_path"]
    general_name = plots_data[line_type]['name']
    enc_name = plots_data[line_type]['enc_name']
    embedding_size = models_info[model_name]['embedding_size']

    num_of_coeffs = None
    if line_type.startswith("kfolds"):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name='',  # Empty name creates blank line
            line=dict(color='rgba(0,0,0,0)'),  # Transparent line
            showlegend=type == "relative",
            legend="legend2"
        ))
        return
    elif line_type.startswith("lasso"):
        coeffs = np.load(coeffs_path)
        num_of_coeffs = np.count_nonzero(coeffs, axis=0) # nonzero_counts
        if type == "relative":
            num_of_coeffs = num_of_coeffs / embedding_size

    elif line_type.startswith("ols"):
        with open(coeffs_path, 'rb') as f:
            ols_model_fitting_params = pickle.load(f)

        if type == "absolute":
            num_of_coeffs = [
                None if ols_param is None else sum(1 for p in ols_param.get('p_values', []) if p < p_threshold)
                for ols_param in ols_model_fitting_params] # sig_lasso_count
        elif type == "relative":
            num_of_coeffs = [
                None if ols_param is None else sum(1 for p in ols_param.get('p_values', []) if p < p_threshold) / embedding_size
                for ols_param in ols_model_fitting_params]  # sig_lasso_count
        else:
            raise ValueError(f"Unknown type: {type}. Expected 'relative' or 'absolute'.")

    elif line_type.startswith("corr"):
        pvals_combined_corrected = np.load(coeffs_path)
        with open(plots_data[line_type]["coeffs_names_path"], 'rb') as f:
            pvals_names = pickle.load(f)
        elec_pvals_corrected = pvals_combined_corrected[:, :, pvals_names.index(elec_name)]
        # is_sig_mask = np.where((elec_pvals_corrected <= p_threshold).sum(axis=1))[0]
        # elec_pvals_corrected_sig = elec_pvals_corrected[is_sig_mask, :]
        num_of_coeffs = np.sum(elec_pvals_corrected<=p_threshold, axis=0)
        if type == "relative":
            num_of_coeffs = num_of_coeffs / embedding_size

    x = np.linspace(-2, 2, len(num_of_coeffs))
    fig.add_trace(go.Scatter(
        x=x,
        y=num_of_coeffs,
        mode='lines',
        name=general_name + " " + plots_data[line_type]["coeffs_ending"],
        legendgroup=coeffs_name,#general_name,
        line=dict(color=f"rgb({','.join(color)})", width=2, dash=dash),
        showlegend=show_legend and (type == "relative"),#not ("enc_path" in plots_data[line_type] and plots_data[line_type]['enc_path'] is not None) and show_legend,
        legend="legend2",
    ), row=row_idx, col=col_idx)
    return

def plot_encoding(row_idx, col_idx, model_name, models_info, line_type, plots_data, show_legend, fig):
    if "enc_path" not in plots_data[line_type] or plots_data[line_type]['enc_path'] is None:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            name='',  # Empty name creates blank line
            line=dict(color='rgba(0,0,0,0)'),  # Transparent line
            showlegend=True,
            legend="legend1"
        ))
        return

    color = models_info[model_name]['colors'][plots_data[line_type]["color_index"]]
    enc_name = plots_data[line_type]['enc_name']
    dash = "solid" #plots_data[line_type]["dash"]
    general_name = plots_data[line_type]['name']

    encoding_path = plots_data[line_type]["enc_path"]
    encoding = np.genfromtxt(encoding_path, delimiter=',')

    if line_type.startswith("kfolds"):
        encoding_mean = encoding.mean(axis=0)
        encoding_std = encoding.std(axis=0)
        y_upper = encoding_mean + encoding_std
        y_lower = encoding_mean - encoding_std

        x = np.linspace(-2, 2, encoding_mean.shape[-1])

        # Add standard deviation to kfolds
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor=f"rgba({','.join(color)},0.2)",
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=enc_name,
            name='±1 SD',
            legend="legend1",
        ), row=row_idx, col=col_idx)

        encoding = encoding_mean

    elif line_type.startswith("ols"):
        encoding = np.sqrt(encoding) # OLS original encoding is r^2

    x = np.linspace(-2, 2, encoding.shape[-1])

    # Add the main line
    fig.add_trace(go.Scatter(
        x=x,
        y=encoding,
        mode='lines',
        name=general_name + " " + plots_data[line_type]["enc_ending"],
        line=dict(color=f"rgb({','.join(color)})", width=2, dash=dash),
        legendgroup=enc_name,
        showlegend=show_legend,
        legend="legend1",
    ), row=row_idx, col=col_idx)

def load_general_files(models_info, filter_type, patient):
    recording_type = get_recording_type(patient)
    electrode_names_csv_path = None
    if filter_type == 160:
        electrode_names_csv_path = "/scratch/gpfs/tk6637/princeton/247-plotting/data/plotting/sig-elecs/podcast_160.csv"
    elif filter_type == 50:
        electrode_names_csv_path = "/scratch/gpfs/tk6637/princeton/247-plotting/data/plotting/sig-elecs/podcast_777_glove_elecs.csv"
    if electrode_names_csv_path:
        electrode_names_df = pd.read_csv(electrode_names_csv_path)
        electrode_names_df["full_elec_name"] = electrode_names_df["subject"].astype(str) + "_" + electrode_names_df[
            "electrode"]

    get_models_path(models_info, patient, recording_type)

    return models_info, electrode_names_df, recording_type


def get_models_path(models_info, patient, recording_type):
    for model_name in models_info.keys():
        model_path = f"../data/encoding/{recording_type}/tk-{recording_type}-{patient}-{model_name}-lag2k-25-all"
        models_info[model_name]['model_path'] = model_path


def get_recording_type(patient):
    recording_type = "tfs"
    if patient == 777:
        recording_type = "podcast"
    return recording_type


if __name__ == '__main__':
    # Choose parameters:
    patient = 777
    mode = "comp"
    models_info = {"glove50": {"layer": 0, "context": 1, "embedding_size": 50,"model_short_name":"glove50", "colors": [('187','222','251'),('100','181','246'),('33','150','243'),('25','118','210'),('13','71','161')]},
                   "gemma-2-2b": {"layer": 13, "context": 32, "embedding_size": 2304, "model_short_name":"gemma", "colors": [('200','230','201'),('129','199','132'),('76','175','80'),('56','142','60'),('27','94','32')]},
                   "gemma-scope-2b-pt-res-canonical": {"layer": 13, "context": 32, "embedding_size": 16384, "model_short_name":"gemma-scope", "colors": [('248','187','208'),('240','98','146'),('233','30','99'),('194','24','91'), ('136','14','79')]},
                   }

    filter_type = 160

    min_alpha = -2
    max_alpha = 30
    num_alphas = 50

    p_threshold = 0.05

    save_dir = "../results/figures"
    sort_coeffs_by = "first_then_last_then_sum"  # Options: "sum", "first_true", "last_true", "neuron_index", "first_then_last"

    # plot_encoding_and_coeffs_lines(patient, mode, models_info, filter_type, min_alpha, max_alpha, num_alphas, p_threshold, save_dir)
    plot_coeffs_heatmap(patient, mode, models_info, filter_type, min_alpha, max_alpha, num_alphas, p_threshold, sort_coeffs_by, save_dir)