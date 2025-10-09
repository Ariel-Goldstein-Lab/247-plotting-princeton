import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_alpha_cv_curve(alpha_range, elec_name, model):
    alphas, best_l1_regs, cv_scores = parse_alpha_range(alpha_range, model)
    plot_all_timepoints(alpha_range, elec_name, alphas, best_l1_regs, cv_scores)


def parse_alpha_range(alpha_range, model):
    with open(f'/scratch/gpfs/HASSON/tk6637/princeton/temp_data/alpha_plots/{alpha_range}/checked_alphas.npy', 'rb') as f:
        alphas = np.load(f)


    with open(f'/scratch/gpfs/HASSON/tk6637/princeton/247-encoding/results/podcast/tk-podcast-777-{model}-lag2k-25-all/tk-200ms-777-lay13-con32-reglasso-alphas_{alpha_range}-all_data/{elec_name}_comp_best_l1_regs.npy', 'rb') as f:
        best_l1_regs = np.load(f)

    with open(f'/scratch/gpfs/HASSON/tk6637/princeton/247-encoding/results/podcast/tk-podcast-777-{model}-lag2k-25-all/tk-200ms-777-lay13-con32-reglasso-alphas_{alpha_range}-all_data/{elec_name}_comp_cv_scores.npy', 'rb') as f:
        cv_scores = np.load(f)

    # file_type = "best_l1_regs"
    # var_name = f"{file_type}_{elec_name}"
    # with open(f'/scratch/gpfs/HASSON/tk6637/princeton/247-encoding/results/podcast/tk-podcast-777-{model}-lag2k-25-all/tk-200ms-777-lay13-con32-reglasso-alphas_{alpha_range}-all_data/{elec_name}_comp_{file_type}.npy', 'rb') as f:
    #     globals()[var_name] = np.load(f)

    return alphas, best_l1_regs, cv_scores


def plot_all_timepoints(alpha_range, elec_name, alphas, best_l1_regs, cv_scores):
    time_points = np.linspace(-2, 2, 161)
    for time_index in range(len(time_points)):
        fig = px.scatter(x=np.flip(alphas), y=np.flip(cv_scores[:, time_index]),
                         title=f'Alpha {alpha_range} CV curve, <br>elec={elec_name} time={time_points[time_index]:.3f} (time_index={time_index})<br>best alpha index={np.argmax(np.flip(cv_scores[:, time_index]))}, chosen_alpha={best_l1_regs[time_index]:.3f}',
                         labels={'x': 'Alpha Values (Log Scale)', 'y': 'Y Values'},
                         log_x=True, range_y=[-0.1, 0.1])
        fig.update_traces(
            hovertemplate='<b>Alpha Index:</b> %{pointNumber}<br>' +
                          '<b>Alpha (actual):</b> %{x:.4f}<br>' +
                          '<b>CV Score:</b> %{y:.4f}<br>' +
                          '<extra></extra>'
        )
        fig.write_html(f"/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/results/figures/alpha_plots/{alpha_range}/{elec_name}-alpha_cv_plot_time_{time_index}.html")
        fig.write_image(f"/scratch/gpfs/HASSON/tk6637/princeton/247-plotting/results/figures/alpha_plots/{alpha_range}/{elec_name}-alpha_cv_plot_time_{time_index}.png")
        # fig.show()

if __name__ == '__main__':
    alpha_range = "-0.7_1.27_30"
    elec_name = "662_EEGPO_02REF"
    model = "gemma-scope-2b-pt-res-canonical"
    plot_alpha_cv_curve(alpha_range, elec_name, model)