from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
import chaospy as cp
from scipy.special import logsumexp
from tqdm import tqdm

from probeye.definition.inverse_problem import InverseProblem
from probeye.inference.bias.solver import EmbeddedPCESolver
from probeye.definition.distribution import Normal, Uniform, LogNormal

from probeye.inference.bias.likelihood_models import (
    EmbeddedLikelihoodBaseModel,
    IndependentNormalModelError,
)

from probeye.inference.emcee.solver import EmceeSolver
from calibrate_ishigami_bias import IshigamiBiased


def log_likelihood_bias(data, predictions_mean, predictions_std):

    # Calculate the log likelihood of the data given the model
    log_likelihood = np.sum(-0.5 * np.log(2 * np.pi * np.square(predictions_std)) - 0.5 * np.square(data - predictions_mean) /  np.square(predictions_std))

    return log_likelihood


def segment_data(data: pd.DataFrame, n_segments: int) -> list:
    # Determine the segment edges
    x1_edges = np.linspace(data["x1"].min(), data["x1"].max(), n_segments + 1)
    x2_edges = np.linspace(data["x2"].min(), data["x2"].max(), n_segments + 1)
    x3_edges = np.linspace(data["x3"].min(), data["x3"].max(), n_segments + 1)

    # Assign each entry to a segment
    segments = []
    for i in range(len(data)):
        x1_segment = np.digitize(data["x1"].iloc[i], x1_edges) - 1
        x2_segment = np.digitize(data["x2"].iloc[i], x2_edges) - 1
        x3_segment = np.digitize(data["x3"].iloc[i], x3_edges) - 1
        segments.append((x1_segment, x2_segment, x3_segment))

    data["segment"] = segments
    # Assign an index to each unique tuple of segments
    unique_segments = list(set(segments))
    segment_indices = {seg: idx for idx, seg in enumerate(unique_segments)}

    # Map each segment tuple to its index
    data["segment_index"] = data["segment"].map(segment_indices)
    return data

def solve_inverse_problem(data: pd.DataFrame, full_data: pd.DataFrame):
    problem = InverseProblem("Ishigami with Gaussian noise", print_header=False)

    problem.add_parameter(
        "a",
        tex="$a$",
        info="Value of a",
        prior=Normal(mean=2.0, std=1.0),
    )
    problem.add_parameter(
        "b",
        info="Value of b",
        tex="$b$",
        prior=Normal(mean=1.0, std=1.0),
    )
    problem.add_parameter(
        "sigma_b",
        info="Bias value of b",
        tex="$\sigma_b$",
        prior=LogNormal(mean=-2, std=0.5),
    )
    problem.add_parameter(
        "sigma",
        domain="(0, +oo)",
        tex=r"$\sigma$",
        info="Standard deviation, of zero-mean Gaussian noise model",
        # prior=Uniform(low=0.0, high=0.8),
        value=0.1,
    )

    # load data

    data_path = "code/input/data/ishigami_dataset.csv"
    data = pd.read_csv(data_path)
    x1 = data["x1"].values
    x2 = data["x2"].values
    x3 = data["x3"].values
    y = data["y"].values

    # experimental data
    problem.add_experiment(
        name="TestSeries_1",
        sensor_data={"x1": x1, "x2": x2, "x3": x3, "y": y},
    )
    degree = 3

    # forward model
    forward_model = IshigamiBiased(degree=degree)
    problem.add_forward_model(forward_model, experiments="TestSeries_1")

    # likelihood model
    dummy_lmodel = EmbeddedLikelihoodBaseModel(
    experiment_name="TestSeries_1", l_model="independent_normal"
    )
    likelihood_model = IndependentNormalModelError(dummy_lmodel)
    problem.add_likelihood_model(likelihood_model)

    solver = EmbeddedPCESolver(problem, show_progress=True)
    inference_data = solver.run(n_steps=600, n_initial_steps=100)

    true_values = {"a": 7.0, "b": 0.1}
    # true_values = {"a": 7.0, "b": 0.1, "sigma": 0.1}

    predictions_input_dict = {
        "x1": full_data["x1"].values,
        "x2": full_data["x2"].values,
        "x3": full_data["x3"].values,
        "a": inference_data.posterior["$a$"].mean().values,
        "b": inference_data.posterior["$b$"].mean().values,
        "sigma_b": inference_data.posterior["$\sigma_b$"].mean().values,    
    }
    predictions_response = forward_model.response(predictions_input_dict)
    predictions_dict = {
        "mean":cp.E(predictions_response["y"], predictions_response["dist"]),
        "std":cp.Std(predictions_response["y"], predictions_response["dist"])
    }
    return predictions_dict, inference_data

def calculate_D_phi(S, Y, predictions_mean, predictions_std):
    """
    Calculate D_phi(S) using log-likelihoods with numerical stability.

    Parameters:
        S: list or array
            Indices of observations to exclude.
        Y: list or array
            Full set of observations.
        log_likelihood_fn: function
            Function computing log-likelihood: log_likelihood_fn(Y_subset, theta).
        inference_data: arviz.InferenceData
            Posterior samples stored in an ArviZ inference data object.

    Returns:
        float
            The value of D_phi(S).
    """

    # Identify Y_S (the subset of Y with S excluded)
    Y_S = Y[Y["segment_index"] != S]["y"].values
    predictions_mean_S = predictions_mean.iloc[:, Y[Y["segment_index"] != S].index].values
    predictions_std_S = predictions_std.iloc[:, Y[Y["segment_index"] != S].index].values     
    
    # Compute log π_S(θ) for each posterior sample
    log_pi_S_theta = []
    for prediction_mean, prediction_std, prediction_mean_S, prediction_std_S in zip(predictions_mean.values, predictions_std.values, predictions_mean_S, predictions_std_S):
        log_likelihood_full = log_likelihood_bias(Y["y"].values, prediction_mean, prediction_std)
        log_likelihood_subset = log_likelihood_bias(Y_S, prediction_mean_S, prediction_std_S)
        log_pi_S_theta.append(log_likelihood_full - log_likelihood_subset)  # log(π_S(θ))
    
    log_pi_S_theta = np.array(log_pi_S_theta)
    N = len(log_pi_S_theta)  # Number of posterior samples
    
    # Compute log E_theta|Y [π_S(θ)^{-1}] using the log-sum-exp trick
    log_E_inv_pi_S = -np.log(N) + logsumexp(-log_pi_S_theta)  

    # Compute E_theta|Y [log π_S(θ)]
    E_log_pi_S = np.mean(log_pi_S_theta)  

    # Compute D_phi(S)
    D_phi_S = log_E_inv_pi_S + E_log_pi_S
    return D_phi_S


def robustness_analysis(data: pd.DataFrame, predictions_mean:pd.DataFrame, predictions_std:pd.DataFrame, n_segments: int, output_path: str):
    
    n_segments = 3
    # Segment the data
    data = segment_data(data, n_segments)

    # Create a 3x3x3 array to store D_phi values
    D_phi_values = np.zeros((n_segments, n_segments, n_segments))

    # Iterate over each segment
    for idx in range(n_segments**3):
        # Recover (i, j, k) coordinates for the 3D grid
        i, j, k = np.unravel_index(idx, (n_segments, n_segments, n_segments))
        
        # Compute D_phi for this segment
        D_phi_S = calculate_D_phi(idx, data, predictions_mean, predictions_std)
        print(f"D_phi({i}, {j}, {k}) = {D_phi_S}")
        
        # Store in the corresponding 3D position
        D_phi_values[i, j, k] = D_phi_S

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # 3 Heatmaps for each depth slice (k = depth)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(3)]
    for k in range(n_segments):
        im = axes[k].imshow(D_phi_values[:, :, k], cmap='viridis', origin='lower', vmin=0, vmax=100)
        axes[k].set_title(f"Slice {k}")
        plt.colorbar(im, ax=axes[k])

    # Prepare data for 3D Scatter Plot
    x, y, z = np.meshgrid(np.arange(0.5, n_segments), np.arange(0.5, n_segments), np.arange(0.5, n_segments), indexing='ij')
    x_flat, y_flat, z_flat, values_flat = x.flatten(), y.flatten(), z.flatten(), D_phi_values.flatten()

    ax = fig.add_subplot(2, 2, 4, projection='3d')

    # Make 3D scatter plot
    sc = ax.scatter(x_flat, y_flat, z_flat, c=values_flat, cmap='viridis', s=100, vmin=0, vmax=100)

    # Highlight max value
    max_idx = np.argmax(values_flat)
    ax.scatter(x_flat[max_idx], y_flat[max_idx], z_flat[max_idx], color='red', s=300, edgecolors='black')

    plt.colorbar(sc, label="D_phi Value")
    ax.set_title("3D Scatter Plot of D_phi")

    plt.tight_layout()
    plt.savefig(output_path)

    # Convert D_phi_values to a LaTeX table
    with open(output_path.replace('.png', '.tex'), 'w') as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("Segment (i, j, k) & D_phi Value \\\\\n")
        f.write("\\hline\n")
        for idx in range(n_segments**3):
            i, j, k = np.unravel_index(idx, (n_segments, n_segments, n_segments))
            f.write(f"({i}, {j}, {k}) & {D_phi_values[i, j, k]:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{D_phi values for each segment}\n")
        f.write("\\label{tab:d_phi_values}\n")
        f.write("\\end{table}\n")


def sobol_analysis():
    
    pass

def influence_analysis(data: pd.DataFrame, n_segments: int, output_path: str):

    segmented_data = segment_data(data, n_segments)
    full_predictions, full_inference_data = solve_inverse_problem(data, data)

    influence_matrix_mean = np.zeros((n_segments**3, n_segments**3))
    influence_matrix_std = np.zeros((n_segments**3, n_segments**3))
    a_mean_list = np.zeros((n_segments**3))
    b_mean_list = np.zeros((n_segments**3))
    sigma_mean_list = np.zeros((n_segments**3))
    a_full = full_inference_data.posterior["$a$"].mean().values
    b_full = full_inference_data.posterior["$b$"].mean().values
    sigma_full = full_inference_data.posterior["$\sigma_b$"].mean().values

    for i in tqdm(range(n_segments**3), desc="Influence Analysis Progress"):
        isegmented_data = segmented_data[segmented_data["segment_index"] != i]
        predictions, inference_data = solve_inverse_problem(isegmented_data, data)
        mean_differences = full_predictions["mean"] - predictions["mean"]
        std_differences = full_predictions["std"] - predictions["std"]
        a_mean_list[i] = inference_data.posterior["$a$"].mean().values
        b_mean_list[i] = inference_data.posterior["$b$"].mean().values
        sigma_mean_list[i] = inference_data.posterior["$\sigma_b$"].mean().values
        for segment_index in range(n_segments**3):
            segment_mean_diff = mean_differences[segmented_data["segment_index"] == segment_index]
            influence_matrix_mean[segment_index, i] = np.mean(segment_mean_diff)
            
            segment_std_diff = std_differences[segmented_data["segment_index"] == segment_index]
            influence_matrix_std[segment_index, i] = np.mean(segment_std_diff)

    influence_matrix_mean = influence_matrix_mean.transpose()
    influence_matrix_std = influence_matrix_std.transpose()

    # Save influence matrices to files
    np.savetxt(output_path.replace('.png', '_influence_mean.csv'), influence_matrix_mean, delimiter=',')
    np.savetxt(output_path.replace('.png', '_influence_std.csv'), influence_matrix_std, delimiter=',')
    np.savetxt(output_path.replace('.png', '_a_mean.csv'), a_mean_list, delimiter=',')
    np.savetxt(output_path.replace('.png', '_b_mean.csv'), b_mean_list, delimiter=',')
    np.savetxt(output_path.replace('.png', '_sigma_mean.csv'), sigma_mean_list, delimiter=',')

    return influence_matrix_mean, influence_matrix_std, a_mean_list, b_mean_list, sigma_mean_list, a_full, b_full, sigma_full

def plot_influence_matrix(influence_matrix_mean, influence_matrix_std, output_path: str):
    plt.figure(figsize=(12, 6))

    # Plot heatmap for influence matrix mean
    plt.subplot(1, 2, 1)
    sns.heatmap(influence_matrix_mean, cmap="viridis", cbar=True, vmin=-0.10, vmax=0.10)
    plt.title("Influence Matrix Mean")
    plt.xlabel("Segment Index")
    plt.ylabel("Data Index")

    # Plot heatmap for influence matrix std
    plt.subplot(1, 2, 2)
    sns.heatmap(influence_matrix_std, cmap="viridis", cbar=True, vmin=-0.10, vmax=0.10)
    plt.title("Influence Matrix Std")
    plt.xlabel("Segment Index")
    plt.ylabel("Data Index")

    plt.tight_layout()
    plt.savefig(output_path)

def plot_predictions_influence(a_mean_list, b_mean_list, sigma_mean_list, a_full, b_full, sigma_full, output_path: str):
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 3)

    # Plot a_mean
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(a_mean_list, color='blue')
    ax1.axhline(y=a_full, color='black', linestyle='--')
    ax1.set_title("Influence on a")
    ax1.set_xlabel("Data Index")
    ax1.set_ylabel("Mean of a")

    # Plot b_mean
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(b_mean_list, color='red')
    ax2.axhline(y=b_full, color='black', linestyle='--')
    ax2.set_title("Influence on b")
    ax2.set_xlabel("Data Index")
    ax2.set_ylabel("Mean of b")

    # Plot sigma_mean
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(sigma_mean_list, color='green')
    ax3.axhline(y=sigma_full, color='black', linestyle='--')
    ax3.set_title("Influence on sigma")
    ax3.set_xlabel("Data Index")
    ax3.set_ylabel("Mean of sigma")

    plt.tight_layout()
    plt.savefig(output_path)


def sensitivity_analysis(influence_matrix_mean, influence_matrix_std, output_path: str):

    sensitivity_matrix_mean = influence_matrix_mean.transpose() @ influence_matrix_mean
    sensitivity_matrix_std = influence_matrix_std.transpose() @ influence_matrix_std

    # Save sensitivity matrices to files
    np.savetxt(output_path.replace('.png', '_sensitivity_mean.csv'), sensitivity_matrix_mean, delimiter=',')
    np.savetxt(output_path.replace('.png', '_sensitivity_std.csv'), sensitivity_matrix_std, delimiter=',')

    return sensitivity_matrix_mean, sensitivity_matrix_std
    
    
def plot_sensitivity_matrix(sensitivity_matrix_mean, sensitivity_matrix_std, output_path:str):
    plt.figure(figsize=(12, 6))
    # Plot heatmap for sensitivity matrix mean
    plt.subplot(1, 2, 1)
    sns.heatmap(sensitivity_matrix_mean, cmap="plasma", cbar=True, vmin=-0.05, vmax=0.05)
    plt.title("Sensitivity Matrix Mean")
    plt.xlabel("Data Index")
    plt.ylabel("Data Index")

    # Plot heatmap for sensitivity matrix std
    plt.subplot(1, 2, 2)
    sns.heatmap(sensitivity_matrix_std, cmap="plasma", cbar=True, vmin=-0.05, vmax=0.05)
    plt.title("Sensitivity Matrix Std")
    plt.xlabel("Data Index")
    plt.ylabel("Data Index")
    plt.title("Sensitivity Matrix Std")
    plt.xlabel("Data Index")
    plt.ylabel("Data Index")

    plt.tight_layout()
    plt.savefig(output_path)
    

if __name__ == "__main__":

    data = pd.read_csv('./code/input/data/ishigami_dataset.csv')
    large_data = pd.read_csv('./code/input/data/ishigami_large_dataset.csv')

    inference_data_bias = az.from_netcdf('./code/output/results/calibrate_bias_ishigami_3.az')
    inference_data_no_bias = az.from_netcdf('./code/output/results/calibrate_no_bias_ishigami_3.az')
    output_path_root_bias = './code/output/figures/ishigami_bias'
    output_path_root_no_bias = './code/output/figures/ishigami_no_bias'

    # convergence_analysis(inference_data_bias, output_path_root_bias + '_convergence.png')
    # fit_analysis(inference_data_bias, output_path_root_bias + '_fit.json')
    # fit_analysis(inference_data_no_bias, output_path_root_no_bias + '_fit.json')

    predictions_no_bias = pd.read_csv('./code/output/results/ishigami_nobias_predictions.csv', header=0)
    predictions_bias_mean = pd.read_csv('./code/output/results/ishigami_bias_predictions_at_data_mean.csv', header=0)
    predictions_bias_std = pd.read_csv('./code/output/results/ishigami_bias_predictions_at_data_std.csv', header=0)

    robustness_analysis(data, predictions_bias_mean, predictions_bias_std, 3, './code/output/figures/ishigami_robustness.png') 

    # influence_matrix_mean, influence_matrix_std, a_mean_list, b_mean_list, sigma_mean_list, a_full, b_full, sigma_full = influence_analysis(data, 3, './code/output/figures/ishigami_influence.png')
    # sensitivity_matrix_mean, sensitivity_matrix_std = sensitivity_analysis(influence_matrix_mean, influence_matrix_std, './code/output/figures/ishigami_sensitivity.png')

    influence_matrix_mean = np.loadtxt('./code/output/figures/ishigami_influence_influence_mean.csv', delimiter=',')
    influence_matrix_std = np.loadtxt('./code/output/figures/ishigami_influence_influence_std.csv', delimiter=',')
    # a_mean_list = np.loadtxt('./code/output/figures/ishigami_a_mean.csv', delimiter=',')
    # b_mean_list = np.loadtxt('./code/output/figures/ishigami_b_mean.csv', delimiter=',')
    # sigma_mean_list = np.loadtxt('./code/output/figures/ishigami_sigma_mean.csv', delimiter=',')
    # a_full = np.loadtxt('./code/output/figures/ishigami_a_full.csv', delimiter=',')
    # b_full = np.loadtxt('./code/output/figures/ishigami_b_full.csv', delimiter=',')
    # sigma_full = np.loadtxt('./code/output/figures/ishigami_sigma_full.csv', delimiter=',')
    sensitivity_matrix_mean, sensitivity_matrix_std = sensitivity_analysis(influence_matrix_mean, influence_matrix_std, './code/output/figures/ishigami_sensitivity.png')

    plot_influence_matrix(influence_matrix_mean, influence_matrix_std, './code/output/figures/ishigami_influence.png')
    plot_sensitivity_matrix(sensitivity_matrix_mean, sensitivity_matrix_std, './code/output/figures/ishigami_sensitivity.png')
    # plot_predictions_influence(a_mean_list, b_mean_list, sigma_mean_list, a_full, b_full, sigma_full, './code/output/figures/ishigami_predictions_influence.png')