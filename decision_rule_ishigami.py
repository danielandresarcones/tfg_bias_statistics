import scipy.stats as stats
import scipy.integrate as integrate
import numpy as np
import arviz as az
import pandas as pd

import matplotlib.pyplot as plt

from qoi_ishigami import evaluate_proportion_bias, evaluate_proportion_no_bias


def decision_loss(mean_z, std_z, zlim, C_FN, C_FP, decision):
    """
    Compute the decision-theoretic loss function.

    Parameters:
    - mean_z: Mean of the random variable Z.
    - std_z: Standard deviation of Z.
    - zlim: Decision threshold.
    - C_FN: Cost associated with false negatives.
    - C_FP: Cost associated with false positives.
    - decision: bool, indicating which decision is taken.

    Returns:
    - Loss value associated with the given decision.
    """
    
    def integrand(z):
        return ((z - zlim) ** 2) * stats.norm.pdf(z, mean_z, std_z)

    if decision:
        loss_a1_integral, _ = integrate.quad(integrand, -3, zlim)
        loss = C_FP * loss_a1_integral
    else:
        loss_a0_integral, _ = integrate.quad(integrand, zlim, 3)
        loss = C_FN * loss_a0_integral

    return loss

def decision_loss_no_bias(mean_z, std_z, zlim, C_FN, C_FP, decision):
    """
    Compute the decision-theoretic loss function.

    Parameters:
    - mean_z: Mean of the random variable Z.
    - std_z: Standard deviation of Z.
    - zlim: Decision threshold.
    - C_FN: Cost associated with false negatives.
    - C_FP: Cost associated with false positives.
    - decision: bool, indicating which decision is taken.

    Returns:
    - Loss value associated with the given decision.
    """    

    if decision:
        loss = C_FP * (mean_z - zlim)**2 if mean_z < zlim else 0.0
    else:
        loss = C_FN  * (mean_z - zlim)**2 if mean_z > zlim else 0.0
    return loss


def bayes_risk(posterior_samples, z_lim, C_FN, C_FP, decision, bias=True):
    """
    Computes the Bayes risk using Monte Carlo integration.

    Parameters:
    - posterior_samples: array-like, samples from the posterior distribution of θ given y
    - z_lim: float, threshold for decision making
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive

    Returns:
    - float, estimated Bayes risk
    """
    if bias:
        losses = [decision_loss(sample[0], sample[1], z_lim, C_FN, C_FP, decision) for sample in posterior_samples]
    else:
        losses = [decision_loss_no_bias(sample[0], sample[1], z_lim, C_FN, C_FP, decision) for sample in posterior_samples]
    losses = [loss for loss in losses if not np.isnan(loss)]
    return np.mean(losses)

def decision_rule(inference_data, z_lim, threshold, C_FN, C_FP, X, num_samples=200, bias=True):
    """
    Computes the optimal decision rule for the given posterior samples.

    Parameters:
    - posterior_samples: array-like, samples from the posterior distribution of θ given y
    - z_lim: float, threshold for decision making
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive

    Returns:
    - bool, decision
    """

    posterior_samples = []
    samples = az.extract(inference_data, num_samples = num_samples)
    try:
        for sample_a, sample_b, sample_sigma in zip(samples["$a$"].values, samples["$b$"].values, samples["$\sigma_b$"].values):
            z_samples_minus, z_samples_mean, z_samples_plus = evaluate_proportion_bias(sample_a,sample_b,sample_sigma, threshold, X)
            posterior_samples.append((z_samples_mean, z_samples_plus-z_samples_mean))
    except KeyError:
        for sample_a, sample_b in zip(samples["$a$"].values, samples["$b$"].values):
            z_samples_minus, z_samples_mean, z_samples_plus = evaluate_proportion_no_bias(sample_a,sample_b, threshold, X)
            posterior_samples.append((z_samples_mean, z_samples_plus-z_samples_mean))

    bayes_risk_0 = bayes_risk(posterior_samples, z_lim, C_FN, C_FP, decision=False, bias=bias)
    bayes_risk_1 = bayes_risk(posterior_samples, z_lim, C_FN, C_FP, decision=True, bias=bias)

    return bayes_risk_0 > bayes_risk_1

def analyze_decision_rule(inference_data, data, C_FN, C_FP, z_lim, thresholds_array, bias=True):
    """
    Analyze the decision rule for the given data.

    Parameters:
    - inference_data: InferenceData object, containing the posterior samples
    - data: DataFrame, containing the data
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive
    - z_lim: float, threshold for decision making

    Returns:
    - bool, decision
    """

    # Extract features
    X = data[['x1', 'x2', 'x3']].values

    # Compute the optimal decision rule
    decisions_array = []
    for threshold in thresholds_array:
        decision = decision_rule(inference_data, z_lim, threshold, C_FN, C_FP, X, bias=bias)
        decisions_array.append(decision)

    # Output decisions_array to a CSV file
    decisions_df = pd.DataFrame(decisions_array, columns=['decision'])
    decisions_df.to_csv('./code/output/decisions_array.csv', index=False)

    return decisions_array

def compare_decision_costs(inference_data, data, C_FN_array, C_FP_array, z_lim, thresholds_array, output_path, bias=True):
    """
    Compare the decision rule for different costs.

    Parameters:
    - inference_data: InferenceData object, containing the posterior samples
    - data: DataFrame, containing the data
    - C_FN_array: array-like, containing the costs of false negatives
    - C_FP_array: array-like, containing the costs of false positives
    - z_lim: float, threshold for decision making
    - threshold: float, threshold for the decision rule

    Returns:
    - DataFrame, containing the decisions for different costs
    """

    decisions_matrix = []
    for C_FN in C_FN_array:
        for C_FP in C_FP_array:
            decision = analyze_decision_rule(inference_data, data, C_FN, C_FP, z_lim, thresholds_array, bias=bias)
            decisions_matrix.append(decision)

    # Output decisions_matrix to a CSV file
    decisions_df = pd.DataFrame(decisions_matrix, index=pd.MultiIndex.from_product([C_FN_array, C_FP_array], names=['C_FN', 'C_FP']), columns=thresholds_array)
    decisions_df.to_csv(output_path)

    return decisions_df

def plot_decision_costs_comparison(decisions_df, output_path):
    """
    Plot the comparison of the decision rule for different costs.

    Parameters:
    - decisions_df: DataFrame, containing the decisions for different costs
    """

    # Plot the decisions
    decisions_df = decisions_df.map(lambda x: 1 if x else 0)
    decisions_df.T.plot(style='-o', figsize=(10, 6))
    plt.xlabel('Thresholds')
    plt.xlim(-5, 5)
    plt.xticks(np.linspace(-5, 5, 11))
    plt.ylabel('Decision (0: False, 1: True)')
    plt.ylim(-0.1,1.1)
    plt.yticks([0, 1])
    plt.title('Decision rule for different costs')
    plt.legend([f'C_FN: {index[0]}, C_FP: {index[1]}' for index in decisions_df.index], title='Cost Parameters')
    plt.savefig(output_path)

if __name__ == "__main__":
    # Load the posterior samples
    inference_data_bias = az.from_netcdf("./code/output/results/calibrate_bias_ishigami_3.az")
    inference_data_no_bias = az.from_netcdf("./code/output/results/calibrate_no_bias_ishigami_3.az")
    data_path = './code/input/data/ishigami_dataset.csv'

    # Set the decision threshold
    z_lim = 0.5
    threshold_array = np.linspace(-5, 5, 20)
    # is the proportion of observations greater than the threshold larger than 0.5?

    # Set the costs
    C_FN_array = [1.0, 10.0]
    C_FP_array = [1.0, 10.0]

    # Compute the optimal decision rule
    data = pd.read_csv(data_path)

    # Extract features
    decisions_no_bias = compare_decision_costs(inference_data_no_bias, data, C_FN_array, C_FP_array, z_lim, threshold_array, output_path='./code/output/results/decisions_no_bias.csv', bias=False)
    plot_decision_costs_comparison(decisions_no_bias, output_path='./code/output/figures/decisions_no_bias.png')
    decisions_bias = compare_decision_costs(inference_data_bias, data, C_FN_array, C_FP_array, z_lim, threshold_array, output_path='./code/output/results/decisions_bias.csv', bias=True)
    # decisions_bias = pd.read_csv('./code/output/results/decisions_bias.csv', header=[0], index_col=[0, 1])
    plot_decision_costs_comparison(decisions_bias, output_path='./code/output/figures/decisions_bias.png')
