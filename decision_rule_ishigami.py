import scipy.stats as stats
import scipy.integrate as integrate
import numpy as np
import arviz as az
import pandas as pd
import kdetools as kt

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
    
    posterior_samples = [sample for sample in posterior_samples if sample[1] >= 1e-6 and not np.isnan(sample[0]) and not np.isnan(sample[1])]

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
    # evpi_array = []
    # if bias:
    #     evmi = []
    #     evoi = []
    for threshold in thresholds_array:
        decision = decision_rule(inference_data, z_lim, threshold, C_FN, C_FP, X, bias=bias)
        decisions_array.append(decision)
        # evpi, optimum_loss = expected_value_of_perfect_information(inference_data, data, C_FN, C_FP, z_lim, threshold, bias=bias)
        # evpi_array.append(evpi)
        # if bias:
        #     evmi.append(expected_value_model_improvement_kde(inference_data, data, C_FN, C_FP, z_lim, threshold, bias=bias, optimum_loss_current_info_value=optimum_loss))
        #     evoi.append(expected_value_observations_improvement_kde(inference_data, data, C_FN, C_FP, z_lim, threshold, bias
        #     =bias, optimum_loss_current_info_value=optimum_loss))

    # Output decisions_array to a CSV file
    decisions_df = pd.DataFrame(decisions_array, columns=['decision'])
    # decisions_df['evpi'] = evpi_array
    # if bias:
    #     decisions_df['evmi'] = evmi
    #     decisions_df['evoi'] = evoi
    decisions_df.to_csv(f'./code/output/decisions_array_{C_FN}_{C_FP}.csv', index=False)

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
    # plt.xlabel('Thresholds')
    plt.xlabel('Valor del umbral')
    plt.xlim(0, 5)
    plt.xticks(np.linspace(0, 5, 11))
    # plt.ylabel('Decision (0: False, 1: True)')
    plt.ylabel('Decision (0: Falso, 1: Verdadero)')
    plt.ylim(-0.1,1.1)
    plt.yticks([0, 1])
    # plt.title('Decision rule for different costs')
    plt.title('Regla de decisión para diferentes costos')
    # plt.legend([f'C_FN: {index[0]}, C_FP: {index[1]}' for index in decisions_df.index], title='Cost Parameters')
    plt.legend([f'C_FN: {index[0]}, C_FP: {index[1]}' for index in decisions_df.index], title='Parámetros de costo')
    plt.savefig(output_path)

def expected_value_of_perfect_information(inference_data, data, C_FN, C_FP, z_lim, threshold, bias=True):
    """
    Compute the Expected Value of Perfect Information (EVPI).

    Parameters:
    - inference_data: InferenceData object, containing the posterior samples
    - data: DataFrame, containing the data
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive
    - z_lim: float, threshold for decision making
    - thresholds_array: array-like, containing decision thresholds
    - bias: bool, whether to include bias in computations

    Returns:
    - float, EVPI value
    """
    
    # Compute the expected loss with perfect information
    posterior_samples = []
    X = data[['x1', 'x2', 'x3']].values
    
    samples = az.extract(inference_data, num_samples=200)
    a_values = samples["$a$"].values
    b_values = samples["$b$"].values
    sigma_b_values = samples["$\sigma_b$"]
    
    # Fit KDE to joint distribution of (a, b, sigma_b)
    joint_samples = np.vstack([a_values, b_values, sigma_b_values])
    kde = stats.gaussian_kde(joint_samples)

    try:
        for sample_a, sample_b, sample_sigma in zip(samples["$a$"].values, samples["$b$"].values, samples["$\sigma_b$"].values):
            z_samples_minus, z_samples_mean, z_samples_plus = evaluate_proportion_bias(sample_a, sample_b, sample_sigma, threshold, X)
            posterior_samples.append((z_samples_mean, z_samples_plus - z_samples_mean))
    except KeyError:
        for sample_a, sample_b in zip(samples["$a$"].values, samples["$b$"].values):
            z_samples_minus, z_samples_mean, z_samples_plus = evaluate_proportion_no_bias(sample_a, sample_b, threshold, X)
            posterior_samples.append((z_samples_mean, z_samples_plus - z_samples_mean))
    
    joint_samples = [sample for i, sample in enumerate(joint_samples.T) if posterior_samples[i][1] >= 1e-6 and not np.isnan(posterior_samples[i][0]) and not np.isnan(posterior_samples[i][1])]
    joint_samples = np.array(joint_samples)
    posterior_samples = [sample for sample in posterior_samples if sample[1] >= 1e-6 and not np.isnan(sample[0]) and not np.isnan(sample[1])]
    # Compute expected loss with perfect information

    weights = np.array([kde(sample) for sample in joint_samples]).squeeze()  # Define posterior_pdf accordingly
    weights /= weights.sum()

    expected_loss_perfect_info_value = expected_loss_perfect_info(posterior_samples, C_FN, C_FP, z_lim, weights, bias=bias)
    optimum_loss_current_info_value = optimum_loss_current_info(posterior_samples, C_FN, C_FP, z_lim, weights, bias=bias)

    # expected_loss_perfect_info_value = 0 if np.isnan(expected_loss_perfect_info_value) else expected_loss_perfect_info_value
    # optimum_loss_current_info_value = 0 if np.isnan(optimum_loss_current_info_value) else optimum_loss_current_info_value
    # Compute EVPI
    evpi = expected_loss_perfect_info_value - optimum_loss_current_info_value
    
    return evpi, optimum_loss_current_info_value

def expected_loss_perfect_info(posterior_samples, C_FN, C_FP, z_lim, weights, bias=True):
    """
    Compute the Expected Value of Perfect Information (EVPI).

    Parameters:
    - posterior_samples: array-like, samples from the posterior distribution of θ given y
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive
    - z_lim: float, threshold for decision making
    - bias: bool, whether to include bias in computations

    Returns:
    - float, EVPI value
    """
    
    # Compute expected loss with perfect information
    expected_loss_perfect_info = np.sum(np.array([
        min(
            bayes_risk([sample], z_lim, C_FN, C_FP, decision=False, bias=bias),
            bayes_risk([sample], z_lim, C_FN, C_FP, decision=True, bias=bias)
        ) for sample in posterior_samples
    ])*weights)
    
    return expected_loss_perfect_info

def optimum_loss_current_info(posterior_samples, C_FN, C_FP, z_lim, weights, bias=True):
    """
    Compute the Expected Value of Perfect Information (EVPI).

    Parameters:
    - posterior_samples: array-like, samples from the posterior distribution of θ given y
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive
    - z_lim: float, threshold for decision making
    - bias: bool, whether to include bias in computations

    Returns:
    - float, EVPI value
    """

    mean_z = np.sum(weights * np.array([sample[0] for sample in posterior_samples]))
    std_z = np.sqrt(np.sum(weights * (np.array([sample[1] for sample in posterior_samples]) ** 2)))
    
    optimum_loss_current_info = min(bayes_risk([(mean_z, std_z)], z_lim, C_FN, C_FP, decision=False, bias=bias), bayes_risk([(mean_z, std_z)], z_lim, C_FN, C_FP, decision=True, bias=bias))
    
    return optimum_loss_current_info
    # return 0

def expected_value_model_improvement(inference_data, data, C_FN, C_FP, z_lim, threshold, bias=True, num_bins=10, optimum_loss_current_info_value=0):
    """
    Compute the Expected Value of Model Improvement (EVMI).

    Parameters:
    - inference_data: InferenceData object, containing the posterior samples
    - data: DataFrame, containing the data
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive
    - z_lim: float, threshold for decision making
    - thresholds_array: array-like, containing decision thresholds
    - bias: bool, whether to include bias in computations

    Returns:
    - float, EVMI value
    """
    
    # Compute the expected loss with perfect information
    posterior_samples = []
    X = data[['x1', 'x2', 'x3']].values
    
    samples = az.extract(inference_data, num_samples=200)
    expected_loss_sigma = []
    a_values = samples["$a$"].values
    b_values = samples["$b$"].values
    sigma_b_values = samples["$\sigma_b$"].values
    bin_edges = np.linspace(sigma_b_values.min()-1E-6, sigma_b_values.max()+1E-6, num_bins + 1)

    

    n_samples = 0
    for i in range(num_bins):
        posterior_samples = []
        bin_min, bin_max = bin_edges[i], bin_edges[i + 1]
        mask = (sigma_b_values >= bin_min) & (sigma_b_values < bin_max)
        conditioned_samples_a = a_values[mask]
        conditioned_samples_b = b_values[mask]
        for sample_a, sample_b in zip(conditioned_samples_a, conditioned_samples_b):
            z_samples_minus, z_samples_mean, z_samples_plus = evaluate_proportion_bias(sample_a, sample_b, bin_min+(bin_max-bin_min)/2, threshold, X)
            posterior_samples.append((z_samples_mean, z_samples_plus - z_samples_mean))
        posterior_samples = [sample for sample in posterior_samples if sample[1] >= 1e-6 and not np.isnan(sample[0]) and not np.isnan(sample[1])]
        expected_value = expected_loss_perfect_info(posterior_samples, C_FN, C_FP, z_lim, threshold, bias=bias)*float(len(posterior_samples))
        if not np.isnan(expected_value):
            expected_loss_sigma.append(expected_value)
            n_samples = n_samples + len(posterior_samples)
    
    # Compute EVPI
    evmi = np.sum(expected_loss_sigma)/n_samples - optimum_loss_current_info_value
    
    return evmi

def expected_value_model_improvement_kde(inference_data, data, C_FN, C_FP, z_lim, threshold, bias=True, num_samples=200, optimum_loss_current_info_value=0):
    """
    Compute the Expected Value of Model Improvement (EVMI) using KDE to model the joint distribution.

    Parameters:
    - inference_data: InferenceData object, containing the posterior samples
    - data: DataFrame, containing the data
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive
    - z_lim: float, threshold for decision making
    - threshold: float, decision threshold
    - bias: bool, whether to include bias in computations
    - num_samples: int, number of samples for KDE evaluation
    - optimum_loss_current_info_value: float, optimal loss with current information

    Returns:
    - float, EVMI value
    """
    
    # Extract posterior samples
    samples = az.extract(inference_data, num_samples=num_samples)
    a_values = samples["$a$"].values
    b_values = samples["$b$"].values
    sigma_b_values = samples["$\sigma_b$"]
    X = data[['x1', 'x2', 'x3']].values
    
    # Fit KDE to joint distribution of (a, b, sigma_b)
    joint_samples = np.vstack([a_values, b_values, sigma_b_values])
    kde = kt.gaussian_kde(joint_samples)
    kde.set_bandwidth(bw_method='cv', bw_type='diagonal')
    
    # Define a range of sigma_b values for integration
    sigma_b_range = np.linspace(sigma_b_values.min(), sigma_b_values.max(), int(num_samples/10))
    expected_loss_sigma = []
    a_b_samples = kde.conditional_resample(int(num_samples/5), x_cond=np.array(sigma_b_range), dims_cond=[2])
    
    for i, sigma_b in enumerate(sigma_b_range):
        posterior_samples = []
        for sample_a, sample_b in a_b_samples[i]:
            z_samples_minus, z_samples_mean, z_samples_plus = evaluate_proportion_bias(sample_a, sample_b, sigma_b, threshold, X)
            posterior_samples.append((z_samples_mean, z_samples_plus - z_samples_mean))
        
        posterior_samples = [sample for sample in posterior_samples if sample[1] >= 1e-6 and not np.isnan(sample[0]) and not np.isnan(sample[1])]
        
        expected_value = expected_loss_perfect_info(posterior_samples, C_FN, C_FP, z_lim, threshold, bias=bias)
        if not np.isnan(expected_value):
            expected_loss_sigma.append(expected_value)
    
    # Compute posterior density of sigma_b
    p_sigma_b = kde.evaluate(np.vstack([np.mean(a_b_samples,axis=1).T,sigma_b_range]))  # Extract the relevant density
    p_sigma_b /= p_sigma_b.sum()  # Normalize to sum to 1

    # Compute EVMI with proper weighting
    evmi = np.sum(np.array(expected_loss_sigma) * p_sigma_b) - optimum_loss_current_info_value

    return evmi

def expected_value_observations_improvement(inference_data, data, C_FN, C_FP, z_lim, threshold, bias=True, num_bins=10, optimum_loss_current_info_value=0):
    """
    Compute the Expected Value of Observations Improvement (EVOI).

    Parameters:
    - inference_data: InferenceData object, containing the posterior samples
    - data: DataFrame, containing the data
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive
    - z_lim: float, threshold for decision making
    - thresholds_array: array-like, containing decision thresholds
    - bias: bool, whether to include bias in computations

    Returns:
    - float, EVMI value
    """
    
    # Compute the expected loss with perfect information
    posterior_samples = []
    X = data[['x1', 'x2', 'x3']].values
    
    samples = az.extract(inference_data, num_samples=200)
    a_values = samples["$a$"].values
    b_values = samples["$b$"].values
    sigma_b_values = samples["$\sigma_b$"].values

    # Define 2D non-overlapping bins for (a, b)
    a_bins = np.linspace(a_values.min()-1E-6, a_values.max()+1E-6, num_bins + 1)
    b_bins = np.linspace(b_values.min()-1E-6, b_values.max()+1E-6, num_bins + 1)
    expected_loss_params = []
    
    n_samples = 0
    for i in range(num_bins):
        a_min, a_max = a_bins[i], a_bins[i + 1]
        for j in range(num_bins):
            # Define bin edges
            b_min, b_max = b_bins[j], b_bins[j + 1]

            posterior_samples = []
            mask = (a_values >= a_min) & (a_values < a_max) & (b_values >= b_min) & (b_values < b_max)
            conditional_sigma_values = sigma_b_values[mask]

            if np.sum(mask) == 0:  # Skip empty bins
                continue

            for sample_sigma in conditional_sigma_values:
                z_samples_minus, z_samples_mean, z_samples_plus = evaluate_proportion_bias(a_min+(a_max-a_min)/2, b_min+(b_max-b_min)/2, sample_sigma, threshold, X)
                posterior_samples.append((z_samples_mean, z_samples_plus - z_samples_mean))
            posterior_samples = [sample for sample in posterior_samples if sample[1] >= 1e-6 and not np.isnan(sample[0]) and not np.isnan(sample[1])]
            expected_value = expected_loss_perfect_info(posterior_samples, C_FN, C_FP, z_lim, threshold, bias=bias)*float(len(posterior_samples))
            if not np.isnan(expected_value):
                n_samples = n_samples + len(posterior_samples)
                expected_loss_params.append(expected_value)
        

    # Compute EVPI
    evoi = np.sum(expected_loss_params)/n_samples - optimum_loss_current_info_value
    
    return evoi

def expected_value_observations_improvement_kde(inference_data, data, C_FN, C_FP, z_lim, threshold, bias=True, num_samples=200, optimum_loss_current_info_value=0):
    """
    Compute the Expected Value of Observations Improvement (EVOI) using KDE to model the joint distribution.

    Parameters:
    - inference_data: InferenceData object, containing the posterior samples
    - data: DataFrame, containing the data
    - C_FN: float, cost of false negative
    - C_FP: float, cost of false positive
    - z_lim: float, threshold for decision making
    - threshold: float, decision threshold
    - bias: bool, whether to include bias in computations
    - num_samples: int, number of samples for KDE evaluation
    - optimum_loss_current_info_value: float, optimal loss with current information

    Returns:
    - float, EVOI value
    """
    
    # Extract posterior samples
    samples = az.extract(inference_data, num_samples=num_samples)
    a_values = samples["$a$"].values
    b_values = samples["$b$"].values
    sigma_b_values = samples["$\sigma_b$"]
    X = data[['x1', 'x2', 'x3']].values
    
    # Fit KDE to joint distribution of (a, b, sigma_b)
    joint_samples = np.vstack([a_values, b_values, sigma_b_values])
    kde = kt.gaussian_kde(joint_samples)
    kde.set_bandwidth(bw_method='cv', bw_type='diagonal')
    
    # Define a 2D grid of (a, b) values
    a_range = np.linspace(a_values.min(), a_values.max(), int(num_samples / 20))
    b_range = np.linspace(b_values.min(), b_values.max(), int(num_samples / 20))
    A_grid, B_grid = np.meshgrid(a_range, b_range)
    ab_grid = np.vstack([A_grid.ravel(), B_grid.ravel()])
    sigma_b_samples = kde.conditional_resample(int(num_samples / 5), x_cond=ab_grid.T, dims_cond=[0, 1]).squeeze()

    # Compute posterior density p(a, b) over this grid
    p_ab = kde.evaluate(np.vstack([ab_grid[0], ab_grid[1], np.mean(sigma_b_samples, axis=1)]))
    p_ab /= p_ab.sum()  # Normalize to sum to 1

    expected_loss_ab = []

    # Loop over the (a, b) grid
    for i, (sample_a, sample_b) in enumerate(ab_grid.T):
        if p_ab[i] < 1e-6:
            expected_loss_ab.append(0)
        else:
            posterior_samples = []
            for sample_sigma in sigma_b_samples[i]:
                z_samples_minus, z_samples_mean, z_samples_plus = evaluate_proportion_bias(sample_a, sample_b, sample_sigma, threshold, X)
                posterior_samples.append((z_samples_mean, z_samples_plus - z_samples_mean))
            
            posterior_samples = [sample for sample in posterior_samples if sample[1] >= 1e-6 and not np.isnan(sample[0]) and not np.isnan(sample[1])]
            
            expected_value = expected_loss_perfect_info(posterior_samples, C_FN, C_FP, z_lim, threshold, bias=bias)
            if not np.isnan(expected_value):
                expected_loss_ab.append(expected_value)
            else:
                expected_loss_ab.append(0)
    
    # Compute posterior density of (a, b)
    evoi = np.sum(np.array(expected_loss_ab) * p_ab) - optimum_loss_current_info_value
    
    return evoi

if __name__ == "__main__":
    # Load the posterior samples
    inference_data_bias = az.from_netcdf("./code/output/results/calibrate_bias_ishigami_3.az")
    inference_data_no_bias = az.from_netcdf("./code/output/results/calibrate_no_bias_ishigami_3.az")
    data_path = './code/input/data/ishigami_dataset.csv'

    # Set the decision threshold
    z_lim = 0.5
    threshold_array = np.linspace(0, 5, 10)
    # is the proportion of observations greater than the threshold larger than 0.5?

    # Set the costs
    C_FN_array = [1.0, 10.0]
    C_FP_array = [1.0, 10.0]

    # Compute the optimal decision rule
    data = pd.read_csv(data_path)

    # Extract features
    decisions_no_bias = compare_decision_costs(inference_data_no_bias, data, C_FN_array, C_FP_array, z_lim, threshold_array, output_path='./code/output/results/decisions_no_bias.csv', bias=False)
    plot_decision_costs_comparison(decisions_no_bias, output_path='./code/output/figures/decisions_ishigami_no_bias.pdf')
    decisions_bias = compare_decision_costs(inference_data_bias, data, C_FN_array, C_FP_array, z_lim, threshold_array, output_path='./code/output/results/decisions_ishigami_bias.csv', bias=True)
    # decisions_bias = pd.read_csv('./code/output/results/decisions_bias.csv', header=[0], index_col=[0, 1])
    plot_decision_costs_comparison(decisions_bias, output_path='./code/output/figures/decisions_ishigami_bias.pdf')
