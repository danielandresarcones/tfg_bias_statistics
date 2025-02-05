import arviz as az
import numpy as np
import pandas as pd
import chaospy as cp
import matplotlib.pyplot as plt

from ishigami_function_3d import IshigamiFunctionPolynomial, IshigamiFunction

def evaluate_threshold(regressor, threshold, X):
    """Evaluate the threshold on the regressor."""
    X = np.array(X)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("Input X must be a 2D array with 3 columns.")
    
    y_pred = np.zeros(X.shape[0])
    y_indices = np.zeros(X.shape[0])
    y_pred = regressor(X)
    y_indices[y_pred > threshold] = 1
    return y_indices

def get_true_proporition(threshold, X):
    """Get the true proportion of points above the threshold."""
    true_values = {"a": 7.0, "b": 0.1}

    # Create an instance of the Ishigami function
    ishigami = IshigamiFunction(**true_values)

    true_threshold_indices = evaluate_threshold(ishigami, threshold, X)
    true_proporition = true_threshold_indices.sum() / true_threshold_indices.size
    # print(f"True proportion of points above threshold: {true_proporition:.2f}")

    return true_proporition

def get_proportion_no_bias(inference_no_bias, threshold, X, sigma=0.1):
    """Get the proportion of points above the threshold for the no bias model."""
    mean_a_no_bias = inference_no_bias.posterior["$a$"].mean().values
    mean_b_no_bias = inference_no_bias.posterior["$b$"].mean().values

    ishigami_no_bias = IshigamiFunctionPolynomial(c=3)
    ishigami_no_bias.a = mean_a_no_bias
    ishigami_no_bias.b = mean_b_no_bias

    ishigami_no_bias_plus_std = lambda x: ishigami_no_bias(x) + sigma
    ishigami_no_bias_minus_std = lambda x: ishigami_no_bias(x) - sigma

    threshold_indices_no_bias = evaluate_threshold(ishigami_no_bias, threshold, X)
    threshold_indices_no_bias_plus_std = evaluate_threshold(ishigami_no_bias_plus_std, threshold, X)
    threshold_indices_no_bias_minus_std = evaluate_threshold(ishigami_no_bias_minus_std, threshold, X)
    proportion_no_bias = threshold_indices_no_bias.sum() / threshold_indices_no_bias.size
    proportion_no_bias_plus_std = threshold_indices_no_bias_plus_std.sum() / threshold_indices_no_bias_plus_std.size
    proportion_no_bias_minus_std = threshold_indices_no_bias_minus_std.sum() / threshold_indices_no_bias_minus_std.size

    return proportion_no_bias_minus_std, proportion_no_bias, proportion_no_bias_plus_std

def get_proportion_bias(inference_bias, threshold, X, sigma=0.1, pce_order=2):

    mean_a_bias = inference_bias.posterior["$a$"].mean().values
    mean_b_bias = inference_bias.posterior["$b$"].mean().values
    mean_sigma_b = inference_bias.posterior["$\sigma_b$"].mean().values

    ishigami_bias = IshigamiFunctionPolynomial(c=3)
    ishigami_bias.a = mean_a_bias
    ishigami_bias.b = mean_b_bias

    b_dist = cp.Normal(mean_b_bias, mean_sigma_b)
    expansion = cp.generate_expansion(pce_order, b_dist)
    sparse_quads = cp.generate_quadrature(pce_order, b_dist, rule="Gaussian")
    sparse_evals = []
    for node in sparse_quads[0][0]:
        ishigami_bias.b = node
        sparse_evals.append(ishigami_bias(X))

    fitted_sparse = cp.fit_quadrature(expansion, sparse_quads[0], sparse_quads[1], sparse_evals)

    ishigami_bias_mean =lambda x: cp.E(fitted_sparse, b_dist)
    ishigami_bias_std = cp.Std(fitted_sparse, b_dist)
    ishigami_bias_plus_std =lambda x: ishigami_bias_mean(x) + ishigami_bias_std
    ishigami_bias_minus_std =lambda x: ishigami_bias_mean(x) - ishigami_bias_std

    threshold_indices_bias = evaluate_threshold(ishigami_bias_mean, threshold, X)
    threshold_indices_bias_plus_std = evaluate_threshold(ishigami_bias_plus_std, threshold, X)
    threshold_indices_bias_minus_std = evaluate_threshold(ishigami_bias_minus_std, threshold, X)
    proportion_bias = threshold_indices_bias.sum() / threshold_indices_bias.size
    proportion_bias_plus_std = threshold_indices_bias_plus_std.sum() / threshold_indices_bias_plus_std.size
    proportion_bias_minus_std = threshold_indices_bias_minus_std.sum() / threshold_indices_bias_minus_std.size

    return proportion_bias_minus_std, proportion_bias, proportion_bias_plus_std

def threshold_analysis(get_proportion_function, inference_data, thresholds, X, sigma=0.1, **kwargs):

    proportions_minus_std = []
    proportions = []
    proportions_plus_std = []

    for threshold in thresholds:
        proportion_minus_std, proportion, proportion_plus_std = get_proportion_function(inference_data, threshold, X, sigma=sigma, **kwargs)
        proportions_minus_std.append(proportion_minus_std)
        proportions.append(proportion)
        proportions_plus_std.append(proportion_plus_std)

    return proportions_minus_std, proportions, proportions_plus_std

def plot_proportion_no_bias(inference_no_bias, X, sigma=0.1):

    thresholds = np.linspace(-10,10,50)
    proportions_no_bias_minus_std, proportions_no_bias, proportions_no_bias_plus_std = threshold_analysis(get_proportion_no_bias, inference_no_bias, thresholds, X, sigma)

    plt.plot(thresholds, proportions_no_bias, label="No Bias", color='blue')
    plt.plot(thresholds, proportions_no_bias_minus_std, linestyle='--', color='blue')
    plt.plot(thresholds, proportions_no_bias_plus_std, linestyle='--', color='blue')
    plt.fill_between(thresholds, proportions_no_bias_minus_std, proportions_no_bias_plus_std, color='blue', alpha=0.3)
    plt.xlabel("Threshold")
    plt.ylabel("Proportion of Points Above Threshold")
    plt.title("Proportion of Points Above Threshold for No Bias Model")
    plt.legend()
    plt.show()

    plt.savefig("./code/output/figures/ishigami_proportion_no_bias.png")
    plt.close()

    return proportions_no_bias_minus_std, proportions_no_bias, proportions_no_bias_plus_std

def plot_proportion_bias(inference_bias, X, sigma=0.1, pce_order=2):

    thresholds = np.linspace(-10,10,50)
    proportions_bias_minus_std, proportions_bias, proportions_bias_plus_std = threshold_analysis(get_proportion_bias, inference_bias, thresholds, X, sigma, pce_order=pce_order)

    plt.plot(thresholds, proportions_bias, label="Bias", color='red')
    plt.plot(thresholds, proportions_bias_minus_std, linestyle='--', color='red')
    plt.plot(thresholds, proportions_bias_plus_std, linestyle='--', color='red')
    plt.fill_between(thresholds, proportions_bias_minus_std, proportions_bias_plus_std, color='red', alpha=0.3)
    plt.xlabel("Threshold")
    plt.ylabel("Proportion of Points Above Threshold")
    plt.title("Proportion of Points Above Threshold for Bias Model")
    plt.legend()
    plt.show()
    
    plt.savefig("./code/output/figures/ishigami_proportion_bias.png")
    plt.close()

    return proportions_bias_minus_std, proportions_bias, proportions_bias_plus_std

def validate_true_proportion(X):

    thresholds = np.linspace(-10,10,50)
    true_proportions = []
    for threshold in thresholds:
        true_proportion = get_true_proporition(threshold, X)
        true_proportions.append(true_proportion)

    plt.plot(thresholds, true_proportions)
    plt.xlabel("Threshold")
    plt.ylabel("True Proportion")
    plt.title("True Proportion of Points Above Threshold")
    # plt.show()

    plt.savefig("./code/output/figures/ishigami_true_proportion.png")
    plt.close()

    return true_proportions

def compare_proportions(inference_no_bias, inference_bias, X, sigma=0.1, pce_order=2):

    thresholds = np.linspace(-10,10,50)
    true_proportions = validate_true_proportion(X)
    proportions_no_bias_minus_std, proportions_no_bias, proportions_no_bias_plus_std = threshold_analysis(get_proportion_no_bias, inference_no_bias, thresholds, X, sigma=sigma)
    proportions_bias_minus_std, proportions_bias, proportions_bias_plus_std = threshold_analysis(get_proportion_bias, inference_bias, thresholds, X, sigma=sigma, pce_order=pce_order)

    plt.plot(thresholds, true_proportions, label="True Proportion", color='black')
    plt.plot(thresholds, proportions_no_bias, label="No Bias", color='blue')
    plt.plot(thresholds, proportions_no_bias_minus_std, linestyle='--', color='blue')
    plt.plot(thresholds, proportions_no_bias_plus_std, linestyle='--', color='blue')
    plt.fill_between(thresholds, proportions_no_bias_minus_std, proportions_no_bias_plus_std, color='blue', alpha=0.3)
    plt.plot(thresholds, proportions_bias, label="Bias", color='red')
    plt.plot(thresholds, proportions_bias_minus_std, linestyle='--', color='red')
    plt.plot(thresholds, proportions_bias_plus_std, linestyle='--', color='red')
    plt.fill_between(thresholds, proportions_bias_minus_std, proportions_bias_plus_std, color='red', alpha=0.3)
    plt.xlabel("Threshold")
    plt.ylabel("Proportion of Points Above Threshold")
    plt.title("Proportion of Points Above Threshold for No Bias and Bias Models")
    plt.legend()
    plt.show()

    plt.savefig("./code/output/figures/ishigami_proportion_comparison.png")
    plt.close()

    return true_proportions, proportions_no_bias_minus_std, proportions_no_bias, proportions_no_bias_plus_std, proportions_bias_minus_std, proportions_bias, proportions_bias_plus_std

if __name__ == "__main__":

    # Load data from CSV file
    data_path = './code/input/data/ishigami_dataset.csv'
    data = pd.read_csv(data_path)

    # Extract features
    X = data[['x1', 'x2', 'x3']].values

    # Set the threshold
    threshold = 7.0

    # Get the true proportion of points above the threshold
    get_true_proporition(threshold, X)

    # Validate the true proportion of points above the threshold

    validate_true_proportion(X)

    # Load the inference data

    inference_no_bias = az.from_netcdf("./code/output/results/calibrate_no_bias_ishigami_3.az")
    inference_bias = az.from_netcdf("./code/output/results/calibrate_bias_ishigami_3.az")

    # Evaluate the point no bias
    compare_proportions(inference_no_bias, inference_bias, X)



    
