import arviz as az
import numpy as np
import pandas as pd
import chaospy as cp
import matplotlib.pyplot as plt

from cantilever_function import CantileverBeam


def evaluate_true_deflection(threshold, dataset_path='./input/data/cantilever_dataset.csv'):
    """Get the true deflection of points above the threshold."""
    # Load dataset
    dataset = pd.read_csv(dataset_path)

    # Use the last row as true values
    true_values= {"length":5.0, "height":0.3, "young_modulus":dataset["Young's Modulus"].values[-1], "yield_stress":20e6, "softening_factor":0.95}

    # Create an instance of the Cantilever function
    cantilever = CantileverBeam(**true_values) 

    threshold_array = []
    for E_module, load_value in zip(dataset["Young's Modulus"].values, dataset["Load"].values):
        cantilever.current_young_modulus = E_module
        threshold_array.append(threshold < cantilever.deflection(load_value))

    return threshold_array

def get_deflection_no_bias(inference_bias, threshold, load, sigma=0.01):

    mean_E_no_bias = inference_bias.posterior["$E$"].mean().values

    return evaluate_deflection_no_bias(mean_E_no_bias, threshold, load, sigma)

def evaluate_deflection_no_bias(sample_E_no_bias, threshold, load, sigma=0.01, values = False):
    """Get the deflection of points above the threshold for the no bias model."""

    beam_values= {"length":5.0, "height":0.3, "young_modulus":sample_E_no_bias, "yield_stress":20e6, "softening_factor":0.95}
    cantilever_no_bias = CantileverBeam(**beam_values)

    cantilever_no_bias_plus_std = lambda x: cantilever_no_bias.deflection(x) + sigma
    cantilever_no_bias_minus_std = lambda x: cantilever_no_bias.deflection(x) - sigma

    if values:
        return  cantilever_no_bias_minus_std(load), cantilever_no_bias.deflection(load), cantilever_no_bias_plus_std(load)
    else:
        deflection_no_bias = (threshold < cantilever_no_bias.deflection(load))
        deflection_no_bias_plus_std = (threshold < cantilever_no_bias_plus_std(load))
        deflection_no_bias_minus_std = (threshold < cantilever_no_bias_minus_std(load))

        return deflection_no_bias_minus_std, deflection_no_bias, deflection_no_bias_plus_std

def evaluate_deflection_bias(sample_E_bias, sample_sigma_bias, threshold, load, sigma=0.01, pce_order=2, values = False):
    """Get the deflection of points above the threshold for the no bias model."""

    beam_values= {"length":5.0, "height":0.3, "young_modulus":sample_E_bias, "yield_stress":20e6, "softening_factor":0.95}
    cantilever_bias = CantileverBeam(**beam_values)

    b_dist = cp.Normal(sample_E_bias, sample_sigma_bias)
    expansion = cp.generate_expansion(pce_order, b_dist)
    sparse_quads = cp.generate_quadrature(pce_order, b_dist, rule="Gaussian")
    sparse_evals = []
    for node in sparse_quads[0][0]:
        cantilever_bias.current_young_modulus = node
        sparse_evals.append(cantilever_bias.deflection(load))

    fitted_sparse = cp.fit_quadrature(expansion, sparse_quads[0], sparse_quads[1], sparse_evals)

    cantilever_bias_mean =lambda x: cp.E(fitted_sparse, b_dist)
    cantilever_bias_std = cp.Std(fitted_sparse, b_dist)
    cantilever_bias_plus_std =lambda x: cantilever_bias_mean(x) + cantilever_bias_std
    cantilever_bias_minus_std =lambda x: cantilever_bias_mean(x) - cantilever_bias_std

    if values:
        return cantilever_bias_minus_std(load),cantilever_bias_mean(load),  cantilever_bias_plus_std(load)
    else:
        deflection_bias = (threshold < cantilever_bias_mean(load))
        deflection_bias_plus_std = (threshold < cantilever_bias_plus_std(load))
        deflection_bias_minus_std = (threshold < cantilever_bias_minus_std(load))

        return deflection_bias_minus_std, deflection_bias, deflection_bias_plus_std


def get_deflection_bias(inference_bias, threshold, load, sigma=0.01, pce_order=2):

    mean_E_bias = inference_bias.posterior["$E$"].mean().values
    mean_sigma_bias = inference_bias.posterior["$\sigma_E$"].mean().values

    return evaluate_deflection_bias(mean_E_bias, mean_sigma_bias, threshold, load, sigma, pce_order)

    

def threshold_analysis(get_deflection_function, inference_data, thresholds, X, sigma=0.01, **kwargs):

    deflections_minus_std = []
    deflections = []
    deflections_plus_std = []

    for threshold in thresholds:
        deflection_minus_std, deflection, deflection_plus_std = get_deflection_function(inference_data, threshold, X, sigma=sigma, **kwargs)
        deflections_minus_std.append(deflection_minus_std)
        deflections.append(deflection)
        deflections_plus_std.append(deflection_plus_std)

    return deflections_minus_std, deflections, deflections_plus_std

def plot_deflection_no_bias(inference_no_bias, X, sigma=0.01):

    thresholds = np.linspace(0,0.3,50)
    deflections_no_bias_minus_std, deflections_no_bias, deflections_no_bias_plus_std = threshold_analysis(get_deflection_no_bias, inference_no_bias, thresholds, X, sigma)

    # plt.plot(thresholds, deflections_no_bias, label="No Bias", color='blue')
    plt.plot(thresholds, deflections_no_bias, label="Sin sesgo", color='blue')
    plt.plot(thresholds, deflections_no_bias_minus_std, linestyle='--', color='blue')
    plt.plot(thresholds, deflections_no_bias_plus_std, linestyle='--', color='blue')
    plt.fill_between(thresholds, deflections_no_bias_minus_std, deflections_no_bias_plus_std, color='blue', alpha=0.3)
    # plt.xlabel("Threshold")
    plt.xlabel("Umbral")
    # plt.ylabel("Proportion of Points Above Threshold")
    plt.ylabel("Proporción de puntos por encima del umbral")
    # plt.title("Proportion of Points Above Threshold for No Bias Model")
    plt.title("Proporción de puntos por encima del umbral para el modelo sin sesgo")
    plt.legend()

    plt.savefig("./output/figures/cantilever_deflection_no_bias.pdf")
    plt.close()

    return deflections_no_bias_minus_std, deflections_no_bias, deflections_no_bias_plus_std

def plot_deflection_bias(inference_bias, X, sigma=0.01, pce_order=2):

    thresholds = np.linspace(0,0.3,50)
    deflections_bias_minus_std, deflections_bias, deflections_bias_plus_std = threshold_analysis(get_deflection_bias, inference_bias, thresholds, X, sigma, pce_order=pce_order)

    # plt.plot(thresholds, deflections_bias, label="Bias", color='red')
    plt.plot(thresholds, deflections_bias, label="Con sesgo", color='red')
    plt.plot(thresholds, deflections_bias_minus_std, linestyle='--', color='red')
    plt.plot(thresholds, deflections_bias_plus_std, linestyle='--', color='red')
    plt.fill_between(thresholds, deflections_bias_minus_std, deflections_bias_plus_std, color='red', alpha=0.3)
    # plt.xlabel("Threshold")
    plt.xlabel("Umbral")
    # plt.ylabel("Proportion of Points Above Threshold")
    plt.ylabel("Proporción de puntos por encima del umbral")
    # plt.title("Proportion of Points Above Threshold for Bias Model")
    plt.title("Proporción de puntos por encima del umbral para el modelo con sesgo")
    plt.legend()
    
    plt.savefig("./output/figures/cantilever_deflection_bias.pdf")
    plt.close()

    return deflections_bias_minus_std, deflections_bias, deflections_bias_plus_std

def validate_true_deflection(thresholds, dataset_path='./input/data/cantilever_dataset.csv'):

    true_deflections = []
    for threshold in thresholds:
        true_deflection = evaluate_true_deflection(threshold, dataset_path)
        true_deflections.append(true_deflection)

    plt.plot(thresholds, [np.sum(itrue_deflections) for itrue_deflections in true_deflections], color='black')
    # plt.xlabel("Threshold")
    plt.xlabel("Umbral")
    # plt.ylabel("True Proportion")
    plt.ylabel("Proporción verdadera")
    # plt.title("True Proportion of Points Above Threshold")
    plt.title("Proporción verdadera de puntos por encima del umbral")

    plt.savefig("./output/figures/cantilever_true_deflection.pdf")
    plt.close()

    return true_deflections

def compare_deflections(inference_no_bias, inference_bias, X, sigma=0.01, pce_order=2):

    thresholds = np.linspace(0.0,0.3,50)
    true_deflections = validate_true_deflection(thresholds)
    deflections_no_bias_minus_std, deflections_no_bias, deflections_no_bias_plus_std = threshold_analysis(get_deflection_no_bias, inference_no_bias, thresholds, X, sigma=sigma)
    deflections_bias_minus_std, deflections_bias, deflections_bias_plus_std = threshold_analysis(get_deflection_bias, inference_bias, thresholds, X, sigma=sigma, pce_order=pce_order)

    # plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in true_deflections], label="True Proportion", color='black')
    plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in true_deflections], label="Proporción verdadera", color='black')
    # plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in deflections_no_bias], label="No Bias", color='blue')
    plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in deflections_no_bias], label="Sin sesgo", color='blue')
    plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in deflections_no_bias_minus_std], linestyle='--', color='blue')
    plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in deflections_no_bias_plus_std], linestyle='--', color='blue')
    plt.fill_between(thresholds, [np.sum(i_deflections) for i_deflections in deflections_no_bias_minus_std], [np.sum(i_deflections) for i_deflections in deflections_no_bias_plus_std], color='blue', alpha=0.3)
    # plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in deflections_bias], label="Bias", color='red')
    plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in deflections_bias], label="Con sesgo", color='red')
    plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in deflections_bias_minus_std], linestyle='--', color='red')
    plt.plot(thresholds, [np.sum(i_deflections) for i_deflections in deflections_bias_plus_std], linestyle='--', color='red')
    plt.fill_between(thresholds, [np.sum(i_deflections) for i_deflections in deflections_bias_minus_std], [np.sum(i_deflections) for i_deflections in deflections_bias_plus_std], color='red', alpha=0.3)
    # plt.xlabel("Threshold")
    plt.xlabel("Umbral")
    # plt.ylabel("Number of Loads Above Threshold")
    plt.ylabel("Número de cargas por encima del umbral")
    # plt.title("Number of Loads Above Threshold for No Bias and Bias Models")
    plt.title("Número de cargas por encima del umbral para los modelos sin sesgo y con sesgo")
    plt.legend()

    plt.savefig("./output/figures/cantilever_deflection_comparison.pdf")
    plt.close()

    return true_deflections, deflections_no_bias_minus_std, deflections_no_bias, deflections_no_bias_plus_std, deflections_bias_minus_std, deflections_bias, deflections_bias_plus_std

if __name__ == "__main__":

    # Load data from CSV file
    data_path = './input/data/cantilever_dataset.csv'
    data = pd.read_csv(data_path)

    # Extract features
    X = data["Load"].values

    # Set the threshold
    threshold = 0.1

    # Get the true deflection of points above the threshold
    print("True deflection: " + str(evaluate_true_deflection(threshold)))

    # Validate the true deflection of points above the threshold

    # Load the inference data

    inference_no_bias = az.from_netcdf("./output/results/calibrate_no_bias_cantilever.az")
    inference_bias = az.from_netcdf("./output/results/calibrate_bias_cantilever.az")

    # Evaluate the point no bias
    compare_deflections(inference_no_bias, inference_bias, X)



    
