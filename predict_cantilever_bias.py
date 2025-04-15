import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import chaospy as cp
import numpy as np

from cantilever_function import CantileverBeam

def fit_polynomial_chaos(cantilever, data_inputs, pce_order, E, sigma_E):
    E_dist = cp.Normal(E, sigma_E)

    # generate the polynomial chaos expansion
    expansion = cp.generate_expansion(pce_order, E_dist)

    # generate quadrature nodes and weights
    sparse_quads = cp.generate_quadrature(
        pce_order, E_dist, rule="Gaussian"
    )
    # evaluate the model at the quadrature nodes
    sparse_evals = []
    for node in sparse_quads[0][0]:
        cantilever.current_young_modulus = node
        sparse_evals.append(cantilever.deflection(data_inputs))

    # fit the polynomial chaos expansion
    fitted_polynomial = cp.fit_quadrature(
        expansion, sparse_quads[0], sparse_quads[1], sparse_evals
    )
    # Evaluate the Cantilever function at the data inputs
    prediction_mean = cp.E(fitted_polynomial, E_dist)
    prediction_std = cp.Std(fitted_polynomial, E_dist)

    return prediction_mean, prediction_std

def predict_cantilever_bias(inference_data: az.InferenceData, output_path: str, data_inputs_path: str, degree: int = 3, n_samples: int = 200, pce_order: int = 2):
    """
    Predict the Cantilever function without bias term
    :param inference_data: InferenceData object
    :param output_path: str
    :param degree: int
    :param pce_order: int
    :return: None
    """
    # Load the model
    cantilever = CantileverBeam(length=5.0, height=0.3, young_modulus=30e9, yield_stress=1e24, softening_factor=0.99)

    # Read data inputs from CSV
    data_inputs = pd.read_csv(data_inputs_path).values
    # Extract samples from inference_data
    samples = inference_data.posterior

    # Initialize an empty list to store predictions
    predictions_mean_list = []
    predictions_std_list = []

    # Number of samples to use
    samples_array= az.extract(inference_data, num_samples=n_samples)

    for E, sigma_E in zip(samples_array["$E$"].values, samples_array["$\sigma_E$"].values):
        
        prediction_mean, prediction_std = fit_polynomial_chaos(cantilever, data_inputs[:, 0], pce_order, E, sigma_E)

        predictions_mean_list.append(prediction_mean)
        predictions_std_list.append(prediction_std)

    # Convert predictions to a DataFrame
    predictions_mean_df = pd.DataFrame(predictions_mean_list)
    predictions_std_df = pd.DataFrame(predictions_std_list)

    # Output the predictions to CSV
    predictions_mean_df.to_csv(output_path.replace(".csv", "_mean.csv"), index=False)
    predictions_std_df.to_csv(output_path.replace(".csv", "_std.csv"), index=False)

    mean_E = samples_array["$E$"].values.mean()
    mean_sigma_E = samples_array["$\sigma_E$"].values.mean()

    mean_prediction, std_prediction = fit_polynomial_chaos(cantilever, data_inputs[:, 0], pce_order, mean_E, mean_sigma_E)
    plus_prediction = mean_prediction + np.sqrt(np.square(std_prediction) +0.01**2)
    minus_prediction = mean_prediction - np.sqrt(np.square(std_prediction) +0.01**2)

    plt.figure()
    sorted_indices = np.argsort(data_inputs[:, 0])
    sorted_data_inputs = data_inputs[sorted_indices]
    sorted_mean_prediction = mean_prediction[sorted_indices]
    sorted_plus_prediction = plus_prediction[sorted_indices]
    sorted_minus_prediction = minus_prediction[sorted_indices]

    plt.plot(sorted_data_inputs[:, 0], sorted_data_inputs[:, 2], 'k.', label="Data")
    plt.plot(sorted_data_inputs[:, 0], sorted_mean_prediction, 'g-', label="Mean")
    plt.plot(sorted_data_inputs[:, 0], sorted_plus_prediction, 'g--', label=r"Mean$\pm\sigma$")
    plt.plot(sorted_data_inputs[:, 0], sorted_minus_prediction, 'g--', label="_no_legend_")
    plt.legend()
    plt.xlabel("Load")
    plt.ylabel("Deflection")

    plt.savefig(output_path[:-3]+"pdf")


if __name__ == "__main__":

    # Load the inference data
    inference_data = az.from_netcdf("./code/output/results/calibrate_bias_cantilever.az")

    # Define the path to the data inputs
    data_inputs_path = "./code/input/data/cantilever_dataset.csv"

    # Define the path to save the predictions
    output_path = "./code/output/results/cantilever_bias_predictions.csv"

    # Predict the Cantilever function without bias term
    predict_cantilever_bias(inference_data, output_path, data_inputs_path, n_samples=500)