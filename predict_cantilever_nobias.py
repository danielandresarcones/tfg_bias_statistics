import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from cantilever_function import CantileverBeam


def predict_cantilever_nobias(inference_data: az.InferenceData, output_path: str, data_inputs_path: str, sigma=0.01, n_samples: int = 200):
    """
    Predict the Canitlever function without bias term
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

    # Initialize an empty list to store predictions
    predictions = []

    # Number of samples to use
    samples_array= az.extract(inference_data, num_samples=n_samples)

    for E in samples_array["$E$"].values:
        # Update parameters a and b from the samples
        cantilever.current_young_modulus = E

        # Evaluate the Canitlever function at the data inputs
        prediction = cantilever.deflection(data_inputs[:,0])
        predictions.append(prediction)

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Output the predictions to CSV
    predictions_df.to_csv(output_path, index=False)

    mean_E = samples_array["$E$"].values.mean()
    std_E = samples_array["$E$"].values.std()

    cantilever.current_young_modulus = mean_E
    mean_prediction = cantilever.deflection(data_inputs[:,0])
    cantilever.current_young_modulus = mean_E + std_E
    plus_prediction = cantilever.deflection(data_inputs[:,0])+sigma
    cantilever.current_young_modulus = mean_E + std_E
    minus_prediction = cantilever.deflection(data_inputs[:,0])-sigma

    sorted_indices = np.argsort(data_inputs[:, 0])
    sorted_data_inputs = data_inputs[sorted_indices]
    sorted_mean_prediction = mean_prediction[sorted_indices]
    sorted_plus_prediction = plus_prediction[sorted_indices]
    sorted_minus_prediction = minus_prediction[sorted_indices]

    plt.plot(sorted_data_inputs[:, 0], sorted_data_inputs[:, 2], 'k.', label="Data")
    plt.plot(sorted_data_inputs[:, 0], sorted_mean_prediction, 'r-', label="Mean")
    plt.plot(sorted_data_inputs[:, 0], sorted_plus_prediction, 'r--', label=r"Mean$\pm\sigma$")
    plt.plot(sorted_data_inputs[:, 0], sorted_minus_prediction, 'r--', label="_no_legend_")
    plt.legend()
    plt.xlabel("Load")
    plt.ylabel("Deflection")
    
    plt.savefig(output_path[:-3]+"pdf")

    

if __name__ == "__main__":

    # Load the inference data
    inference_data = az.from_netcdf("./code/output/results/calibrate_no_bias_cantilever.az")

    # Define the path to the data inputs
    data_inputs_path = "./code/input/data/cantilever_dataset.csv"

    # Define the path to save the predictions
    output_path = "./code/output/results/cantilever_nobias_predictions.csv"

    # Predict the Canitlever function without bias term
    predict_cantilever_nobias(inference_data, output_path, data_inputs_path, n_samples=500)