import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import chaospy as cp

from ishigami_function_3d import IshigamiFunctionPolynomial


def predict_ishigami_bias(inference_data: az.InferenceData, output_path: str, data_inputs_path: str, degree: int = 3, n_samples: int = 200, pce_order: int = 2):
    """
    Predict the Ishigami function without bias term
    :param inference_data: InferenceData object
    :param output_path: str
    :param degree: int
    :param pce_order: int
    :return: None
    """
    # Load the model
    ishigami = IshigamiFunctionPolynomial(c=degree)

    # Read data inputs from CSV
    data_inputs = pd.read_csv(data_inputs_path).values
    data_inputs = data_inputs[:, :3]
    # Extract samples from inference_data
    samples = inference_data.posterior

    # Initialize an empty list to store predictions
    predictions_mean_list = []
    predictions_std_list = []

    # Number of samples to use
    samples_array= az.extract(inference_data, num_samples=n_samples)

    for a, b, sigma_b in zip(samples_array["$a$"].values, samples_array["$b$"].values, samples_array["$\sigma_b$"].values):
        # Update parameters a and b from the samples
        ishigami.a = a

        b_dist = cp.Normal(b, sigma_b)
        poly_exp = cp.generate_expansion(pce_order, b_dist)
        quadrature_nodes, quadrature_weights = cp.generate_quadrature(pce_order, b_dist, rule="G")
        evaluation = []
        for node in quadrature_nodes[0]:
            ishigami.b = node
            prediction = ishigami.evaluate(data_inputs)
            evaluation.append(prediction)

        fitted_polynomial = cp.fit_quadrature(poly_exp, quadrature_nodes, quadrature_weights, evaluation)

        # Evaluate the Ishigami function at the data inputs
        prediction_mean = cp.E(fitted_polynomial, b_dist)
        prediction_std = cp.Std(fitted_polynomial, b_dist)

        predictions_mean_list.append(prediction_mean)
        predictions_std_list.append(prediction_std)

    # Convert predictions to a DataFrame
    predictions_mean_df = pd.DataFrame(predictions_mean_list)
    predictions_std_df = pd.DataFrame(predictions_std_list)

    # Output the predictions to CSV
    predictions_mean_df.to_csv(output_path.replace(".csv", "_mean.csv"), index=False)
    predictions_std_df.to_csv(output_path.replace(".csv", "_std.csv"), index=False)


if __name__ == "__main__":

    # Load the inference data
    inference_data = az.from_netcdf("./code/output/results/calibrate_bias_ishigami_3.az")

    # Define the path to the data inputs
    data_inputs_path = "./code/input/data/ishigami_dataset.csv"

    # Define the path to save the predictions
    output_path = "./code/output/results/ishigami_bias_predictions_at_data.csv"

    # Predict the Ishigami function without bias term
    predict_ishigami_bias(inference_data, output_path, data_inputs_path, n_samples=500)