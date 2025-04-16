import arviz as az
import pandas as pd

from ishigami_function_3d import IshigamiFunctionPolynomial


def predict_ishigami_nobias(inference_data: az.InferenceData, output_path: str, data_inputs_path: str, degree: int = 3, n_samples: int = 200):
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

    # Initialize an empty list to store predictions
    predictions = []

    # Number of samples to use
    samples_array= az.extract(inference_data, num_samples=n_samples)

    for a, b in zip(samples_array["$a$"].values, samples_array["$b$"].values):
        # Update parameters a and b from the samples
        ishigami.a = a
        ishigami.b = b

        # Evaluate the Ishigami function at the data inputs
        prediction = ishigami.evaluate(data_inputs.transpose())
        predictions.append(prediction)

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Output the predictions to CSV
    predictions_df.to_csv(output_path, index=False)

if __name__ == "__main__":

    # Load the inference data
    inference_data = az.from_netcdf("./output/results/calibrate_no_bias_ishigami_3.az")

    # Define the path to the data inputs
    data_inputs_path = "./input/data/ishigami_large_dataset.csv"

    # Define the path to save the predictions
    output_path = "./output/results/ishigami_nobias_predictions.csv"

    # Predict the Ishigami function without bias term
    predict_ishigami_nobias(inference_data, output_path, data_inputs_path, n_samples=500)