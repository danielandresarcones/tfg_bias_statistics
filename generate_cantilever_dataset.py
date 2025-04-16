from cantilever_function import CantileverBeam
import numpy as np

PATH = "input/data/cantilever_dataset.csv"

def generate_random_dataset(n_samples, length=5.0, height=0.3, young_modulus=30e9, yield_stress=20e6, softening_factor=0.95, ranges=[(2e4, 1e5)], noise_std=0.01):
    beam = CantileverBeam(length, height, young_modulus, yield_stress, softening_factor)
    input_Y_vector = np.random.uniform(*ranges[0], (n_samples,))
    # input_Y_vector = np.linspace(*ranges[0], n_samples)
    dataset = beam.generate_data(input_Y_vector, noise_std)

    dataset.to_csv(PATH, index=False)
    return dataset

def plot_dataset(dataset, output_path=None):

    CantileverBeam.plot_load_displacement(dataset, output_path)

if __name__ == "__main__":
    dataset = generate_random_dataset(200)
    plot_dataset(dataset, output_path="output/figures/cantilever_dataset.pdf")
    # plot_dataset(dataset, output_path="output/figures/cantilever_dataset.pdf")
    print(f"Dataset saved to {PATH}")
    print("Plot saved to output/figures/cantilever_dataset.pdf")