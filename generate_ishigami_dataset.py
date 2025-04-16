from ishigami_function_3d import IshigamiFunction
import numpy as np


PATH = "input/data/ishigami_dataset.csv"

def generate_random_dataset(n_samples, a = 7, b = 0.1, ranges=[(-np.pi, np.pi)]*3, noise_std=0.1):
    ishigami = IshigamiFunction(a=a, b=b)
    X = np.random.uniform(*zip(*ranges), (n_samples, 3))
    y = ishigami.evaluate(X)
    y += np.random.normal(0, noise_std, n_samples)

    np.savetxt(PATH, np.hstack((X, y[:, None])), delimiter=",", header="x1,x2,x3,y", comments="")
    return X, y

if __name__ == "__main__":

    generate_random_dataset(100)
    print(f"Dataset saved to {PATH}")
    PATH = "input/data/ishigami_large_dataset.csv"
    generate_random_dataset(1000)
    print(f"Dataset saved to {PATH}")