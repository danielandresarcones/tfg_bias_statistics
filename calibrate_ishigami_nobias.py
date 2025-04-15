import numpy as np


# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

from probeye.inference.emcee.solver import EmceeSolver

from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot

from ishigami_function_3d import IshigamiFunctionPolynomial
import pandas as pd
import arviz as az


class IshigamiBiased(ForwardModelBase):
    def __init__(self, name = "Ishigami", degree=2):
        super().__init__(name=name)
        self.degree = degree
        self.ishigami = IshigamiFunctionPolynomial(c=self.degree)

    def interface(self):
        self.parameters = ["a", "b"]
        self.input_sensors = [Sensor("x1"), Sensor("x2"), Sensor("x3")]
        self.output_sensors = Sensor("y", std_model="sigma")

    def response(self, inp: dict) -> dict:
        x1 = inp["x1"]
        x2 = inp["x2"]
        x3 = inp["x3"]
        a = inp["a"]
        b = inp["b"]

        self.ishigami.a = a
        self.ishigami.b = b

        return {"y": self.ishigami([x1, x2, x3])}
    
if __name__ == "__main__":
   
    problem = InverseProblem("Ishigami with Gaussian noise", print_header=False)

    problem.add_parameter(
        "a",
        tex="$a$",
        info="Value of a",
        prior=Normal(mean=2.0, std=1.0),
    )
    problem.add_parameter(
        "b",
        info="Value of b",
        tex="$b$",
        prior=Normal(mean=1.0, std=1.0),
    )
    problem.add_parameter(
        "sigma",
        domain="(0, +oo)",
        tex=r"$\sigma$",
        info="Standard deviation, of zero-mean Gaussian noise model",
        # prior=Uniform(low=0.0, high=0.8),
        value=0.1,
    )

    # load data

    data_path = "code/input/data/ishigami_dataset.csv"
    data = pd.read_csv(data_path)
    x1 = data["x1"].values
    x2 = data["x2"].values
    x3 = data["x3"].values
    y = data["y"].values

    # experimental data
    problem.add_experiment(
        name="TestSeries_1",
        sensor_data={"x1": x1, "x2": x2, "x3": x3, "y": y},
    )

    # forward model
    degree = 3
    forward_model = IshigamiBiased(degree=degree)
    problem.add_forward_model(forward_model, experiments="TestSeries_1")

    # likelihood model
    problem.add_likelihood_model(
        GaussianLikelihoodModel(experiment_name="TestSeries_1", model_error="additive")
    )

    problem.info(print_header=True)

    emcee_solver = EmceeSolver(problem, show_progress=True)
    inference_data = emcee_solver.run(n_steps=2000, n_initial_steps=200)

    true_values = {"a": 7.0, "b": 0.1}
    # true_values = {"a": 7.0, "b": 0.1, "sigma": 0.1}


    output_path = f"code/output/results/calibrate_no_bias_ishigami_{degree}.az"
    az.to_netcdf(inference_data, output_path)

    pair_plot_array = create_pair_plot(
        inference_data,
        emcee_solver.problem,
        focus_on_posterior=True,
        show_legends=True,
        title="Gráfico de pares (sin sesgo)",
        # title="Sampling results from emcee-Solver (pair plot)",
    )