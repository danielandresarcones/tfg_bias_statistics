import numpy as np
import chaospy as cp

# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform, LogNormal
from probeye.definition.sensor import Sensor
from probeye.inference.bias.solver import EmbeddedPCESolver

from probeye.inference.bias.likelihood_models import (
    EmbeddedLikelihoodBaseModel,
    MomentMatchingModelError,
    GlobalMomentMatchingModelError,
    RelativeGlobalMomentMatchingModelError,
    IndependentNormalModelError,
)

from probeye.inference.emcee.solver import EmceeSolver

from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot

from ishigami_function_3d import IshigamiFunctionPolynomial
import pandas as pd
import arviz as az


class IshigamiBiased(ForwardModelBase):
    def __init__(self, name = "Ishigami", degree=2, pce_order=2):
        super().__init__(name=name)
        self.degree = degree
        self.ishigami = IshigamiFunctionPolynomial(c=self.degree)
        self.pce_order = pce_order

    def interface(self):
        self.parameters = ["a", "b", "sigma_b"]
        self.input_sensors = [Sensor("x1"), Sensor("x2"), Sensor("x3")]
        self.output_sensors = Sensor("y", std_model="sigma")

    def response(self, inp: dict) -> dict:
        x1 = inp["x1"]
        x2 = inp["x2"]
        x3 = inp["x3"]
        a = inp["a"]
        b = inp["b"]
        sigma_b = inp["sigma_b"]

        self.ishigami.a = a
        self.ishigami.b = b
         # define the distribution for the bias term
        b_dist = cp.Normal(b, sigma_b)

        # generate the polynomial chaos expansion
        expansion = cp.generate_expansion(self.pce_order, b_dist)

        # generate quadrature nodes and weights
        sparse_quads = cp.generate_quadrature(
            self.pce_order, b_dist, rule="Gaussian"
        )
        # evaluate the model at the quadrature nodes
        sparse_evals = []
        for node in sparse_quads[0][0]:
            self.ishigami.b = node
            sparse_evals.append(self.ishigami(np.array([x1, x2, x3]).transpose()))

        # fit the polynomial chaos expansion
        fitted_sparse = cp.fit_quadrature(
            expansion, sparse_quads[0], sparse_quads[1], sparse_evals
        )
        return {"y": fitted_sparse, "dist": b_dist}

    
def run_calibrate_ishigami_bias(degree=3):
   
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
        "sigma_b",
        info="Bias value of b",
        tex="$\sigma_b$",
        prior=LogNormal(mean=-2, std=0.5),
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
    forward_model = IshigamiBiased(degree=degree)
    problem.add_forward_model(forward_model, experiments="TestSeries_1")

    # likelihood model
    dummy_lmodel = EmbeddedLikelihoodBaseModel(
    experiment_name="TestSeries_1", l_model="independent_normal"
    )
    likelihood_model = IndependentNormalModelError(dummy_lmodel)
    problem.add_likelihood_model(likelihood_model)

    problem.info(print_header=True)

    solver = EmbeddedPCESolver(problem, show_progress=True)
    inference_data = solver.run(n_steps=2000, n_initial_steps=20)

    true_values = {"a": 7.0, "b": 0.1}
    # true_values = {"a": 7.0, "b": 0.1, "sigma": 0.1}


    output_path = f"code/output/results/calibrate_bias_ishigami_{degree}.az"
    az.to_netcdf(inference_data, output_path)

    pair_plot_array = create_pair_plot(
        inference_data,
        solver.problem,
        focus_on_posterior=True,
        show_legends=True,
        show=False,
        title="Sampling results from emcee-Solver (pair plot)",
    )
    pair_plot_array.ravel()[0].figure.savefig(f"code/output/figures/pair_plot_ishigami_{degree}.png")
  
    trace_plot_array = create_trace_plot(
        inference_data,
        solver.problem,
        show=False,
        title="Sampling results from emcee-Solver (trace plot)",
    )
    trace_plot_array.ravel()[0].figure.savefig(f"code/output/figures/trace_plot_ishigami_{degree}.png")

if __name__ == "__main__":
    run_calibrate_ishigami_bias(degree=3)