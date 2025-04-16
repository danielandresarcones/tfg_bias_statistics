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
    IndependentNormalModelError,
)

from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot

from cantilever_function import CantileverBeam
import pandas as pd
import arviz as az


class CantileverBiased(ForwardModelBase):
    def __init__(self, name = "Cantilever"):
        super().__init__(name=name)
        self.cantilever = CantileverBeam(length=5.0, height=0.3, young_modulus=30e9, yield_stress=1e24, softening_factor=0.99)
        self.pce_order = 2
    def interface(self):
        self.parameters = ["E", "sigma_E"]
        self.input_sensors = [Sensor("Load")]
        self.output_sensors = Sensor("Displacement", std_model="sigma")

    def response(self, inp: dict) -> dict:
        E=inp["E"]
        sigma_E = inp["sigma_E"]
        loads = inp["Load"]

        E_dist = cp.Normal(E, sigma_E)

        # generate the polynomial chaos expansion
        expansion = cp.generate_expansion(self.pce_order, E_dist)

        # generate quadrature nodes and weights
        sparse_quads = cp.generate_quadrature(
            self.pce_order, E_dist, rule="Gaussian"
        )
        # evaluate the model at the quadrature nodes
        sparse_evals = []
        for node in sparse_quads[0][0]:
            self.cantilever.current_young_modulus = node
            sparse_evals.append(self.cantilever.deflection(loads))

        # fit the polynomial chaos expansion
        fitted_sparse = cp.fit_quadrature(
            expansion, sparse_quads[0], sparse_quads[1], sparse_evals
        )
        return {"Displacement": fitted_sparse, "dist": E_dist}
    
if __name__ == "__main__":
   
    problem = InverseProblem("Cantilever with Gaussian noise", print_header=False)

    problem.add_parameter(
        "E",
        domain="(0, +oo)",
        tex=r"$E$",
        info="Young's modulus",
        prior=LogNormal(mean=23, std=0.5)
    )
    problem.add_parameter(
        "sigma_E",
        info="Bias value of E",
        tex="$\sigma_E$",
        prior=LogNormal(mean=21, std=1),
    )
    problem.add_parameter(
        "sigma",
        domain="(0, +oo)",
        tex=r"$\sigma$",
        info="Standard deviation, of zero-mean Gaussian noise model",
        # prior=Uniform(low=0.0, high=0.8),
        value=0.01,
    )

    # load data

    data_path = "input/data/cantilever_dataset.csv"
    data = pd.read_csv(data_path)
    loads = data["Load"].values
    displacements = data["Deflection"].values

    # experimental data
    problem.add_experiment(
        name="TestSeries_1",
        sensor_data={"Load": loads, "Displacement": displacements},
    )

    # forward model
    forward_model = CantileverBiased()
    problem.add_forward_model(forward_model, experiments="TestSeries_1")

    dummy_lmodel = EmbeddedLikelihoodBaseModel(
    experiment_name="TestSeries_1", l_model="independent_normal"
    )
    likelihood_model = IndependentNormalModelError(dummy_lmodel)
    problem.add_likelihood_model(likelihood_model)

    problem.info(print_header=True)

    solver = EmbeddedPCESolver(problem, show_progress=True)
    inference_data = solver.run(n_steps=2000, n_initial_steps=200)

    # true_values ={ "E":30e9}

    output_path = f"output/results/calibrate_bias_cantilever.az"
    az.to_netcdf(inference_data, output_path)

    pair_plot_array = create_pair_plot(
        inference_data,
        solver.problem,
        focus_on_posterior=True,
        show_legends=True,
        # true_values= true_values,
        # title="Sampling results from emcee-Solver (pair plot)",
        title = "Gráfico de pares (con sesgo)",
    )
    pair_plot_array.ravel()[0].figure.savefig(f"output/figures/pair_plot_cantilever_bias.pdf")
    trace_plot_array = create_trace_plot(
        inference_data,
        solver.problem,
        show=False,
        # title="Sampling results from emcee-Solver (trace plot)",
        title = "Gráfico de trazas (con sesgo)",
    )
    trace_plot_array.ravel()[0].figure.savefig(f"output/figures/trace_plot_cantilever_bias.pdf")
    posterior_plot = create_posterior_plot(
        inference_data,
        solver.problem,
        show=False,
        # title="Sampling results from emcee-Solver (trace plot)",
        title="Gráfico de posterior (con sesgo)",
    )
    posterior_plot.ravel()[0].figure.savefig(f"output/figures/posterior_plot_cantilever_bias.pdf")