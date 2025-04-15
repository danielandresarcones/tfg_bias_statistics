import numpy as np


# local imports (problem definition)
from probeye.definition.inverse_problem import InverseProblem
from probeye.definition.forward_model import ForwardModelBase
from probeye.definition.distribution import Normal, Uniform, LogNormal
from probeye.definition.sensor import Sensor
from probeye.definition.likelihood_model import GaussianLikelihoodModel

from probeye.inference.emcee.solver import EmceeSolver

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

    def interface(self):
        self.parameters = ["E"]
        self.input_sensors = [Sensor("Load")]
        self.output_sensors = Sensor("Displacement", std_model="sigma")

    def response(self, inp: dict) -> dict:
        self.cantilever.current_young_modulus=inp["E"]
        loads = inp["Load"]
        delta = self.cantilever.deflection(loads)

        return {"Displacement": delta}
    
if __name__ == "__main__":
   
    problem = InverseProblem("Cantilever with Gaussian noise", print_header=False)

    problem.add_parameter(
        "E",
        domain="(0, +oo)",
        tex=r"$E$",
        info="Young's modulus",
        prior=LogNormal(mean=11, std=0.1)
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

    data_path = "code/input/data/cantilever_dataset.csv"
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

    # likelihood model
    problem.add_likelihood_model(
        GaussianLikelihoodModel(experiment_name="TestSeries_1", model_error="additive")
    )

    problem.info(print_header=True)

    emcee_solver = EmceeSolver(problem, show_progress=True)
    inference_data = emcee_solver.run(n_steps=5000, n_initial_steps=500)

    true_values ={ "E":30e9}

    output_path = f"code/output/results/calibrate_no_bias_cantilever.az"
    az.to_netcdf(inference_data, output_path)

    # pair_plot_array = create_pair_plot(
    #     inference_data,
    #     emcee_solver.problem,
    #     focus_on_posterior=True,
    #     show_legends=True,
    #     true_values= true_values,
    #     title="Sampling results from emcee-Solver (pair plot)",
    # )
    # pair_plot_array.ravel()[0].figure.savefig(f"code/output/figures/pair_plot_cantilever_no_bias.png")
    trace_plot_array = create_trace_plot(
        inference_data,
        emcee_solver.problem,
        show=False,
        # title="Sampling results from emcee-Solver (trace plot)",
        title="Trace plot (no bias)",
    )
    trace_plot_array.ravel()[0].figure.savefig(f"code/output/figures/trace_plot_cantilever_no_bias.png")
    posterior_plot = create_posterior_plot(
        inference_data,
        emcee_solver.problem,
        show=True,
        # title="Sampling results from emcee-Solver (trace plot)",
        title="Posterior plot (no bias)",
    )
    posterior_plot.figure.savefig(f"code/output/figures/posterior_plot_cantilever_no_bias.png")