"""
Dodo file for running all the functions related to the Ishigami model analysis.
This file defines the tasks and their dependencies for the doit task runner.
"""

import os
import sys
from pathlib import Path

# Configuration for doit
DOIT_CONFIG = {
    'default_tasks': ['all'],
    'verbosity': 2,
}

# Ensure output directories exist
def create_directories():
    """Create necessary directories if they don't exist."""
    paths = [
        "output/results",
        "output/figures",
        "input/data"
    ]
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def task_create_directories():
    """Task to create necessary directories."""
    return {
        'actions': [create_directories],
        'verbosity': 2,
    }

def task_generate_ishigami_dataset():
    """Generate the Ishigami datasets."""
    return {
        'actions': ['python generate_ishigami_dataset.py'],
        'file_dep': ['generate_ishigami_dataset.py', 'ishigami_function_3d.py'],
        'targets': ['input/data/ishigami_dataset.csv', 'input/data/ishigami_large_dataset.csv'],
        'task_dep': ['create_directories'],
        'verbosity': 2,
    }

def task_calibrate_ishigami_nobias():
    """Run calibration of Ishigami model without bias."""
    return {
        'actions': ['python calibrate_ishigami_nobias.py'],
        'file_dep': ['calibrate_ishigami_nobias.py', 'ishigami_function_3d.py', 'input/data/ishigami_dataset.csv'],
        'targets': ['output/results/calibrate_no_bias_ishigami_3.az'],
        'task_dep': ['generate_ishigami_dataset'],
        'verbosity': 2,
    }

def task_calibrate_ishigami_bias():
    """Run calibration of Ishigami model with bias."""
    return {
        'actions': ['python calibrate_ishigami_bias.py'],
        'file_dep': ['calibrate_ishigami_bias.py', 'ishigami_function_3d.py', 'input/data/ishigami_dataset.csv'],
        'targets': [
            'output/results/calibrate_bias_ishigami_3.az',
            'output/figures/pair_plot_ishigami_3.png',
            'output/figures/trace_plot_ishigami_3.png'
        ],
        'task_dep': ['generate_ishigami_dataset'],
        'verbosity': 2,
    }

def task_predict_ishigami_nobias():
    """Run prediction with the no-bias Ishigami model."""
    return {
        'actions': ['python predict_ishigami_nobias.py'],
        'file_dep': [
            'predict_ishigami_nobias.py',
            'ishigami_function_3d.py',
            'output/results/calibrate_no_bias_ishigami_3.az',
            'input/data/ishigami_large_dataset.csv'
        ],
        'targets': ['output/results/ishigami_nobias_predictions.csv'],
        'task_dep': ['calibrate_ishigami_nobias'],
        'verbosity': 2,
    }

def task_predict_ishigami_bias():
    """Run prediction with the bias-aware Ishigami model."""
    return {
        'actions': ['python predict_ishigami_bias.py'],
        'file_dep': [
            'predict_ishigami_bias.py',
            'ishigami_function_3d.py',
            'output/results/calibrate_bias_ishigami_3.az',
            'input/data/ishigami_dataset.csv'
        ],
        'targets': [
            'output/results/ishigami_bias_predictions_at_data_mean.csv',
            'output/results/ishigami_bias_predictions_at_data_std.csv'
        ],
        'task_dep': ['calibrate_ishigami_bias'],
        'verbosity': 2,
    }

def task_qoi_ishigami():
    """Run quantities of interest analysis for Ishigami models."""
    return {
        'actions': ['python qoi_ishigami.py'],
        'file_dep': [
            'qoi_ishigami.py',
            'ishigami_function_3d.py',
            'input/data/ishigami_dataset.csv',
            'output/results/calibrate_no_bias_ishigami_3.az',
            'output/results/calibrate_bias_ishigami_3.az'
        ],
        'targets': [
            'output/figures/ishigami_true_proportion.png',
            'output/figures/ishigami_proportion_no_bias.png',
            'output/figures/ishigami_proportion_bias.png',
            'output/figures/ishigami_proportion_comparison.png'
        ],
        'task_dep': ['calibrate_ishigami_nobias', 'calibrate_ishigami_bias'],
        'verbosity': 2,
    }

def task_ishigami_statistical_analysis():
    """Run statistical analysis on Ishigami models."""
    return {
        'actions': ['python ishigami_statistical_analysis.py'],
        'file_dep': [
            'ishigami_statistical_analysis.py',
            'input/data/ishigami_dataset.csv',
            'input/data/ishigami_large_dataset.csv',
            'output/results/calibrate_bias_ishigami_3.az',
            'output/results/calibrate_no_bias_ishigami_3.az',
            'output/results/ishigami_nobias_predictions.csv',
            'output/results/ishigami_bias_predictions_at_data_mean.csv',
            'output/results/ishigami_bias_predictions_at_data_std.csv'
        ],
        'targets': [
            'output/figures/ishigami_no_bias_variance_decomposition.png',
            'output/figures/ishigami_bias_variance_decomposition.png',
            'output/figures/ishigami_no_bias_variance_decomposition_residuals_sigma.png',
            'output/figures/ishigami_bias_variance_decomposition_residuals_sigma.png'
        ],
        'task_dep': ['predict_ishigami_nobias', 'predict_ishigami_bias'],
        'verbosity': 2,
    }

def task_ishigami_robustness_analysis():
    """Run robustness analysis on Ishigami models."""
    return {
        'actions': ['python ishigami_robustness_analysis.py'],
        'file_dep': [
            'ishigami_robustness_analysis.py',
            'calibrate_ishigami_bias.py',  # Using functions from this file
            'input/data/ishigami_dataset.csv',
            'output/results/calibrate_bias_ishigami_3.az',
            'output/results/calibrate_no_bias_ishigami_3.az',
            'output/results/ishigami_bias_predictions_at_data_mean.csv',
            'output/results/ishigami_bias_predictions_at_data_std.csv'
        ],
        'targets': [
            'output/figures/ishigami_robustness.png',
            'output/figures/ishigami_robustness.tex',
            'output/figures/ishigami_influence.png',
            'output/figures/ishigami_sensitivity.png'
        ],
        'task_dep': ['predict_ishigami_bias'],
        'verbosity': 2,
    }

def task_decision_rule_ishigami():
    """Run decision rule analysis on Ishigami models."""
    return {
        'actions': ['python decision_rule_ishigami.py'],
        'file_dep': [
            'decision_rule_ishigami.py',
            'qoi_ishigami.py',  # Using functions from this file
            'input/data/ishigami_dataset.csv',
            'output/results/calibrate_bias_ishigami_3.az',
            'output/results/calibrate_no_bias_ishigami_3.az'
        ],
        'targets': [
            'output/figures/decisions_ishigami_no_bias.pdf',
            'output/figures/decisions_ishigami_bias.pdf',
            'output/results/decisions_no_bias.csv',
            'output/results/decisions_ishigami_bias.csv'
        ],
        'task_dep': ['qoi_ishigami'],
        'verbosity': 2,
    }

# Cantilever beam tasks
def task_generate_cantilever_dataset():
    """Generate the Cantilever beam dataset."""
    return {
        'actions': ['python generate_cantilever_dataset.py'],
        'file_dep': ['generate_cantilever_dataset.py', 'cantilever_function.py'],
        'targets': ['input/data/cantilever_dataset.csv', 'output/figures/cantilever_dataset.png'],
        'task_dep': ['create_directories'],
        'verbosity': 2,
    }

def task_calibrate_cantilever_nobias():
    """Run calibration of Cantilever beam model without bias."""
    return {
        'actions': ['python calibrate_cantilever_nobias.py'],
        'file_dep': ['calibrate_cantilever_nobias.py', 'cantilever_function.py', 'input/data/cantilever_dataset.csv'],
        'targets': [
            'output/results/calibrate_no_bias_cantilever.az',
            'output/figures/trace_plot_cantilever_no_bias.png',
            'output/figures/posterior_plot_cantilever_no_bias.png'
        ],
        'task_dep': ['generate_cantilever_dataset'],
        'verbosity': 2,
    }

def task_calibrate_cantilever_bias():
    """Run calibration of Cantilever beam model with bias."""
    return {
        'actions': ['python calibrate_cantilever_bias.py'],
        'file_dep': ['calibrate_cantilever_bias.py', 'cantilever_function.py', 'input/data/cantilever_dataset.csv'],
        'targets': [
            'output/results/calibrate_bias_cantilever.az',
            'output/figures/pair_plot_cantilever_bias.png',
            'output/figures/trace_plot_cantilever_bias.png',
            'output/figures/posterior_plot_cantilever_bias.png'
        ],
        'task_dep': ['generate_cantilever_dataset'],
        'verbosity': 2,
    }

def task_predict_cantilever_nobias():
    """Run prediction with the no-bias Cantilever beam model."""
    return {
        'actions': ['python predict_cantilever_nobias.py'],
        'file_dep': [
            'predict_cantilever_nobias.py',
            'cantilever_function.py',
            'output/results/calibrate_no_bias_cantilever.az',
            'input/data/cantilever_dataset.csv'
        ],
        'targets': [
            'output/results/cantilever_nobias_predictions.csv',
            'output/results/cantilever_nobias_predictions.pdf'
        ],
        'task_dep': ['calibrate_cantilever_nobias'],
        'verbosity': 2,
    }

def task_predict_cantilever_bias():
    """Run prediction with the bias-aware Cantilever beam model."""
    return {
        'actions': ['python predict_cantilever_bias.py'],
        'file_dep': [
            'predict_cantilever_bias.py',
            'cantilever_function.py',
            'output/results/calibrate_bias_cantilever.az',
            'input/data/cantilever_dataset.csv'
        ],
        'targets': [
            'output/results/cantilever_bias_predictions_mean.csv',
            'output/results/cantilever_bias_predictions_std.csv',
            'output/results/cantilever_bias_predictions.pdf'
        ],
        'task_dep': ['calibrate_cantilever_bias'],
        'verbosity': 2,
    }

def task_qoi_cantilever():
    """Run quantities of interest analysis for Cantilever beam models."""
    return {
        'actions': ['python qoi_cantilever.py'],
        'file_dep': [
            'qoi_cantilever.py',
            'cantilever_function.py',
            'input/data/cantilever_dataset.csv',
            'output/results/calibrate_no_bias_cantilever.az',
            'output/results/calibrate_bias_cantilever.az'
        ],
        'targets': [
            'output/figures/cantilever_true_deflection.png',
            'output/figures/cantilever_deflection_no_bias.png',
            'output/figures/cantilever_deflection_bias.png',
            'output/figures/cantilever_deflection_comparison.png'
        ],
        'task_dep': ['calibrate_cantilever_nobias', 'calibrate_cantilever_bias'],
        'verbosity': 2,
    }

def task_cantilever_statistical_analysis():
    """Run statistical analysis on Cantilever beam models."""
    return {
        'actions': ['python cantilever_statistical_analysis.py'],
        'file_dep': [
            'cantilever_statistical_analysis.py',
            'input/data/cantilever_dataset.csv',
            'output/results/calibrate_bias_cantilever.az',
            'output/results/calibrate_no_bias_cantilever.az',
            'output/results/cantilever_nobias_predictions.csv',
            'output/results/cantilever_bias_predictions_mean.csv',
            'output/results/cantilever_bias_predictions_std.csv'
        ],
        'targets': [
            'output/figures/cantilever_no_bias_variance_decomposition.png',
            'output/figures/cantilever_bias_variance_decomposition.png',
            'output/figures/cantilever_no_bias_variance_decomposition_residuals_sigma.png',
            'output/figures/cantilever_bias_variance_decomposition_residuals_sigma.png',
            'output/figures/cantilever_no_bias_z_value_1d.png',
            'output/figures/cantilever_bias_z_value_1d.png',
            'output/figures/cantilever_no_bias_residuals_1d.png',
            'output/figures/cantilever_bias_residuals_1d.png'
        ],
        'task_dep': ['predict_cantilever_nobias', 'predict_cantilever_bias'],
        'verbosity': 2,
    }

def task_cantilever_robustness_analysis():
    """Run robustness analysis on Cantilever beam models."""
    return {
        'actions': ['python cantilever_robustness_analysis.py'],
        'file_dep': [
            'cantilever_robustness_analysis.py',
            'cantilever_function.py',
            'calibrate_cantilever_bias.py',
            'input/data/cantilever_dataset.csv',
            'output/results/calibrate_bias_cantilever.az',
            'output/results/calibrate_no_bias_cantilever.az',
            'output/results/cantilever_bias_predictions_mean.csv',
            'output/results/cantilever_bias_predictions_std.csv'
        ],
        'targets': [
            'output/figures/cantilever_influence.png',
            'output/figures/cantilever_sensitivity.png'
        ],
        'task_dep': ['predict_cantilever_bias'],
        'verbosity': 2,
    }

def task_decision_rule_cantilever():
    """Run decision rule analysis on Cantilever beam models."""
    return {
        'actions': ['python decision_rule_cantilever.py'],
        'file_dep': [
            'decision_rule_cantilever.py',
            'qoi_cantilever.py',
            'input/data/cantilever_dataset.csv',
            'output/results/calibrate_bias_cantilever.az',
            'output/results/calibrate_no_bias_cantilever.az'
        ],
        'targets': [
            'output/figures/decisions_cantilever_no_bias.png',
            'output/figures/decisions_cantilever_bias.png',
            'output/results/decisions_cantilever_no_bias.csv',
            'output/results/decisions_cantilever_bias.csv'
        ],
        'task_dep': ['qoi_cantilever'],
        'verbosity': 2,
    }

def task_all():
    """Run all tasks in the correct order."""
    return {
        'actions': None,
        'task_dep': [
            'generate_ishigami_dataset',
            'calibrate_ishigami_nobias',
            'calibrate_ishigami_bias',
            'predict_ishigami_nobias',
            'predict_ishigami_bias',
            'qoi_ishigami',
            'ishigami_statistical_analysis',
            'ishigami_robustness_analysis',
            'decision_rule_ishigami',
            'generate_cantilever_dataset',
            'calibrate_cantilever_nobias',
            'calibrate_cantilever_bias',
            'predict_cantilever_nobias',
            'predict_cantilever_bias',
            'qoi_cantilever',
            'cantilever_statistical_analysis',
            'cantilever_robustness_analysis',
            'decision_rule_cantilever'
        ],
        'verbosity': 2,
    }
