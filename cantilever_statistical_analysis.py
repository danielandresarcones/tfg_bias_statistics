from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns

from tqdm import tqdm

        
def utility_plot_1d(df: pd.DataFrame, output_path: str, value: str):

    plt.figure(figsize=(12, 6))

    # Plot z values
    plt.subplot(1, 2, 1)
    plt.plot(df['z'], label=value)
    # plt.axhline(y=df['z'].mean(), color='r', linestyle='--', label='Mean')
    plt.axhline(y=df['z'].median(), color='r', linestyle='--', label='Mediana')
    # plt.xlabel('Index')
    plt.xlabel('Carga')
    # plt.ylabel('z')
    plt.ylabel(value)
    # plt.title('z values and Mean')
    plt.title(f'{value} y Mediana')
    plt.legend()
    plt.grid(True)

    # Plot histogram
    plt.subplot(1, 2, 2)
    plt.hist(df['z'], bins=30, edgecolor='k', alpha=0.7)
    # plt.axvline(df['z'].mean(), color='r', linestyle='--', label='Mean')
    plt.axvline(df['z'].median(), color='r', linestyle='--', label='Mediana')
    plt.text(df['z'].median() * 1.1, plt.gca().get_ylim()[1] * 0.9, f'Mediana: {df["z"].median():.2f}', color='r', va='bottom', ha='left')
    # plt.xlabel('z')
    plt.xlabel(value)
    # plt.ylabel('Frequency')
    plt.ylabel('Frecuencia')
    # plt.title('Histogram of z values')
    plt.title(f'Histograma de {value}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)

def convergence_analysis(inference_data: az.InferenceData, output_path: str):

    ess = az.ess(inference_data)
    print("Effective Sample Size (ESS):")
    print(ess)
    ess_list_E = []
    ess_list_sigma_E = []

    for i in tqdm(range(1, int(len(az.extract(inference_data)["$E$"].values)/len(inference_data.posterior["$E$"])) + 1), desc="Calculating ESS"):
        ess_subset = az.ess(inference_data.sel(draw=slice(None, i)))
        ess_list_E.append(ess_subset["$E$"].values)
        ess_list_sigma_E.append(ess_subset["$\sigma_E$"].values)

    ess_list_E = [x for x in ess_list_E if not np.isnan(x)]
    ess_list_sigma_E = [x for x in ess_list_sigma_E if not np.isnan(x)]
    
    plt.figure(figsize=(10, 6))
    plt.axhline(y=837, color='r', linestyle='--')
    plt.text(20, 837, r'$\widehat{\text{ESS}}=837$', color='r', va='bottom', ha='center')
    plt.plot(ess_list_E, label=r'$\widehat{\text{ESS}}$ $E$')
    plt.plot(ess_list_sigma_E, label=r'$\widehat{\text{ESS}}$ $\sigma_E$')
    # plt.xlabel('Number of samples')
    plt.xlabel('Número de muestras')
    plt.xlim(0, len(ess_list_E))
    # plt.ylabel(r'Effective Sample Size ($\widehat{\text{ESS}}$)')
    plt.ylabel(r'Tamaño de muestra efectivo ($\widehat{\text{ESS}}$)')
    # plt.title(r'Convergence Analysis of $\widehat{\text{ESS}}$')
    plt.title(r'Análisis de convergencia de $\widehat{\text{ESS}}$')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

def fit_analysis(inference_data: az.InferenceData, output_path: str, alpha: float = 0.05):

    hdi = az.hdi(inference_data.posterior, hdi_prob=1-alpha)
    print("Highest Density Interval (HDI):")
    print(hdi)
    print(az.summary(inference_data, hdi_prob=1-alpha))
    # Save summary to LaTeX table format
    summary_df = az.summary(inference_data, hdi_prob=1-alpha).reset_index()
    with open(output_path.replace('.json', '_summary.tex'), 'w') as f:
        summary_df.to_latex(f, index=False)
        

def predictions_analysis(data:pd.DataFrame, predictions: pd.DataFrame, bias: bool, sigma: Union[float, pd.DataFrame], index: int = 0, output_path: str = None):

    if bias:
        z_values = np.abs(data["Deflection"].values[index] - predictions.values.transpose()[index]) / sigma.values.transpose()[index]
    else:
        z_values = np.abs(data["Deflection"].values[index] - predictions.values.transpose()[index]) / sigma
    # z_values = z_values.values.flatten()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(z_values, fill=True)
    # plt.axvline(np.mean(z_values), color='r', linestyle='--', label='Mean')
    plt.axvline(np.median(z_values), color='r', linestyle='--', label='Mediana')
    plt.text(np.median(z_values) * 1.1, plt.gca().get_ylim()[1] * 0.9, f'Mediana: {np.median(z_values):.2f}', color='r', va='bottom', ha='left')
    plt.axvline(1.96, color='k', linestyle='--', label=r'Normal $3\sigma$')
    # plt.xlabel('Z-values')
    plt.xlabel('Valores Z')
    # plt.ylabel('Density')
    plt.ylabel('Densidad')
    # plt.title('Density Plot of Z-values at index {}'.format(index))
    plt.title('Gráfico de densidad de valores Z en el índice {}'.format(index))
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path.replace('.pdf', '_{}.pdf'.format(index)))

    return z_values


def plot_comparison_z_values(z_values_no_bias: np.ndarray, z_values_bias: np.ndarray, output_path: str):

    plt.figure(figsize=(10, 6))
    # sns.kdeplot(z_values_no_bias, fill=True, color = "r", label='No Bias')
    sns.kdeplot(z_values_no_bias, fill=True, color="r", label='Sin Sesgo')
    # sns.kdeplot(z_values_bias, fill=True, color="g", label='Bias')
    sns.kdeplot(z_values_bias, fill=True, color="g", label='Con Sesgo')
    # plt.axvline(np.median(z_values_no_bias), color='r', linestyle='--', label='Median No Bias')
    plt.axvline(np.median(z_values_no_bias), color='r', linestyle='--', label='Mediana Sin Sesgo')
    # plt.axvline(np.median(z_values_bias), color='g', linestyle='--', label='Median Bias')
    plt.axvline(np.median(z_values_bias), color='g', linestyle='--', label='Mediana Con Sesgo')
    plt.axvline(1.96, color='k', linestyle='--', label=r'Normal $3\sigma$')
    # plt.xlabel('Z-values')
    plt.xlabel('Valores Z')
    # plt.ylabel('Density')
    plt.ylabel('Densidad')
    # plt.title('Density Plot of Z-values')
    plt.title('Gráfico de densidad de valores Z')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)

def plot_z_values_1d(data: pd.DataFrame, predictions: pd.DataFrame, bias: bool, sigma: Union[float, pd.DataFrame], output_path: str):

    if bias:
        z_values = np.abs(data["Deflection"].values - np.mean(predictions.values, axis=0)) / np.mean(sigma.values, axis=0)
    else:
        z_values = np.abs(data["Deflection"].values - np.mean(predictions.values, axis=0)) / float(sigma)

    df = pd.DataFrame({'Load': data["Load"],'z': z_values})

    utility_plot_1d(df, output_path, "Valores Z")

def plot_residuals_1d(data: pd.DataFrame, predictions: pd.DataFrame, output_path: str):
    
    residual = np.abs(data["Deflection"].values - np.mean(predictions.values, axis=0))

    df = pd.DataFrame({'Load': data['Load'],'z': residual})
    utility_plot_1d(df, output_path, "Residuos")

def variance_decomposition_analysis(data: pd.DataFrame, predictions: pd.DataFrame, bias: bool, sigma: Union[float, pd.DataFrame], output_path: str):

    residuals = np.abs(data["Deflection"].values - np.mean(predictions.values, axis=0))
    total_variance = np.sum(np.square(residuals))

    # Handle bias and sigma
    if bias:
        # Ensure sigma is a 1D array or a scalar
        sigma = np.array(np.mean(sigma, axis=0))
        if sigma.ndim == 0:  # Scalar
            variance_vector = np.square(sigma) * np.ones(len(residuals))
        elif sigma.shape == residuals.shape:  # Same length as residuals
            variance_vector = np.square(sigma)
        else:
            raise ValueError(
                f"Shape of sigma ({sigma.shape}) does not match residuals ({residuals.shape})"
            )
    else:
        variance_vector = np.square(float(sigma) * np.ones(len(residuals)))  # Scalar case

    # Compute explained and unexplained variance
    explained_variance = min(np.sum(variance_vector), total_variance)
    unexplained_variance = total_variance - explained_variance

    # Print variance values
    # print("Total Variance:", total_variance)
    print("Varianza total:", total_variance)
    # print("Explained Variance:", explained_variance)
    print("Varianza explicada:", explained_variance)
    # print("Unexplained Variance:", unexplained_variance)
    print("Varianza no explicada:", unexplained_variance)

    # Plot variance decomposition
    plt.figure(figsize=(10, 6))
    # labels = ['Total Variance', 'Explained Variance', 'Unexplained Variance']
    labels = ['Varianza total', 'Varianza explicada', 'Varianza no explicada']
    variances = [total_variance, explained_variance, unexplained_variance]
    plt.bar(labels, variances, color=['blue', 'green', 'red'])
    # plt.xlabel('Variance Components')
    plt.xlabel('Componentes de varianza')
    # plt.ylabel('Variance')
    plt.ylabel('Varianza')
    # plt.title('Variance Decomposition')
    plt.title('Descomposición de varianza')
    plt.grid(True)
    plt.savefig(output_path.replace('.pdf', '_variance_decomposition.pdf'))

    # Plot residuals and variance vector
    plt.figure(figsize=(12, 6))

    indices = np.arange(len(residuals))  # Indices for predictions
    # plt.bar(indices, residuals, color='blue', alpha=0.7, label='Residuals')
    plt.bar(indices, residuals, color='blue', alpha=0.7, label='Residuos')
    # plt.bar(indices, variance_vector, color='green', alpha=0.7, label='Variance (sigma^2)')
    plt.bar(indices, variance_vector, color='green', alpha=0.7, label=r'Varianza ($\sigma^2$)')

    # plt.xlabel('Index')
    plt.xlabel('Carga')
    # plt.ylabel('Value')
    plt.ylabel('Valor')
    # plt.title('Residuals and Variance at Each Prediction Point')
    plt.title('Residuos y varianza en cada punto de predicción')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path.replace('.pdf', '_residuals_sigma.pdf'))
    # plt.show()

if __name__ == "__main__":

    data = pd.read_csv('./input/data/cantilever_dataset.csv')
    # large_data = pd.read_csv('./input/data/cantilever_large_dataset.csv')

    inference_data_bias = az.from_netcdf('./output/results/calibrate_bias_cantilever.az')
    inference_data_no_bias = az.from_netcdf('./output/results/calibrate_no_bias_cantilever.az')
    output_path_root_bias = './output/figures/cantilever_bias'
    output_path_root_no_bias = './output/figures/cantilever_no_bias'

    # convergence_analysis(inference_data_bias, output_path_root_bias + '_convergence.pdf')
    # fit_analysis(inference_data_bias, output_path_root_bias + '_fit.json')
    # fit_analysis(inference_data_no_bias, output_path_root_no_bias + '_fit.json')

    predictions_no_bias = pd.read_csv('./output/results/cantilever_nobias_predictions.csv', header=0)
    predictions_bias_mean = pd.read_csv('./output/results/cantilever_bias_predictions_mean.csv', header=0)
    predictions_bias_std = pd.read_csv('./output/results/cantilever_bias_predictions_std.csv', header=0)

    z_values_no_bias = predictions_analysis(data, predictions_no_bias, bias=False, sigma= 0.01, index = 0, output_path = output_path_root_no_bias + '_predictions.pdf')
    z_values_bias = predictions_analysis(data, predictions_bias_mean, bias=True, sigma = predictions_bias_std, index=0, output_path = output_path_root_bias + '_predictions.pdf')
    plot_comparison_z_values(z_values_no_bias, z_values_bias, output_path_root_bias + '_comparison_0.pdf')

    z_values_no_bias = predictions_analysis(data, predictions_no_bias, bias=False, sigma= 0.01, index = 50, output_path = output_path_root_no_bias + '_predictions.pdf')
    z_values_bias = predictions_analysis(data, predictions_bias_mean, bias=True, sigma = predictions_bias_std, index=50, output_path = output_path_root_bias + '_predictions.pdf')
    plot_comparison_z_values(z_values_no_bias, z_values_bias, output_path_root_bias + '_comparison_50.pdf')

    z_values_no_bias = predictions_analysis(data, predictions_no_bias, bias=False, sigma= 0.01, index = 90, output_path = output_path_root_no_bias + '_predictions.pdf')
    z_values_bias = predictions_analysis(data, predictions_bias_mean, bias=True, sigma = predictions_bias_std, index=90, output_path = output_path_root_bias + '_predictions.pdf')
    plot_comparison_z_values(z_values_no_bias, z_values_bias, output_path_root_bias + '_comparison_90.pdf')

    plot_z_values_1d(data, predictions_no_bias, bias=False, sigma= 0.01, output_path= output_path_root_no_bias + '_z_value_1d.pdf')
    plot_z_values_1d(data, predictions_bias_mean, bias=True, sigma = predictions_bias_std,output_path= output_path_root_bias + '_z_value_1d.pdf')

    plot_residuals_1d(data, predictions_no_bias,output_path= output_path_root_no_bias + '_residuals_1d.pdf')
    plot_residuals_1d(data, predictions_bias_mean,output_path= output_path_root_bias + '_residuals_1d.pdf')

    variance_decomposition_analysis(data, predictions_no_bias, bias=False, sigma= 0.01,output_path= output_path_root_no_bias + '_variance_decomposition.pdf')
    variance_decomposition_analysis(data, predictions_bias_mean, bias=True, sigma = predictions_bias_std,output_path= output_path_root_bias + '_variance_decomposition.pdf')