from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns

from tqdm import tqdm

def utility_plot_3d(df: pd.DataFrame, output_path: str, valor: str, colorbar_limit: tuple = None):
    avg_z_x3 = df.groupby(['x1', 'x2'])['z'].mean().reset_index()

    # Average z over x1 for x2 vs x3
    avg_z_x1 = df.groupby(['x2', 'x3'])['z'].mean().reset_index()

    # Average z over x2 for x1 vs x3
    avg_z_x2 = df.groupby(['x1', 'x3'])['z'].mean().reset_index()

    # Step 4: Create the plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.delaxes(axs[1, 1])

    # 2D Scatter Plot: x1 vs x2, avg(z) over x3
    scatter1 = axs[0, 0].scatter(
        avg_z_x3['x1'], avg_z_x3['x2'], c=avg_z_x3['z'], cmap='viridis', s=40,
        vmin=colorbar_limit[0] if colorbar_limit else None,
        vmax=colorbar_limit[1] if colorbar_limit else None
    )
    # axs[0, 0].set_title('x1 vs x2 (avg over x3)')
    axs[0, 0].set_title(r'$x_1$ vs $x_2$ ($x_3$ promedio)')
    axs[0, 0].set_xlabel(r'$x_1$')
    axs[0, 0].set_ylabel(r'$x_2$')
    fig.colorbar(scatter1, ax=axs[0, 0], shrink=0.8)

    # 2D Scatter Plot: x2 vs x3, avg(z) over x1
    scatter2 = axs[0, 1].scatter(avg_z_x1['x2'], avg_z_x1['x3'], c=avg_z_x1['z'], cmap='viridis', s=40,
        vmin=colorbar_limit[0] if colorbar_limit else None,
        vmax=colorbar_limit[1] if colorbar_limit else None
    )
    # axs[0, 1].set_title('x2 vs x3 (avg over x1)')
    axs[0, 1].set_title(r'$x_2$ vs $x_3$ ($x_1$ promedio)')
    axs[0, 1].set_xlabel(r'$x_2$')
    axs[0, 1].set_ylabel(r'$x_3$')
    fig.colorbar(scatter2, ax=axs[0, 1], shrink=0.8)

    # 2D Scatter Plot: x1 vs x3, avg(z) over x2
    scatter3 = axs[1, 0].scatter(avg_z_x2['x1'], avg_z_x2['x3'], c=avg_z_x2['z'], cmap='viridis', s=40,
        vmin=colorbar_limit[0] if colorbar_limit else None,
        vmax=colorbar_limit[1] if colorbar_limit else None
    )
    # axs[1, 0].set_title('x1 vs x3 (avg over x2)')
    axs[1, 0].set_title(r'$x_1$ vs $x_3$ ($x_2$ promedio)')
    axs[1, 0].set_xlabel(r'$x_1$')
    axs[1, 0].set_ylabel(r'$x_3$')
    fig.colorbar(scatter3, ax=axs[1, 0], shrink=0.8)

    # 3D Scatter Plot: x1, x2, x3 with z as colormap
    ax = fig.add_subplot(224, projection='3d')
    sc = ax.scatter(df['x1'], df['x2'], df['x3'], c=df['z'], cmap='viridis', s=40,
        vmin=colorbar_limit[0] if colorbar_limit else None,
        vmax=colorbar_limit[1] if colorbar_limit else None
    )  # Increased dot size
    # ax.set_title('3D Scatter Plot')
    ax.set_title('Gráfico de dispersión 3D')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    # ax.set_zlabel(r'$x_3$')
    fig.colorbar(sc, ax=ax, shrink=0.5, orientation='vertical')

    plt.savefig(output_path)
        
def utility_plot_1d(df: pd.DataFrame, output_path: str, value: str):

    plt.figure(figsize=(12, 6))

    # Plot z values
    plt.subplot(1, 2, 1)
    # plt.plot(df['z'], label='z values')
    plt.plot(df['z'], label=value)
    plt.ylim(0, np.percentile(df['z'], 99))
    # plt.axhline(y=df['z'].mean(), color='r', linestyle='--', label='Mean')
    plt.axhline(y=df['z'].median(), color='r', linestyle='--', label='Mediana')
    plt.text(len(df['z']) * 0.8, df['z'].median(), f'Mediana: {df["z"].median():.2f}', color='r', va='bottom', ha='center')
    # plt.xlabel('Index')
    plt.xlabel('Índice')
    # plt.ylabel('z')
    plt.ylabel(value)
    # plt.title('z values and Mean')
    plt.title(value + ' y mediana')
    plt.legend()
    plt.grid(True)

    # Plot histogram
    plt.subplot(1, 2, 2)
    plt.hist(df['z'], bins=30, range=(0, np.percentile(df['z'], 95)), edgecolor='k', alpha=0.7)
    # plt.axvline(df['z'].mean(), color='r', linestyle='--', label='Mean')
    plt.axvline(df['z'].median(), color='r', linestyle='--', label='Mediana')
    plt.text(df['z'].median() * 1.1, plt.gca().get_ylim()[1] * 0.9, f'Mediana: {df["z"].median():.2f}', color='r', va='bottom', ha='left')
    # plt.xlabel('z')
    plt.xlabel(value)
    # plt.ylabel('Frequency')
    plt.ylabel('Frecuencia')
    # plt.title('Histogram of z values')
    plt.title('Histograma de ' + value)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)

def convergence_analysis(inference_data: az.InferenceData, output_path: str):

    ess = az.ess(inference_data)
    # print("Effective Sample Size (ESS):")
    print("Tamaño de muestra efectivo (ESS):")
    print(ess)
    ess_list_a = []
    ess_list_b = []
    ess_list_sigma_b = []

    for i in tqdm(range(1, int(len(az.extract(inference_data)["$a$"].values)/len(inference_data.posterior["$a$"])) + 1), desc="Calculating ESS"):
        ess_subset = az.ess(inference_data.sel(draw=slice(None, i)))
        ess_list_a.append(ess_subset["$a$"].values)
        ess_list_b.append(ess_subset["$b$"].values)
        ess_list_sigma_b.append(ess_subset["$\sigma_b$"].values)

    ess_list_a = [x for x in ess_list_a if not np.isnan(x)]
    ess_list_b = [x for x in ess_list_b if not np.isnan(x)]
    ess_list_sigma_b = [x for x in ess_list_sigma_b if not np.isnan(x)]
    
    plt.figure(figsize=(10, 6))
    plt.axhline(y=837, color='r', linestyle='--')
    plt.text(20, 837, r'$\widehat{\text{ESS}}=837$', color='r', va='bottom', ha='center')
    plt.plot(ess_list_a, label=r'$\widehat{\text{ESS}}$ $a$')
    plt.plot(ess_list_b, label=r'$\widehat{\text{ESS}}$ $b$')
    plt.plot(ess_list_sigma_b, label=r'$\widehat{\text{ESS}}$ $\sigma_b$')
    # plt.xlabel('Number of samples')
    plt.xlabel('Número de muestras')
    plt.xlim(0, len(ess_list_a))
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
    # print("Highest Density Interval (HDI):")
    print("Intervalo de mayor densidad (HDI):")
    print(hdi)
    print(az.summary(inference_data, hdi_prob=1-alpha))
    # Save summary to LaTeX table format
    summary_df = az.summary(inference_data, hdi_prob=1-alpha).reset_index()
    with open(output_path.replace('.json', '_summary.tex'), 'w') as f:
        summary_df.to_latex(f, index=False)
        

def predictions_analysis(data:pd.DataFrame, predictions: pd.DataFrame, bias: bool, sigma: Union[float, pd.DataFrame], index: int = 0, output_path: str = None):

    if bias:
        z_values = np.abs(data["y"].values[index] - predictions.values.transpose()[index]) / sigma.values.transpose()[index]
    else:
        z_values = np.abs(data["y"].values[index] - predictions.values.transpose()[index]) / sigma
    # z_values = z_values.values.flatten()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(z_values, fill=True)
    # plt.axvline(np.mean(z_values), color='r', linestyle='--', label='Mean')
    plt.axvline(np.mean(z_values), color='r', linestyle='--', label='Media')
    plt.axvline(1.96, color='k', linestyle='--', label=r'Normal $3\sigma$')
    # plt.xlabel('Z-values')
    plt.xlabel('Valores Z')
    # plt.ylabel('Density')
    plt.ylabel('Densidad')
    # plt.title('Density Plot of Z-values at index {}'.format(index))
    plt.title('Gráfico de densidad de los valores Z en el índice {}'.format(index))
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path.replace('.pdf', '_{}.pdf'.format(index)))

    return z_values


def plot_comparison_z_values(z_values_no_bias: np.ndarray, z_values_bias: np.ndarray, output_path: str):

    plt.figure(figsize=(10, 6))
    # sns.kdeplot(z_values_no_bias, fill=True, color = "r", label='No Bias')
    sns.kdeplot(z_values_no_bias, fill=True, color="r", label='Sin sesgo')
    # sns.kdeplot(z_values_bias, fill=True, color="g", label='Bias')
    sns.kdeplot(z_values_bias, fill=True, color="g", label='Sesgo')
    # plt.axvline(np.mean(z_values_no_bias), color='r', linestyle='--', label='Mean No Bias')
    plt.axvline(np.mean(z_values_no_bias), color='r', linestyle='--', label='Media sin sesgo')
    # plt.axvline(np.mean(z_values_bias), color='g', linestyle='--', label='Mean Bias')
    plt.axvline(np.mean(z_values_bias), color='g', linestyle='--', label='Media con sesgo')
    plt.axvline(1.96, color='k', linestyle='--', label=r'Normal $3\sigma$')
    # plt.xlabel('Z-values')
    plt.xlabel('Valores Z')
    # plt.ylabel('Density')
    plt.ylabel('Densidad')
    # plt.title('Density Plot of Z-values')
    plt.title('Gráfico de densidad de los valores Z')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)

def plot_z_values_3d(data: pd.DataFrame, predictions: pd.DataFrame, bias: bool, sigma: Union[float, pd.DataFrame], output_path: str):

    if bias:
        z_values = np.abs(data["y"].values - np.mean(predictions.values, axis=0)) / np.mean(sigma.values, axis=0)
    else:
        z_values = np.abs(data["y"].values - np.mean(predictions.values, axis=0)) / sigma

    df = pd.DataFrame({'x1': data['x1'], 'x2': data['x2'], 'x3':data['x3'],'z': z_values})
    colorbar_limit = (0, np.percentile(df['z'], 95))
    utility_plot_3d(df, output_path, "Valores Z", colorbar_limit=colorbar_limit)

def plot_z_values_1d(data: pd.DataFrame, predictions: pd.DataFrame, bias: bool, sigma: Union[float, pd.DataFrame], output_path: str):

    if bias:
        z_values = np.abs(data["y"].values - np.mean(predictions.values, axis=0)) / np.mean(sigma.values, axis=0)
    else:
        z_values = np.abs(data["y"].values - np.mean(predictions.values, axis=0)) / sigma

    df = pd.DataFrame({'x1': data['x1'], 'x2': data['x2'], 'x3':data['x3'],'z': z_values})

    utility_plot_1d(df, output_path, "Valores Z")

def plot_residuals_1d(data: pd.DataFrame, predictions: pd.DataFrame, output_path: str):
    
    residual = np.abs(data["y"].values - np.mean(predictions.values, axis=0))

    df = pd.DataFrame({'x1': data['x1'], 'x2': data['x2'], 'x3':data['x3'],'z': residual})
    utility_plot_1d(df, output_path, "Residuos")

def plot_residuals_3d(data: pd.DataFrame, predictions: pd.DataFrame, output_path: str):
    
    residual = np.abs(data["y"].values - np.mean(predictions.values, axis=0))

    df = pd.DataFrame({'x1': data['x1'], 'x2': data['x2'], 'x3':data['x3'],'z': residual})
    utility_plot_3d(df, output_path, "Residuos")

def variance_decomposition_analysis(data: pd.DataFrame, predictions: pd.DataFrame, bias: bool, sigma: Union[float, pd.DataFrame], output_path: str):

    residuals = np.abs(data["y"].values - np.mean(predictions.values, axis=0))
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
        variance_vector = np.square(sigma * np.ones(len(residuals)))  # Scalar case

    # Compute explained and unexplained variance
    explained_variance = np.sum(variance_vector)
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
    plt.xlabel('Índice')
    # plt.ylabel('Value')
    plt.ylabel('Valor')
    # plt.title('Residuals and Variance at Each Prediction Point')
    plt.title('Residuos y varianza en cada punto de predicción')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path.replace('.pdf', '_residuals_sigma.pdf'))
    plt.show()


if __name__ == "__main__":

    data = pd.read_csv('./input/data/ishigami_dataset.csv')
    large_data = pd.read_csv('./input/data/ishigami_large_dataset.csv')

    inference_data_bias = az.from_netcdf('./output/results/calibrate_bias_ishigami_3.az')
    inference_data_no_bias = az.from_netcdf('./output/results/calibrate_no_bias_ishigami_3.az')
    output_path_root_bias = './output/figures/ishigami_bias'
    output_path_root_no_bias = './output/figures/ishigami_no_bias'

    convergence_analysis(inference_data_bias, output_path_root_bias + '_convergence.pdf')
    fit_analysis(inference_data_bias, output_path_root_bias + '_fit.json')
    fit_analysis(inference_data_no_bias, output_path_root_no_bias + '_fit.json')

    predictions_no_bias = pd.read_csv('./output/results/ishigami_nobias_predictions.csv', header=0)
    predictions_bias_mean = pd.read_csv('./output/results/ishigami_bias_predictions_mean.csv', header=0)
    predictions_bias_std = pd.read_csv('./output/results/ishigami_bias_predictions_std.csv', header=0)

    z_values_no_bias = predictions_analysis(data, predictions_no_bias, bias=False, sigma= 0.1, index = 0, output_path = output_path_root_no_bias + '_predictions.pdf')
    z_values_bias = predictions_analysis(data, predictions_bias_mean, bias=True, sigma = predictions_bias_std, index=0, output_path = output_path_root_bias + '_predictions.pdf')
    plot_comparison_z_values(z_values_no_bias, z_values_bias, output_path_root_bias + '_comparison_0.pdf')

    z_values_no_bias = predictions_analysis(data, predictions_no_bias, bias=False, sigma= 0.1, index = 50, output_path = output_path_root_no_bias + '_predictions.pdf')
    z_values_bias = predictions_analysis(data, predictions_bias_mean, bias=True, sigma = predictions_bias_std, index=50, output_path = output_path_root_bias + '_predictions.pdf')
    plot_comparison_z_values(z_values_no_bias, z_values_bias, output_path_root_bias + '_comparison_50.pdf')

    z_values_no_bias = predictions_analysis(data, predictions_no_bias, bias=False, sigma= 0.1, index = 90, output_path = output_path_root_no_bias + '_predictions.pdf')
    z_values_bias = predictions_analysis(data, predictions_bias_mean, bias=True, sigma = predictions_bias_std, index=90, output_path = output_path_root_bias + '_predictions.pdf')
    plot_comparison_z_values(z_values_no_bias, z_values_bias, output_path_root_bias + '_comparison_90.pdf')

    plot_z_values_3d(large_data, predictions_no_bias, bias=False, sigma= 0.1,output_path= output_path_root_no_bias + '_z_value_3d.pdf')
    plot_z_values_3d(large_data, predictions_bias_mean, bias=True, sigma = predictions_bias_std,output_path= output_path_root_bias + '_z_value_3d.pdf')

    plot_residuals_3d(large_data, predictions_no_bias,output_path= output_path_root_no_bias + '_residuals_3d.pdf')
    plot_residuals_3d(large_data, predictions_bias_mean,output_path= output_path_root_bias + '_residuals_3d.pdf')

    plot_z_values_1d(large_data, predictions_no_bias, bias=False, sigma= 0.1,output_path= output_path_root_no_bias + '_z_value_1d.pdf')
    plot_z_values_1d(large_data, predictions_bias_mean, bias=True, sigma = predictions_bias_std,output_path= output_path_root_bias + '_z_value_1d.pdf')

    plot_residuals_1d(large_data, predictions_no_bias,output_path= output_path_root_no_bias + '_residuals_1d.pdf')
    plot_residuals_1d(large_data, predictions_bias_mean,output_path= output_path_root_bias + '_residuals_1d.pdf')

    variance_decomposition_analysis(large_data, predictions_no_bias, bias=False, sigma= 0.1,output_path= output_path_root_no_bias + '_variance_decomposition.pdf')
    variance_decomposition_analysis(large_data, predictions_bias_mean, bias=True, sigma = predictions_bias_std,output_path= output_path_root_bias + '_variance_decomposition.pdf')