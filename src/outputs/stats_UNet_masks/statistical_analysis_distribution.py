"""
Created on Sun May 19 13:44:03 2024

@author: ELiana Lousada
"""
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd

# Load the CSV file
df = pd.read_csv('clean_data.csv')

# Save the output to a file
with open('analysis_output.txt', 'w') as f:
    # Display the first few rows of the DataFrame
    f.write("First few rows of the DataFrame:\n")
    f.write(str(df.head()) + "\n\n")

    # Plotting the distributions for each metric
    metrics = ['White_Area']
    cell_types = df['Label'].unique()

    for metric in metrics:
        plt.figure(figsize=(14, 8))
        sns.histplot(data=df, x=metric, hue='Label', kde=True, element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {metric} for different cell types')
        plt.savefig(f'{metric}_distribution.png')
        plt.close()
        f.write(f"Saved {metric}_distribution.png\n\n")

    # Assess normality for each metric and each cell type
    f.write("Shapiro-Wilk test results for normality:\n")
    for metric in metrics:
        for label in cell_types:
            data = df[df['Label'] == label][metric]
            stat, p = stats.shapiro(data)
            f.write(f'Shapiro-Wilk test for {metric} of {label}:\n')
            f.write(f'Statistic: {stat}, p-value: {p}\n\n')

    # Anderson-Darling test results
    f.write("Anderson-Darling test results for normality:\n")
    for metric in metrics:
        for label in cell_types:
            data = df[df['Label'] == label][metric]
            result = stats.anderson(data, dist='norm')
            f.write(f'Anderson-Darling test for {metric} of {label}:\n')
            f.write(f'Statistic: {result.statistic}, critical values: {result.critical_values}, significance levels: {result.significance_level}\n\n')

    # Q-Q plots
    for metric in metrics:
        for label in cell_types:
            data = df[df['Label'] == label][metric]
            plt.figure(figsize=(6, 6))
            stats.probplot(data, dist="norm", plot=plt)
            plt.title(f'Q-Q plot for {metric} of {label}')
            plt.savefig(f'{metric}_qqplot_{label}.png')
            plt.close()
            f.write(f"Saved Q-Q plot for {metric} of {label}\n\n")
