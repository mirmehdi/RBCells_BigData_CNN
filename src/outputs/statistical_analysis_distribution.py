# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:44:03 2024

@author: ELiana Lousada
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Load the CSV file
df = pd.read_csv('DataSet_segmentation_openCV.csv')

# Save the output to a file
with open('analysis_output.txt', 'w') as f:
    # Display the first few rows of the DataFrame
    f.write("First few rows of the DataFrame:\n")
    f.write(str(df.head()) + "\n\n")

    # Plotting the distributions for each metric
    metrics = ['CellArea', 'Cell_perimeter', 'cell_circularity']
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
