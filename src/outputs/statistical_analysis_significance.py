# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:05:15 2024

@author: Eliana Lousada
"""

# Import Required Libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import DataFrame 
from statistical_analysis_distribution import df

# Define Variables
metrics = ['CellArea', 'Cell_perimeter', 'cell_circularity']
cell_types = df['Label'].unique()

# Define significance level after Bonferroni correction
alpha = 0.05 / len(metrics)  # Adjusted significance level

# Create a dictionary to store pairwise test results
pairwise_results = {}

# Perform pairwise Mann-Whitney U tests with Bonferroni correction
for metric in metrics:
    pairwise_results[metric] = {}
    for i in range(len(cell_types)):
        for j in range(i + 1, len(cell_types)):
            label1 = cell_types[i]
            label2 = cell_types[j]
            data1 = df[df['Label'] == label1][metric]
            data2 = df[df['Label'] == label2][metric]
            stat, p = stats.mannwhitneyu(data1, data2)
            adjusted_alpha = alpha / (len(cell_types) * (len(cell_types) - 1) / 2)  # Bonferroni correction
            pairwise_results[metric][(label1, label2)] = (stat, p)

# Plot the distributions using violin plots
for metric in metrics:
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x='Label', y=metric, data=df, order=cell_types)
    ax.set_title(f'{metric} for different cell types')
    ax.set_ylabel(metric)
    ax.set_xlabel('Cell Type')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{metric}_violinplot.png')
    plt.close()

# Create significance matrices for each metric
significance_matrices = {}

for metric in metrics:
    significance_matrix = np.zeros((len(cell_types), len(cell_types)))

    for i, label1 in enumerate(cell_types):
        for j, label2 in enumerate(cell_types):
            if i < j:  # Ensure only comparing pairs once
                data1 = df[df['Label'] == label1][metric]
                data2 = df[df['Label'] == label2][metric]
                _, p = stats.mannwhitneyu(data1, data2)
                adjusted_alpha = alpha / (len(cell_types) * (len(cell_types) - 1) / 2)  # Bonferroni correction
                if p < alpha:
                    significance_matrix[i, j] = 1  # Significantly different, set to 1
                else:
                    significance_matrix[i, j] = 0  # Not significantly different, set to 0
                significance_matrix[j, i] = significance_matrix[i, j]  # Since the matrix is symmetric
                
    significance_matrices[metric] = significance_matrix

# Generate heatmaps for each significance matrix
for metric, significance_matrix in significance_matrices.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(significance_matrix, annot=True, cmap='coolwarm', fmt=".0f", xticklabels=cell_types, yticklabels=cell_types)
    plt.title(f'Significance Matrix for {metric} Cell Type Comparisons')
    plt.xlabel('Cell Type')
    plt.ylabel('Cell Type')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'significance_matrix_heatmap_{metric}.png')
    plt.close()  # Close the figure to prevent displaying it

