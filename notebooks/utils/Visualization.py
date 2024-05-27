import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set the theme for seaborn
sns.set_theme(style="ticks", context="talk", palette="bright")

class Visualization:
    def __init__(self, df):
        self.df = df

    def hist_plot(self,path):
        # Create the plot using Seaborn
        g = sns.displot(self.df['Label'], kind='hist')
        plt.show()  # Display the plot

        # Save the figure
        # Since displot returns a FacetGrid, we access the figure to save it
        g.figure.savefig(path)  # Use savefig with the path

    def dist_cell_area(self,path):
        # Create the plot using Seaborn
            a = self.df.groupby('Label')['CellArea'].mean().reset_index()
            # Create the plot using Seaborn
            plt.figure(figsize=(10, 6))
            bar_plot = sns.barplot(x='Label', y='CellArea', data=a)
            plt.title('Mean Cell Area per Label')
            plt.xlabel('Label')
            plt.ylabel('Mean Cell Area')
            plt.show()
    
        # Save the figure
            bar_plot.get_figure().savefig(path) 


    def img_size_dist(self,path):
        df = self.df
        df['Dimension'] = df['ImageSize_D1'].astype(str) + 'x' + df['ImageSize_D2'].astype(str)

        plt.figure(figsize=(12, 6))
        sns.countplot(x='Label', hue='Dimension', data=df)
        plt.title('Image Dimensions per Label')
        plt.xlabel('Label')
        plt.ylabel('Frequency')
        plt.legend(title='Dimension', loc='upper right')
        plt.show()
        
        plt.figure.savefig(path)  # Use savefig with the path

