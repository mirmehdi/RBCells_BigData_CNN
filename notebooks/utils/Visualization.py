import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style="ticks", context="talk", palette="bright")

class Visualization:
    def __init__(self, df):
        self.df = df

    def hist_plot(self):
        sns.displot(self.df['Label'], kind='hist')
        plt.show()
