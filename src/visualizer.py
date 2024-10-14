import matplotlib.pyplot as plt
import gensim
import numpy as np
import os
from src.clusters_data_saver import ClustersData


class Visualizer:

    def __init__(self, cluster_saver: ClustersData, vectors: dict) -> None:
        self.cluster_saver = cluster_saver
        self.healthy_data_clean = cluster_saver.get_df('healthy')[['ID',
                                                                   'C6(a)-clean',
                                                                   'C6(b)-clean',
                                                                   'C6(c)-clean']]
        self.healthy_data_all = cluster_saver.get_df('healthy')[['ID',
                                                                 'C6(a)-clean-all-lexemes',
                                                                 'C6(b)-clean-all-lexemes',
                                                                 'C6(c)-clean-all-lexemes']]
        self.aphasia_data_clean = cluster_saver.get_df('aphasia')[['ID',
                                                                   'C6(a)-clean',
                                                                   'C6(b)-clean',
                                                                   'C6(c)-clean']]
        self.aphasia_data_all = cluster_saver.get_df('aphasia')[['ID',
                                                                 'C6(a)-clean-all-lexemes',
                                                                 'C6(b)-clean-all-lexemes',
                                                                 'C6(c)-clean-all-lexemes']]
        self.vectors = vectors

    def cosine_similarity(self, w1, w2):
        v1 = np.array(self.vectors.get(w1))
        v2 = np.array(self.vectors.get(w2))

        return np.dot(gensim.matutils.unitvec(v1),
                      gensim.matutils.unitvec(v2))

    def visualize_linear(self, sheet: str, id: str, lexemes: str):
        """
    dataset - one of 4 dataframes
    """
        if sheet == 'healthy':
            if lexemes == 'clean':
                dataset = self.healthy_data_clean
            else:
                dataset = self.healthy_data_all
        else:
            if lexemes == 'clean':
                dataset = self.aphasia_data_clean
            else:
                dataset = self.aphasia_data_all

        data = dataset[dataset['ID'] == id]  # data for a specific user

        fig, axs = plt.subplots(3, figsize=(10, 15))
        custom_lines = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='First Response'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Switch'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster')]

        for i, columns in enumerate(data.columns[1:]):  # getting names of 3 columns we need
            words = [item for sublist in data[columns].values[0] for item in sublist]  # list of all words
            first_words = [sublist[0] for sublist in data[columns].values[0]]
            ax = axs[i]
            y_min = 0

            for idx in range(len(words)):
                label = words[idx]
                if idx == 0:
                    y = 0.0
                else:
                    y = self.cosine_similarity(words[idx], words[idx - 1])

                if y < y_min:
                    y_min = y

                color = 'red' if idx == 0 else 'yellow' if words[idx] in first_words else 'blue'
                ax.scatter(idx, y, label=label, color=color)

                if idx == 1:
                    ax.plot([idx - 1, idx], [0.0, y], color='gray', linewidth=0.8)

                if idx > 1:
                    ax.plot([idx - 1, idx], [self.cosine_similarity(words[idx - 2], words[idx - 1]), y], color='gray',
                            linewidth=0.8)

                ax.annotate(label, (idx, y), textcoords="offset points", xytext=(0, 10), fontsize=8, ha='center')

            ax.set_ylim(y_min - 0.2, 1)
            ax.set_title(f'{columns} for {id}')
            ax.set_ylabel('Word2Vec Similarity')
            ax.legend(handles=custom_lines, loc='upper right')

        plt.tight_layout()

        directory = self.create_dir(dataset, id)

        self.save_image(directory)

    def visualize_avg_cluster_size(self):
        labels = ['Mean_cluster_size_clean', 'Mean_cluster_size_all']

        temp_df_healthy = self.cluster_saver.get_df('healthy')[labels]
        temp_df_aphasia = self.cluster_saver.get_df('aphasia')[labels]

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
        fig.suptitle('Average cluster number')

        bplot1 = ax1.boxplot(temp_df_healthy,
                             notch=True,
                             vert=True,
                             patch_artist=True,
                             labels=labels)
        ax1.set_title('Healthy control group')

        bplot2 = ax2.boxplot(temp_df_aphasia,
                             notch=True,
                             vert=True,
                             patch_artist=True,
                             labels=labels)
        ax2.set_title('Aphasia group')

        color1 = 'lightgreen'
        color2 = 'lightblue'
        for bplot in (bplot1, bplot2):
            for idx, patch in enumerate(bplot['boxes']):
                if idx == 0:
                    patch.set_facecolor(color1)
                else:
                    patch.set_facecolor(color2)

        plt.tight_layout()
        save_dir = '/visualization/metrics_visualization/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        directory = os.path.join(save_dir, 'mean_cluster_size.jpg')

        self.save_image(directory)

    def visualize_switch_num(self):
        labels = ['Switch_number_animals_clean',
                  'Switch_number_professions_clean',
                  'Switch_number_cities_clean',
                  'Switch_number_animals_clean-all-lexemes',
                  'Switch_number_professions_clean-all-lexemes',
                  'Switch_number_cities_clean-all-lexemes']

        temp_df_healthy = self.cluster_saver.get_df('healthy')[labels]
        temp_df_aphasia = self.cluster_saver.get_df('aphasia')[labels]

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
        fig.suptitle('Number of switches')

        bplot1 = ax1.boxplot(temp_df_healthy,
                             notch=True,
                             vert=True,
                             patch_artist=True,
                             labels=labels)
        ax1.set_title('Healthy control group')

        bplot2 = ax2.boxplot(temp_df_aphasia,
                             notch=True,
                             vert=True,
                             patch_artist=True,
                             labels=labels)
        ax2.set_title('Aphasia group')

        color1 = 'lightgreen'
        color2 = 'lightblue'
        for bplot in (bplot1, bplot2):
            for idx, patch in enumerate(bplot['boxes']):
                if idx < 3:
                    patch.set_facecolor(color1)
                else:
                    patch.set_facecolor(color2)

        plt.tight_layout()
        save_dir = '/visualization/metrics_visualization/'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        directory = os.path.join(save_dir, 'switch_num.jpg')

        self.save_image(directory)

    def create_dir(self, dataset, id):
        """
        Creating/finding a sufficient directory
        """

        if dataset is self.healthy_data_clean:
            folder_dir = 'healthy/clean'

        elif dataset is self.healthy_data_all:
            folder_dir = 'healthy/all_lexemes'

        elif dataset is self.aphasia_data_clean:
            folder_dir = 'aphasia/clean'

        else:
            folder_dir = 'aphasia/all_lexemes'

        save_dir = os.path.join('visualization/', folder_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        return os.path.join(save_dir, f'{id}.jpg')

    @staticmethod
    def save_image(save_dir):
        plt.savefig(save_dir, format='jpg')
        plt.close()

    # @staticmethod
    # def show_image(path):
    #   img = Image.open(path)
    #   img.show()

    def visualize_all(self, sheet):
        """
        Build 3 linear graphs for each datatype for a particular id,
        draw box plots for some metrics
        """
        for lexemes in ['clean', 'all_lexemes']:
            for id in self.cluster_saver.get_df(sheet)['ID'].values:
                self.visualize_linear(sheet=sheet, lexemes=lexemes, id=id)

        self.visualize_avg_cluster_size()
        self.visualize_switch_num()
