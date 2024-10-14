import gensim
import pandas as pd

from src.data_extraction import DataExtractionPDTexts
from src.clusters_data_saver import ClustersDataPDTexts
from src.clusterizer import Clusterizer
from src.vectorizer import Vectorizer

project_path = r'C:\Users\vchemsmisl\Desktop\programming\CLB_workproject'


def main():
    # defining classes
    model_path = rf'{project_path}\models\geowac\model.model'
    geowac_model = gensim.models.KeyedVectors.load(model_path)
    extractor = DataExtractionPDTexts(rf'{project_path}\data\control_pd_preprocessed.xlsx')
    vectoriser = Vectorizer(geowac_model)
    cluster_saver = ClustersDataPDTexts(extractor, geowac_model)
    clusters_getter = Clusterizer(geowac_model)

    # general principle: clustering one cell at a time
    DB_values_page = []
    silhouette_values_page = []

    for page in ['healthy', 'pd']:
        DB_values_lexemes_kind = []
        silhouette_values_lexemes_kind = []

        for category in extractor.category_types:
            sequence_series = extractor.get_series(page, category)  # getting words lists from a column
            clusters_list = []  # a list of lists of clusters for current column

            DB_values_column = []
            silhouette_values_column = []

            for words_string in sequence_series:
                if not isinstance(words_string, str):  # dealing with NaNs or other non-string values
                    clusters_list.append([])
                    continue

                tokens_sequence = vectoriser.get_sequence(words_string)
                # string of words coverted to list with special tags

                cell_clusters = clusters_getter.cluster(tokens_sequence)
                # converting list of words to list of clusters
                clusters_list.append(cell_clusters)

                DB_value = clusters_getter.davies_bouldin_index(cell_clusters)
                # calculating Davies Bouldin index for each cell
                if DB_value:
                    DB_values_column.append(DB_value)

                silhouette_value = clusters_getter.silhouette_score(cell_clusters)
                # calculating Silhouette score for each cell
                silhouette_values_column.append(silhouette_value)

            cluster_saver.add_column(page, category,
                                     pd.Series(clusters_list))
            # adding clusters column in a table

            # counting metrics
            cluster_saver.count_num_switches(page, category)
            cluster_saver.count_mean_cluster_size(page, category)
            cluster_saver.count_mean_distances(page, category)
            cluster_saver.count_mean_silhouette_score(page, category)
            cluster_saver.count_cluster_t_scores(page, category)

            DB_values_lexemes_kind.extend(DB_values_column)
            silhouette_values_lexemes_kind.extend(silhouette_values_column)

        DB_values_page.extend(DB_values_lexemes_kind)
        silhouette_values_page.extend(silhouette_values_lexemes_kind)

    clusters_getter.evaluate_clustering(DB_values_page, silhouette_values_page)
    cluster_saver.save_excel(rf'{project_path}\result\pd_texts\clusters_metrics_dataset.xlsx')
    # vectors = vectoriser.get_dictionary()

    # visualizer = Visualizer(cluster_saver, vectors)
    #
    # visualizer.visualize_all('healthy')
    # visualizer.visualize_all('aphasia')


if __name__ == '__main__':
    main()
