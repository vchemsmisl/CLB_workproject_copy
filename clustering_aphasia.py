import gensim
import pandas as pd

from src.data_extraction import DataExtractionAphasia
from src.vectorizer import Vectorizer
from src.clusters_data_saver import ClustersDataAphasia
from src.clusterizer import Clusterizer
from src.visualizer import Visualizer


def main():
    model_path = 'C:\pyproj\CLB_workproject\models\geowac\model.model'
    geowac_model = gensim.models.KeyedVectors.load(model_path)
    extractor = DataExtractionAphasia('/content/cleaned_dataset.xlsx')
    vectoriser = Vectorizer(geowac_model)
    cluster_saver = ClustersDataAphasia(extractor, geowac_model)
    clusters_getter = Clusterizer(geowac_model)

    # general principle: clusterising one cell at a time
    DB_values_page = []
    silhouette_values_page = []
    for page in ['healthy', 'aphasia']:
        DB_values_lexemes_kind = []
        silhouette_values_lexemes_kind = []

        for lexemes in ['clean', 'clean-all-lexemes']:
            DB_values_category = []
            silhouette_values_category = []

            for category in ['animals', 'professions', 'cities']:
                sequence_series = extractor.get_series(page, category, lexemes)  # getting words lists from a column
                clusters_list = []  # a list of lists of clusters for current column

                DB_values_column = []
                silhouette_values_column = []
                for words_string in sequence_series:
                    if not isinstance(words_string, str):  # dealing with NaNs or other non-string values
                        clusters_list.append([])
                        continue

                    tokens_sequence = vectoriser.get_sequence(
                        words_string)  # string of words coverted to list with special tags
                    vectoriser.update_dict(words_string)  # adding words embeddings to a dict
                    cell_clusters = clusters_getter.cluster(
                        tokens_sequence)  # converting list of words to list of clusters
                    clusters_list.append(cell_clusters)

                    DB_value = clusters_getter.davies_bouldin_index(
                        cell_clusters)  # calculating Davies Bouldin index for each cell
                    if DB_value:
                        DB_values_column.append(DB_value)
                    silhouette_value = clusters_getter.silhouette_score(
                        cell_clusters)  # calculating Silhouette score for each cell
                    silhouette_values_column.append(silhouette_value)

                cluster_saver.add_column(page, category, lexemes,
                                         pd.Series(clusters_list))  # adding clusters column in a table
                cluster_saver.count_switches(page, category, lexemes)  # counting number of switches
                vectoriser.update_json()  # updating a json-file with words embeddings from a dict for each category

                DB_values_category.extend(DB_values_column)
                silhouette_values_category.extend(silhouette_values_column)

            DB_values_lexemes_kind.extend(DB_values_category)
            silhouette_values_lexemes_kind.extend(silhouette_values_category)

        cluster_saver.count_mean(page)  # counting mean number of clusters for each person
        DB_values_page.extend(DB_values_lexemes_kind)
        silhouette_values_page.extend(silhouette_values_lexemes_kind)

    cluster_saver.save_excel()
    # vectors = vectoriser.get_dictionary()

    # visualizer = Visualizer(cluster_saver, vectors)
    #
    # visualizer.visualize_all('healthy')
    # visualizer.visualize_all('aphasia')


if __name__ == '__main__':
    main()
