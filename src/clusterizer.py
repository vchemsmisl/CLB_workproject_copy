# import fasttext
import gensim
import numpy as np


class Clusterizer:

    def __init__(self, model: gensim.models.fasttext.FastTextKeyedVectors) -> None:
        self._model = model

    def get_vector(self, word):
        """
        Getting vector depending on the model
        """
        if isinstance(self._model, gensim.models.fasttext.FastTextKeyedVectors):
            return self._model[word]

        return self._model.get_word_vector(word)

    def get_cosine_similarity(self, w1, w2):
        """
        Getting cosine similarity depending on model
        """
        if isinstance(self._model, gensim.models.fasttext.FastTextKeyedVectors):
            return self._model.similarity(w1, w2)

        v1 = self._model.get_word_vector(w1)
        v2 = self._model.get_word_vector(w2)

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def cluster(self, word_sequence: list[str]) -> list[list[str]]:
        """
        An implementation of words clustering algorithm
        for verbal fluency test results,
        from N. Lundin et al., 2022.
        b -- current word
        c -- next word
        d -- next word after the next
        """
        words_by_clusters = []
        cluster = []

        for idx, word in enumerate(word_sequence):
            if word == 'PEOS':
                break

            if word == 'BOS':
                b = word
                c = word_sequence[idx + 1]
                d = word_sequence[idx + 2]
                b_c_sim = self.get_cosine_similarity(b, c)
                c_d_sim = self.get_cosine_similarity(c, d)
                continue

            cluster.append(word)
            a_b_sim = b_c_sim   # S(A,B) equals S(B,C) from previous iteration
            b_c_sim = c_d_sim   # S(B,C) equals S(C,D) from previous iteration
            c_d_sim = self.get_cosine_similarity(word_sequence[idx + 1], word_sequence[idx + 2])

            if a_b_sim > b_c_sim and b_c_sim < c_d_sim:  # a condition of a switch
                words_by_clusters.append(cluster)
                cluster = []

        if cluster:
            words_by_clusters.append(cluster)
        return words_by_clusters

    @staticmethod
    def _custom_similarity(embedding_1, embedding_2):  # с этим что-то надо сделать, оно работает для фасттехта?
        return np.dot(gensim.matutils.unitvec(embedding_1),
                      gensim.matutils.unitvec(embedding_2))

    def davies_bouldin_index(self, cluster_sequence: list[list[str]]) -> float:
        """
        The Davies Bouldin index implementation,
        based on https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
        Si -- the average distance between each point of cluster i
        and the centroid of that cluster – also known as cluster diameter;
        Sj -- the average distance between each point of cluster j
        and the centroid of that cluster – also known as cluster diameter;
        Dij -- the distance between cluster centroids i and j;
        Rij -- similarity between clusters i and j.
        """
        centroids_dict = {}
        for cluster in cluster_sequence:
            centroid = sum(self.get_vector(word) for word in cluster) / len(cluster)
            centroids_dict[tuple(cluster)] = centroid

        Si_values_dict = {}
        for cluster in cluster_sequence:
            cluster_centroid = centroids_dict[tuple(cluster)]
            Si = sum(self._custom_similarity(self.get_vector(word), cluster_centroid)
                    for word in cluster) / len(cluster)
            Si_values_dict[tuple(cluster)] = Si

        Rij_max_values = []
        for cluster_1 in cluster_sequence:
            Rij_values = []
            Si = Si_values_dict[tuple(cluster)]

            for cluster_2 in cluster_sequence:
                if cluster_2 == cluster_1:
                    continue
                Sj = Si_values_dict[tuple(cluster_2)]
                Dij = self._custom_similarity(centroids_dict[tuple(cluster_1)], centroids_dict[tuple(cluster_2)])
                Rij = (Si + Sj) / Dij
                Rij_values.append(Rij)

            if Rij_values:
                Rij_max_values.append(max(Rij_values))

        return sum(Rij_max_values) / len(Rij_max_values) if Rij_max_values else None

    def silhouette_score(self, cluster_sequence: list[list[str]]) -> float:
        """
        The Silhouette score implementation,
        based on https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
        a -- the mean distance between a sample and all other points in the same class;
        b -- the mean distance between a sample and all other points in the next nearest cluster.
        """
        silhouette_coefs = []

        for idx, cluster in enumerate(cluster_sequence):
            for word_1 in cluster:

                a = sum(self.get_cosine_similarity(word_1, word_2)
                    for word_2 in cluster if word_1 != word_2) / len(cluster)

                if idx != len(cluster_sequence) - 1:
                    b = sum(self.get_cosine_similarity(word_1, word_2)
                    for word_2 in cluster_sequence[idx + 1]) / len(cluster_sequence[idx + 1])
                else:
                    b = sum(self.get_cosine_similarity(word_1, word_2)
                    for word_2 in cluster_sequence[idx - 1]) / len(cluster_sequence[idx - 1])

                if a == 0 or b == 0:
                    s = 0
                else:
                    s = (b - a) / max(a, b)
                silhouette_coefs.append(s)

        return sum(silhouette_coefs) / len(silhouette_coefs)

    @staticmethod
    def evaluate_clustering(DB_values_page: list[float], silhouette_values: list[float]) -> None:
        """
        The computation of all clustering metrics
        to evaluate given clustering model.
        """
        mean_DB_index_value = sum(DB_values_page) / len(DB_values_page)
        mean_silhouette_score_value = sum(silhouette_values) / len(silhouette_values)
        print('The performance of this clustering algorithm: ')
        print(f'Mean value of Davies Bouldin index: {mean_DB_index_value}')
        print(f'Mean value of Silhouette score: {mean_silhouette_score_value}')