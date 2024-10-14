import gensim
import json


class Vectorizer:
    def __init__(self, model: gensim.models.fasttext.FastTextKeyedVectors) -> None:
        self.model = model
        self._vectors_dictionary = {'BOS': self.get_vector('BOS').tolist(),
                                    'EOS': self.get_vector('EOS').tolist(),
                                    'PEOS': self.get_vector('PEOS').tolist()}

    def get_vector(self, word):
        """
        Getting vectors depending on the model, that is given
        """
        if isinstance(self.model, gensim.models.fasttext.FastTextKeyedVectors):
            return self.model[word]
        return self.model.get_word_vector(word)

    def update_dict(self, words: str) -> None:
        """
        Updating the dictionary during each cell vectorising
        """
        for one_word in words.split(', '):
            if one_word not in self._vectors_dictionary:
                self._vectors_dictionary[one_word] = self.get_vector(one_word).tolist()

    def update_json(self) -> None:
        """
        Updating and saving the json file
        """
        with open("/content/vectors.json", "w") as fp:
            json.dump(self._vectors_dictionary, fp, ensure_ascii=False)

    def get_dictionary(self) -> dict:
        """
        In case we need to get the dictionary
        """
        return self._vectors_dictionary

    @staticmethod
    def get_sequence(words_string: str) -> list[str]:
        """
        Getting a list of tokens + tags of beginning and ending
        BOS -- Beginning of Sentence
        PEOS -- pre-End of Sentence
        EOS -- End of Sentence
        """
        return ['BOS'] + words_string.split(', ') + ['PEOS', 'EOS']

