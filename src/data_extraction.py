import pandas as pd


class DataExtractionBase:

    def __init__(self, link: str) -> None:
        self.category_types = None

    def get_ids(self, sheet_name: str):
        pass

    def get_column_name(self, category: str):
        pass

    def get_series(self, sheet_name: str, category: str):
        pass


class DataExtractionAphasia(DataExtractionBase):

    def __init__(self, link: str) -> None:
        super().__init__(link)
        self.dataset_norm = pd.read_excel(link, sheet_name='healthy')
        self.dataset_aphasia = pd.read_excel(link, sheet_name='aphasia')
        self.category_types = {'animals': 'a',
                               'professions': 'b',
                               'cities': 'c'}

    def get_ids(self, sheet_name: str = 'healthy') -> int:
        """
        Getting ID column
        """
        if sheet_name == 'healthy':
            return self.dataset_norm['ID']
        return self.dataset_aphasia['ID']

    def get_column_name(self, category: str, lexemes: str) -> str:
        """
        Get column name from category and type of included lexemes;
        lexemes: clean | clean-all-lexemes
        """
        return f'C6({self.category_types.get(category)})-{lexemes}'

    def get_series(self,
                   sheet_name: str,
                   category: str,
                   lexemes: str = 'clean') -> pd.DataFrame:
        """
        Getting one of 12 columns:
          from one of the 2 pages of the dataset
          from one of the 3 categories
          from one of the types of lexemes

        sheet_name: healthy | aphasia
        category: animals | professions | cities
        lexemes: clean | clean-all-lexemes
        """
        if sheet_name == 'healthy':
            return self.dataset_norm[self.get_column_name(category, lexemes)]

        return self.dataset_aphasia[self.get_column_name(category, lexemes)]


class DataExtractionPDTexts(DataExtractionBase):

    def __init__(self, link: str) -> None:
        super().__init__(link)
        self.dataset_norm = pd.read_excel(link, sheet_name='healthy')
        self.dataset_pd = pd.read_excel(link, sheet_name='general_massive')
        self.category_types = ['tokens', 'tokens_without_stops', 'lemmas', 'lemmas_without_stops']

    def get_ids(self, sheet_name: str = 'healthy') -> int:
        """
        Getting ID column
        """
        if sheet_name == 'healthy':
            return self.dataset_norm['speakerID']
        return self.dataset_pd['ID']

    def get_series(self,
                   sheet_name: str,
                   category: str) -> pd.DataFrame:
        """
        Getting one of 8 columns:
          from one of the 2 pages of the dataset
          from one of the 4 categories

        sheet_name: healthy | PD
        category: tokens | tokens_without_stops | lemmas | lemmas_without_stops
        """
        if sheet_name == 'healthy':
            return self.dataset_norm[category]

        return self.dataset_pd[category]
