"""
Tool for normalizing terms based on an example based coding.
"""

__version__ = "0.1"
__author__ = "Tomohiro Nishiyama"
__credits__ = "Social Computing Laboratory"

from collections import Counter
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm


class EntityDictionary:
    def __init__(self, path, source_column, target_column, index: bool = False):
        self.df = pd.read_csv(path)

        source_column = self.__parse_column(source_column, index)
        target_column = self.__parse_column(target_column, index)

        self.source_column = self.df.iloc[:, source_column].to_list()
        self.target_column = self.df.iloc[:, target_column].to_list()

    def __parse_column(self, column: str, index: bool) -> int:
        if index:
            return int(column)
        return self.df.columns.to_list().index(column)

    def get_candidates_list(self):
        return self.source_column

    def get_normalization_list(self):
        return self.target_column

    def get_normalized_term(self, term):
        return self.target_column[self.source_column.index(term)]


class EntityNormalizer:
    def __init__(
        self,
        database: EntityDictionary,
        matching_method=fuzz.ratio,
        matching_threshold=0,
    ):
        self.database = database
        self.matching_method = matching_method
        self.matching_threshold = matching_threshold
        self.candidates = [
            x for x in self.database.get_candidates_list()
        ]
        

    def normalize(self, term) -> str:
        limit = 10

        
        preferred_candidate = process.extract(
            term, self.candidates, scorer=self.matching_method, limit=limit
        )
        
        # 最大値を持つタプルのみを抽出する
        numbers = [item[1] for item in preferred_candidate]
        preferred_candidate = [item for item in preferred_candidate if item[1] ==  max(numbers)]
        preferred_candidates = [candidate[0] for candidate in preferred_candidate]
        scores = [candidate[1] for candidate in preferred_candidate]
        
        normalized_candidates = [self.database.get_normalized_term(candidate) for candidate in preferred_candidates]        
        normalized_candidates = [candidate for candidate in normalized_candidates if isinstance(candidate, str)]
        
        dict_score = dict(zip(normalized_candidates, scores))

        # タプル内の一つ目の要素の出現回数を数える
        counts = Counter(item for item in normalized_candidates)
        
        if len(counts) == 0:
            return None, 0
        else:
            # 最もカウントが多い要素とそのカウント数を返す
            most_common_element, count = counts.most_common(1)[0]

            preferred_candidate = most_common_element
            score = count / limit # どのくらいの割合で出現しているか
            return preferred_candidate, dict_score[preferred_candidate]


def normalize(
    entities: list,
    dictionary: EntityDictionary,
    matching_method=fuzz.ratio,
    matching_threshold=0,
) -> list:
    normalizer = EntityNormalizer(dictionary, matching_method, matching_threshold)
    normalized = []
    scores = []
    for entity in tqdm(entities):
        normalization, score = normalizer.normalize(entity)
        normalized.append(str(normalization))
        scores.append(score)
    return normalized, scores
