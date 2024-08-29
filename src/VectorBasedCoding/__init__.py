"""
Tool for normalizing terms based on an example based coding.
"""

__version__ = "0.1"
__author__ = "Tomohiro Nishiyama"
__credits__ = "Social Computing Laboratory"

import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss


class EntityDictionary:
    def __init__(self, path, source_column, target_column, MODEL_NAME, MODEL_WEIGHT, index: bool = False):
        self.df = pd.read_csv(path)

        source_column = self.__parse_column(source_column, index)
        target_column = self.__parse_column(target_column, index)

        self.source_column = self.df.iloc[:, source_column].to_list()
        self.target_column = self.df.iloc[:, target_column].to_list()
        
        self.model = SentenceTransformer(MODEL_NAME)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.model_weight = torch.load(MODEL_WEIGHT, map_location=device)
        self.model.load_state_dict(self.model_weight)
        self.model.to(device)

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
    
    def get_model(self):
        return self.model
    
    def get_surface_form_embeddings(self):
        return np.array(self.model.encode([str(x) for x in self.source_column]))

    def get_index_general_form_embeddings(self):
        corpus_embeddings_np = np.array(self.model.encode([str(x) for x in self.target_column]))
        
        # FAISSインデックスの作成
        index = faiss.IndexFlatL2(corpus_embeddings_np.shape[1])
        index.add(corpus_embeddings_np)
        return index
    
    def get_general_form(self):
        return [str(x) for x in self.target_column]

class EntityNormalizer:
    def __init__(
        self,
        database: EntityDictionary,
    ):
        self.database = database
        self.candidates = [
            x for x in self.database.get_candidates_list()
        ]
        self.targets = [
            x for x in self.database.get_normalization_list()
        ]
        self.model = self.database.get_model()
        self.surface_form_embeddings = self.database.get_surface_form_embeddings()

    def normalize(self, terms) -> list:        
        normalized_term = []
        score = []
        
        print("vectorizing...")
        surface_form_embeddings = self.model.encode([str(x) for x in terms])
        # クエリのベクトル化
        query_embeddings_np = np.array(surface_form_embeddings)
        closest_n = 2
        # クエリに対して最も近いk個のベクトルを検索
        distances, indices = self.database.get_index_general_form_embeddings().search(query_embeddings_np, closest_n)
        
        print("vectorizing done.")
            
        for i, query in enumerate([str(x) for x in terms]):
            normalized_term.append(self.database.get_general_form()[indices[i][0]])
            score.append(1 - distances[i][0])
        return normalized_term, score
    

def normalize(
    entities: list,
    dictionary: EntityDictionary
) -> list:
    normalizer = EntityNormalizer(dictionary)
    normalized, scores = normalizer.normalize(entities)
    return normalized, scores
