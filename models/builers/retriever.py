from abc import ABC, abstractmethod
import pickle
from data.dataset import Dataset
import numpy as np
from numpy.linalg import norm
import os

class Retriever(ABC):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        if not index_path is None:
            self.index = self.__LoadIndex(index_path)
        else:
            self.index = self.__BuildIndex(documents)
    
    def __BuildIndex(self, documents: list[dict]):
        """
        @param dataset: The dataset for which an index containing embeddings should be built.
        """
        index = Dataset(documents)
        return index
    
    def __LoadIndex(self, index_path: str):
        """
        @param index_path: The path to the pre-computed index.
        """
        file = open(index_path, 'rb')
        index = pickle.load(file)
        file.close()
        
        return index
    
    def SaveIndex(self, index_path: str):
        """
        @param index_path: The path to save the pre-computed index.
        """
        #self.index.embedding_matrix = None # TEST whether embedding matrix is the one which fills memory
        if not os.path.exists(os.path.dirname(index_path)):
            os.makedirs(os.path.dirname(index_path))
        file = open(index_path, 'wb')
        pickle.dump(self.index, file)
        file.close()
        
    def CosineSimilarity(self, vector_1: list[float], vector_2: list[float]):
        return np.dot(vector_1, vector_2)/(norm(vector_1)*norm(vector_2) + 1e-10)

    def InnerProduct(self, matrix_1, matrix_2):
        """
        Cosine similarity becomes an inner product because vectors are normalized!
        """
        return matrix_1 @ matrix_2

    @abstractmethod
    def CalculateScores(self, query: str):
        raise NotImplementedError("Must overwrite")
    
    def Lookup(self, queries: list[str], k: int):
        """
        @param queries: The input text(s) to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve for each input query.
        """
        scores = self.CalculateScores(queries)
        ranked_documents = [[d for _, d in sorted(zip(query_scores, self.index.GetDocuments()), key=lambda pair: pair[0], reverse=True)] for query_scores in scores]
        return [ranked_document[:min(k, len(ranked_documents[0]))] for ranked_document in ranked_documents]