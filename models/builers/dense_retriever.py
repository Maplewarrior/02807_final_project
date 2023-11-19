import numpy as np
from abc import ABC, abstractmethod
from data.embedding_dataset import EmbeddingDataset
from models.builers.retriever import Retriever
from utils.misc import time_func, batch

class DenseRetriever(Retriever, ABC):
    def __init__(self, documents: list[dict] = None, index_path: str = None, batch_size: int = 100) -> None:
        print("DENSE RETRIEVER IDX PATH: ", index_path)
        self.batch_size = batch_size
        if not index_path is None:
            super(DenseRetriever, self).__init__(documents, index_path)
        else:
            self.index = self.__BuildIndex(documents)
        
        self.index.GetEmbeddingMatrix() # initialize embedding matrix
    
    def __BuildIndex(self, documents: list[dict]):
        """
        Batched version of __BuildIndex
        """
        
        index = EmbeddingDataset(documents) # initialize index
        count = 0
        print(f'Building embedding index using device {self.device}. Running this on GPU is strongly adviced!')
        # add embeddings
        for documents in batch(index.GetDocuments(), self.batch_size):
            embeddings = self.EmbedQueries([doc.GetText() for doc in documents]) # [batch_size x 768]
            for j, document in enumerate(documents):
                document.SetEmbedding(embeddings[j][:, None]) # save embeddings to index
            
            # verbosity
            count += self.batch_size
            if count % 5000 == 0:
                print(f'iter: {count}/{len(index.GetDocuments())}')
    
        return index

    @abstractmethod
    def EmbedQuery(self, query: str):
        """
        @param query: The input text for which relevant passages should be found.
        returns: An embedding of the query.
        """
        raise NotImplementedError("Must overwrite")
    
    @abstractmethod
    def EmbedQueries(self, queries: str):
        """
        @param query: The input text for which relevant passages should be found.
        returns: An embedding of the queries.
        """
        raise NotImplementedError("Must overwrite")
    
    # @time_func
    # def CalculateScores(self, query: str):
    #     print("DPR CalcScores")
    #     query_embedding = self.EmbedQuery(query)
    #     scores = [self.CosineSimilarity(d.GetEmbedding().squeeze(1), query_embedding) for d in self.index.GetDocuments()]
    #     return scores

    def CalculateScores(self, queries: list[str]):
        query_embeddings = self.EmbedQueries(queries)
        scores = [[self.InnerProduct(query_embedding, d.GetEmbedding()) for d in self.index.GetDocuments()] for query_embedding in query_embeddings]
        return scores