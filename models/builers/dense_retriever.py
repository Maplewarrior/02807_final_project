from abc import ABC, abstractmethod
from data.embedding_dataset import EmbeddingDataset
from models.builers.retriever import Retriever


class DenseRetriever(Retriever, ABC):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        if not index_path is None:
            super(DenseRetriever, self).__init__(documents, index_path)
        else:
            self.index = self.__BuildIndex(documents)
        
    def __BuildIndex(self, documents: list[dict]):
        """
        @param dataset: The dataset for which an index containing embeddings should be built.
        """
        index = EmbeddingDataset(documents)
        for document in index.GetDocuments():
            document.SetEmbedding(self.EmbedQuery(document.GetText()))
        return index
    
    @abstractmethod
    def EmbedQuery(self, query: str):
        """
        @param query: The input text for which relevant passages should be found.
        returns: An embedding of the query.
        """
        raise NotImplementedError("Must overwrite")
    
    def CalculateScores(self, query: str):
        query_embedding = self.EmbedQuery(query)
        scores = [self.CosineSimilarity(d.GetEmbedding(), query_embedding) for d in self.index.GetDocuments()]
        return scores