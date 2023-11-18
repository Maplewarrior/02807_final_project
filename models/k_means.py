from data.embedding_document import EmbeddingDocument
from models.builers.dense_retriever import DenseRetriever
from transformers import BertModel, BertTokenizer
import random
import numpy as np
import torch

class KMeans(DenseRetriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = 'bert-base-uncased', k: int = 10) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        super(KMeans, self).__init__(documents, index_path)
        self.clusters = self.__CreateClusters(k)
        
    def EmbedQuery(self, query: str):
        input_ids = self.tokenizer.encode(query, add_special_tokens=True, 
                                            max_length=512, truncation=True)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        last_hidden_states = last_hidden_states.mean(1)
        return last_hidden_states[0].numpy()
        
    def __CreateClusters(self, k: int):
        if k <= len(self.index.GetDocuments()):
            clusters = ClusterCollection(k, self.index.GetDocuments())
            prev_error = np.inf
            while clusters.GetError() < prev_error:
                clusters.AssignDocuments()
                prev_error = clusters.GetError()
                clusters.UpdateCentroids()
            return clusters
        else:
            raise ValueError("Can not create more clusters, than there is documents.")
    
    def CalculateScores(self, query: str):
        query_embedding = self.EmbedQuery(query)
        cluster_documents = self.clusters.GetMostSimilarCluster(query_embedding).GetDocuments()
        scores = [self.CosineSimilarity(d.GetEmbedding(), query_embedding) for d in cluster_documents]
        return scores, cluster_documents
    
    def Lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        scores, cluster_documents = self.CalculateScores(query)
        ranked_documents = [d for _, d in sorted(zip(scores, cluster_documents), key=lambda pair: pair[0], reverse=True)]
        return ranked_documents[:min(k,len(ranked_documents))]
    
class ClusterCollection:
    def __init__(self, k: int, documents: list[EmbeddingDocument]) -> None:
        self.documents = documents
        self.clusters: list[Cluster] = self.__InitializeClusters(k)
    
    def __InitializeClusters(self, k: int):
        clusters = []
        initial_centroids = random.choices(self.documents, k=k)
        initial_centroids = [initial_centroid.GetEmbedding() for initial_centroid in initial_centroids]
        for i in range(k):
            clusters.append(Cluster(initial_centroids[i]))
        return clusters
        
    def __ClearDocuments(self):
        for cluster in self.clusters:
            cluster.ResetDocuments()
            
    def AssignDocuments(self):
        self.__ClearDocuments()
        for document in self.documents:
            min_distance = np.inf
            assigned_cluster = None
            for cluster in self.clusters:
                distance_to_cluster = cluster.GetDistanceToCentroid(document)
                if distance_to_cluster < min_distance:
                    min_distance = distance_to_cluster
                    assigned_cluster = cluster
            assigned_cluster.AsignDocument(document)
            
    def UpdateCentroids(self):
        for cluster in self.clusters:
            cluster.UpdateCentroid()
            
    def GetError(self):
        error = 0
        for cluster in self.clusters:
            error += cluster.GetError()
        return error
    
    def GetMostSimilarCluster(self, query_embedding: list[float]):
        min_distance = np.inf
        similar_cluster = None
        for cluster in self.clusters:
            distance_to_query = cluster.GetDistanceToQuery(query_embedding)
            if distance_to_query < min_distance:
                min_distance = distance_to_query
                similar_cluster = cluster
        return similar_cluster
    
class Cluster:
    def __init__(self, initial_centroid: list[float]) -> None:
        self.centroid = initial_centroid
        self.documents: list[EmbeddingDocument] = []
    
    def UpdateCentroid(self):
        if len(self.documents) > 0:
            self.centroid = np.mean([document.GetEmbedding() for document in self.documents], axis=0)
        
    def AsignDocument(self, document: EmbeddingDocument):
        self.documents.append(document)
        
    def ResetDocuments(self):
        self.documents = []
        
    def GetDistanceToCentroid(self, document: EmbeddingDocument):
        return np.sum((document.GetEmbedding()-self.centroid)**2)
    
    def GetDistanceToQuery(self, query_embedding: list[float]):
        return np.sum((query_embedding-self.centroid)**2)
    
    def GetError(self):
        error = 0
        for document in self.documents:
            error += self.GetDistanceToCentroid(document)
        return error
    
    def GetDocuments(self):
        return self.documents