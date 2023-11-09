from data.embedding_document import EmbeddingDocument
from models.builers.dense_retriever import DenseRetriever
from transformers import BertModel, BertTokenizer
import torch
import random
import numpy as np

class CURE(DenseRetriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = 'bert-base-uncased', k: int = 10, n: int = 2, shrinkage_fraction: float = 0.2) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        super(CURE, self).__init__(documents, index_path)
        self.clusters = self.__CreateClusters(k, n, shrinkage_fraction)
        
    def EmbedQuery(self, query: str):
        input_ids = self.tokenizer.encode(query, add_special_tokens=True)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        last_hidden_states = last_hidden_states.mean(1)
        return last_hidden_states[0].numpy()
    
    def __CreateClusters(self, k: int, n: int, shrinkage_fraction: float):
        clusters = CureClusterCollection(k, n, shrinkage_fraction, self.index.GetDocuments())
        clusters.Compute()
        return clusters
    
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
    

class CureObservation:
    def __init__(self, document: EmbeddingDocument) -> None:
        self.document = document
        self.cluster = None
        
    def SetCluster(self, cluster):
        self.cluster = cluster
        
    def GetCluster(self):
        return self.cluster
        
    def GetPoint(self):
        return self.document.GetEmbedding()
    
    def Shrink(self, centroid: list[float], shrinkage_fraction: float):
        self.document.SetEmbedding(self.GetPoint() + ((centroid - self.GetPoint()) * shrinkage_fraction))
        
    def GetDistanceTo(self, observation):
        return np.sum((observation.GetPoint()-self.GetPoint())**2)
    
    def GetDocument(self):
        return self.document

class CureCluster:
    def __init__(self, k: int, n: int, observation: CureObservation) -> None:
        self.observations = [observation]
        self.centroid = observation.GetPoint()
        observation.SetCluster(self)
        self.n = n
        self.k = k
        self.representative_observations = self.__AssignRepresentativePoints()
        
    def ShrinkRepresentativeObservations(self, shrinkage_fraction: float):
        for observation in self.representative_observations:
            observation.Shrink(self.centroid, shrinkage_fraction)
            
    def __AssignRepresentativePoints(self):
        max_distance_sorted_observations = sorted(self.observations, key=lambda obs: self.GetDistanceToCentroid(obs), reverse=False)
        return max_distance_sorted_observations[:int(min(self.n, len(self.observations)))]
    
    def UpdateCentroid(self):
        self.centroid = np.mean([observation.GetPoint() for observation in self.observations], axis=0)
        
    def GetObservations(self):
        return self.observations
    
    def SetObservationClusters(self):
        for observation in self.observations:
            observation.SetCluster(self)
        
    def Merge(self, cluster, shrinkage_fraction: float):
        self.observations.extend(cluster.GetObservations())
        self.SetObservationClusters()
        self.representative_observations = self.__AssignRepresentativePoints()
        self.ShrinkRepresentativeObservations(shrinkage_fraction)
        self.UpdateCentroid()
        
    def GetDistanceToCentroid(self, observation: CureObservation):
        return np.sum((observation.GetPoint()-self.centroid)**2)
        
    def GetError(self):
        error = 0
        for observation in self.observations:
            error += self.GetDistanceToCentroid(observation)
        return error
    
    def GetDistanceToQuery(self, query_embedding: list[float]):
        return np.sum((query_embedding-self.centroid)**2)
    
    def GetDocuments(self):
        return [observation.GetDocument() for observation in self.observations]
        
class CureClusterCollection:
    def __init__(self, k: int, n: int, shrinkage_fraction: float, documents: list[EmbeddingDocument]) -> None:
        self.k = k
        self.n = n
        self.shrinkage_fraction = shrinkage_fraction
        self.observations = self.__InitializeObservations(documents)
        self.clusters = self.__InitializeClusters()
        
    def __InitializeObservations(self, documents: list[EmbeddingDocument]):
        return [CureObservation(document) for document in documents]
    
    def __InitializeClusters(self):
        return [CureCluster(self.k, self.n, observation) for observation in self.observations]
    
    def GetError(self):
        error = 0
        for cluster in self.clusters:
            error += cluster.GetError()
        return error
    
    def GetMergeObservations(self):
        merge_observation_one = None
        merge_observation_two = None
        min_distance = np.inf
        for observation_one in self.observations:
            for observation_two in self.observations:
                if (not observation_one.GetCluster() == observation_two.GetCluster()):
                    distance = observation_one.GetDistanceTo(observation_two)
                    if (distance < min_distance):
                        merge_observation_one = observation_one
                        merge_observation_two = observation_two
                        min_distance = distance
        return merge_observation_one.GetCluster(), merge_observation_two.GetCluster()
    
    def GetNumberOfClusters(self):
        return len(self.clusters)
    
    def Compute(self):
        while self.GetNumberOfClusters() > self.k:
            merge_cluster_one, merge_cluster_two = self.GetMergeObservations()
            merge_cluster_one.Merge(merge_cluster_two, self.shrinkage_fraction)
            self.clusters.remove(merge_cluster_two)
            
    def GetMostSimilarCluster(self, query_embedding: list[float]):
        min_distance = np.inf
        similar_cluster = None
        for cluster in self.clusters:
            distance_to_query = cluster.GetDistanceToQuery(query_embedding)
            if distance_to_query < min_distance:
                min_distance = distance_to_query
                similar_cluster = cluster
        return similar_cluster