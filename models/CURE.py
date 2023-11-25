from data.embedding_document import EmbeddingDocument
from models.builers.dense_retriever import DenseRetriever
from transformers import BertModel, BertTokenizer
import torch
import random
import numpy as np
import time

class CURE(DenseRetriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = 'bert-base-uncased', n: int = 2, initial_clusters: int = 10, shrinkage_fraction: float = 0.2, threshold: float = 1, subsample_fraction: float = 0.5, batch_size: int = 5) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        super(CURE, self).__init__(documents, index_path, model_name, batch_size)
        start = time.time()
        self.clusters = self.__CreateClusters(n, initial_clusters, shrinkage_fraction, threshold, subsample_fraction)
        end = time.time()
        print("\n\nTime", end-start, "\n\n")
        
    def EmbedQuery(self, query: str):
        input_ids = self.tokenizer.encode(query, add_special_tokens=True, 
                                            max_length=512, truncation=True)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        last_hidden_states = last_hidden_states.mean(1)
        return last_hidden_states[0].numpy()

    def EmbedQueries(self, queries: list[str]):
        """
        Batched version of EmbedQuery
            - Embeddings returned are normalized!
        """
        tokenized_queries = self.tokenizer(queries, add_special_tokens=True, 
                                           padding=True, max_length=512, 
                                           truncation=True, return_tensors='pt').to(self.device)
        # model inference
        with torch.no_grad():
            last_hidden_states = self.model(**tokenized_queries)[0]
        # average embedding over tokens
        last_hidden_states = last_hidden_states.mean(1).cpu().numpy()
        # Normalize embeddings
        norms = np.linalg.norm(last_hidden_states, ord=2, axis=1)[:, None] # compute norms for batch and unsqueeze 2nd dim
        return last_hidden_states / norms # returns [Batch_size x 768]
    
    def __CreateClusters(self, n: int, initial_clusters: int, shrinkage_fraction: float, threshold: float, subsample_fraction: float):
        clusters = CureClusterCollection(n, initial_clusters, threshold, subsample_fraction, shrinkage_fraction, self.index.GetDocuments())
        clusters.Compute()
        return clusters
    
    def CalculateScores(self, queries: list[str]):
        query_embeddings = self.EmbedQueries(queries)
        cluster_documents = [self.clusters.GetMostSimilarCluster(query_embedding).GetDocuments() for query_embedding in query_embeddings]
        scores = [[self.InnerProduct(query_embedding, d.GetEmbedding()) for d in cluster_documents[i]] for i, query_embedding in enumerate(query_embeddings)]
        return scores, cluster_documents
    
    def Lookup(self, queries: list[str], k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        scores, cluster_documents = self.CalculateScores(queries)
        score_document_pairs = [list(zip(scores[i], cluster_documents[i])) for i in range(len(queries))]
        ranked_documents_batch = [[d for _, d in sorted(pairs, key=lambda pair: pair[0], reverse=True)] for pairs in score_document_pairs]
        return [ranked_documents[:min(k, len(ranked_documents))] for ranked_documents in ranked_documents_batch]
    
class CureObservation:
    def __init__(self, document: EmbeddingDocument) -> None:
        self.document = document
        self.embedding = document.GetEmbedding()
        self.cluster = None
        
    def SetCluster(self, cluster):
        self.cluster = cluster
        
    def GetCluster(self):
        return self.cluster
        
    def GetPoint(self):
        return self.embedding
    
    def Shrink(self, centroid: list[float], shrinkage_fraction: float):
        self.embedding = self.embedding + ((centroid - self.GetPoint()) * shrinkage_fraction)
        
    def GetDistanceTo(self, observation):
        return np.sum((observation.GetPoint()-self.GetPoint())**2)
    
    def GetDocument(self):
        return self.document

class CureCluster:
    def __init__(self, n: int, observation: CureObservation) -> None:
        self.observations = [observation]
        self.centroid = observation.GetPoint()
        observation.SetCluster(self)
        self.n = n # representative points
        
    def ShrinkRepresentativeObservations(self, shrinkage_fraction: float):
        for observation in self.representative_observations:
            observation.Shrink(self.centroid, shrinkage_fraction)
            
    def GetRepresentativePoints(self):
        return self.representative_observations
            
    def AddObservation(self, observation: CureObservation):
        self.observations.append(observation)
        observation.SetCluster(self)
        
    def RemoveObservations(self):
        self.observations = []
    
    def __FindMostDispersedPoints(self, distance_matrix, n):
        # Initialize variables
        selected_points = []
        remaining_points = set(range(len(distance_matrix)))

        # Greedy selection
        while len(selected_points) < min(n, len(self.observations)):
            if not selected_points:
                # Select the point with the maximum distance to any other point
                current_point = max(remaining_points, key=lambda p: np.max(distance_matrix[p]))
            else:
                # Select the point with the maximum cumulative distance to the already selected points
                current_point = max(remaining_points, key=lambda p: sum(distance_matrix[p, q] for q in selected_points))

            selected_points.append(current_point)
            remaining_points.remove(current_point)

        return selected_points
    
    def __GetRepresentativePoints(self) -> list[CureObservation]:
        distance_matrix = np.array([[0 for _ in self.observations] for _ in self.observations])
        for i in range(len(self.observations)):
            for j in range(len(self.observations)):
                if i < j:
                    distance = self.observations[i].GetDistanceTo(self.observations[j])
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance 
        indexes = self.__FindMostDispersedPoints(distance_matrix, self.n)
        return [self.observations[index] for index in indexes]
    
    def AssignRepresentativePoints(self):
        self.representative_observations = self.__GetRepresentativePoints()
    
    def UpdateCentroid(self):
        if len(self.observations) > 0:
            self.centroid = np.mean([observation.GetPoint() for observation in self.observations], axis=0)
        
    def GetCentroid(self):
        return self.centroid
        
    def GetObservations(self):
        return self.observations
    
    def SetObservationClusters(self):
        for observation in self.observations:
            observation.SetCluster(self)
        
    def Merge(self, cluster, shrinkage_fraction: float):
        self.observations.extend(cluster.GetObservations())
        self.SetObservationClusters()
        self.representative_observations = self.__GetRepresentativePoints()
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
    def __init__(self, n: int, initial_clusters: int, threshold: float, subsample_fraction: float, shrinkage_fraction: float, documents: list[EmbeddingDocument]) -> None:
        self.n = n # Representative points
        self.initial_clusters = initial_clusters
        self.subsample_fraction = subsample_fraction
        self.threshold = threshold # Threshold for merging
        self.shrinkage_fraction = shrinkage_fraction
        self.observations = self.__InitializeObservations(documents)
        self.clusters = self.__InitializeClusters()
        
    def __InitializeObservations(self, documents: list[EmbeddingDocument]):
        return [CureObservation(document) for document in documents]
    
    def __InitializeClusters(self):
        return self.__RunKMeans()
        
    def __RunKMeans(self):
        observations = random.choices(self.observations, k=int(len(self.observations)*self.subsample_fraction))
        clusters = [CureCluster(self.n, observation) for observation in random.choices(observations, k=self.initial_clusters)]
        sum_centroid_change = 1
        while sum_centroid_change > 1e-02:
            for cluster in clusters:
                cluster.RemoveObservations()
            for observation in observations:
                min_distance = np.inf
                min_cluster = None
                for cluster in clusters:
                    if cluster.GetDistanceToCentroid(observation) < min_distance:
                        min_distance = cluster.GetDistanceToCentroid(observation)
                        min_cluster = cluster
                min_cluster.AddObservation(observation)
            sum_centroid_change = 0
            for cluster in clusters:
                old_centroid = cluster.GetCentroid()
                cluster.UpdateCentroid()
                new_centroid = cluster.GetCentroid()
                sum_centroid_change += np.sum((old_centroid-new_centroid)**2)
        return clusters
    
    def GetError(self):
        error = 0
        for cluster in self.clusters:
            error += cluster.GetError()
        return error
    
    def GetObservationsAndClustersToMerge(self):
        for observation in self.observations:
            if observation.GetCluster() is None:
                min_distance = np.inf
                assigned_cluster = None
                for cluster in self.clusters:
                    for representative_point in cluster.GetRepresentativePoints():
                        if observation.GetDistanceTo(representative_point) < min_distance:
                            min_distance = observation.GetDistanceTo(representative_point)
                            assigned_cluster = cluster
                yield observation, assigned_cluster
            
    def __GetRepresentativePointsDistanceMatrix(self, all_representative_points):
        distance_matrix = np.array([[np.inf for _ in all_representative_points] for _ in all_representative_points])
        for i in range(len(all_representative_points)):
            for j in range(len(all_representative_points)):
                if i < j and not all_representative_points[i].GetCluster() == all_representative_points[j].GetCluster():
                    distance = all_representative_points[i].GetDistanceTo(all_representative_points[j])
                    distance_matrix[i][j] = distance
                    distance_matrix[j][i] = distance 
        return distance_matrix
            
    def GetClustersToMerge(self):
        while True:
            all_representative_points = [p for cluster in self.clusters for p in cluster.GetRepresentativePoints()]
            distance_matrix = self.__GetRepresentativePointsDistanceMatrix(all_representative_points)
            i,j = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
            if distance_matrix[i][j] < self.threshold:
                yield all_representative_points[i].GetCluster(), all_representative_points[j].GetCluster()
            else:
                break
    
    def GetNumberOfClusters(self):
        return len(self.clusters)
    
    def Compute(self):
        print("\n\nInitial Clusters", len(self.clusters))
        ## Assign Representative Points and clusters
        for cluster in self.clusters:
            cluster.SetObservationClusters()
            cluster.AssignRepresentativePoints()
        ## Merging clusters
        for cluster, other_cluster in self.GetClustersToMerge():
            cluster.Merge(other_cluster, self.shrinkage_fraction)
            self.clusters.remove(other_cluster)
        ## Add remaining observations
        for observation, assigned_cluster in self.GetObservationsAndClustersToMerge():
            assigned_cluster.AddObservation(observation)
            
    def GetMostSimilarCluster(self, query_embedding: list[float]):
        min_distance = np.inf
        similar_cluster = None
        for cluster in self.clusters:
            distance_to_query = cluster.GetDistanceToQuery(query_embedding)
            if distance_to_query < min_distance:
                min_distance = distance_to_query
                similar_cluster = cluster
        return similar_cluster