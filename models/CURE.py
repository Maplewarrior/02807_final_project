from data.embedding_document import EmbeddingDocument
from models.builers.dense_retriever import DenseRetriever
# from transformers import BertModel, BertTokenizer
import torch
import random
import numpy as np
import time
import pdb
from sklearn.cluster import AgglomerativeClustering
from utils.distance_utils import GetSimilarity
# np.random.seed(2)

class CURE(DenseRetriever):
    def __init__(self, 
                 documents: list[dict] = None, 
                 index_path: str = None, 
                 model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                 n: int = 2, 
                 initial_clusters: int = 10, 
                 shrinkage_fraction: float = 0.2, 
                 threshold: float = 1, 
                 subsample_fraction: float = 0.5, 
                 batch_size: int = 5,
                 similarity_measure: str = "cosine",
                 initial_clustering_algorithm: str = "agglomerative") -> None:
        self.similarity_measure = similarity_measure
        self.initial_clustering_algorithm = initial_clustering_algorithm
        super(CURE, self).__init__(documents, index_path, model_name, batch_size)
        start = time.time()
        self.clusters = self.__CreateClusters(n, initial_clusters, shrinkage_fraction, threshold, subsample_fraction, similarity_measure, initial_clustering_algorithm)
        end = time.time()
        print("\n\nTime", end-start, "\n\n")
        self.__CreateEmbeddingMatrices()        

    def EmbedQueries(self, queries: list[str]):
        """
        Batched version of EmbedQuery
            - Embeddings returned are normalized!
        """
        tokenized_queries = self.tokenizer(queries, add_special_tokens=True, 
                                           padding=True, max_length=512, 
                                           truncation=True, return_tensors='pt').to(self.device)
        
        # dec_input_ids = self.tokenizer.batch_decode(tokenized_queries['input_ids']) # decode to ensure sentences are encoded correctly
        # model inference

        with torch.no_grad():
            last_hidden_states = self.model(**tokenized_queries)[0]
        # average embedding over tokens
        last_hidden_states = last_hidden_states.mean(1)#.cpu().numpy()
        # Normalize embeddings
        norms = torch.linalg.norm(last_hidden_states, 2, dim=1, keepdim=False).unsqueeze(1) # NOTE: documentation says kwarg "ord" should be "p", but it thorws an error  
        # norms = np.linalg.norm(last_hidden_states, ord=2, axis=1)[:, None] # compute norms for batch and unsqueeze 2nd dim
        return last_hidden_states / norms # returns [Batch_size x 768]
    
    def __CreateClusters(self, n: int, initial_clusters: int, shrinkage_fraction: float, threshold: float, subsample_fraction: float, similarity_measure: str, initial_clustering_algorithm: str):
        clusters = CureClusterCollection(n, initial_clusters, threshold, subsample_fraction, shrinkage_fraction, self.index.GetDocuments(), similarity_measure, initial_clustering_algorithm)
        clusters.Compute()
        return clusters
    
    def __CreateEmbeddingMatrices(self):
        for cluster in self.clusters.clusters:
            cluster.SetEmbeddingMatrix()
            cluster.embedding_matrix = cluster.embedding_matrix.to(self.device)
    
    def CalculateScores(self, queries: list[str], k: int):
        query_embeddings = self.EmbedQueries(queries)
        most_similar_clusters = [self.clusters.GetMostSimilarCluster(query_embedding.cpu().numpy(), k) for query_embedding in query_embeddings]
        scores = []
        all_documents = []
        for i,query_embedding in enumerate(query_embeddings):
            score = []
            documents = []
            for most_similar_cluster in most_similar_clusters[i]:
                score.extend(self.InnerProduct(query_embedding, most_similar_cluster.embedding_matrix).cpu())
                documents.extend(most_similar_cluster.GetDocuments())
            scores.append(score)
            all_documents.append(documents)
        # scores = [self.InnerProduct(query_embedding, most_similar_clusters[i].embedding_matrix).cpu() for i, query_embedding in enumerate(query_embeddings)]
        return scores, all_documents
        # return scores, [c.GetDocuments() for c in most_similar_clusters]
    
    def Lookup(self, queries: list[str], k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        scores, cluster_documents = self.CalculateScores(queries, k)
        score_document_pairs = [list(zip(scores[i], cluster_documents[i])) for i in range(len(queries))]
        ranked_documents_batch = [[d for _, d in sorted(pairs, key=lambda pair: pair[0], reverse=True)] for pairs in score_document_pairs]
        return [ranked_documents[:min(k, len(ranked_documents))] for ranked_documents in ranked_documents_batch]
    
class CureObservation:
    def __init__(self, document: EmbeddingDocument, similarity_measure: str) -> None:
        self.document = document
        self.embedding = document.GetEmbedding()
        self.cluster = None
        self.similarity_measure = similarity_measure
        
    def SetCluster(self, cluster):
        self.cluster = cluster
        
    def GetCluster(self):
        return self.cluster
        
    def GetPoint(self):
        return self.embedding
    
    def Shrink(self, centroid: list[float], shrinkage_fraction: float):
        self.embedding = self.embedding + ((centroid - self.GetPoint()) * shrinkage_fraction)
        
    def GetDistanceTo(self, observation):
        return GetSimilarity(observation.GetPoint(), self.GetPoint(), self.similarity_measure)
    
    def GetDistanceToQuery(self, observation):
        return GetSimilarity(observation, self.GetPoint(), self.similarity_measure)
    
    def GetDocument(self):
        return self.document

class CureCluster:
    def __init__(self, n: int, observation: CureObservation, similarity_measure) -> None:
        self.observations = [observation]
        self.similarity_measure = similarity_measure
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
        for observation in self.observations:
            observation.SetCluster(None)
        self.observations = []
    
    def __FindMostDispersedPoints(self, distance_matrix):
        selected_points = []
        remaining_points = set(range(len(distance_matrix)))
        
        while len(selected_points) < min(self.n, len(self.observations)):
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
        indexes = self.__FindMostDispersedPoints(distance_matrix)
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
        self.UpdateCentroid()
        self.representative_observations = self.__GetRepresentativePoints()
        self.ShrinkRepresentativeObservations(shrinkage_fraction)
        
    def GetDistanceToCentroid(self, observation: CureObservation):
        return GetSimilarity(observation.GetPoint(), self.centroid, self.similarity_measure)
        
    def GetError(self):
        error = 0
        for observation in self.observations:
            error += self.GetDistanceToCentroid(observation)
        return error
    
    def GetDistanceToQuery(self, query_embedding: list[float]):
        return GetSimilarity(query_embedding, self.centroid, self.similarity_measure)
    
    def GetDocuments(self):
        return [observation.GetDocument() for observation in self.observations]
    
    def SetEmbeddingMatrix(self):
        emb_matrix = [torch.from_numpy(doc.GetEmbedding()) for doc in self.GetDocuments()]
        self.embedding_matrix = torch.cat(emb_matrix, dim=1)
        
class CureClusterCollection:
    def __init__(self, n: int, initial_clusters: int, threshold: float, subsample_fraction: float, shrinkage_fraction: float, documents: list[EmbeddingDocument], similarity_measure: str, initial_clustering_algorithm: str) -> None:
        self.n = n # Representative points
        self.initial_clusters = initial_clusters
        self.subsample_fraction = subsample_fraction
        self.threshold = threshold # Threshold for merging
        self.shrinkage_fraction = shrinkage_fraction
        self.similarity_measure = similarity_measure
        self.observations = self.__InitializeObservations(documents)
        self.clusters = self.__InitializeClusters(initial_clustering_algorithm)
        
    def __InitializeObservations(self, documents: list[EmbeddingDocument]):
        return [CureObservation(document, self.similarity_measure) for document in documents]
    
    def __InitializeClusters(self, initial_clustering_algorithm: str):
        if initial_clustering_algorithm == "agglomerative":
            clusters = self.__RunAgglomerativeClustering()
        elif initial_clustering_algorithm == "k_means":
            clusters = self.__RunKMeans()
        print(f'Created clusters using {initial_clustering_algorithm} of sizes', " ".join([str(len(cluster.GetObservations())) for cluster in clusters]))
        return clusters
        
    def __RunAgglomerativeClustering(self):
        """ Run allgomerative clustering on subsample of observations
        
        Returns:
            list[CureCluster]: List of clusters
        """
        subsample = np.random.choice(self.observations, size=int(len(self.observations)*self.subsample_fraction), replace=False)
        # Extract points from observations
        points = np.array([obs.GetPoint().flatten() for obs in subsample])
        
        # Perform Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=self.initial_clusters, linkage="average", metric="cosine")
        clustering.fit(points)
        
        # Extract cluster labels
        cluster_labels = clustering.labels_

        # Cluster dictionary to keep track of clusters
        clusters: dict[str, CureCluster] = {}

        for obs, label in zip(subsample, cluster_labels):
            if label in clusters:
                clusters[label].AddObservation(obs)
            else:
                clusters[label] = CureCluster(self.n, obs, self.similarity_measure)
        computed_clusters = list(clusters.values())
        random.shuffle(computed_clusters)
        return computed_clusters

    def __RunKMeans(self):
        observations = np.random.choice(self.observations, size=int(len(self.observations)*self.subsample_fraction), replace=False)
        clusters = [CureCluster(self.n, observation, self.similarity_measure) for observation in np.random.choice(observations, size=self.initial_clusters, replace=False)]
        sum_centroid_change = 1
        while sum_centroid_change > 1e-02:
            for cluster in clusters:
                cluster.RemoveObservations()
            for observation in observations:
                min_distance = np.inf
                min_cluster = None
                for cluster in clusters:
                    distance = cluster.GetDistanceToCentroid(observation)
                    if distance < min_distance:
                        min_distance = distance
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
                        distance = observation.GetDistanceTo(representative_point)
                        if distance < min_distance:
                            min_distance = distance
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
        ## Assign Representative Points and clusters
        for cluster in self.clusters:
            cluster.AssignRepresentativePoints()
            cluster.ShrinkRepresentativeObservations(self.shrinkage_fraction)
        ## Merging clusters
        for cluster, other_cluster in self.GetClustersToMerge():
            cluster.Merge(other_cluster, self.shrinkage_fraction)
            self.clusters.remove(other_cluster)
        # Add remaining observations
        for observation, assigned_cluster in self.GetObservationsAndClustersToMerge():
            assigned_cluster.AddObservation(observation)
    
    def GetMostSimilarCluster(self, query_embedding: list[float], k: int):
        num_docs_in_clusters = 0
        cluster_to_return = []
        
        while num_docs_in_clusters < k:
            min_distance = np.inf
            most_similar_cluster = None
            for cluster in self.clusters:
                if not cluster in cluster_to_return:
                    for representative_point in cluster.GetRepresentativePoints():
                        distance_to_query = representative_point.GetDistanceToQuery(query_embedding)
                        if distance_to_query < min_distance:
                            min_distance = distance_to_query
                            most_similar_cluster = cluster
            cluster_to_return.append(most_similar_cluster)
            num_docs_in_clusters += len(most_similar_cluster.GetDocuments())
        return cluster_to_return