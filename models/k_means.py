from data.embedding_document import EmbeddingDocument
from models.builers.dense_retriever import DenseRetriever
from transformers import BertModel, BertTokenizer
import random
import numpy as np
import torch

class KMeans(DenseRetriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1", k: int = 10, batch_size: int = None) -> None:  
        super(KMeans, self).__init__(documents, index_path, model_name, batch_size)
        print(f'KMeans running on: {self.device}')
        self.clusters = self.__CreateClusters(k)
        print('Creating embedding matrices for clusters!')
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
        
    def __CreateClusters(self, k: int):
        print(f'Computing {k} cluster centroids')
        if k <= len(self.index.GetDocuments()):
            clusters = ClusterCollection(k, self.index.GetDocuments())
            prev_error = np.inf
            tol = 1e-02
            while np.abs(clusters.GetError() - prev_error) > tol:
                clusters.AssignDocuments()
                prev_error = clusters.GetError()
                clusters.UpdateCentroids()
                print(f'Error difference: {np.abs(clusters.GetError() - prev_error)}')
            
            return clusters
        else:
            raise ValueError("Cannot create more clusters, than there are documents.")
    
    def __CreateEmbeddingMatrices(self):
        for cluster in self.clusters.clusters:
            cluster.SetEmbeddingMatrix()
            cluster.embedding_matrix = cluster.embedding_matrix.to(self.device)
        
    def CalculateScores(self, queries: list[str]):
        query_embeddings = self.EmbedQueries(queries)
        most_similar_clusters = [self.clusters.GetMostSimilarCluster(query_embedding.cpu().numpy()) for query_embedding in query_embeddings]
        scores = [self.InnerProduct(query_embedding, most_similar_clusters[i].embedding_matrix).cpu() for i, query_embedding in enumerate(query_embeddings)]
        scores = [score.tolist() for score in scores]
        return scores, [c.GetDocuments() for c in most_similar_clusters]
    
    def Lookup(self, queries: list[str], k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        scores, cluster_documents = self.CalculateScores(queries)
        score_document_pairs = [list(zip(scores[i], cluster_documents[i])) for i in range(len(queries))]
        ranked_documents_batch = [[d for _, d in sorted(pairs, key=lambda pair: pair[0], reverse=True)] for pairs in score_document_pairs]
        return [ranked_documents[:min(k, len(ranked_documents))] for ranked_documents in ranked_documents_batch]
        # scores = self.CalculateScores(queries)
        # ranked_documents = [[d for _, d in sorted(zip(query_scores, self.index.GetDocuments()), key=lambda pair: pair[0], reverse=True)] for query_scores in scores]
        # return [ranked_document[:min(k, len(ranked_documents[0]))] for ranked_document in ranked_documents]
    
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
    
    ### TODO: Check if inner product performs better
    def GetDistanceToQuery(self, query_embedding: list[float]):
        return np.sum((query_embedding-self.centroid)**2)
    
    def GetError(self):
        error = 0
        for document in self.documents:
            error += self.GetDistanceToCentroid(document)
        return error
    
    def GetDocuments(self):
        return self.documents
    
    def SetEmbeddingMatrix(self):
        emb_matrix = [torch.from_numpy(document.GetEmbedding()) for document in self.documents]
        self.embedding_matrix = torch.cat(emb_matrix, dim=1)