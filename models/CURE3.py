import numpy as np
from scipy.spatial.distance import cdist
from data.embedding_document import EmbeddingDocument
from models.builers.dense_retriever import DenseRetriever
from transformers import BertModel, BertTokenizer
import torch
import random
import numpy as np
import pdb
from sklearn.cluster import AgglomerativeClustering
import itertools
import math

def timeit(method):
    """
    Decorator for timing methods.
    """
    import time
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time() - ts
        
        print('%r %2.2f sec' % (method.__name__, te))
        return result
    
    return timed

class CureObservation:
    def __init__(self, document: EmbeddingDocument, cluster=None):
        self.document = document
        self.point = document.GetEmbedding()
        self.cluster = cluster
        
    def GetPoint(self):
        return self.point
    
    def GetDocument(self):
        return self.document
    
    def AssignCluster(self, cluster):
        self.cluster = cluster

    def GetShrinkedCopy(self, centroid, shrink_factor):
        shrinked_point = self.point + ((centroid - self.point) * shrink_factor)
        new_doc = EmbeddingDocument(title=None, text=self.document.GetText(), _id=None)
        new_doc.SetEmbedding(shrinked_point)
        return CureObservation(new_doc, cluster=self.cluster)

class CURE(DenseRetriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = 'bert-base-uncased', n_clusters: int = 2, n_representatives: int = 3, shrink_factor: float = 0.2, subsample_fraction=0.7, merge_threshold = 1, batch_size: int =1) -> None:
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_clusters = n_clusters
        self.n_representatives = n_representatives
        self.shrink_factor = shrink_factor
        self.subsample_fraction = subsample_fraction
        self.merge_threshold = merge_threshold
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        super(CURE, self).__init__(documents, index_path, model_name=model_name, batch_size=batch_size)

        # Extract embeddings from the index
        self.observations = self._CreateObservations()
        # Remove last dimension of embeddings
        # self.observations = np.squeeze(self.observations, axis=2)

        # Apply CURE clustering
        # self.clusters = self._ApplyCure()
        self.cluster_collection = self._ApplyCure()

    def _ExtractEmbeddings(self):
        """Extract embeddings from the index
        
        Returns:
            np.array -- Array of embeddings
        """    
        embedding_list = np.array([doc.GetEmbedding() for doc in self.index.documents])
        
        
        
        return embedding_list

    def _CreateObservations(self):
        """Create points from the embeddings
        
        Returns:
            list -- List of points
        """
        # Create points from the embeddings
        observations = [CureObservation(doc) for doc in self.index.documents]
        return observations

    def _ApplyCure(self):
        """Apply CURE clustering on the embeddings
        
        Returns:
            list -- List of clusters
        """
        # 1. Take random subsample of points 
        fraction = self.subsample_fraction
        print(f"Drawing subsample from observations. Fraction of full dataset: {fraction}")
        subsample = random.sample(self.observations, int(len(self.observations) * fraction))

        # 2. Cluster using hierarchical clustering method
        print(f"Clustering subsample with Agglomerative Clustering, k={self.n_clusters}")
        points = np.array([obs.GetPoint().flatten() for obs in subsample])

        # Perform Agglomerative Clustering
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters)
        clustering.fit(points)

        cluster_labels = clustering.labels_
        # Assign cluster labels back to the observations
        for i, obs in enumerate(subsample):
            obs.AssignCluster(cluster_labels[i])
        

        # 3 Find representatives for each cluster
        print(f"Finding n={self.n_representatives} representatives for each cluster")
        representatives = self._FindRepresentatives(subsample, self.n_representatives)

        # 4 Move representatives towards centroid
        shrinked_reps = []
        for rep_cluster in representatives:
            centroid = np.mean([obs.GetPoint() for obs in rep_cluster], axis=0)
            shrinked_reps.append([obs.GetShrinkedCopy(centroid, self.shrink_factor) for obs in rep_cluster])


        # TODO
        # 5. Merge clusters if representatives are closer than a threshold
        print("Merging clusters if representatives are closer than a threshold")
        

        # TODO
        # Repeat 3, 4 and 5 until no more clusters are merged


        # 6. Assign observations not in subsample to closest cluster i.e. closest representative
        print("Assigning observations not in subsample to closest cluster")
        for obs in self.observations:
            if obs not in subsample:
                closest_rep = self._FindClosestRepresentative(obs, shrinked_reps)
                obs.AssignCluster(closest_rep.cluster)
        

        print("Now all observations have been assigned to a cluster. Congrats!")
        # Now all observations are assigned to a cluster. Congrats!
        


    def _FindClosestRepresentative(self, observation: CureObservation, representatives: list[list[CureObservation]]):
        """Find the closest representative to an observation
        
        Arguments:
            observation {CureObservation} -- Observation to find closest representative to
            representatives {list[list[CureObservation]]} -- List of representatives for each cluster
        
        Returns:
            CureObservation -- Closest representative
        """
        min_distance = math.inf
        closest_rep = None
        for rep_cluster in representatives:
            for rep in rep_cluster:
                distance = self._CalculateDistance(observation.GetPoint(), rep.GetPoint())
                if distance < min_distance:
                    min_distance = distance
                    closest_rep = rep

        return closest_rep


    def _CalculateDistance(self, point1, point2):
        """Calculate the Euclidean distance between two points using NumPy."""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def _FindFurthestPoints(self, cluster: list[CureObservation], n: int):
        """Find n points that are progressively farther away from each other."""
        c_points = np.array([obs.GetPoint() for obs in cluster])
        
        if n <= 0 or not c_points.any() or 2 > len(c_points):
            # If there is only one point in the cluster, return it as the furthest point
            if len(c_points) == 1 and n >= 0:
                return [cluster[0]]

            return []

        # Start by finding the two farthest points
        max_distance = 0
        farthest_pair_indices = None
        for i, j in itertools.combinations(range(len(c_points)), 2):
            distance = self._CalculateDistance(c_points[i], c_points[j])
            if distance > max_distance:
                max_distance = distance
                farthest_pair_indices = (i, j)

        selected_indices = list(farthest_pair_indices)

        # Iteratively find the next point that is farthest from all previously selected points
        while len(selected_indices) < n and len(selected_indices) < len(c_points):
            next_index = None
            max_distance_to_set = 0

            for i, point in enumerate(c_points):
                if i not in selected_indices:
                    min_distance_to_set = min(self._CalculateDistance(point, c_points[selected_index]) for selected_index in selected_indices)
                    if min_distance_to_set > max_distance_to_set:
                        max_distance_to_set = min_distance_to_set
                        next_index = i

            if next_index is not None:
                selected_indices.append(next_index)

        return [cluster[i] for i in selected_indices]

    def _FindRepresentatives(self, subsample, n_representatives=2):
        """Find representatives for each cluster
        
        Arguments:
            subsample {list} -- List of observations
        
        Returns:
            list -- List of clusters
        """
        clusters = []
        for i in range(self.n_clusters):
            cluster = [obs for obs in subsample if obs.cluster == i]
            
            # find points furthest away from each other 
            representatives = self._FindFurthestPoints(cluster, n_representatives)
            clusters.append(representatives)
            
        
        return clusters
    
    def EmbedQuery(self, query: str):
        input_ids = self.tokenizer.encode(query, add_special_tokens=True, 
                                            max_length=512, truncation=True)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        last_hidden_states = last_hidden_states.mean(1)
        return last_hidden_states[0].numpy().reshape(1, -1)

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
        out = last_hidden_states / norms
        return last_hidden_states / norms # returns [Batch_size x 768]
    # TODO Use predict function in cluster collection to closest cluster and then closest document within that cluster
    # def CalculateScores(self, queries: list[str]):
        # return scores, cluster_documents

    def predict(self, query: str):
        query_embedding = self.EmbedQuery(query)
        cluster_idx = self.cluster_collection.predict(query_embedding)
        return cluster_idx