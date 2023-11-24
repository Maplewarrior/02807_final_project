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
import matplotlib.pyplot as plt

# set a seed for reproducibility
# random.seed(42)

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
    def __init__(self, document: EmbeddingDocument, cluster=None, slice=None):
        """Create a point from an embedding
        
        Args:
            document (EmbeddingDocument): Document to create point from
            cluster (int, optional): Cluster label. Defaults to None.
            slice (int, optional): Slice the embedding (only for testing purposes). Defaults to None.
        """
        self.document = document
        if slice:
            # Slice the embedding (only for testing purposes). Useful when visualizing the algorithm
            self.point = document.GetEmbedding()[:slice]
        else:
            self.point = document.GetEmbedding()
        self.cluster = cluster
        
    def GetPoint(self):
        return self.point
    
    def GetDocument(self):
        return self.document
    
    def AssignCluster(self, cluster):
        self.cluster = cluster

    def GetShrinkedCopy(self, centroid, shrink_factor):
        # normalize centroid 
        centroid = centroid / np.linalg.norm(centroid)

        # Linearly interpolate the point towards the centroid
        shrinked_point = ((1 - shrink_factor) * self.point + shrink_factor * centroid)

        # Re-normalize the shrinked point to ensure its length is 1
        shrinked_point = shrinked_point / np.linalg.norm(shrinked_point)

        # shrinked_point = self.point + ((centroid - self.point) * shrink_factor)
        new_doc = EmbeddingDocument(title=None, text=self.document.GetText(), _id=None)
        new_doc.SetEmbedding(shrinked_point)
        return CureObservation(new_doc, cluster=self.cluster)

class CURE(DenseRetriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = 'bert-base-uncased', n_clusters: int = 2, n_representatives: int = 3, shrink_factor: float = 0.2, subsample_fraction=0.7, merge_threshold = 1, batch_size: int =1, slice_and_plot=False) -> None:
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
        self.slice_and_plot = slice_and_plot
        if self.slice_and_plot:
            self.observations = self._CreateObservations(slice=True)
        else:
            self.observations = self._CreateObservations()

        # Apply CURE clustering. 
        # This updates the observations with cluster labels 
        # It also creates a dictionary of clusters with cluster label as key
        # and a dictionary of centroids with cluster label as key
        self._ApplyCure()

    def _ExtractEmbeddings(self):
        """Extract embeddings from the index
        
        Returns:
            np.array -- Array of embeddings
        """    
        embedding_list = np.array([doc.GetEmbedding() for doc in self.index.documents])
        
        
        
        return embedding_list

    def _CreateObservations(self, slice=False):
        """Create points from the embeddings
        
        Returns:
            list -- List of points
        """
        # Create points from the embeddings
        if slice:
            observations = [CureObservation(doc, slice=2) for doc in self.index.documents]
        else:
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
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="average", metric="cosine")
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
        cents = []
        for rep_cluster in representatives:
            centroid = np.mean([obs.GetPoint() for obs in rep_cluster], axis=0)
            cents.append(centroid)
            shrinked_reps.append([obs.GetShrinkedCopy(centroid, self.shrink_factor) for obs in rep_cluster])

        # TODO
        # 5. Merge clusters if representatives are closer than a threshold
        print("Merging clusters if representatives are closer than a threshold")
        

        # TODO
        # Repeat 3, 4 and 5 until no more clusters are merged


        newly_assigned = []
        # 6. Assign observations not in subsample to closest cluster i.e. closest representative
        print("Assigning observations not in subsample to closest cluster")
        for obs in self.observations:
            if obs not in subsample:
                closest_rep = self._FindClosestRepresentative(obs, shrinked_reps)
                obs.AssignCluster(closest_rep.cluster)
                newly_assigned.append(obs)

        if self.slice_and_plot:
            # PLOTS
            cmap = plt.cm.viridis  # Choose any available colormap
            colors = cmap(np.linspace(0, 1, self.n_clusters ))

            for i in range(self.n_clusters):
                points = np.array([obs.GetPoint().flatten() for obs in subsample if obs.cluster == i])
                if len(points) > 0:
                    plt.scatter(points[:, 0], points[:, 1],marker='o', c=[colors[i]], alpha=0.6, label=f'Cluster ({i})')
                    plt.scatter(cents[i][0], cents[i][1], label=f'Centroid ({i})', marker='P', alpha=0.8, c=[colors[i]])


            # plot points not in subsample
            points = np.array([obs.GetPoint().flatten() for obs in self.observations if obs not in subsample])
            plt.scatter(points[:, 0], points[:, 1], c='gray', marker='x', alpha=0.5, label='Not in subsample')
            # plot representatives from each cluster
            for i in range(self.n_clusters):
                points = np.array([obs.GetPoint().flatten() for obs in representatives[i]])
                # plt.scatter(points[:, 0], points[:, 1], label=f'Cluster rep ({i})', marker='*', alpha=0.4, c=colors[i])
                if len(points) > 0:
                    plt.scatter(points[:, 0], points[:, 1], label=f'Cluster rep ({i})', marker='*', alpha=0.4, c=[colors[i]])

            # plot shrinked representatives from each cluster
            for i in range(self.n_clusters):
                points = np.array([obs.GetPoint().flatten() for obs in shrinked_reps[i]])
                # plt.scatter(points[:, 0], points[:, 1], label=f'Shrinked cluster rep ({i})', marker='*', alpha=0.8, c=colors[i])
                if len(points) > 0:
                    plt.scatter(points[:, 0], points[:, 1], label=f'Shrinked cluster rep ({i})', marker='*', alpha=0.8, c=[colors[i]])
            

            # plot newly assigned
            for i in range(self.n_clusters):
                points = np.array([obs.GetPoint().flatten() for obs in newly_assigned if obs.cluster == i])
                if len(points) > 0:
                    plt.scatter(points[:, 0], points[:, 1], label=f'Newly assigned ({i})', marker='x', alpha=0.8, c=[colors[i]])

            # plt.legend(loc='upper left')
            plt.show()
            # END PLOTS


        print("Now all observations have been assigned to a cluster. Congrats!")
        # Now all observations are assigned to a cluster. Congrats!
        
        # Make dictionary of clusters with cluster label as key
        cluster_dict = {}
        [cluster_dict.setdefault(obs.cluster, []).append(obs) for obs in self.observations]
        self.cluster_dict = cluster_dict
        # make centroids dict 
        self.centroids = {cluster: np.mean([obs.GetPoint() for obs in cluster_dict[cluster]], axis=0) for cluster in cluster_dict.keys()}


    def GetMostSimilarCluster(self, embeddedQuery: np.array):
        """ Get most similar cluster to query.
        
        Args: 
            embeddedQuery (np.array): Query embedding
        
        Returns:
            (int, float): Closest cluster label and distance to this cluster
        """
        # get closest_cluster and the distance to this min_distance
        min_distance, closest_cluster = min(
            [(self._CalculateDistance(embeddedQuery, centroid), cluster_label) for cluster_label, centroid in self.centroids.items()],
            key=lambda x: x[0]
        )

        return closest_cluster, min_distance

    def CalculateScores(self, queries: list[str]):
        """ Calculate scores for each query and cluster

        Args:
            queries (list[str]): List of queries
    
        Returns:
            (list[list[float]], list[list[CureObservation]]): List of scores for each query and cluster, and list of clusters for each query (list of lists
        """
        query_embeddings = self.EmbedQueries(queries)
        
        cluster_documents = [self.cluster_dict[self.GetMostSimilarCluster(query_embedding)[0]] for query_embedding in query_embeddings]
        # calculate distance for each document in cluster
        scores = [[self._CalculateDistance(query_embedding, doc.GetPoint()) for doc in cluster] for query_embedding, cluster in zip(query_embeddings, cluster_documents)]
        # scores = [[self.InnerProduct(query_embedding, d.GetPoint()) for d in cluster_documents[i]] for i, query_embedding in enumerate(query_embeddings)]
        return scores, cluster_documents

    
    def Lookup(self, queries: list[str], k: int):
        """
        Find the k most relevant passages for each query in the list of queries.

        Args: 
            (list[str]): List of queries (strings)
            (int): Number of passages to retrieve for each query
        
        Returns:
            (list[list[EmbeddingDocument]]): List of lists of EmbeddingDocuments. Each inner list contains the k most relevant passages for the corresponding query.
        """

        scores, cluster_documents = self.CalculateScores(queries)
        score_document_pairs = [list(zip(scores[i], cluster_documents[i])) for i in range(len(queries))]
        ranked_documents_batch = [[d for _, d in sorted(pairs, key=lambda pair: pair[0], reverse=True)] for pairs in score_document_pairs]
        return [ranked_documents[:min(k, len(ranked_documents))] for ranked_documents in ranked_documents_batch]


    def _FindClosestRepresentative(self, observation: CureObservation, representatives: list[list[CureObservation]]):
        """Find the closest representative to an observation
        
        Arguments:
            observation {CureObservation} -- Observation to find closest representative to
            representatives {list[list[CureObservation]]} -- List of representatives for each cluster
        
        Returns:
            CureObservation -- Closest representative
        """
        max_similarity = 0
        closest_rep = None
        for rep_cluster in representatives:
            for rep in rep_cluster:
                similarity = self._CalculateDistance(observation.GetPoint(), rep.GetPoint())
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_rep = rep

        return closest_rep


    def _CalculateDistance(self, point1, point2):
        """Calculate the Euclidean distance between two points using NumPy. """
        # Calculate cosine similarity. Remember that embeddings are already normalized
        return self.InnerProduct(point1.flatten(), point2.flatten())

    def _FindFurthestPoints(self, cluster: list[CureObservation], n: int):
        """Find n points that are progressively farther away from each other."""
        c_points = np.array([obs.GetPoint() for obs in cluster])
        
        if n <= 0 or not c_points.any() or 2 > len(c_points):
            # If there is only one point in the cluster, return it as the furthest point
            if len(c_points) == 1 and n >= 0:
                return [cluster[0]]

            return []
        # Start by finding the two farthest points

        min_similarity = np.inf
        farthest_pair_indices = None
        for i, j in itertools.combinations(range(len(c_points)), 2):
            distance = self._CalculateDistance(c_points[i], c_points[j])
            # We use cosine similarity which means we need to get the points with the smallest similarity to get the ones furthest away from each other.
            if distance < min_similarity:
                min_similarity = distance
                farthest_pair_indices = (i, j)

        selected_indices = list(farthest_pair_indices)

        # Iteratively find the next point that is farthest from all previously selected points
        while len(selected_indices) < n and len(selected_indices) < len(c_points):
            next_index = None
            min_similarity_to_set = np.inf

            for i, point in enumerate(c_points):
                if i not in selected_indices:
                    max_similarity_to_set = max(self._CalculateDistance(point, c_points[selected_index]) for selected_index in selected_indices)
                    if max_similarity_to_set < min_similarity_to_set:
                        min_similarity_to_set = max_similarity_to_set
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
        return out # returns [Batch_size x 768]
    # TODO Use predict function in cluster collection to closest cluster and then closest document within that cluster
    # def CalculateScores(self, queries: list[str]):
        # return scores, cluster_documents

    def predict(self, query: str):
        query_embedding = self.EmbedQuery(query)
        cluster_idx = self.cluster_collection.predict(query_embedding)
        return cluster_idx