from data.dataloader import DataLoader
import configparser
import pdb
import numpy as np
import time
import pdb
import random

from models.k_means import KMeans
from models.CURE import CURE

# load config.ini 
config = configparser.ConfigParser()
config.read('configs/config.ini')
data_handler = DataLoader(config)
dataset = "fiqa"
corpus, queries = data_handler.get_dataset(dataset)

documents = np.random.choice(corpus, size=100, replace=False)
del corpus


# model = CURE(
#     documents=documents,
#     n = 25,
#     initial_clusters=25,
#     shrinkage_fraction=0.1,
#     threshold=0.35,
#     subsample_fraction = 0.5,
#     similarity_measure="cosine",
#     initial_clustering_algorithm="agglomerative",
# )

model = KMeans(
    documents = documents,
    k=5,
    batch_size = 5,
)

random_docs = np.random.choice(documents, size=5, replace=False)
queries = [rand_doc["text"] for rand_doc in random_docs]
# chosen_docs = [random.choice(cluster.observations) for cluster in model.clusters.clusters]
# queries = [rand_doc.GetDocument().GetText() for rand_doc in chosen_docs]
sims = model.Lookup(queries, k=3)

for i in range(len(queries)):
    print("QUERY")
    print(queries[i])
    print("DOC")
    print(sims[i][0].GetText())
    print("\n\n")

# print("Num clusters", model.clusters.GetNumberOfClusters())
# print("Cluster dist.", [len(cluster.observations) for cluster in model.clusters.clusters])
pdb.set_trace()