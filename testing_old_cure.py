from data.dataloader import DataLoader
import configparser
import pdb
import numpy as np
import time
import pdb
import random

# load config.ini 
config = configparser.ConfigParser()
config.read('configs/config.ini')
data_handler = DataLoader(config)
dataset = "fiqa"
corpus, queries = data_handler.get_dataset(dataset)

documents = corpus[:2500]
del corpus

from models.CURE import CURE


model = CURE(
    documents=documents,
    n = 25,
    initial_clusters=50,
    shrinkage_fraction=0.1,
    threshold=0.25,
    subsample_fraction = 0.5,
    similarity_measure="cosine"
)

# random_docs = np.random.choice(documents, size=5, replace=False)
chosen_docs = [random.choice(cluster.observations) for cluster in model.clusters.clusters]
queries = [rand_doc.GetDocument().GetText() for rand_doc in chosen_docs]
sims = model.Lookup(queries, k=3)

for i in range(len(queries)):
    print("QUERY")
    print(queries[i])
    print("DOC")
    print(sims[i][0].GetText())
    print("\n\n")

print("Num clusters", model.clusters.GetNumberOfClusters())
print("Cluster dist.", [len(cluster.observations) for cluster in model.clusters.clusters])
pdb.set_trace()