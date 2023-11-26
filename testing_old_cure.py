from data.dataloader import DataLoader
import configparser
import pdb
import numpy as np
import time

# load config.ini 
config = configparser.ConfigParser()
config.read('configs/config.ini')
data_handler = DataLoader(config)
dataset = "fiqa"
corpus, queries = data_handler.get_dataset(dataset)

documents = corpus[:500]
del corpus

from models.CURE import CURE
model = CURE(
    documents=documents,
    n = 25,
    initial_clusters=10,
    shrinkage_fraction=0.05,
    threshold=0.5,
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)

random_docs = np.random.choice(documents, size=5, replace=False)
queries = [rand_doc["text"] for rand_doc in random_docs]
sims = model.Lookup(queries, k=3)

for i in range(len(queries)):
    print("QUERY")
    print(queries[i])
    print("DOC")
    print(sims[i][0].GetText())
    print("\n\n")

print("Num clusters", model.clusters.GetNumberOfClusters())