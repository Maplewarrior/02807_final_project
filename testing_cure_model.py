import os
from models.builers.retriever import Retriever

from data.dataloader import DataLoader
import configparser
import pdb
import numpy as np

# load config.ini 
config = configparser.ConfigParser()
config.read('configs/config.ini')
data_handler = DataLoader(config)
dataset = "fiqa"
corpus, queries = data_handler.get_dataset(dataset)

documents = corpus[:10]
print(documents[0])
del corpus

from models.model_loader_helpers import createModels

# models_to_create = {"CURE": {}}

# models = createModels(documents=documents, dataset_name=dataset, models=models_to_create, save=True)
from models.CURE3 import CURE
model = CURE(documents=documents, model_name= 'bert-base-uncased', batch_size=5, n_clusters = 2, n_representatives = 2, shrink_factor = 0.3, subsample_fraction = 0.7, slice_and_plot=False, merge_threshold=0.8)

# Remember it is random, so the results will be different each time
# random 5 numbers
random_docs = np.random.choice(documents, size=5, replace=False)
queries = [rand_doc["text"] for rand_doc in random_docs]
# queries = [documents[0], documents[5], documents[10]]
sims = model.Lookup(queries, k=3)
pdb.set_trace()

