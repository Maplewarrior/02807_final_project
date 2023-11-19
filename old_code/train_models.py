from data.dataloader import DataLoader
import configparser

# load config.ini 
config = configparser.ConfigParser()
config.read('config.ini')
data_handler = DataLoader(config)
dataset = "fiqa"
corpus, queries = data_handler.get_dataset(dataset)

from models.model_loader_helpers import createModels

models_to_create = {"TF-IDF": {},
                    "BM25": {},
                    "DPR": {},
                    "Crossencoder": {"n":25},
                    "KMeans": {"k":4},
                    "CURE": {"k": 2, "n": 2, "shrinkage_fraction":0.2}}

documents = corpus

createModels(documents=documents, dataset_name=dataset, models=models_to_create, save=True)