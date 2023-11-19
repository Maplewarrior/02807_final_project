
import csv
import os

def createModels(documents, 
                 dataset_name, models = {"TF-IDF": {}}, 
                 save = True,
                 embedding_index_path: str = None):
    
    import pickle
    from models.builers.retriever import Retriever

    models_ = {}
    for model_name in list(models.keys()):
        if model_name == "TF-IDF":
            print("Creating TF-IDF model")
            from models.TFIDF import TFIDF
            models_[model_name] = TFIDF(documents=documents, **models[model_name])
        elif model_name == "BM25":
            print("Creating BM25 model")
            from models.BM25 import BM25
            models_[model_name] = BM25(documents=documents, **models[model_name])
        elif model_name == "DPR":
            print("Creating DPR model")
            from models.DPR import DPR
            models_[model_name] = DPR(documents=documents, **models[model_name], index_path=embedding_index_path)
        elif model_name == "Crossencoder":
            print("Crossencoder model")
            from models.DPR_crossencoder import DPRCrossencoder
            models_[model_name] = DPRCrossencoder(documents=documents, **models[model_name], index_path=embedding_index_path)
        elif model_name == "KMeans":
            print("KMeans model")
            from models.k_means import KMeans
            models_[model_name] = KMeans(documents=documents, **models[model_name], index_path=embedding_index_path)
        elif model_name == "CURE":
            print("CURE model")
            from models.CURE import CURE
            models_[model_name] = CURE(documents=documents, **models[model_name], index_path=embedding_index_path)
        else:
            raise Exception(f"Model '{model_name}' not implemented")
            
        if save: 
            if not os.path.exists(f"models/pickled_models/{dataset_name}"):
                print("Creating directory: models/pickled_models")
                os.makedirs(f"models/pickled_models/{dataset_name}")

            # Make parameters into string
            s = [f"{k}{v}" for k, v in models[model_name].items()]
            s = "_".join(s)
            if s:
                path = f"models/pickled_models/{dataset_name}/{model_name}_{s}.pickle"
            else:
                path = f"models/pickled_models/{dataset_name}/{model_name}.pickle"
            with open(path, "wb") as f:
                print(f"Saving model '{model_name}' at: {path}")

                pickle.dump(models_[model_name] , f)
    return models_

def loadModels(dataset_name, models={"TF-IDF":{}}):
    import pickle
    models_ = {}
    for model_name in list(models.keys()):
        s = [f"{k}{v}" for k, v in models[model_name].items()]
        s = "_".join(s)
        if s:
            path = f"models/pickled_models/{dataset_name}/{model_name}_{s}.pickle"
        else:
            path = f"models/pickled_models/{dataset_name}/{model_name}.pickle"
        with open(path, "rb") as f:
            models_[model_name] = pickle.load(f)

    return models_