import torch
from abc import ABC, abstractmethod
from data.embedding_dataset import EmbeddingDataset
from models.builers.retriever import Retriever
from utils.misc import time_func, batch
import numpy as np

class DenseRetriever(Retriever, ABC):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = None, batch_size: int = None) -> None:
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.__InitRetrievalModels(model_name)

        if not index_path is None: # load index if previously saved
            super(DenseRetriever, self).__init__(documents, index_path)
            self.index.GetEmbeddingMatrix() # initialize embedding matrix
            self.index.embedding_matrix = self.index.embedding_matrix.to(self.device) # send to device
            print(f'Embedding matrix initialized to {self.device}!')
        
        else: # build index
            self.index = self.__BuildIndex(documents)

    def __InitRetrievalModels(self, model_name):
        print("Initializing retrieval model!")
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # self.model = BertModel.from_pretrained(model_name).to(self.device)
        if "bert" in model_name:
            from transformers import BertModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        else:
            from transformers import MPNetTokenizer, MPNetModel
            self.tokenizer = MPNetTokenizer.from_pretrained(model_name)
            self.model = MPNetModel.from_pretrained(model_name).to(self.device)
            self.model.eval()

    @time_func
    def __BuildIndex(self, documents: list[dict]):
        """
        Function that precomputes embeddings for an EmbeddingDocument
            - Inference is run on GPU.
            - Vectors are converted to numpy arrays prior to saving the index to reduce memory consumption.
        """
        index = EmbeddingDataset(documents) # initialize index
        itercount = 0
        print(f'Building embedding index using device: {self.device}. Running this on GPU is strongly adviced!')
        # add embeddings
        for documents in batch(index.GetDocuments(), self.batch_size):
            # convert to cpu if it is a tensor else nothing
            embeddings = self.EmbedQueries([doc.GetText() for doc in documents]).cpu().unsqueeze(2).numpy() if self.device == 'cuda' else np.expand_dims(self.EmbedQueries([doc.GetText() for doc in documents]), axis=2)
            for j, document in enumerate(documents):
                document.SetEmbedding(embeddings[j]) # save embeddings to index

            itercount += self.batch_size
            if itercount % 5000 == 0:
                print(f'iter: {itercount}/{len(index.GetDocuments())}')

        return index
    
    @abstractmethod
    def EmbedQueries(self, queries: str):
        """
        @param query: The input text for which relevant passages should be found.
        returns: An embedding of the queries.
        """
        raise NotImplementedError("Must overwrite")
    
    def mean_pooling(self, model_output, attention_mask):
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:,0]
    
    #@time_func
    def CalculateScores(self, queries: list[str]):
        query_embeddings = self.EmbedQueries(queries)
        # scores = [[self.InnerProduct(query_embedding, d.GetEmbedding()) for d in self.index.GetDocuments()] for query_embedding in query_embeddings]
        scores = self.InnerProduct(query_embeddings, self.index.embedding_matrix).cpu()
        scores = [score.tolist() for score in scores]
        return scores