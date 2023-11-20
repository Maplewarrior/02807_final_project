from models.builers.dense_retriever import DenseRetriever
import torch
import numpy as np
from utils.misc import time_func

class DPR(DenseRetriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1", batch_size: int = 25) -> None:
        super(DPR, self).__init__(documents, index_path, model_name, batch_size)
        print(f'DPR running on {self.device}')
        print(f"Embedding model is:\n{model_name}")
    
    def EmbedQueries(self, queries: list[str]):
        """
        This function takes in a batch of querys and returns their assoicated embeddings.
        Depending on the model, either average pooling or the embedding associated with the [CLS] token is returned.
        The embeddings are normalized --> cosine similarity becomes an inner product.
        """

        # tokenize inputs
        tokenized_queries = self.tokenizer(queries, add_special_tokens=True, 
                                           padding=True, max_length=512, 
                                           truncation=True, return_tensors='pt').to(self.device)
        
        # dec_input_ids = self.tokenizer.batch_decode(tokenized_queries['input_ids']) # decode to ensure sentences are encoded correctly
        # model inference
        with torch.no_grad():
            model_output = self.model(**tokenized_queries)
        
        # last_hidden_states = self.mean_pooling(model_output, tokenized_queries['attention_mask']) # average embedding over tokens
        last_hidden_states = self.cls_pooling(model_output) # perform pooling at the CLS token
        
        # Normalize embeddings
        norms = torch.linalg.norm(last_hidden_states, 2, dim=1, keepdim=False).unsqueeze(1) # NOTE: documentation says kwarg "ord" should be "p", but it thorws an error  

        return last_hidden_states / norms # returns [Batch_size x 768]
        