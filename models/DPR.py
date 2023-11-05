from models.builers.dense_retriever import DenseRetriever
from transformers import BertModel, BertTokenizer
import torch

class DPR(DenseRetriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = 'bert-base-uncased') -> None:
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        super(DPR, self).__init__(documents, index_path)

    def EmbedQuery(self, query: str):
        input_ids = self.tokenizer.encode(query, add_special_tokens=True)
        input_ids = torch.tensor([input_ids])
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        last_hidden_states = last_hidden_states.mean(1)
        return last_hidden_states[0].numpy()