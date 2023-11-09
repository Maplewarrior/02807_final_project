from data.dataset import Dataset
from data.embedding_document import EmbeddingDocument

class EmbeddingDataset(Dataset):
    def __init__(self, documents: list[dict]) -> None:
        self.documents = self.__BuildDocuments(documents)
    
    def __BuildDocuments(self, documents):
        return [EmbeddingDocument(title=document["title"], text=document["text"], _id=document["_id"]) for i,document in enumerate(documents)]
    
    def GetDim(self):
        return len(self.documents[0].GetEmbedding())