from data.dataset import Dataset
from data.embedding_document import EmbeddingDocument

class EmbeddingDataset(Dataset):
    def __init__(self, documents: list[dict]) -> None:
        self.documents = self.__BuildDocuments(documents)
    
    def __BuildDocuments(self, documents):
        return [EmbeddingDocument(document["text"], document["id"]) for i,document in enumerate(documents)]