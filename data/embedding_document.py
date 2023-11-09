from data.document import Document

class EmbeddingDocument(Document):
    def __init__(self, title: str, text: str, _id: str) -> None:
        super(EmbeddingDocument, self).__init__(title, text, _id)
        
    def SetEmbedding(self, embedding: list[float]):
        self.embedding = embedding
        
    def GetEmbedding(self):
        return self.embedding