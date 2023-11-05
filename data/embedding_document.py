from data.document import Document

class EmbeddingDocument(Document):
    def __init__(self, text: str, id: str) -> None:
        super(EmbeddingDocument, self).__init__(text, id)
        
    def SetEmbedding(self, embedding: list[float]):
        self.embedding = embedding
        
    def GetEmbedding(self):
        return self.embedding