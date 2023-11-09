from data.document import Document

class Dataset:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = self.__BuildDocuments(documents)
        
    def __BuildDocuments(self, documents):
        return [Document(title=document["title"], text=document["text"], _id = document["_id"]) for _,document in enumerate(documents)]
        
    def GetDocuments(self):
        return self.documents