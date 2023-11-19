
from data.document import Document


class Query:
    def __init__(self, text: str, id: str, relevant_document_ids: list[str], label: str = None) -> None:
        self.text = text
        self.id = id
        self.relevant_document_ids = relevant_document_ids
        self.label = label
        
    def getQuery(self):
        return self.text
    
    def getNumberOfRelevantDocuments(self):
        return len(self.relevant_document_ids)
    
    def isDocumentRelevant(self, document: Document):
        return document.GetId() in self.relevant_document_ids
    
    def getLabel(self):
        return self.label