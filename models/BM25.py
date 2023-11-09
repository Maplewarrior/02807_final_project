from data.document import Document
from models.builers.retriever import Retriever
import numpy as np

class BM25(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.coprus_vocabulary = self.GetCorpusVocabulary()
        self.idfs = self.GetInverseDocumentFrequencies()
        self.document_lengths, self.average_document_length = self.GetDocumentLengths()
        self.tfs = self.GetTermFrequencies()
        self.k1 = k1
        self.b = b
        
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.GetTotalVocabulary(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    def PreprocessText(self, text: str):
        to_removes = [".",",","?","!",":",";"]
        for to_remove in to_removes:
            text = text.replace(to_remove, "")
        text = text.lower()
        return text.split(" ")
    
    def GetTotalVocabulary(self, string: str):
        return self.PreprocessText(string)
    
    def GetDocumentVocabulary(self, document: Document):
        return list(set(self.PreprocessText(document.GetText())))
    
    def GetCorpusVocabulary(self):
        coprus_vocabulary = set()
        for document in self.index.GetDocuments():
            coprus_vocabulary = coprus_vocabulary.union(set(self.PreprocessText(document.GetText())))
        return coprus_vocabulary
    
    def GetDocumentFrequencies(self):
        term_frequencies = {}
        for term in self.coprus_vocabulary:
            term_frequencies[term] = sum(1 for document in self.index.GetDocuments() if term in self.PreprocessText(document.GetText()))
        return term_frequencies
    
    def GetInverseDocumentFrequencies(self):
        idfs = {}
        dfs = self.GetDocumentFrequencies()
        n_documents = len(self.index.GetDocuments())
        for term, df in dfs.items():
            idfs[term] = np.log(n_documents / (df + 1))
        return idfs
    
    def GetTermFrequencies(self):
        tfs: dict[Document, dict[str, int]] = {}
        for document in self.index.GetDocuments():
            document_terms = self.GetDocumentVocabulary(document)
            tfs[document] = {}
            for term in self.coprus_vocabulary:
                tfs[document][term] = document_terms.count(term)
        return tfs
    
    def CalculateScores(self, query: str):
        scores = [0 for _ in self.index.GetDocuments()]
        query_vocabulary = self.GetTotalVocabulary(query)
        for term in query_vocabulary:
            if term in self.coprus_vocabulary:
                for i, document in enumerate(self.index.GetDocuments()):
                    scores[i] += (self.idfs[term] * 
                                (self.tfs[document][term] * (self.k1 + 1)) / 
                                (self.tfs[document][term] + self.k1 * (1 - self.b + self.b*(self.document_lengths[document] / self.average_document_length))))
        return scores