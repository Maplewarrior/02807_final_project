from data.document import Document
from models.builers.retriever import Retriever
import numpy as np

class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.coprus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()
        self.tfidf_vectors = self.GetDocumentsTFIDFVectors()
    
    def PreprocessText(self, text: str):
        to_removes = [".",",","?","!",":",";"]
        for to_remove in to_removes:
            text = text.replace(to_remove, "")
        text = text.lower()
        return text.split(" ")
        
    def GetQueryVocabulary(self, query: str):
        return list(set(self.PreprocessText(query)))
        
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
    
    def GetDocumentsTFIDFVectors(self):
        tfidf_vectors = []
        for document in self.index.GetDocuments():
            document_terms = self.GetDocumentVocabulary(document)
            tfidf_vector = []
            for term in self.coprus_vocabulary:
                tf = document_terms.count(term) / len(document_terms)
                tfidf = tf * self.idf[term]
                tfidf_vector.append(tfidf)
            tfidf_vectors.append(tfidf_vector)
        return tfidf_vectors
    
    def CalculateScores(self, query: str):
        scores = [0 for _ in self.tfidf_vectors]
        query_vocabulary = self.GetQueryVocabulary(query)
        for i, term in enumerate(self.coprus_vocabulary):
            if term in query_vocabulary:
                for j,tfidf_vector in enumerate(self.tfidf_vectors):
                    scores[j] += tfidf_vector[i]
        return scores