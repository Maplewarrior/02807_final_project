from data.document import Document
from models.builers.retriever import Retriever
from models.TFIDF import TFIDF
import numpy as np


class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.coprus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()
        self.bm25_vectorss = self.GetDocumentsBM25Vectors()


    def GetAverageDocumentLength(self):
        return sum([len(self.GetDocumentVocabulary(document)) for document in self.index.GetDocuments()]) / len(self.index.GetDocuments())
       
    def GetDocumentsBM25Vectors(self):
        """ Get the BM25 vectors for all documents in the index.
        
        @return: A list of BM25 vectors, where each vector corresponds to a document in the index.
        """
        avg_doc_length = self.GetAverageDocumentLength()
        bm25_vectors = []
        for document in self.index.GetDocuments():
            document_terms = self.GetDocumentVocabulary(document)
            bm25_vector = []
            for term in self.coprus_vocabulary:
                tf = document_terms.count(term) / len(document_terms)
                # BM25 calculation, see report for details
                bm25 = self.idf[term] * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (len(document_terms) / avg_doc_length))))
                bm25_vector.append(bm25)
            bm25_vectors.append(bm25_vector)
        return bm25_vectors
    
    def CalculateScores(self, query: str):
        """ Calculate the BM25 scores for all documents in the index.
        
        Args:
            query: The query for which BM25 scores should be calculated.
        
        Returns:
            A list of BM25 scores, where each score corresponds to a document in the index.
        """
        scores = [0 for _ in self.bm25_vectorss]
        query_vocabulary = self.GetQueryVocabulary(query)
        for i, term in enumerate(self.coprus_vocabulary):
            if term in query_vocabulary:
                for j,bm25_vectors in enumerate(self.bm25_vectorss):
                    scores[j] += bm25_vectors[i]
        return scores