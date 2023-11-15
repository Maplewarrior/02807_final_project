from data.document import Document
from models.builers.retriever import Retriever
import numpy as np
from models.TFIDF import TFIDF
import time
from scipy.sparse import lil_matrix

def time_func(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__} Elapsed: {(end-start)}s")
    return out
  return wrapper

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        # self.corpus_vocabulary = self.GetCorpusVocabulary() # Already Calculated
        # self.idfs = self.GetInverseDocumentFrequencies() # Already Calculated
        self.document_lengths, self.average_document_length = self.GetDocumentLengths()
        self.tfs = self.GetTermFrequencies()
        self.k1 = k1
        self.b = b
    
    @time_func
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.GetDocWords(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    def GetDocWords(self, string: str):
        return self.PreprocessText(string)
    
    @time_func
    def GetTermFrequencies(self):
        [document]
        tfs: dict[Document, dict[str, int]] = {}
        for document in self.index.GetDocuments():
            document_terms = self.GetDocumentVocabulary(document)
            tfs[document] = {}
            for term in self.corpus_vocabulary:
                tfs[document][term] = document_terms.count(term)
        return tfs
    
    @time_func
    def GetDocumentBM25Vectors(self):
        self.idf # Inverse Document Frequencies
        term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}
        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        for doc_idx, document in enumerate(self.index.GetDocuments()):

            document_terms = self.GetDocumentTermCounts(document)
            doc_factor = self.document_lengths[document] / self.average_document_length # |D_j| / D_avg

            for term, count in document_terms.items():
                if term in term_to_index:
                    term_idx = term_to_index[term]
                    enum = self.idf.get(term, 0) * count * (self.k1 + 1)
                    denom = count + self.k1 * (1- self.b + self.b * doc_factor)

                    bm25_matrix[doc_idx, term_idx] = enum/denom
        
        return bm25_matrix.tocsr()

    
    def CalculateScores(self, query: str):
        scores = [0 for _ in self.index.GetDocuments()]
        query_vocabulary = self.GetDocWords(query)
        for term in query_vocabulary:
            if term in self.corpus_vocabulary:
                for i, document in enumerate(self.index.GetDocuments()):
                    scores[i] += (self.idfs[term] * 
                                (self.tfs[document][term] * (self.k1 + 1)) / 
                                (self.tfs[document][term] + self.k1 * (1 - self.b + self.b*(self.document_lengths[document] / self.average_document_length))))
        return scores