from data.document import Document
from models.builers.retriever import Retriever
import numpy as np
from models.TFIDF import TFIDF
import time
from scipy.sparse import lil_matrix
from collections import Counter

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
        self.bm25_matrix = self.GetDocumentBM25Vectors()
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
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}
        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        for doc_idx, document in enumerate(self.index.GetDocuments()):

            document_terms = self.GetDocumentTermCounts(document)
            doc_factor = self.document_lengths[document] / self.average_document_length # |D_j| / D_avg

            for term, count in document_terms.items():
                if term in self.term_to_index:
                    term_idx = self.term_to_index[term]
                    enum = self.idf.get(term, 0) * count * (self.k1 + 1)
                    denom = count + self.k1 * (1- self.b + self.b * doc_factor)

                    bm25_matrix[doc_idx, term_idx] = enum/denom
        
        return bm25_matrix.tocsr()
    
   
    @time_func
    def QueryToVector(self, query: str):
        """ Convert a query to a vector
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        # Preprocess the query and get term counts
        query_terms = self.PreprocessText(query)
        term_freq = Counter(query_terms)

        # Create an empty LIL matrix for the query vector
        query_vector = lil_matrix((1, len(self.corpus_vocabulary)), dtype=np.float32)

        # Fill the vector with term frequencies
        for term, count in term_freq.items():
            if term in self.term_to_index:
                query_vector[0, self.term_to_index[term]] = count

        # Convert to CSR format for efficient multiplication
        return query_vector.tocsr()

    @time_func
    def CalculateScores(self, query: str):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        # Convert the query to a vector
        query_vector = self.QueryToVector(query)
        # Maybe we should calculate dot product in batches
        scores = self.bm25_matrix.dot(query_vector.T).toarray().flatten()
        return scores