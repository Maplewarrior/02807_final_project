from data.document import Document
from models.builers.retriever import Retriever
import numpy as np
import pdb
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

class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()
        self.tfidf_vectors = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str):
        """
        Remove unwanted charatcters from text and lowercase
        """
        to_removes = [".",",","?","!",":",";", "(", ")"]
        for to_remove in to_removes:
            text = text.replace(to_remove, "")
        text = text.lower()
        return text.split(" ")
        
    def GetQueryVocabulary(self, query: str):
        return self.PreprocessText(query)

    def GetDocumentVocabulary(self, document: Document):
        return list(set(self.PreprocessText(document.GetText())))
    
    @time_func
    def GetCorpusVocabulary(self):
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self):
        """
        Function calculates |q_i \in D|
        """
        term_doc_counts = {term: 0 for term in self.corpus_vocabulary}
        for document in self.index.GetDocuments():
            unique_terms = set(self.PreprocessText(document.GetText()))
            for term in unique_terms:
                if term in term_doc_counts:
                    term_doc_counts[term] += 1
        return term_doc_counts
    
    @time_func
    def GetInverseDocumentFrequencies(self):
        idfs = {}
        dfs = self.GetDocumentFrequencies()
        n_documents = len(self.index.GetDocuments())
        for term, df in dfs.items():
            idfs[term] = (np.log2(n_documents / (df)) if df > 0 else 0)
        return idfs
    
    def GetDocumentTermCounts(self, document: Document):
        terms = list(self.PreprocessText(document.GetText()))
        return Counter(terms)
    
    @time_func
    def GetDocumentsTFIDFVectors(self):
        # Create a mapping from terms to indices
        term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        for doc_idx, document in enumerate(self.index.GetDocuments()):
            document_terms = self.GetDocumentTermCounts(document)
            max_freq = max(document_terms.values(), default=1)

            for term, count in document_terms.items():
                if term in term_to_index:
                    # Calculate TF
                    tf = count / max_freq
                    # Retrieve the index of the term and update the matrix
                    term_idx = term_to_index[term]
                    tfidf_matrix[doc_idx, term_idx] = tf * self.idf.get(term, 0)
        
        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def CalculateScores(self, query: str):
        scores = [0 for _ in self.tfidf_vectors]
        query_vocabulary = self.GetQueryVocabulary(query)
        for i, term in enumerate(self.coprus_vocabulary):
            if term in query_vocabulary:
                for j,tfidf_vector in enumerate(self.tfidf_vectors):
                    scores[j] += tfidf_vector[i]
        return scores

