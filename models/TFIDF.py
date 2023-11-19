from data.document import Document
from models.builers.retriever import Retriever
import numpy as np
import time
from scipy.sparse import lil_matrix
from collections import Counter
from utils.misc import time_func

class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

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
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)

        for doc_idx, document in enumerate(self.index.GetDocuments()):
            document_terms = self.GetDocumentTermCounts(document)
            max_freq = max(document_terms.values(), default=1)

            for term, count in document_terms.items():
                if term in self.term_to_index:
                    # Calculate TF
                    tf = count / max_freq
                    # Retrieve the index of the term and update the matrix
                    term_idx = self.term_to_index[term]
                    tfidf_matrix[doc_idx, term_idx] = tf * self.idf.get(term, 0)
        
        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    #@time_func
    def QueryToVector(self, queries: list[str]):
        """ Convert a query to a vector
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        # Preprocess the query and get term counts
        query_terms_batched = [self.PreprocessText(query) for query in queries]
        term_freqs = [Counter(query_terms) for query_terms in query_terms_batched]

        # Create an empty sparse matrix for the query vectors
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        for i in range(len(queries)):
            # Fill the vector with term frequencies
            for term, count in term_freqs[i].items():
                if term in self.term_to_index:
                    query_vector[i, self.term_to_index[term]] = count

        # Convert to CSR format for efficient multiplication
        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries)
        
        # scores = self.tfidf_matrix.dot(query_vectors.T).toarray()
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray()
        return scores