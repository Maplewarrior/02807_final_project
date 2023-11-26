from models.DPR import DPR
from sentence_transformers import CrossEncoder

class DPRCrossencoder(DPR):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2", n: int = 25) -> None:
        super(DPRCrossencoder, self).__init__(documents, index_path)
        self.crossencoder = CrossEncoder(model_name, device=self.device)
        self.n = n
        
    def Lookup(self, queries: list[str], k: int, n: int = None):
        """
        @param queries: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        @param n: The number of documents to include in reranking.
        """

        if n is None:
            n = self.n
        
        ### DPR steps
        scores = self.CalculateScores(queries)
        ranked_documents = [[d for _, d in sorted(zip(query_scores, self.index.GetDocuments()), key=lambda pair: pair[0], reverse=True)] for query_scores in scores]
        ranked_documents = [ranked_document[:min(n, len(ranked_documents[0]))] for ranked_document in ranked_documents]
        
        ### crossencoder steps
        reranked_scores = [self.crossencoder.predict([(document.GetText(), pair[1]) for document in pair[0]]) for pair in list(zip(ranked_documents, queries))] # Batched variant
        # reranked_scores = [[self.crossencoder.predict([document.GetText(), pair[1]]) for document in pair[0]] for pair in list(zip(ranked_documents, queries))] # Unbatched variant
        reranked_scores_document_pairs = [list(zip(reranked_scores[i], ranked_documents[i])) for i in range(len(queries))]
        reranked_documents = [[doc for _, doc in sorted(batch, reverse=True, key= lambda x: x[0])] for batch in reranked_scores_document_pairs]
        return [reranked_document[:min(k, n)] for reranked_document in reranked_documents]