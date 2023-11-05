from models.dpr import DPR
from sentence_transformers import CrossEncoder

class DPRCrossencoder(DPR):
    def __init__(self, documents: list[dict] = None, index_path: str = None, model_name: str = "cross-encoder/stsb-roberta-large", n: int = 25) -> None:
        self.crossencoder = CrossEncoder(model_name)
        self.n = n
        super(DPRCrossencoder, self).__init__(documents, index_path)
        
    def Lookup(self, query: str, k: int, n: int = None):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        @param n: The number of documents to include in reranking.
        """
        if n is None:
            n = self.n
        query_embedding = self.EmbedQuery(query)
        ranked_documents = sorted(self.index.GetDocuments(), key=lambda d: self.CosineSimilarity(d.GetEmbedding(), query_embedding), reverse=True)
        ranked_documents = ranked_documents[:min(n,len(ranked_documents))]
        reranked_scores = self.crossencoder.predict([(document.GetText(),query) for document in ranked_documents])
        reranked_documents = [d for _,d in sorted(zip(reranked_scores,ranked_documents), reverse=True)]
        return reranked_documents[:min(k,len(ranked_documents))]