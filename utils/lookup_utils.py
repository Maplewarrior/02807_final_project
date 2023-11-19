from models.builers.retriever import Retriever
from data.query import Query

def retrieveQueryAndGetRelevancies(model: Retriever, queries: list[Query], k: int):
    N = len(queries)
    retrieved_documents = model.Lookup(queries=[query.getQuery() for query in queries], k=k)
    relevancies = []
    for i in range(N):
        query_relevancies = []
        for document in retrieved_documents[i]:
            if queries[i].isDocumentRelevant(document):
                query_relevancies.append(True)
            else:
                query_relevancies.append(False)
        relevancies.append(query_relevancies)
    return relevancies