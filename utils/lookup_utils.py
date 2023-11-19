from models.builers.retriever import Retriever
from data.query import Query

def retrieveQueryAndGetRelevancies(model: Retriever, query: Query, k: int):
    retrieved_documents = model.Lookup(query=query.getQuery(), k=k)
    relevancies = []
    for document in retrieved_documents:
        print(document.label) # <-- Debugging for pissing retriever
        if query.isDocumentRelevant(document):
            relevancies.append(True)
        else:
            relevancies.append(False)
    return relevancies