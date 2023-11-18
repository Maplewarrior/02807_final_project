from data.dataloader import DataLoader
from data.query import Query

def getCorpus(data_loader: DataLoader, dataset: str):
    documents, _ = data_loader.get_dataset(dataset)
    relevant_doc_ids_for_all_queries = data_loader.get_relevants(dataset)
    return documents, relevant_doc_ids_for_all_queries

def getQueries(data_loader: DataLoader, relevant_doc_ids_for_all_queries) -> list[Query]:
    query_dicts = data_loader.get_queries() # Get queries in the correct format
    queries: list[Query] = []
    for query_id, relevant_docs in relevant_doc_ids_for_all_queries.items():
        relevant_doc_ids = [r[0] for r in relevant_docs]
        query = query_dicts[query_id]
        queries.append(Query(text=query['text'], 
                             id=query_id, 
                             relevant_document_ids=relevant_doc_ids))
    return queries