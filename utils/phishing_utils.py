from data.phishing import PhishingDataset, PhishingEmail
from data.query import Query

def evaluatePhishingByMajorityVote(retrieved_labels: list[list[str]]):
    return [max(set(query_labels), key = query_labels.count) for query_labels in retrieved_labels]

def calculatePhishingAccuracy(preds: list[str], labels: list[str]) -> float:
    return sum([1 if preds[i] == labels[i] else 0 for i in range(len(preds))]) / len(preds)

def getPhishingQueries(dataset: PhishingDataset):
    return [Query(text=email.GetText(), 
                         id=email.GetId(), 
                         relevant_document_ids=[], 
                         label=email.GetLabel()) for email in dataset]