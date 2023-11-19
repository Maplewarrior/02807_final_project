from data.phishing import PhishingDataset, PhishingEmail
from data.query import Query

def evaluatePhishingByMajorityVote(retrieved_labels: list[str]):
    return max(set(retrieved_labels), key = retrieved_labels.count)

def calculatePhishingAccuracy(preds: list[str], labels: list[str]) -> float:
    return sum([1 if preds[i] == labels[i] else 0 for i in range(len(preds))]) / len(preds)

def getPhishingQueries(dataset: PhishingDataset):
    return [Query(text=email.GetText(), 
                         id=email.GetId(), 
                         relevant_document_ids=[], 
                         label=email.GetLabel()) for email in dataset]