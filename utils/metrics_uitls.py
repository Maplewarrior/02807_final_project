from data.query import Query
import time

def timeFunction(function, **args):
    time_before = time.perf_counter()
    output = function(**args)
    time_after = time.perf_counter()
    return time_after - time_before, output

def calculateReciprocalRank(relevancies):
    for i, relevancy in enumerate(relevancies):
        if relevancy:
            return 1/(i+1)
    return 0

def calculatePrecision(relevancies):
    return sum(relevancies) / len(relevancies)

def calculateRecall(relevancies, query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def calculateMetrics(results: list[list[bool]], queries: list[Query]):
    total_precision = 0
    total_recall = 0
    total_reciprocal_rank = 0
    for relevancies, query in zip(results, queries):
        total_precision += calculatePrecision(relevancies)
        total_recall += calculateRecall(relevancies, query)
        total_reciprocal_rank += calculateReciprocalRank(relevancies)
    return total_precision/len(results), total_recall/len(results), total_reciprocal_rank/len(results)