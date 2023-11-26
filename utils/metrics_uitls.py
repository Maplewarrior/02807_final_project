from data.query import Query
import time
from itertools import product

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

# def calculatePrecision(relevancies):
#     return sum(relevancies) / len(relevancies)

def calculateRPrecision(relevancies, query):
    n_relevant = query.getNumberOfRelevantDocuments()
    return sum(relevancies[:n_relevant]) / min(len(relevancies), n_relevant)

def calculateRecall(relevancies, query: Query):
    return sum(relevancies) / min(len(relevancies), query.getNumberOfRelevantDocuments())

def calculateMetrics(results: list[list[bool]], queries: list[Query], subset_factors: list[int]):
    topk = len(results[0])
    metric_types = ['Rprecision', 'recall', 'MRR']
    variants = list(product(metric_types, subset_factors))
    metric_scores = {e : None for e in variants}

    for subset_factor in subset_factors:

        subset_result = [res[:int(topk/subset_factor)] for res in results]
        total_precision = 0
        total_recall = 0
        total_reciprocal_rank = 0
        for relevancies, query in zip(subset_result, queries):
            total_precision += calculateRPrecision(relevancies, query)
            total_recall += calculateRecall(relevancies, query)
            total_reciprocal_rank += calculateReciprocalRank(relevancies)
        
        
        metric_scores[('Rprecision', subset_factor)] = total_precision/len(results)
        metric_scores[('recall', subset_factor)] = total_recall/len(results)
        metric_scores[('MRR', subset_factor)] = total_reciprocal_rank/len(results)

    return metric_scores