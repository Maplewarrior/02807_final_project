
from data.dataset import Dataset
from models.bm25 import BM25
from models.cure import CURE
from models.dpr import DPR
from models.dpr_crossencoder import DPRCrossencoder
from models.tfidf import TFIDF

documents = [
    {"text": "A cat is an animal.", "id": 1}, 
    {"text": "The city of new york is big.", "id": 2},
    {"text": "The city of chicago is big.", "id": 3},
    {"text": "The city of gentofte is big.", "id": 4},
    {"text": "The city of copenhagen is big.", "id": 5}
    ]

# dpr = DPR(documents = documents)
# result = dpr.Lookup("what is a city?", 1)

# crossencoder = DPRCrossencoder(documents = documents)
# result = crossencoder.Lookup("where is manhattan?", 1, 2)

# for document in result:
#     print(document.GetText())

# tfidf = TFIDF(documents = documents)
# result = tfidf.Lookup("The city of new york might be cool but not as cool as a cat.", 2)

# for document in result:
#     print(document.GetText())