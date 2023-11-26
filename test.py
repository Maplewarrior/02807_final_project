from sentence_transformers import CrossEncoder

crossencoder = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")

predicts = crossencoder.predict([("I have a bird in my garden.", "I have a large bird in my garden."),("How many songs are there?", "The bridge under the field is long.")])

print(predicts)

reranked_documents = sorted(predicts, reverse=True, key= lambda x: x)

print(reranked_documents)