import numpy as np
from numpy.linalg import norm

def GetSimilarity(x1, x2, similarity_measure):
    if similarity_measure == "cosine":
        return CosineDistance(x1, x2)
    elif similarity_measure == "l2_norm":
        return L2Norm(x1, x2)
    raise ValueError("Similarity measure not implemented")

def L2Norm(x1, x2):
    return np.sum((x1.flatten()-x2.flatten())**2)

def CosineDistance(x1, x2):
    return 1-np.dot(x1.flatten(), x2.flatten())/(norm(x1.flatten())*norm(x2.flatten()) + 1e-10)