

class TFIDF_retriever:
    def __init__(self) -> None:
        pass

    def lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        raise NotImplementedError

class BM25_retriever:
    def __init__(self) -> None:
        pass

    def lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        raise NotImplementedError

class DPR_retriever:
    def lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        raise NotImplementedError

    def lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        raise NotImplementedError

class DPR_rerank_retriever:
    def __init__(self) -> None:
        pass

    def lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        raise NotImplementedError
class Kmeans_retriever:
    def __init__(self) -> None:
        pass

    def lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        raise NotImplementedError
class CURE_retriever:
    def __init__(self) -> None:
        pass

    def lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        raise NotImplementedError