class KMeans:
    def __init__(self) -> None:
        pass

    def build_index(self, dataset):
        """
        @param dataset: The dataset for which an index containing embeddings should be built.
        """
        raise NotImplementedError
    
    def load_index(self, index_path: str):
        """
        @param index_path: The path to the pre-computed index.
        """
        raise NotImplementedError

    def embed_query(self, query: str):
        """
        @param query: The input text for which relevant passages should be found.
        returns: An embedding of the query.
        """
        raise NotImplementedError

    def lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        raise NotImplementedError