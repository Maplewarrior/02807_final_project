class BM25:
    def __init__(self) -> None:
        pass

    def lookup(self, query: str, k: int):
        """
        @param query: The input text to which relevant passages should be found.
        @param k: The number of relevant passages to retrieve.
        """
        raise NotImplementedError