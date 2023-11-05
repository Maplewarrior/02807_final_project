from models.builers.retriever import Retriever


class KMeans(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(KMeans, self).__init__(documents, index_path)