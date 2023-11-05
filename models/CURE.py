from models.builers.dense_retriever import DenseRetriever


class CURE(DenseRetriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(CURE, self).__init__(documents, index_path)