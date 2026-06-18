import numpy as np


class Retriever:

    def __init__(
        self,
        vector_store,
        embedding_model
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        question,
        top_k=3
    ):

        question_embedding = (
            self.embedding_model.encode_text(
                question
            )
        )

        question_embedding = np.array(
            [question_embedding],
            dtype="float32"
        )

        distances, indices = (
            self.vector_store.index.search(
                question_embedding,
                top_k
            )
        )

        results = []

        for idx in indices[0]:

            if idx < len(
                self.vector_store.chunks
            ):
                results.append(
                    self.vector_store.chunks[idx]
                )

        return results