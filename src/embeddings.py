from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def encode_text(self, text):

        return self.model.encode(text)

    def encode_chunks(self, chunks):

        return self.model.encode(chunks)