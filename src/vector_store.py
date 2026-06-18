import os
import faiss
import pickle
import numpy as np


INDEX_PATH = "data/faiss_index/index.faiss"
CHUNKS_PATH = "data/faiss_index/chunks.pkl"


class VectorStore:

    def __init__(self):

        self.index = None
        self.chunks = []

    def create_index(self, embeddings, chunks):

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(embeddings)

        self.chunks = chunks

    def save(self):

        os.makedirs(
            "data/faiss_index",
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            INDEX_PATH
        )

        with open(
            CHUNKS_PATH,
            "wb"
        ) as f:

            pickle.dump(
                self.chunks,
                f
            )

    def load(self):

        self.index = faiss.read_index(
            INDEX_PATH
        )

        with open(
            CHUNKS_PATH,
            "rb"
        ) as f:

            self.chunks = pickle.load(f)