from typing import Iterable
from sentence_transformers import SentenceTransformer, util
import torch


class SearchAgent:

    def __init__(self, pretrained: str = "all-MiniLM-L6-v2", corpus=None):
        self.model = SentenceTransformer(pretrained)
        self.corpus = corpus
        self.corpus_embeddings = self.embed(self.corpus)

    def embed(self, texts: Iterable[str]):
        return self.model.encode(texts, convert_to_tensor=True)

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.embed(query)
        scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(scores, k=top_k)
        top_results = [
            {"candidate": self.corpus[idx], "score": score} for score, idx in zip(top_results[0], top_results[1])
        ]
        return top_results
