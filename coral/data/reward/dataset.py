import os
import torch
import pickle

from torch.utils.data import Dataset


class ListwiseRankingDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_title_chars: int = 256,
        max_review_chars: int = 256,
        max_doc_chars: int | None = None, 
    ):
        assert isinstance(data_path, str)
        assert os.path.isfile(data_path)

        self.max_title_chars = max_title_chars
        self.max_review_chars = max_review_chars
        self.max_doc_chars = max_doc_chars

        with open(data_path, "rb") as f:
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def _truncate(self, text: str, max_chars: int) -> str:
        return text[:max_chars] if max_chars is not None else text

    def _merge_title_review(self, title: str, review: str) -> str:
        """
        Merge title + review with independent caps.
        Title is always preserved first.
        """
        title = self._truncate(title.strip(), self.max_title_chars)
        review = self._truncate(review.strip(), self.max_review_chars)

        if review:
            merged = f"{title}. {review}"
        else:
            merged = title

        if self.max_doc_chars is not None:
            merged = merged[: self.max_doc_chars]

        return merged

    def __getitem__(self, idx):
        sample = self.samples[idx]

        query = sample["query"]
        candidates = sample["candidates"]

        docs = [
            self._merge_title_review(
                c.get("title", ""),
                c.get("review", ""),
            )
            for c in candidates
        ]

        labels = torch.tensor(
            [c["relevance"] for c in candidates],
            dtype=torch.long,
        )

        return {
            "query": query,
            "docs": docs,  
            "labels": labels,
        }