from __future__ import annotations


class CharTokenizer:
    """Simple character tokenizer with 0 reserved for padding and 1 for unknown."""

    def __init__(self, alphabet: str | None = None) -> None:
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.stoi = {self.pad_token: 0, self.unk_token: 1}
        if alphabet:
            self.fit(alphabet)

    def fit(self, texts: str | list[str]) -> None:
        chars = sorted(set("".join(texts) if isinstance(texts, list) else texts))
        for ch in chars:
            if ch not in self.stoi:
                self.stoi[ch] = len(self.stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(ch, self.stoi[self.unk_token]) for ch in str(text)[:max_len]]
        if len(ids) < max_len:
            ids.extend([self.stoi[self.pad_token]] * (max_len - len(ids)))
        return ids


class KmerTokenizer(CharTokenizer):
    """Overlapping k-mer tokenizer with the same padding/unknown convention."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = int(k)
        if self.k <= 0:
            raise ValueError("k must be a positive integer")

    def _tokens(self, text: str) -> list[str]:
        text = str(text)
        if len(text) < self.k:
            return [text] if text else []
        return [text[i : i + self.k] for i in range(len(text) - self.k + 1)]

    def fit(self, texts: str | list[str]) -> None:
        values = texts if isinstance(texts, list) else [texts]
        tokens = sorted({token for text in values for token in self._tokens(text)})
        for token in tokens:
            if token not in self.stoi:
                self.stoi[token] = len(self.stoi)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(token, self.stoi[self.unk_token]) for token in self._tokens(text)[:max_len]]
        if len(ids) < max_len:
            ids.extend([self.stoi[self.pad_token]] * (max_len - len(ids)))
        return ids
