from abc import ABCMeta, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModel, logging, BertJapaneseTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer


class Similarity(metaclass=ABCMeta):
    def __init__(self, model_name: str|None = None, device: str|None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @abstractmethod
    def embed_text(self, text: str):
        pass

    def calculate_similarity(self, text1: str, text2: str):
        embed_text1 = self.embed_text(text1).to(self.device)
        embed_text2 = self.embed_text(text2).to(self.device)
        # 1次元のテンソルを2次元に変換
        if embed_text1.ndimension() == 1:
            embed_text1 = embed_text1.reshape(1, -1)
        if embed_text2.ndimension() == 1:
            embed_text2 = embed_text2.reshape(1, -1)
        similarity_score = cosine_similarity(embed_text1.cpu(), embed_text2.cpu())[0][0]
        return similarity_score
    
    
    def k_similar(self, embed_text, embed_texts, targets, k: int, rank: str|None = "high", order: str|None = "desc"):
        # embed_textが1次元の場合は2次元に拡張
        if embed_text.ndimension() == 1:
            embed_text = embed_text.reshape(1, -1)

        # 各embed_text_iとのコサイン類似度を計算し、リストに保存
        similarities = np.array([
            cosine_similarity(embed_text.cpu().numpy(), embed_text_i.cpu().numpy().reshape(1, -1))[0][0]
            for embed_text_i in embed_texts
        ])
        if rank == "high":
            k_indices = similarities.argsort()[::-1][:k]
            if order == "asc":
                k_indices = k_indices[::-1]
        elif rank == "low":
            k_indices = similarities.argsort()[:k]
            if order == "desc":
                k_indices = k_indices[::-1]
        else:
            raise ValueError("rankの値が不正です。")
        return [targets[i] for i in k_indices]


class SimilarityJaColBERT(Similarity):
    def __init__(self, model_name: str|None = "answerdotai/JaColBERTv2.5", device: str|None = None):
        super().__init__(device)
        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def embed_text(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


class SimilaritySentenceBERT(Similarity):
    def __init__(self, model_name: str|None = "sonoisa/sentence-bert-base-ja-mean-tokens-v2", device: str|None = None):
        super().__init__(device)
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_text(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


class SimilaritySimCSE(Similarity):
    def __init__(self, model_name: str|None = "cl-nagoya/sup-simcse-ja-large", device: str|None = None):
        super().__init__(device)
        self.model = SentenceTransformer(model_name).to(self.device)

    def embed_text(self, text: str):
        return torch.tensor(self.model.encode(text, device=str(self.device)))


class SimilarityGLuCoSE(Similarity):
    def __init__(self, model_name: str|None = "pkshatech/GLuCoSE-base-ja", device: str|None = None):
        super().__init__(device)
        self.model = SentenceTransformer(model_name).to(self.device)

    def embed_text(self, text: str):
        return torch.tensor(self.model.encode(text, device=str(self.device)))


def k_similar(embed_text, embed_texts, targets, k: int, rank: str|None = "high", order: str|None = "desc"):
    # embed_textが1次元の場合は2次元に拡張
    if embed_text.ndimension() == 1:
        embed_text = embed_text.reshape(1, -1)

    # 各embed_text_iとのコサイン類似度を計算し、リストに保存
    similarities = np.array([
        cosine_similarity(embed_text.cpu().numpy(), embed_text_i.cpu().numpy().reshape(1, -1))[0][0]
        for embed_text_i in embed_texts
    ])
    if rank == "high":
        k_indices = similarities.argsort()[::-1][:k]
        if order == "asc":
            k_indices = k_indices[::-1]
    elif rank == "low":
        k_indices = similarities.argsort()[:k]
        if order == "desc":
            k_indices = k_indices[::-1]
    else:
        raise ValueError("rankの値が不正です。")
    return [targets[i] for i in k_indices]



if __name__ == "__main__":
    text1 = "今日はいい天気ですね。"
    text2 = "天気が悪いですね。"

    # JaColBERTテスト
    jacolbert_model = SimilarityJaColBERT()
    print("JaColBERT類似度:", jacolbert_model.calculate_similarity(text1, text2))

    # Sentence-BERTテスト
    sbert_model = SimilaritySentenceBERT()
    print("Sentence-BERT類似度:", sbert_model.calculate_similarity(text1, text2))

    # SimCSEテスト
    simcse_model = SimilaritySimCSE()
    print("SimCSE類似度:", simcse_model.calculate_similarity(text1, text2))

    # GLuCoSEテスト
    glucose_model = SimilarityGLuCoSE()
    print("GLuCoSE類似度:", glucose_model.calculate_similarity(text1, text2))