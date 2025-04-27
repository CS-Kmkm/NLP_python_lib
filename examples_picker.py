import pandas as pd
import sys
import os
from abc import ABCMeta, abstractmethod

sys.path.append("/workspace/b4nlp/my_python_lib")
sys.path.append(r"/home/koshi/b4nlp/my_python_lib")
from similarity import Similarity, SimilarityJaColBERT, SimilarityGLuCoSE, SimilaritySimCSE, SimilaritySentenceBERT
from dataframe_metrics import DataFrameMetrics
from bertscore import BERTScoreTiiger


class PickExample(metaclass=ABCMeta):
    def __init__(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def pick_example(self, num: int = 10):
        pass


class PickExampleRandom(PickExample):
    def __init__(self, df: pd.DataFrame, seed: int = 42):
        self.df = df
        self.seed = seed

    def change_seed(self, seed: int):
        self.seed = seed

    def pick_example(self, num: int = 10):
        self.picked_df = self.df.sample(n=num, random_state=self.seed)
        return self.picked_df
    
class PickExampleSimilarity(PickExample):
    def __init__(
            self,
            df: pd.DataFrame,
            similarity: Similarity,
            text_col_name: str = "text",
            # model_name: str|None = None,
            # device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.df = df
        self.df["embed_text"] = self.df[text_col_name].apply(similarity.embed_text)
        self.similarity = similarity

    def pick_example(self, num: int = 10):
        # ToDo: 類似度が高いものを選ぶ
        self.picked_df = self.df.sample(n=num)
        return self.picked_df


class PickExampleLength(PickExample):
    def __init__(
            self,
            df: pd.DataFrame,
            text_col_name: str = "text",
            method: str = "mora"
    ):
        self.df = df
        self.metrics = DataFrameMetrics()
        self.text_col_name = text_col_name
        self.func_dict = {
            "units": self.metrics.count_units,
            "kana": self.metrics.count_kana,
            "yomi": self.metrics.count_kana,
            "mora": self.metrics.count_mora,
            "syllable": self.metrics.count_syllable
        }
        self.calc_length_func = self.func_dict[method]
        self.df["length"] = self.df[self.text_col_name].apply(self.calc_length_func)

    def pick_example(
            self,
            num: int = 10,
            mode: str = "long",
            reversed: bool = False
    ):
        if mode == "long":
            self.picked_df = self.df.sort_values("length", ascending=False).head(num)
        elif mode == "short":
            self.picked_df = self.df.sort_values("length", ascending=True).head(num)
        if reversed:
            self.picked_df = self.picked_df[::-1]
        return self.picked_df

    def change_method(self, method: str):
        self.calc_length_func = self.func_dict[method]
        self.df["length"] = self.df[self.text_col_name].apply(self.calc_length_func)
        return self.df

if __name__ == "__main__":
    path = r"/home/koshi/b4nlp/output/sentences_paraphrase_res_w_slash_annotated.csv"
    df = pd.read_csv(path)
    pick_example_length = PickExampleLength(df)
    for i in range(10):
        df = pick_example_length.pick_example(i)
        print(df["response_id"].to_list())

    # similarity_jcolbert = SimilarityJaColBERT()
    # pick_example_similarity = PickExampleSimilarity(df, similarity_jcolbert)
    # df = pick_example_similarity.pick_example(10)
