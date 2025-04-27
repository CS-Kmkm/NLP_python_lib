import re
import pandas as pd
from fugashi import Tagger
import unidic
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER
from rouge_score import rouge_scorer


class TextInfo:
    SUTEGANA = ['ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ャ', 'ュ', 'ョ', 'ヮ', "ッ"] # 捨て仮名
    CONTENT_POS = {"名詞", "代名詞", "形状詞", "連体詞", "動詞", "形容詞", "副詞"} # 内容語の品詞一覧(代名詞は内容語ではないが、言い換えにおいて頻出のためこのリストに追加)

    def __init__(self, dic=unidic.DICDIR):
        self.Wakati = Tagger(f"-Owakati -d {dic}")
        self.Tagger = Tagger(f"-d {dic}")

    def process_text(self, text: str) -> str:
        text = re.sub(r"\s+", "", text)
        text = re.sub("＊", "", text)
        text = re.sub("\*", "", text)
        except_list = ["。", "、",]
        for except_word in except_list:
            text = re.sub(except_word, "", text)
        return text

    def ngram(self, text: str, n: int = 1) -> list:
        # n-gramを計算
        text = self.process_text(text)
        parsed = self.Wakati.parse(text).split()
        return [parsed[i:i+n] for i in range(len(parsed)-n+1)]

    def count_units(self, text: str) -> int:
        # 形態素数をカウント
        return len(self.ngram(text, 1))

    def count_kana(self, text: str) -> int:
        # 仮名に変換したときの文字数をカウント
        text = self.process_text(text)
        kana = "".join(w.feature.kana for w in self.Tagger(text) if w.feature.kana is not None)
        return len(kana)

    def yomi(self, text: str) -> str:
        text = self.process_text(text)
        yomi = "".join(w.feature.pron for w in self.Tagger(text) if w.feature.pron is not None)
        return yomi

    def count_yomi(self, text: str) -> int:
        # 仮名に変換したときの文字数をカウント（≠音節, モーラ）
        yomi = self.yomi(text)
        return len(yomi)

    def count_mora(self, text: str) -> int:
        # モーラ数をカウント
        mora_exception = TextInfo.SUTEGANA.copy()
        mora_exception.remove("ッ")  # 促音はカウントする
        text = self.process_text(text)
        yomi = self.yomi(text)

        for kana in yomi:
            if kana in mora_exception:
                yomi = yomi.replace(kana, "")
        mora_cnt = len(yomi)
        return mora_cnt

    def count_syllable(self, text: str) -> int:
        # 音節数のカウント
        syllable_exceptions = TextInfo.SUTEGANA.copy()
        syllable_exceptions += ['ン', 'ー']  # 音節ではカウントしないものをリスト化
        text = self.process_text(text)
        yomi = self.yomi(text)

        for kana in yomi:
            if kana in syllable_exceptions:
                yomi = yomi.replace(kana, "")
        syllable_cnt = len(yomi)
        return syllable_cnt

    def collect_content_words(self, text: str) -> set[str]:
        # 内容語の基本形を抽出する関数
        content_words_lemma = [word.feature.lemma for word in self.Tagger(text) if word.feature.pos1 in TextInfo.CONTENT_POS]
        # print([word.feature.lemma for word in Tagger(text)])
        # print([word.feature.pos1 for word in Tagger(text)])
        return set(content_words_lemma)

    def overlap(self, text1: str, text2: str):
        # 2つの文が言い換えかどうかを判定する関数
        content_words_lemma1 = self.collect_content_words(text1)
        content_words_lemma2 = self.collect_content_words(text2)
        overlap = content_words_lemma1 & content_words_lemma2
        paraphrased = content_words_lemma1 ^ content_words_lemma2
        return overlap, paraphrased


class Metrics(TextInfo):
    def __init__(
            self,
            dic=unidic.DICDIR,
            use_stemmer: bool = False,
            metrics: list[str] = ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL'],
            score_type: str = "fmeasure"
    ):
        super().__init__(dic)
        self._stemmer = use_stemmer
        self.BLEU = BLEU(
            tokenize ="ja-mecab",
            smooth_method = "exp", # 'floor', 'add-k', 'exp' or 'none'
            smooth_value = None, # for 'floor' or 'add-k', default floor - 0.1
            # max_ngram_order=1,
        )
        self.CHRF = CHRF()
        self.TER = TER()
        # ROUGE関係の設定
        self.RougeScorer = rouge_scorer.RougeScorer(metrics, use_stemmer=use_stemmer, tokenizer=self)
        self.rouge_metrics = metrics
        self.score_type = score_type

    def tokenize(self, text: str) -> list[str]:
        if self._stemmer:
            return [word.feature.lemma for word in self.Tagger(text)]
        return [word.surface for word in self.Wakati(text)]

    def bleu_score(self, sys: list[str], refs: list[list[str]]):
        score = self.BLEU.corpus_score(sys, refs)
        return score.score
        # return score

    def chrf_score(self, sys: list[str], refs: list[list[str]]):
        score = self.CHRF.corpus_score(sys, refs)
        return score.score

    def ter_score(self, sys: list[str], refs: list[list[str]]):
        score = self.TER.corpus_score(sys, refs)
        return score.score

    def rouge_score(self, sys: str, refs: str):
        rouge_score = self.RougeScorer.score(sys, refs)
        return rouge_score


class DataFrameMetrics(Metrics):
    def __init__(self, dic=unidic.DICDIR, use_stemmer: bool = False):
        super().__init__(dic, use_stemmer)

    def calc_bleu_from_df(
            self, 
            df: pd.DataFrame, 
            sys_col: str = "paraphrase", 
            ref_col: str = "text"
        ) -> float:
        # refs = [[text] for text in df[ref_col].tolist()]
        refs = [df[ref_col].tolist()]
        sys = df[sys_col].tolist()
        # for text in refs:
        #     print(text)
        return self.bleu_score(sys, refs)

    def calc_chrf_from_df(
            self, 
            df: pd.DataFrame, 
            sys_col: str = "paraphrase", 
            ref_col: str = "text"
        ) -> float:
        refs = [df[ref_col].tolist()]
        sys = df[sys_col].tolist()
        return self.chrf_score(sys, refs)

    def calc_ter_from_df(
            self, 
            df: pd.DataFrame, 
            sys_col: str = "paraphrase", 
            ref_col: str = "text"
        ) -> float:
        refs = df[ref_col].tolist()
        sys = df[sys_col].tolist()
        return self.ter_score(sys, refs)

    def calc_rouge_from_df(
            self, 
            df: pd.DataFrame, 
            sys_col: str = "paraphrase", 
            ref_col: str = "text"
        ) -> float:
        refs = df[ref_col].tolist()
        sys = df[sys_col].tolist()
        results = [self.rouge_score(sy, ref) for sy, ref in zip(sys, refs)]
        metrics_scores = {}
        for metrics in self.rouge_metrics:
            if self.score_type == "fmeasure":
                scores = [result[metrics].fmeasure for result in results]
            elif self.score_type == "precision":
                scores = [result[metrics].precision for result in results]
            elif self.score_type == "recall":
                scores = [result[metrics].recall for result in results]
            score = sum(scores) / len(scores)
            metrics_scores[metrics] = score
        return metrics_scores


    def calc_overlap_from_df(
            self, 
            df: pd.DataFrame, 
            sys_col: str = "paraphrase", 
            ref_col: str = "context"
        ) -> float:
        refs = df[ref_col].tolist()
        sys = df[sys_col].tolist()
        overlap_cnt_ls = [len(self.overlap(sy, ref)[0]) for sy, ref in zip(sys, refs)]
        return sum(overlap_cnt_ls) / len(overlap_cnt_ls)

    def calc_length_from_df(
            self, 
            df: pd.DataFrame, 
            text_col: str = "paraphrase"
        ) -> float:
        texts = df[text_col].tolist()
        return sum([len(text) for text in texts]) / len(texts)

    def calc_units_from_df(
            self, 
            df: pd.DataFrame, 
            text_col: str = "paraphrase"
        ) -> float:
        texts = df[text_col].tolist()
        return sum([self.count_units(text) for text in texts]) / len(texts)

    def calc_kana_from_df(
            self, 
            df: pd.DataFrame, 
            text_col: str = "paraphrase"
        ) -> float:
        texts = df[text_col].tolist()
        return sum([self.count_kana(text) for text in texts]) / len(texts)

    def calc_mora_from_df(
            self, 
            df: pd.DataFrame, 
            text_col: str = "paraphrase"
        ) -> float:
        texts = df[text_col].tolist()
        return sum([self.count_mora(text) for text in texts]) / len(texts)

    def calc_syllable_from_df(
            self, 
            df: pd.DataFrame, 
            text_col: str = "paraphrase"
        ) -> float:
        texts = df[text_col].tolist()
        return sum([self.count_syllable(text) for text in texts]) / len(texts)

    def check_mora_limit(
        self,
        df: pd.DataFrame,
        text_col_name: str = "text",
        llm_col_name: str = "paraphrase",
        output_col_name: str = "mora_check",
        margin: int = 0,
    )-> int:
        df.loc[:, "text_mora"] = df[text_col_name].apply(self.count_mora)
        df.loc[:, "llm_mora"] = df[llm_col_name].apply(self.count_mora)
        df.loc[:, output_col_name] = df["text_mora"] - df["llm_mora"] <= margin
        return df[output_col_name].sum()

if __name__ == "__main__":
    text = "私は猫です。"
    text_info = TextInfo()
    # print(f"ngram : {text_info.ngram(text, 1)}")
    # print(f"units : {text_info.count_units(text)}")
    # print(f"kana : {text_info.count_kana(text)}")
    # print(f"yomi : {text_info.count_yomi(text)}")
    # print(f"mora : {text_info.count_mora(text)}")
    # print(f"syllable : {text_info.count_syllable(text)}")
    # print(f"content words : {text_info.collect_content_words(text)}")

    refs = [
        ["私は犬が好きです。", "私は犬が好きです。"],  # 1つ目の生成文に対する複数の参照
        # ["私は猫が好きです。", "私は猫が好きです。"],   # 2つ目の生成文に対する複数の参照
    ]
    sys = [
        "私は猫が好きです。",  # 生成された文
        # "私は犬が好きです。"   # 2つ目の生成された文
    ]
    metrics = Metrics()
    print(f"bleu : {metrics.bleu_score(sys, refs)}")
    # print(f"chrf : {metrics.chrf_score(sys, refs)}")
    # print(f"ter : {metrics.ter_score(sys, refs)}")
    # print(f"rouge : {metrics.rouge_score(refs[0][0], sys[0], metrics=['rouge1', 'rougeL'], use_stemmer=True)}")

    # text1 = "私は猫です。"
    # text2 = "私は犬です。"
    # print(f"overlap : {metrics.overlap(text1, text2)[0]}")
    # print(f"paraphrased : {metrics.overlap(text1, text2)[1]}")

    # dataframe関係
    # df = pd.read_csv("/home/koshi/b4nlp/output/units_paraphrase_res.csv")
    # df_metrics = DataFrameMetrics()
    # print(f"{df_metrics.calc_rouge_from_df(df, 'text', 'context')}")
    # print(f"{df_metrics.calc_rouge_from_df(df, 'context', 'text')}")
    # print(f"{df_metrics.calc_bleu_from_df(df, 'text', 'context')}")
