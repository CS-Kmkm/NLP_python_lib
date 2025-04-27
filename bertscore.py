import numpy as np
import pandas as pd
import evaluate
import bert_score
import torch


class BERTScoreEvaluate:
    # evaluateのBERTScoreを利用するクラス。
    # scorerを利用していないので、多数評価する場合の使用は注意
    # https://huggingface.co/spaces/evaluate-metric/bertscore
    def __init__(
            self,
            model_type: str | None = None,
            lang: str = "ja",
            rescale_with_baseline: bool = False,
            type: str = "mean",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        self.device = device
        self.bertscore_eval = evaluate.load("bertscore")

    def _compute(self, refs: list[str], hyps: list[str]):
        P, R, F1 = self.bertscore_eval.compute(
            cands=hyps, 
            refs=refs, 
            lang=self.lang, 
            model_type=self.model_type, 
            device = self.device,
            rescale_with_baseline=self.rescale_with_baseline
        )
        return {"precision": P, "recall": R, "f1": F1}

    def compute_precision(self, refs: list[str], hyps: list[str]):
        return self._compute(refs, hyps)["precision"].numpy().tolist()
    
    def compute_recall(self, refs: list[str], hyps: list[str]):
        return self._compute(refs, hyps)["recall"].numpy().tolist()

    def compute_f1(self, refs: list[str], hyps: list[str]):
        return self._compute(refs, hyps)["f1"].numpy().tolist()
    
class BERTScoreTiiger:
    def __init__(
            self, 
            model_type:None|str = None,
            num_layers:None|int = None,
            lang: str = 'ja',
            rescale_with_baseline: bool = False,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            verbose: bool = False,
            type: str = "mean",
        ):
        self.scorer = bert_score.BERTScorer(
            model_type=model_type, 
            num_layers=num_layers,
            lang=lang, 
            rescale_with_baseline=rescale_with_baseline,
            device=device,
        )
        self.verbose = verbose
        self.type = type

    def _compute(self, refs: list[str], hyps: list[str]):
        P, R, F1 = self.scorer.score(
            cands=hyps, 
            refs=refs,
            verbose=self.verbose
        )
        return P, R, F1

    def compute_precision(self, refs: list[str], hyps: list[str]):
        P, _, _ = self._compute(refs, hyps)
        if self.type == "mean":
            return P.numpy().mean()
        else:
            return P.numpy().tolist()
    
    def compute_recall(self, refs: list[str], hyps: list[str]):
        _, R, _ = self._compute(refs, hyps)
        if self.type == "mean":
            return R.numpy().mean()
        else:
            return R.numpy().tolist()

    def compute_f1(self, refs: list[str], hyps: list[str]):
        _, _, F1 = self._compute(refs, hyps)
        if self.type == "mean":
            return F1.numpy().mean()
        else:
            return F1.numpy().tolist()

    def compute_f1_from_df(
            self,
            df: pd.DataFrame,
            ref_col_name: str = "text",
            sys_col_name: str = "paraphrase"
    ) -> float:
        sys = df[sys_col_name].tolist()
        refs = df[ref_col_name].tolist()
        return self.compute_f1(refs, sys)



if __name__ == "__main__":
    # refs = [
    #     "夕食には寿司を食べるのが好きです。", 
    #     "今日はいい天気ですね", 
    #     "今日は本当にいい天気",
    #     "暇な時間にはビデオゲームをするのが好きです。", 
    #     "太陽が空で輝いています。", 
    #     "今週末、海に旅行に行くつもりです。",
    # ]
    # cands = [
    #     "夕食に食べるのは寿司が一番好きな食べ物です。", 
    #     "今日は良くない天気ですね",
    #     "今日は本当にいい天気", 
    #     "暇な時間にはビデオゲームをするのは楽しいです。", 
    #     "外では今、激しい雨が降っています。", 
    #     "週末は仕事で、楽しいことをすることができません。",
    # ]
    
    # # BERTScoreEvaluateクラスの利用
    # bertscore_evaluate = BERTScoreEvaluate()
    # print("evaluateモジュールによる精度:", bertscore_evaluate.compute_precision(refs, cands))
    
    # BERTScoreTiigerクラスの利用
    bert_score_tiiger = BERTScoreTiiger()
    ref = "アウトドアですね"
    sys1 = "外で遊ぶのが好きなんですね"
    sys2 = "アウトドア派ですね"
    print("bert-scoreモジュールF1:", bert_score_tiiger.compute_f1([ref], [sys1]))
    print("bert-scoreモジュールF1:", bert_score_tiiger.compute_f1([ref], [sys2]))
    # print("bert-scoreモジュールP:", bert_score_tiiger.compute_precision(refs, cands))
    # print("bert-scoreモジュールR:", bert_score_tiiger.compute_recall(refs, cands))
    # print("bert-scoreモジュールF1:", bert_score_tiiger.compute_f1(refs, cands))
    # print("bert-scoreモジュールF1:", bert_score_tiiger.compute_f1(cands, refs))

    # # BERTScoreの計算
    # BERTScore_tohoku_bert = BERTScoreTiiger(model_type="tohoku-nlp/bert-base-japanese", num_layers=12)
    # print("tohoku-nlp/bert-base-japanese P:", BERTScore_tohoku_bert.compute_precision(refs, cands))
    # print("tohoku-nlp/bert-base-japanese R:", BERTScore_tohoku_bert.compute_recall(refs, cands))
    # print("tohoku-nlp/bert-base-japanese F1:", BERTScore_tohoku_bert.compute_f1(refs, cands))
