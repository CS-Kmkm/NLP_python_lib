# 異なる乱数シードに対して生成した結果の比較
# 言い換えのテキストは必ずshot:{shot}_seed:{random_seed}
import pandas as pd
from dataframe_metrics import Metrics, DataFrameMetrics
from bertscore import BERTScoreTiiger


def process_text(text: str) -> str:
    text = "".join(text.split(" "))
    text = "".join(text.split("/"))
    return text

def process_text_clause(text: str) -> str:
    text = "".join(text.split(" "))
    return text


def evaluate_different_random(
    df: pd.DataFrame,
    output_file_path: str,
    shots: list[int] = list(range(11)),
    random_seeds: list[int] = list(range(5)),
    ref_col: str = "text",
    BERTScorer: BERTScoreTiiger = BERTScoreTiiger(),
    df_metrics: DataFrameMetrics = DataFrameMetrics()
) -> pd.DataFrame:

    bleu_scores = []
    chrf_scores = []
    ter_scores = []
    overlap_cnts = []
    length_cnts = []
    units_cnts = []
    kana_cnts = []
    mora_cnts = []
    syllable_cnts = []
    BERTScores = []
    rouge1_scores = []
    rouge2_scores = []
    rouge3_scores = []
    rouge4_scores = []
    rougeL_scores = []

    for shot in shots:
        bleu_each = 0
        chrf_each = 0
        ter_each = 0
        overlap_each = 0
        length_each = 0
        units_each = 0
        kana_each = 0
        mora_each = 0
        syllable_each = 0
        BERTScore_each = 0
        rouge1_each = 0
        rouge2_each = 0
        rouge3_each = 0
        rouge4_each = 0
        rougeL_each = 0

        for random_seed in random_seeds:
            sys_col = f"shot:{shot}_seed:{random_seed}"
            bleu_each += df_metrics.calc_bleu_from_df(df, sys_col, ref_col)
            chrf_each += df_metrics.calc_chrf_from_df(df, sys_col, ref_col)
            ter_each += df_metrics.calc_ter_from_df(df, sys_col, ref_col)
            BERTScore_each += BERTScorer.compute_f1(df[ref_col].tolist(), df[sys_col].tolist())
            overlap_each += df_metrics.calc_overlap_from_df(df, sys_col, ref_col)
            length_each += df_metrics.calc_length_from_df(df, sys_col)
            units_each += df_metrics.calc_units_from_df(df, sys_col)
            kana_each += df_metrics.calc_kana_from_df(df, sys_col)
            mora_each += df_metrics.calc_mora_from_df(df, sys_col)
            syllable_each += df_metrics.calc_syllable_from_df(df, sys_col)
            rouge_scores = df_metrics.calc_rouge_from_df(df, sys_col, ref_col)
            rouge1_each += rouge_scores["rouge1"]
            rouge2_each += rouge_scores["rouge2"]
            rouge3_each += rouge_scores["rouge3"]
            rouge4_each += rouge_scores["rouge4"]
            rougeL_each += rouge_scores["rougeL"]

        random_seeds_length = len(random_seeds)
        bleu_scores.append(bleu_each / random_seeds_length)
        chrf_scores.append(chrf_each / random_seeds_length)
        ter_scores.append(ter_each / random_seeds_length)
        BERTScores.append(BERTScore_each / random_seeds_length)
        overlap_cnts.append(overlap_each / random_seeds_length)
        length_cnts.append(length_each / random_seeds_length)
        units_cnts.append(units_each / random_seeds_length)
        kana_cnts.append(kana_each / random_seeds_length)
        mora_cnts.append(mora_each / random_seeds_length)
        syllable_cnts.append(syllable_each / random_seeds_length)
        rouge1_scores.append(rouge1_each / random_seeds_length)
        rouge2_scores.append(rouge2_each / random_seeds_length)
        rouge3_scores.append(rouge3_each / random_seeds_length)
        rouge4_scores.append(rouge4_each / random_seeds_length)
        rougeL_scores.append(rougeL_each / random_seeds_length)

    min_length = min(len(bleu_scores), len(chrf_scores), len(ter_scores), len(overlap_cnts), len(length_cnts), len(units_cnts), len(kana_cnts), len(mora_cnts), len(syllable_cnts))
    df_results = pd.DataFrame({
        "id": shots[:min_length],
        "bleu_scores": bleu_scores[:min_length],
        "chrf_scores": chrf_scores[:min_length],
        "ter_scores": ter_scores[:min_length],
        "overlap_cnts": overlap_cnts[:min_length],
        "length_cnts": length_cnts[:min_length],
        "units_cnts": units_cnts[:min_length],
        "kana_cnts": kana_cnts[:min_length],
        "mora_cnts": mora_cnts[:min_length],
        "syllable_cnts": syllable_cnts[:min_length],
        "BERTScores": BERTScores[:min_length],
        "rouge1": rouge1_scores[:min_length],
        "rouge2": rouge2_scores[:min_length],
        "rouge3": rouge3_scores[:min_length],
        "rouge4": rouge4_scores[:min_length],
        "rougeL": rougeL_scores[:min_length]
    })
    # CSVファイルに保存
    df_results.to_csv(output_file_path, index=False)
    return df_results