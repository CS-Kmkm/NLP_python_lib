from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
import sys
import os
from copy import deepcopy
import torch
from abc import ABCMeta, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append("/workspace/b4nlp/my_python_lib")
sys.path.append(r"/home/koshi/b4nlp/my_python_lib")
from examples_picker import PickExample, PickExampleRandom, PickExampleSimilarity, PickExampleLength
from dataframe_metrics import DataFrameMetrics
from bertscore import BERTScoreTiiger
from evaluate_different_random import evaluate_different_random, process_text

class LLM_Meta(metaclass=ABCMeta):
    def __init__(
            self,
            model_name: str = "gpt2",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # クラスのインスタンス作成
        self.metrics = DataFrameMetrics()
        self.BERTScorer = BERTScoreTiiger()
        self.device = torch.device(device)

        # modelの準備
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def set_user_template(self, template: list[dict[str, str]]):
        self.user_template = template

    def set_assistant_template(self, template: list[dict[str, str]]):
        self.assistant_template = template

    def add_system_message(self, system_message: list[dict[str, str]]):
        self.system_message = system_message

    def set_examples(self, examples: list[dict[str, str]]):
        self.examples = examples

    def message(self, prompt: list[str]):
        texts = deepcopy(self.user_template)
        texts[0]["content"] = texts[0]["content"].format(*prompt)
        message = deepcopy(self.examples)
        message.extend(texts) # userを追加
        # assistant = deepcopy(self.assistant_template)
        # assistant[0]["content"] = assistant[0]["content"].format("")
        # message.extend(assistant) # assistantを追加
        print(message)
        return message

    @abstractmethod
    def generate(self, texts: list[str]) -> str:
        pass

    def generate_from_df(
        self,
        df: pd.DataFrame,
        text_col_names: list[str] = ["context"],
        paraphrased_col_name: str = "paraphrase",
    ) -> pd.DataFrame:
        df[paraphrased_col_name] = df.apply(lambda x: self.generate([x[col] for col in text_col_names]), axis=1)
        return df


class VLLM(LLM_Meta):
    def __init__(
            self,
            model_name: str = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model_name, device)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,  # default is 0.9
            # enable_prefix_caching=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=512, stop="<|eot_id|>"
        )

    def generate(self, texts: list[str]):
        prompt = self.tokenizer.apply_chat_template(
            self.message(texts), tokenize=False, add_generation_prompt=True
        )
        output = self.llm.generate(prompt, self.sampling_params)
        return output[0].outputs[0].text


class Transformer_LLM(LLM_Meta):
    def __init__(
            self,
            model_name: str = "llm-jp/llm-jp-3-3.7b-instruct",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model_name, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

class Transformer_LLM_gemma(Transformer_LLM):
    # gemma2b-jpn-itのデモに合わせた形
    def __init__(
            self,
            model_name: str = "google/gemma-2-2b-jpn-it",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model_name, device)

    def generate(self, texts: list[str]):
        prompt = self.tokenizer.apply_chat_template(
                self.message(texts),return_tensors="pt", add_generation_prompt=True, return_dict=True
            ).to(self.model.device)
        outputs = self.model.generate(**prompt, max_new_tokens=256, temperature=0.0,)
        generated_text = self.tokenizer.batch_decode(outputs[:, prompt['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        print(generated_text.strip())
        return generated_text.strip()

class Transformer_LLM_qwen(Transformer_LLM):
    # gemma2b-jpn-itのデモに合わせた形
    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model_name, device)

    def generate(self, texts: list[str]):
        prompt = self.tokenizer.apply_chat_template(
                self.message(texts),return_tensors="pt", add_generation_prompt=True, tokenize=False
            )
        prompt = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**prompt, max_new_tokens=256, do_sample=False)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt.input_ids, generated_ids)
        ]
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text.strip())
        return generated_text.strip()

class Transformer_LLM_llmjp(Transformer_LLM):
    # llm-jp/llm-jp-3-3.7b-instructのデモに合わせた形
    def __init__(
            self,
            model_name: str = "llm-jp/llm-jp-3-3.7b-instruct",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model_name, device)

    def generate(self, texts: list[str]):
        tokenized_prompt = self.tokenizer.apply_chat_template(
                self.message(texts),return_tensors="pt", tokenize=True, add_generation_prompt=True
            ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                tokenized_prompt, 
                max_new_tokens=256,
                do_sample=False
            )[0]
        generated_text = self.tokenizer.decode(outputs)
        return generated_text.strip()


class Transformer_LLM_terminators(Transformer_LLM):
    # rinna/llama-3-youko-8b-instruct
    def __init__(
            self,
            model_name: str = "rinna/llama-3-youko-8b-instruct",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model_name, device)
        self.terminators = [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    def generate(self, texts: list[str]):
        prompt = self.tokenizer.apply_chat_template(
                self.message(texts), return_tensors="pt", add_generation_prompt=True
            ).to(self.model.device)

        outputs = self.model.generate(
            prompt,
            max_new_tokens=512,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.0,
        )
        response = outputs[0][prompt.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        return response.strip()

if __name__ == "__main__":
    llm = VLLM()
    print(llm.llm.get_default_sampling_params())

    # path = "/workspace/b4nlp/output/sentences_paraphrase_res.csv"
    # df = pd.read_csv(path)
    # # df.apply(lambda x: print(x["context"], x["text"]), axis=1)
    # df["context"] = df["context"].apply(process_text)
    # df["text"] = df["text"].apply(process_text)
    # df_train = df[df["data_type"] == "train"]
    # df_dev = df[df["data_type"] == "dev"]
    # example_picker = PickExampleRandom(df_train)
    # example_picker.pick_example()
    # llm.add_system_message([
    #     # {
    #     #     "role": "system",
    #     #     "content":
    #     #         "あなたは役に立つアシスタントです。\n"
    #     #         "あなたはユーザの発話に対して傾聴応答を行うことができます。\n"
    #     #         "傾聴応答はユーザの発話に対する傾聴態度を示すために行われる発話を指します。\n"
    #     #         "言い換え応答は、発話の一部を言い換えることにより、ユーザの発話を理解して共有する応答です。\n"
    #     # }
    # ])
    # llm.set_user_template([
    #         {
    #             "role": "user",
    #             "content": "次の発話に対する言い換え応答を考えてください。\n発話：{}"
    #         }
    # ])
    # llm.set_assistant_template([
    #         {
    #             "role": "assistant",
    #             "content": "応答：{}"
    #         }
    # ])

    # message = deepcopy(llm.system_message)
    # for _, row in example_picker.picked_df.iterrows():
    #     user = deepcopy(llm.user_template)
    #     user[0]["content"] = user[0]["content"].format(row["context"])
    #     assistant = deepcopy(llm.assistant_template)
    #     assistant[0]["content"] = assistant[0]["content"].format(row["text"])
    #     message.extend(user)
    #     message.extend(assistant)
    # print(message)
    # llm.set_examples(message)

    # df_ret = llm.generate_from_df(df_dev)
    # df_ret.to_csv("/workspace/b4nlp/output/test_swallow.csv", index=False)


