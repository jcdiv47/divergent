"""
Convert Qwen 2 models to LLama 3 models

Still WIP with bugs
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import fire
import torch
from huggingface_hub import split_torch_state_dict_into_shards
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.modeling_utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    shard_checkpoint,
)
from transformers.utils import check_min_version


try:
    check_min_version("4.41.2")
except Exception:
    raise ValueError("Please upgrade `transformers` to >=4.41.2")


CONFIG_FILE = "config.json"


def save_weight(
    qwen2_dir: str,
    llama3_dir: str,
    shard_size: str,
    save_safetensors: bool
) -> str:
    """ save weights """
    qwen2_path = Path(qwen2_dir)
    llama3_path = Path(llama3_dir)
    qwen2_state_dict: dict[str, torch.Tensor] = dict()

    for filepath in tqdm(qwen2_path.glob("*.safetensors"), desc="Load weights"):
        with safe_open(filepath, framework="pt", device="cpu") as f: # type: ignore[reportGeneralTypeIssues]
            for key in f.keys():
                qwen2_state_dict[key] = f.get_tensor(key)

    llama3_state_dict = deepcopy(qwen2_state_dict)
    torch_dtype = None
    for key, value in tqdm(qwen2_state_dict.items(), desc="Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
            break

    weights_name = SAFE_WEIGHTS_NAME if save_safetensors else WEIGHTS_NAME
    # TODO: Note that `shard_checkpoint` is deprecated and will be removed in v4.44.
    # We recommend you using split_torch_state_dict_into_shards from huggingface_hub library
    # split_torch_state_dict_into_shards(llama3_state_dict, max_shard_size=shard_size, weights_name=weights_name)
    shards, index = shard_checkpoint(llama3_state_dict, max_shard_size=shard_size, weights_name=weights_name)

    for shard_file, shard in tqdm(shards.items(), desc="Save weights"):
        if save_safetensors:
            save_file(shard, llama3_path / shard_file, metadata={"format": "pt"})
        else:
            torch.save(shard, llama3_path / shard_file)

    if index is None:
        logger.info(f"Model weights saved in {llama3_path / weights_name}")
    else:
        index_name = SAFE_WEIGHTS_INDEX_NAME if save_safetensors else WEIGHTS_INDEX_NAME
        with open(llama3_path / index_name, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        logger.info("Model weights saved in {llama3_dir}")

    return str(torch_dtype).replace("torch.", "")


def save_config(
    qwen2_dir: str,
    llama3_dir: str,
    torch_dtype: str
):
    """ save `config.json` """
    qwen2_path = Path(qwen2_dir)
    llama3_path = Path(llama3_dir)

    with open(qwen2_path / CONFIG_FILE, encoding="utf-8") as f:
        qwen2_config: dict[str, Any] = json.load(f)

    llama3_config: dict[str, Any] = dict()
    llama3_config["architectures"] = ["LlamaForCausalLM"]
    llama3_config["attention_bias"] = True
    llama3_config["attention_dropout"] = 0.0
    # llama3_config["bos_token_id"] = 128000
    # llama3_config["eos_token_id"] = 128009
    llama3_config["hidden_act"] = "silu"
    llama3_config["hidden_size"] = qwen2_config["hidden_size"]
    llama3_config["initializer_range"] = 0.02
    llama3_config["intermediate_size"] = qwen2_config["intermediate_size"]
    llama3_config["max_position_embeddings"] = 32768
    llama3_config["model_type"] = "llama"
    llama3_config["num_attention_heads"] = qwen2_config["num_attention_heads"]
    llama3_config["num_hidden_layers"] = qwen2_config["num_hidden_layers"]
    llama3_config["num_key_value_heads"] = qwen2_config["num_key_value_heads"]
    llama3_config["pretraining_tp"] = 1
    llama3_config["rms_norm_eps"] = 1e-06
    llama3_config["rope_scaling"] = None
    llama3_config["rope_theta"] = 1000000.0
    llama3_config["sliding_window"] = qwen2_config["sliding_window"]
    llama3_config["tie_word_embeddings"] = qwen2_config["tie_word_embeddings"]
    llama3_config["torch_dtype"] = torch_dtype
    llama3_config["transformers_version"] = qwen2_config["transformers_version"]
    llama3_config["use_cache"] = True
    llama3_config["vocab_size"] = qwen2_config["vocab_size"]

    with open(llama3_path / CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(llama3_config, f, indent=2)
    logger.info(f"Model config saved in {llama3_path / CONFIG_FILE}")


# TODO: what about the tokenizer?
def convert_qwen2_to_llama3(
    qwen2_dir: str,
    llama3_dir: str,
    shard_size: str = "2GB",
    save_safetensors: bool = False,
):
    """
    Converts the Qwen models in the same format as LLaMA2.


    >>> python convert_qwen2_to_llama3.py --qwen2_dir input --llama3_dir output
    """
    # try:
    #     Path(llama3_dir).mkdir(parents=True, exist_ok=False)
    # except Exception as e:
    #     raise print("Output dir already exists", e)

    # torch_dtype = save_weight(qwen2_dir, llama3_dir, shard_size, save_safetensors)
    torch_dtype = "bfloat16"
    save_config(qwen2_dir, llama3_dir, torch_dtype)


if __name__ == "__main__":
    fire.Fire(convert_qwen2_to_llama3)