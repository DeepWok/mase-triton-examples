# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jsonargparse[all]",
#     "lm-eval",
#     "mase-triton>=0.0.5",
#     "transformers==4.52.4",
# ]
# ///
"""
This script shows how to replace the linear layers of a Hugging Face model with MXFPLinearPTQ layers,
and evaluate the model's perplexity on the Wikitext dataset using lm-eval.

Usage:
    python mxfp/02_lm_eval_ppl-w-only.py --model_name <model_name> --w_meta <MXFPMeta>
    where <model_name> is the name of the Hugging Face model and <MXFPMeta> is one of the presets or None.

Example:
    python mxfp/02_lm_eval_ppl-w-only.py --model_name "unsloth/Llama-3.2-1B" --w_meta "MXFP8_E4M3"
"""

from typing import Literal, Union

import torch
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from mase_triton.mxfp.layers import MXFPLinearPTQ
from mase_triton.mxfp.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    MXFP8_E4M3_fn,
    MXFP8_E5M2_fn,
    MXFPMeta,
)
from mase_triton.utils.torch_module import set_layer_by_name
from transformers import AutoModelForCausalLM, AutoTokenizer


def replace_fc(model: torch.nn.Module, fc_kwargs: dict, skip_layers: list[str] = None):
    """
    Replace all Linear layers in the model with MXFPLinearPTQ layers.
    """
    replaced = 0
    for name, layer in model.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        if skip_layers is not None and name in skip_layers:
            print(f"Skipping layer: {name}")
            continue
        new_fc = MXFPLinearPTQ.from_linear(layer=layer, **fc_kwargs)
        set_layer_by_name(model, name, new_fc)
        replaced += 1

    print(f"Replaced {replaced} layers")
    return model


PRESET_MAP = {
    "MXFP8_E4M3": MXFP8_E4M3_fn,
    "MXFP8_E5M2": MXFP8_E5M2_fn,
    "MXFP6_E2M3": OCP_MXFP6_E2M3,
    "MXFP6_E3M2": OCP_MXFP6_E3M2,
    "MXFP4_E2M1": OCP_MXFP4_E2M1,
}


def eval_wikitext_ppl(
    model_name: str = "unsloth/Llama-3.2-1B",
    w_meta: Union[
        MXFPMeta,
        Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"],
        None,
    ] = "MXFP8_E4M3",
):
    """
    Evaluate the perplexity of a model on the Wikitext dataset with all weights of the linear layers (except lm_head) quantized with MXFP.

    Args:
        model_name (str): The name of the model to evaluate.
        w_meta (Union[MXFPMeta, str, None]): The metadata for the weights of the linear layers. If None, no quantization is applied.
            Can also be one of the following strings: "MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1".
    """
    tasks = ["wikitext"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create the quantization config (MXFPMeta) for MXFPLinearPTQ
    if w_meta is not None and not isinstance(w_meta, MXFPMeta):
        w_meta = PRESET_MAP[w_meta]

    q_config = {
        "x_mxfp_meta": None,  # No quantization for input activation
        "w_mxfp_meta": w_meta,  # Quantization metadata for weights
        "b_mxfp_meta": None,  # No quantization for bias
        "layer_type": "XWqB",  # Only the weight is quantized (Wq)
        "backend": "separate",  # Use 'separate' backend for quantization. For now the fused backend is not implemented.
    }

    # create the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model = model.eval()
    # replace the linear layers with MXFPLinearPTQ layers, except for the lm_head
    if w_meta is not None:
        model = replace_fc(model, fc_kwargs=q_config, skip_layers=["lm_head"])
    else:
        print("Skipping quantization, using original model weights.")
    # wrap the model with lm-eval's HFLM
    model_lm_eval = HFLM(pretrained=model, tokenizer=tokenizer, max_length=2048)
    # pass the wrapped model to the lm-eval's evaluator
    results = simple_evaluate(
        model=model_lm_eval, tasks=tasks, batch_size="auto", log_samples=False
    )
    # print the results
    table = make_table(results)
    print(w_meta)
    print(table)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(eval_wikitext_ppl)
