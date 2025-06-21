"""
This script shows a more complex example of how to quantize the linear layers, attention matmuls (QK^T and AV), and KV cache of Llama models using MXFP quantization,
and evaluate the quantized model using lm-eval on various tasks.

Usage:
    python mxfp/03_lm_eval_ppl-complex.py --help

Example:
    # Note that the default model is unsloth/Llama-3.2-1B. Change it to your model of choice using --model_name <MODEL_NAME>

    # Using preset to quantize the model linear weights, attention activations, and KV cache
    python mxfp/03_lm_eval_ppl-complex.py --preset XqWqKVq --preset_dtype MXFP8_E4M3
    # Weight-only quantization of the model linear weights
    python mxfp/03_lm_eval_ppl-complex.py --preset XWqKV --preset_dtype MXFP8_E4M3
    # Original model without quantization
    python mxfp/03_lm_eval_ppl-complex.py --preset original
    # Only quantize the KV cache
    python mxfp/03_lm_eval_ppl-complex.py --preset null --kv_cache_meta MXFP8_E4M3
"""

from pprint import pformat
from typing import Literal, Optional, Tuple, Union

import torch
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from mase_triton.mxfp import functional as MXFP_F
from mase_triton.mxfp.layers import MXFPLinearPTQ
from mase_triton.mxfp.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    OCP_MXFP8_E4M3,
    OCP_MXFP8_E5M2,
    MXFPMeta,
)
from mase_triton.utils.torch_module import set_layer_by_name
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    Cache,
    FlashAttentionKwargs,
    LlamaAttention,
    LlamaForCausalLM,
    Unpack,
    apply_rotary_pos_emb,
    repeat_kv,
)


class LlamaAttentionMXFP(LlamaAttention):
    def __init__(
        self,
        config,
        layer_idx,
        qk_q_meta: MXFPMeta | None,
        qk_k_meta: MXFPMeta | None,
        qk_func_type: Literal["XW", "XqW", "XWq", "XqWq"] | None,
        av_a_meta: MXFPMeta | None,
        av_v_meta: MXFPMeta | None,
        av_func_type: Literal["XW", "XqW", "XWq", "XqWq"] | None,
        kv_cache_meta: MXFPMeta | None = None,
        mxfp_mm_backend: Literal["separate", "fused"] = "separate",
    ):
        super().__init__(config, layer_idx)
        self.qk_q_meta = qk_q_meta
        self.qk_k_meta = qk_k_meta
        self.qk_func_type = qk_func_type
        self.av_a_meta = av_a_meta
        self.av_v_meta = av_v_meta
        self.av_func_type = av_func_type
        self.mxfp_mm_backend = mxfp_mm_backend
        self.kv_cache_meta = kv_cache_meta

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # [batch_size, num_heads, seq_length, head_dim]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # *: quantize KV cache if meta is not None
            if self.kv_cache_meta is not None:
                key_states = MXFP_F.quantize_dequantize(
                    key_states, block_dim=-1, mxfp_meta=self.kv_cache_meta
                )
                value_states = MXFP_F.quantize_dequantize(
                    value_states, block_dim=-1, mxfp_meta=self.kv_cache_meta
                )
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: callable = eager_attention_forward_mxfp

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            qk_q_meta=self.qk_q_meta,
            qk_k_meta=self.qk_k_meta,
            qk_func_type=self.qk_func_type,
            av_a_meta=self.av_a_meta,
            av_v_meta=self.av_v_meta,
            av_func_type=self.av_func_type,
            mxfp_mm_backend=self.mxfp_mm_backend,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def from_attention(
        cls,
        attention: LlamaAttention,
        qk_q_meta: MXFPMeta | None = None,
        qk_k_meta: MXFPMeta | None = None,
        qk_func_type: Literal["XW", "XqW", "XWq", "XqWq"] | None = None,
        av_a_meta: MXFPMeta | None = None,
        av_v_meta: MXFPMeta | None = None,
        av_func_type: Literal["XW", "XqW", "XWq", "XqWq"] | None = None,
        mxfp_mm_backend: Literal["separate", "fused"] = "separate",
        kv_cache_meta: MXFPMeta | None = None,
    ):
        new_attn = cls(
            config=attention.config,
            layer_idx=attention.layer_idx,
            qk_q_meta=qk_q_meta,
            qk_k_meta=qk_k_meta,
            qk_func_type=qk_func_type,
            av_a_meta=av_a_meta,
            av_v_meta=av_v_meta,
            av_func_type=av_func_type,
            mxfp_mm_backend=mxfp_mm_backend,
            kv_cache_meta=kv_cache_meta,
        )
        new_attn.to(attention.q_proj.weight.dtype)
        # load q/k/v/o projections
        # this assumes that the projections are not quantized yet
        new_attn.load_state_dict(attention.state_dict(), strict=True)
        return new_attn


def eager_attention_forward_mxfp(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    qk_q_meta: MXFPMeta | None = None,
    qk_k_meta: MXFPMeta | None = None,
    qk_func_type: Literal["XW", "XqW", "XWq", "XqWq"] | None = None,
    av_a_meta: MXFPMeta | None = None,
    av_v_meta: MXFPMeta | None = None,
    av_func_type: Literal["XW", "XqW", "XWq", "XqWq"] | None = None,
    mxfp_mm_backend: Literal["separate", "fused"] = "separate",
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # *: quantized QK matmul if meta is not None
    # attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = MXFP_F.mxfp_matmul(
        query,
        key_states.transpose(2, 3),
        input_meta=qk_q_meta,
        other_meta=qk_k_meta,
        func_type=qk_func_type,
        backend=mxfp_mm_backend,
    )
    attn_weights = attn_weights * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    # *: quantized AV matmul if meta is not None
    # attn_output = torch.matmul(attn_weights, value_states)
    attn_output = MXFP_F.mxfp_matmul(
        attn_weights,
        value_states,
        input_meta=av_a_meta,
        other_meta=av_v_meta,
        func_type=av_func_type,
        backend=mxfp_mm_backend,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def replace_attn(model: torch.nn.Module, attn_kwargs: dict):
    assert isinstance(model, LlamaForCausalLM)
    replaced = 0
    for name, layer in model.named_modules():
        if not isinstance(layer, LlamaAttention):
            continue
        if isinstance(layer, LlamaAttentionMXFP):
            print(f"Skipping already replaced layer: {name}")
            continue
        new_attn = LlamaAttentionMXFP.from_attention(attention=layer, **attn_kwargs)
        set_layer_by_name(model, name, new_attn)
        replaced += 1

    print(f"Replaced {replaced} attention layers")
    return model


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


PRESET_META_MAP = {
    "MXFP8_E4M3": OCP_MXFP8_E4M3,
    "MXFP8_E5M2": OCP_MXFP8_E5M2,
    "MXFP6_E2M3": OCP_MXFP6_E2M3,
    "MXFP6_E3M2": OCP_MXFP6_E3M2,
    "MXFP4_E2M1": OCP_MXFP4_E2M1,
}


def parse_mxfp_meta(
    mxfp_meta: Union[
        MXFPMeta,
        Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"],
        None,
    ],
) -> Optional[MXFPMeta]:
    if mxfp_meta is None:
        return None
    if isinstance(mxfp_meta, MXFPMeta):
        return mxfp_meta
    if isinstance(mxfp_meta, str):
        if mxfp_meta in PRESET_META_MAP:
            return PRESET_META_MAP[mxfp_meta]
        else:
            raise ValueError(f"Unknown MXFP meta: {mxfp_meta}")
    raise TypeError(f"Invalid type for mxfp_meta: {type(mxfp_meta)}")


def mxfp_lm_eval(
    model_name: str = "unsloth/Llama-3.2-1B",
    tasks: Union[str, list[str]] = "wikitext",
    fc_x_meta: Union[
        MXFPMeta,
        Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"],
        None,
    ] = None,
    fc_w_meta: Union[
        MXFPMeta,
        Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"],
        None,
    ] = None,
    qk_q_meta: Union[
        MXFPMeta,
        Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"],
        None,
    ] = None,
    qk_k_meta: Union[
        MXFPMeta,
        Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"],
        None,
    ] = None,
    av_a_meta: Union[
        MXFPMeta,
        Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"],
        None,
    ] = None,
    av_v_meta: Union[
        MXFPMeta,
        Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"],
        None,
    ] = None,
    kv_cache_meta: Union[
        MXFPMeta,
        Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"],
        None,
    ] = None,
    preset: Union[Literal["XqWqKVq", "XWqKV", "XWKVq", "original"], None] = "XqWqKVq",
    preset_dtype: Literal[
        "MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"
    ] = "MXFP8_E4M3",
    mxfp_mm_backend: Literal["separate"] = "separate",
):
    """
    Evaluate the perplexity of a model on lm-eval's tasks with MXFP quantization applied to the linear layers, attention matmuls, and KV cache. When preset is not None, it uses the preset configuration for quantization parameters; else it uses the custom parameters provided by the user.

    Args:
        model_name (str): The name of the model to evaluate.
        tasks (Union[str, list[str]]): The tasks to evaluate on. Can be a single task or a list of tasks.
        fc_x_meta (Union[MXFPMeta, str, None]): The metadata for the input activation of the linear layers. If None, no quantization is applied.
        fc_w_meta (Union[MXFPMeta, str, None]): The metadata for the weights of the linear layers. If None, no quantization is applied.
        qk_q_meta (Union[MXFPMeta, str, None]): The metadata for the query of the QK^T matmul. If None, no quantization is applied.
        qk_k_meta (Union[MXFPMeta, str, None]): The metadata for the key of the QK^T. If None, no quantization is applied.
        av_a_meta (Union[MXFPMeta, str, None]): The metadata for the attention scores of the AV matmul. If None, no quantization is applied.
        av_v_meta (Union[MXFPMeta, str, None]): The metadata for the values of the AV matmul. If None, no quantization is applied.
        kv_cache_meta (Union[MXFPMeta, str, None]): The metadata for the KV cache. If None, no quantization is applied.
        preset (Optional[Literal["XqWqKVq", "XWqKV", "XWKVq", "original"]]): A preset configuration for quantization parameters. If None, custom parameters are used. X/Xq indicates whether the input activations (of FC and QK/AV matmul) are quantized; W/Wq indicates whether the weights of FC are quantized; KVq indicates whether the KV cache is quantized.
        preset_dtype (Literal["MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1"]): The data type for the preset configuration.
        mxfp_mm_backend (Literal["separate"]): The backend to use for MXFP matmuls.
    """
    fc_kwargs = {}
    attn_kwargs = {}
    if preset is not None:
        if preset == "original":
            fc_kwargs = {
                "x_mxfp_meta": None,
                "w_mxfp_meta": None,
                "b_mxfp_meta": None,
                "layer_type": "XWB",
                "backend": mxfp_mm_backend,
            }
            attn_kwargs = {
                "qk_q_meta": None,
                "qk_k_meta": None,
                "av_a_meta": None,
                "av_v_meta": None,
                "qk_func_type": "XW",
                "av_func_type": "XW",
                "kv_cache_meta": None,
                "mxfp_mm_backend": mxfp_mm_backend,
            }
            print("Using original parameters, no quantization applied.")
        else:
            fc_layer_type = ""
            if "Xq" in preset:
                fc_kwargs["x_mxfp_meta"] = PRESET_META_MAP[preset_dtype]
                fc_layer_type += "Xq"
            else:
                fc_kwargs["x_mxfp_meta"] = None
                fc_layer_type += "X"
            if "Wq" in preset:
                fc_kwargs["w_mxfp_meta"] = PRESET_META_MAP[preset_dtype]
                fc_layer_type += "Wq"
            else:
                fc_kwargs["w_mxfp_meta"] = None
                fc_layer_type += "W"
            fc_kwargs["b_mxfp_meta"] = None
            fc_layer_type += "B"
            fc_kwargs["layer_type"] = fc_layer_type
            fc_kwargs["backend"] = mxfp_mm_backend

            attn_kwargs["qk_q_meta"] = None
            attn_kwargs["qk_k_meta"] = None
            attn_kwargs["av_a_meta"] = None
            attn_kwargs["av_v_meta"] = None
            attn_kwargs["qk_func_type"] = "XW"
            attn_kwargs["av_func_type"] = "XW"

            if "Xq" in preset:
                attn_kwargs["qk_q_meta"] = PRESET_META_MAP[preset_dtype]
                attn_kwargs["qk_k_meta"] = PRESET_META_MAP[preset_dtype]
                attn_kwargs["av_a_meta"] = PRESET_META_MAP[preset_dtype]
                attn_kwargs["av_v_meta"] = PRESET_META_MAP[preset_dtype]
                attn_kwargs["qk_func_type"] = "XqWq"
                attn_kwargs["av_func_type"] = "XqWq"

            if "KVq" in preset:
                attn_kwargs["kv_cache_meta"] = PRESET_META_MAP[preset_dtype]
            else:
                attn_kwargs["kv_cache_meta"] = None
            attn_kwargs["mxfp_mm_backend"] = mxfp_mm_backend
            print(f"Using preset {preset}, which sets the following parameters:\n")
            print(f"fc_kwargs:\n{pformat(fc_kwargs)}")
            print(f"attn_kwargs:\n{pformat(attn_kwargs)}")
    else:
        fc_layer_type = ""
        fc_kwargs["x_mxfp_meta"] = parse_mxfp_meta(fc_x_meta)
        if fc_x_meta is None:
            fc_layer_type += "X"
        else:
            fc_layer_type += "Xq"
        fc_kwargs["w_mxfp_meta"] = parse_mxfp_meta(fc_w_meta)
        if fc_w_meta is None:
            fc_layer_type += "W"
        else:
            fc_layer_type += "Wq"
        fc_kwargs["b_mxfp_meta"] = None
        fc_layer_type += "B"
        fc_kwargs["layer_type"] = fc_layer_type
        fc_kwargs["backend"] = mxfp_mm_backend

        qk_func_type = ""
        av_func_type = ""
        attn_kwargs["qk_q_meta"] = parse_mxfp_meta(qk_q_meta)
        if qk_q_meta is None:
            qk_func_type += "X"
        else:
            qk_func_type += "Xq"
        attn_kwargs["qk_k_meta"] = parse_mxfp_meta(qk_k_meta)
        if qk_k_meta is None:
            qk_func_type += "W"
        else:
            qk_func_type += "Wq"
        attn_kwargs["av_a_meta"] = parse_mxfp_meta(av_a_meta)
        if av_a_meta is None:
            av_func_type += "X"
        else:
            av_func_type += "Xq"
        attn_kwargs["av_v_meta"] = parse_mxfp_meta(av_v_meta)
        if av_v_meta is None:
            av_func_type += "W"
        else:
            av_func_type += "Wq"
        attn_kwargs["qk_func_type"] = qk_func_type
        attn_kwargs["av_func_type"] = av_func_type
        attn_kwargs["kv_cache_meta"] = parse_mxfp_meta(kv_cache_meta)
        attn_kwargs["mxfp_mm_backend"] = mxfp_mm_backend
        print("Using custom parameters:\n")
        print(f"fc_kwargs:\n{pformat(fc_kwargs)}")
        print(f"attn_kwargs:\n{pformat(attn_kwargs)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # create the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    replace_attn(model, attn_kwargs=attn_kwargs)
    replace_fc(model, fc_kwargs=fc_kwargs, skip_layers=["lm_head"])

    model = model.to(device)
    # wrap the model with lm-eval's HFLM
    model_lm_eval = HFLM(pretrained=model, tokenizer=tokenizer, max_length=2048)
    # pass the wrapped model to the lm-eval's evaluator
    if isinstance(tasks, str):
        tasks = [tasks]
    results = simple_evaluate(
        model=model_lm_eval, tasks=tasks, batch_size="auto", log_samples=False
    )
    # print the results
    table = make_table(results)
    print(table)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(
        mxfp_lm_eval,
    )
