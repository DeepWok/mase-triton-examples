# Software Emulated MXFP

- [01_downcast_and_upcast.py](/mxfp/01_downcast_and_upcast.py) demonstrates how to convert between different types of MXFP objects (quantization & de-quantization).
- [02_lm_eval_ppl-w-only.py](/mxfp/02_lm_eval_ppl-w-only.py) shows how to apply **w-only** MXFP quantization to a HuggingFace model and evaluate its WikiText perplexity using the `lm_eval` library.
- [03_lm_eval-complex.py](/mxfp/03_lm_eval-complex.py) demonstrates how to apply wq MXFP quantization to a Llama model, including **FC layer weights/ FC layer activations/attention matmul activations (QK^T, AV)/KV cache**, and evaluate it using `lm_eval` library.