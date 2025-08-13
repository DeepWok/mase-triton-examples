# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jsonargparse[all]",
#     "lm-eval",
#     "mase-triton>=0.0.1",
#     "transformers==4.52.4",
# ]
# ///
"""
This script demonstrates how to downcast/quantize a BF16 tensor to MXFP, and upcast it back to BF16.
"""

import torch
from mase_triton.mxfp.functional import compose_mxfp_tensor, extract_mxfp_components
from mase_triton.mxfp.meta import (
    OCP_MXFP4_E2M1,
    OCP_MXFP6_E2M3,
    OCP_MXFP6_E3M2,
    MXFP8_E4M3_fn,
    MXFP8_E5M2_fn,
    MXFPMeta,
)
from mase_triton.utils.bit_repr import get_binary_repr, get_binary_repr_bf16
from mase_triton.utils.train_utils import set_seed

set_seed(42)  # for reproducibility


def quantize():
    torch.set_printoptions(linewidth=200)
    # define a mxfp format
    mxfp_format = MXFPMeta(
        block_size=2,  # a small block size for demonstration
        scale_exp_bits=8,  # the bit width of the shared exponent
        element_exp_bits=4,  # the bit width of the element (MiniFloat) exponent
        element_frac_bits=3,  # the bit width of the element (MiniFloat) fraction
        element_is_finite=True, # saturation instead of Inf/NaN
        round_mode="rn", # round to nearest even
    )

    w = torch.randn((3, 2), dtype=torch.bfloat16, device="cuda") * 100.0
    print("Original tensor:")
    print(get_binary_repr_bf16(w))
    # quantize the tensor to MXFP format, extracting shared scales and elements
    # tensor_meta contains the metadata of the MXFP tensor, like original shape, the dimension to share scales, device, etc.
    scales, elements, tensor_meta = extract_mxfp_components(
        w, block_dim=1, mxfp_meta=mxfp_format
    )
    print("Tensor Meta data")
    print(tensor_meta)
    print("Shared scales:")
    print(get_binary_repr(scales))
    print("Elements (Minifloat)")
    print(get_binary_repr(elements))
    """outputs:
    Original tensor:
    [['1 10000110 0001011' '0 10000010 1010011']
    ['0 10000100 0101111' '0 10000110 1000101']
    ['1 10000111 0001001' '1 10000100 1011111']]
    Tensor Meta data
    MXFPTensorMeta(device='cuda:0', shape=(3, 2), block_dim=1, meta=MXFPMeta(block_size=2, scale_exp_bits=8, element_exp_bits=4, element_frac_bits=3))
    Shared scales:
    [['1000 0110']
    ['1000 0110']
    ['1000 0111']]
    Elements (Minifloat)
    [['1011 1000' '0001 1101']
    ['0010 1010' '0011 1100']
    ['1011 1000' '1010 0101']]
    """


def quantize_and_dequantize():
    torch.set_printoptions(linewidth=200)
    mxfp_format = MXFP8_E5M2_fn

    w = torch.randn((3, 64), dtype=torch.bfloat16, device="cuda") * 100.0
    scales, elements, tensor_meta = extract_mxfp_components(
        w, block_dim=1, mxfp_meta=mxfp_format
    )
    # dequantize the MXFP tensor back to BF16
    w_mxfp8 = compose_mxfp_tensor(scales, elements, tensor_meta)
    avg_err = (w - w_mxfp8).abs().mean().item()
    err_ratio = avg_err / w.abs().mean().item()
    print(f"Tensor Meta data: {tensor_meta}")
    print("Mean abs error: ", avg_err)
    print("Error ratio: ", err_ratio)
    """outputs:
    Tensor Meta data: MXFPMeta(block_size=32, scale_exp_bits=8, element_exp_bits=5, element_frac_bits=2)
    Mean abs error:  6.25
    Error ratio:  0.08223684210526316
    """


if __name__ == "__main__":
    quantize()
    quantize_and_dequantize()
