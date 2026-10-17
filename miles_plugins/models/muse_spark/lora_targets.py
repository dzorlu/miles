"""The linears Muse Spark's native LoRA adapts, as Megatron module paths within a decoder layer.

Kept free of Megatron imports: argument validation reads it in every miles process."""

ADAPTER_TARGETS = (
    "self_attention.linear_qkv",
    "self_attention.linear_proj",
    "mlp.experts.linear_fc1",
    "mlp.experts.linear_fc2",
)
