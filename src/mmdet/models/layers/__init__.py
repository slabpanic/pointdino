"""Lazy model-layer exports.

Importing one detector should not import every optional MMDetection layer.
The names remain compatible with the original package API and are loaded
only when requested by a concrete model.
"""

from importlib import import_module


_MODULES = {
    "SiLU": ".activations",
    "fast_nms": ".bbox_nms",
    "multiclass_nms": ".bbox_nms",
    "AdaptiveAvgPool2d": ".brick_wrappers",
    "FrozenBatchNorm2d": ".brick_wrappers",
    "adaptive_avg_pool2d": ".brick_wrappers",
    "ConvUpsample": ".conv_upsample",
    "CSPLayer": ".csp_layer",
    "DropBlock": ".dropblock",
    "ExpMomentumEMA": ".ema",
    "InvertedResidual": ".inverted_residual",
    "mask_matrix_nms": ".matrix_nms",
    "MSDeformAttnPixelDecoder": ".msdeformattn_pixel_decoder",
    "NormedConv2d": ".normed_predictor",
    "NormedLinear": ".normed_predictor",
    "PixelDecoder": ".pixel_decoder",
    "TransformerEncoderPixelDecoder": ".pixel_decoder",
    "LearnedPositionalEncoding": ".positional_encoding",
    "SinePositionalEncoding": ".positional_encoding",
    "SinePositionalEncoding3D": ".positional_encoding",
    "ResLayer": ".res_layer",
    "SimplifiedBasicBlock": ".res_layer",
    "ChannelAttention": ".se_layer",
    "DyReLU": ".se_layer",
    "SELayer": ".se_layer",
}

_MODULES.update({
    name: f".transformer.{module}"
    for module, names in {
        "conditional_detr_layers": (
            "ConditionalDetrTransformerDecoder",
            "ConditionalDetrTransformerDecoderLayer",
        ),
        "dab_detr_layers": (
            "DABDetrTransformerDecoder",
            "DABDetrTransformerDecoderLayer",
            "DABDetrTransformerEncoder",
        ),
        "ddq_detr_layers": ("DDQTransformerDecoder",),
        "deformable_detr_layers": (
            "DeformableDetrTransformerDecoder",
            "DeformableDetrTransformerDecoderLayer",
            "DeformableDetrTransformerEncoder",
            "DeformableDetrTransformerEncoderLayer",
        ),
        "detr_layers": (
            "DetrTransformerDecoder",
            "DetrTransformerDecoderLayer",
            "DetrTransformerEncoder",
            "DetrTransformerEncoderLayer",
        ),
        "dino_layers": (
            "CdnQueryGenerator",
            "DinoTransformerDecoder",
            "PointCdnQueryGenerator",
        ),
        "grounding_dino_layers": (
            "GroundingDinoTransformerDecoder",
            "GroundingDinoTransformerDecoderLayer",
            "GroundingDinoTransformerEncoder",
        ),
        "mask2former_layers": (
            "Mask2FormerTransformerDecoder",
            "Mask2FormerTransformerDecoderLayer",
            "Mask2FormerTransformerEncoder",
        ),
        "utils": (
            "MLP",
            "AdaptivePadding",
            "ConditionalAttention",
            "DynamicConv",
            "PatchEmbed",
            "PatchMerging",
            "coordinate_to_encoding",
            "inverse_sigmoid",
            "nchw_to_nlc",
            "nlc_to_nchw",
        ),
    }.items()
    for name in names
})

__all__ = tuple(_MODULES)


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
