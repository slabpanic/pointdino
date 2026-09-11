"""Lazy transformer-layer exports."""

from importlib import import_module


_MODULES = {
    "ConditionalDetrTransformerDecoder": ".conditional_detr_layers",
    "ConditionalDetrTransformerDecoderLayer": ".conditional_detr_layers",
    "DABDetrTransformerDecoder": ".dab_detr_layers",
    "DABDetrTransformerDecoderLayer": ".dab_detr_layers",
    "DABDetrTransformerEncoder": ".dab_detr_layers",
    "DDQTransformerDecoder": ".ddq_detr_layers",
    "DeformableDetrTransformerDecoder": ".deformable_detr_layers",
    "DeformableDetrTransformerDecoderLayer": ".deformable_detr_layers",
    "DeformableDetrTransformerEncoder": ".deformable_detr_layers",
    "DeformableDetrTransformerEncoderLayer": ".deformable_detr_layers",
    "DetrTransformerDecoder": ".detr_layers",
    "DetrTransformerDecoderLayer": ".detr_layers",
    "DetrTransformerEncoder": ".detr_layers",
    "DetrTransformerEncoderLayer": ".detr_layers",
    "CdnQueryGenerator": ".dino_layers",
    "DinoTransformerDecoder": ".dino_layers",
    "PointCdnQueryGenerator": ".dino_layers",
    "GroundingDinoTransformerDecoder": ".grounding_dino_layers",
    "GroundingDinoTransformerDecoderLayer": ".grounding_dino_layers",
    "GroundingDinoTransformerEncoder": ".grounding_dino_layers",
    "Mask2FormerTransformerDecoder": ".mask2former_layers",
    "Mask2FormerTransformerDecoderLayer": ".mask2former_layers",
    "Mask2FormerTransformerEncoder": ".mask2former_layers",
    "MLP": ".utils",
    "AdaptivePadding": ".utils",
    "ConditionalAttention": ".utils",
    "DynamicConv": ".utils",
    "PatchEmbed": ".utils",
    "PatchMerging": ".utils",
    "coordinate_to_encoding": ".utils",
    "inverse_sigmoid": ".utils",
    "nchw_to_nlc": ".utils",
    "nlc_to_nchw": ".utils",
}

__all__ = tuple(_MODULES)


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
