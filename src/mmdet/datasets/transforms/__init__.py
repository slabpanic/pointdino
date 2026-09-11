"""Lazy data-transform exports.

Only the transform modules named by a pipeline are imported. This keeps
image-only inference independent of mask and augmentation dependencies.
"""

from importlib import import_module


_MODULES = {}
for module, names in {
    "augment_wrappers": ("AutoAugment", "RandAugment"),
    "colorspace": (
        "AutoContrast", "Brightness", "Color", "ColorTransform",
        "Contrast", "Equalize", "Invert", "Posterize", "Sharpness",
        "Solarize", "SolarizeAdd",
    ),
    "formatting": (
        "ImageToTensor", "PackDetInputs", "PackReIDInputs",
        "PackTrackInputs", "ToTensor", "Transpose",
    ),
    "frame_sampling": ("BaseFrameSample", "UniformRefFrameSample"),
    "geometric": (
        "GeomTransform", "Rotate", "ShearX", "ShearY", "TranslateX",
        "TranslateY",
    ),
    "instaboost": ("InstaBoost",),
    "loading": (
        "FilterAnnotations", "InferencerLoader", "LoadAnnotations",
        "LoadEmptyAnnotations", "LoadMultiChannelImageFromFiles",
        "LoadPanopticAnnotations", "LoadProposals", "LoadTrackAnnotations",
        "LoadImageFromNDArray",
    ),
    "text_transformers": ("LoadTextAnnotations", "RandomSamplingNegPos"),
    "transformers_glip": ("GTBoxSubOne_GLIP", "RandomFlip_GLIP"),
    "transforms": (
        "Albu", "CachedMixUp", "CachedMosaic", "CopyPaste", "CutOut",
        "Expand", "FixScaleResize", "FixShapeResize", "MinIoURandomCrop",
        "MixUp", "Mosaic", "Pad", "PhotoMetricDistortion", "RandomAffine",
        "RandomCenterCropPad", "RandomCrop", "RandomErasing", "RandomFlip",
        "RandomShift", "Resize", "ResizeShortestEdge", "SegRescale",
        "YOLOXHSVRandomAug",
    ),
    "wrappers": ("MultiBranch", "ProposalBroadcaster", "RandomOrder"),
}.items():
    for name in names:
        _MODULES[name] = f".{module}"

__all__ = tuple(_MODULES)


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
