"""Lazy dataset exports.

Dataset classes are needed by training registration, but image-only
inference should not import every annotation format and evaluator.
"""

from importlib import import_module


_MODULES = {}
for module, names in {
    "ade20k": ("ADE20KInstanceDataset", "ADE20KPanopticDataset", "ADE20KSegDataset"),
    "base_det_dataset": ("BaseDetDataset",),
    "base_semseg_dataset": ("BaseSegDataset",),
    "base_video_dataset": ("BaseVideoDataset",),
    "cityscapes": ("CityscapesDataset",),
    "coco": ("CocoDataset",),
    "coco_caption": ("CocoCaptionDataset",),
    "coco_panoptic": ("CocoPanopticDataset",),
    "coco_semantic": ("CocoSegDataset",),
    "crowdhuman": ("CrowdHumanDataset",),
    "dataset_wrappers": ("ConcatDataset", "MultiImageMixDataset"),
    "deepfashion": ("DeepFashionDataset",),
    "dod": ("DODDataset",),
    "dsdl": ("DSDLDetDataset",),
    "flickr30k": ("Flickr30kDataset",),
    "isaid": ("iSAIDDataset",),
    "lvis": ("LVISDataset", "LVISV1Dataset", "LVISV05Dataset"),
    "mdetr_style_refcoco": ("MDETRStyleRefCocoDataset",),
    "mot_challenge_dataset": ("MOTChallengeDataset",),
    "objects365": ("Objects365V1Dataset", "Objects365V2Dataset"),
    "odvg": ("ODVGDataset",),
    "openimages": ("OpenImagesChallengeDataset", "OpenImagesDataset"),
    "refcoco": ("RefCocoDataset",),
    "reid_dataset": ("ReIDDataset",),
    "v3det": ("V3DetDataset",),
    "voc": ("VOCDataset",),
    "wider_face": ("WIDERFaceDataset",),
    "xml_style": ("XMLDataset",),
    "youtube_vis_dataset": ("YouTubeVISDataset",),
    "utils": ("get_loading_pipeline",),
    "samplers": (
        "AspectRatioBatchSampler", "ClassAwareSampler", "CustomSampleSizeSampler",
        "GroupMultiSourceSampler", "MultiSourceSampler",
        "TrackAspectRatioBatchSampler", "TrackImgSampler",
    ),
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
