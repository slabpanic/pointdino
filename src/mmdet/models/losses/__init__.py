"""Lazy loss exports."""

from importlib import import_module


_MODULES = {}
for module, names in {
    "accuracy": ("Accuracy", "accuracy"),
    "ae_loss": ("AssociativeEmbeddingLoss",),
    "balanced_l1_loss": ("BalancedL1Loss", "balanced_l1_loss"),
    "cross_entropy_loss": (
        "CrossEntropyCustomLoss", "CrossEntropyLoss", "binary_cross_entropy",
        "cross_entropy", "mask_cross_entropy",
    ),
    "ddq_detr_aux_loss": ("DDQAuxLoss",),
    "dice_loss": ("DiceLoss",),
    "eqlv2_loss": ("EQLV2Loss",),
    "focal_loss": ("FocalCustomLoss", "FocalLoss", "sigmoid_focal_loss"),
    "gaussian_focal_loss": ("GaussianFocalLoss",),
    "gfocal_loss": ("DistributionFocalLoss", "QualityFocalLoss"),
    "ghm_loss": ("GHMC", "GHMR"),
    "iou_loss": (
        "BoundedIoULoss", "CIoULoss", "DIoULoss", "EIoULoss", "GIoULoss",
        "IoULoss", "SIoULoss", "bounded_iou_loss", "iou_loss",
    ),
    "kd_loss": ("KnowledgeDistillationKLDivLoss",),
    "l2_loss": ("L2Loss",),
    "margin_loss": ("MarginL2Loss",),
    "mse_loss": ("MSELoss", "mse_loss"),
    "multipos_cross_entropy_loss": ("MultiPosCrossEntropyLoss",),
    "pisa_loss": ("carl_loss", "isr_p"),
    "seesaw_loss": ("SeesawLoss",),
    "smooth_l1_loss": ("L1Loss", "SmoothL1Loss", "l1_loss", "smooth_l1_loss"),
    "triplet_loss": ("TripletLoss",),
    "utils": ("reduce_loss", "weight_reduce_loss", "weighted_loss"),
    "varifocal_loss": ("VarifocalLoss",),
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
