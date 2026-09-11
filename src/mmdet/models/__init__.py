# Copyright (c) OpenMMLab. All rights reserved.
"""Model namespace.

Model families are registered by ``register_all_modules`` for training. The
empty package initializer keeps library inference from importing every model
family; importing a concrete submodule registers only that family.
"""
