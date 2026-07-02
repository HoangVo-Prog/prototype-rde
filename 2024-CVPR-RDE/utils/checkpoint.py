# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
from collections import OrderedDict

import torch


def _strip_module_prefix_key(key):
    return key[len("module."):] if key.startswith("module.") else key


def _is_model_prototype_key(key):
    return _strip_module_prefix_key(key).startswith("prototype_branch.")


class Checkpointer:
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

    def resume(self, f=None):
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            raise IOError(f"No Checkpoint file found on {f}")
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            try:
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            except Exception as exc:
                self.logger.warning(
                    "Skipping optimizer state from %s because it is incompatible with the current model: %s",
                    f,
                    exc,
                )
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            try:
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
            except Exception as exc:
                self.logger.warning(
                    "Skipping scheduler state from %s because it is incompatible with the current optimizer: %s",
                    f,
                    exc,
                )
        # return any further checkpoint data
        return checkpoint

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, except_keys=None):
        load_state_dict(self.model, checkpoint.pop("model"), except_keys)


def check_key(key, except_keys):
    if except_keys is None:
        return False
    else:
        for except_key in except_keys:
            if except_key in key:
                return True
        return False


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys=None):
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    logger = logging.getLogger("PersonSearch.checkpoint")
    if not loaded_keys:
        logger.warning("Checkpoint model state is empty; keeping current model initialization.")
        return

    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    matched_loaded_keys = set()
    missing_keys = []
    shape_mismatches = []
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            missing_keys.append(current_keys[idx_new])
            continue

        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if check_key(key, except_keys):
            continue
        if tuple(model_state_dict[key].shape) != tuple(loaded_state_dict[key_old].shape):
            shape_mismatches.append((
                key,
                key_old,
                tuple(model_state_dict[key].shape),
                tuple(loaded_state_dict[key_old].shape),
            ))
            continue
        model_state_dict[key] = loaded_state_dict[key_old]
        matched_loaded_keys.add(key_old)
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )

    unexpected_keys = [key for key in loaded_keys if key not in matched_loaded_keys]
    missing_proto = [key for key in missing_keys if _is_model_prototype_key(key)]
    unexpected_proto = [key for key in unexpected_keys if _is_model_prototype_key(key)]
    if missing_proto:
        logger.warning(
            "Checkpoint is missing %d prototype tensor(s) for the current model; keeping initialized values. Examples: %s",
            len(missing_proto),
            ", ".join(missing_proto[:5]),
        )
    if unexpected_proto:
        logger.warning(
            "Checkpoint contains %d unmatched prototype tensor(s) that are ignored by the current model. Examples: %s",
            len(unexpected_proto),
            ", ".join(unexpected_proto[:5]),
        )
    if shape_mismatches:
        examples = [
            f"{key} <= {key_old} current={current_shape} checkpoint={loaded_shape}"
            for key, key_old, current_shape, loaded_shape in shape_mismatches[:5]
        ]
        logger.warning(
            "Skipped %d checkpoint tensor(s) with incompatible shapes. Examples: %s",
            len(shape_mismatches),
            "; ".join(examples),
        )


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, except_keys=None):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys)

    # use strict loading
    model.load_state_dict(model_state_dict)
