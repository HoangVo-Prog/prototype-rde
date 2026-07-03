def prototype_requested(args):
    return bool(
        getattr(args, "prototype", False)
        or getattr(args, "use_loss_id", False)
    )


def pbt_enabled(args):
    return prototype_requested(args) and not bool(getattr(args, "no_pbt", False))


def ira_enabled(args):
    return prototype_requested(args) and not bool(getattr(args, "no_ira", False))


def proto_id_loss_enabled(args):
    return prototype_requested(args) and bool(getattr(args, "use_loss_id", False))


def effective_prototype_id_weight(args):
    return float(getattr(args, "prototype_id_weight", 0.2)) if proto_id_loss_enabled(args) else 0.0


def ablation_suffix(args):
    tags = []
    if getattr(args, "no_pbt", False):
        tags.append("no_pbt")
    if getattr(args, "no_ira", False):
        tags.append("no_ira")
    return "" if not tags else "_" + "_".join(tags)


def finalize_ablation_args(args):
    args.prototype_enabled = prototype_requested(args)
    args.pbt_enabled = pbt_enabled(args)
    args.ira_enabled = ira_enabled(args)
    args.proto_id_loss_enabled = proto_id_loss_enabled(args)
    args.prototype_id_weight_effective = effective_prototype_id_weight(args)
    return args


def log_ablation_config(args, logger):
    finalize_ablation_args(args)

    if getattr(args, "no_pbt", False) and prototype_requested(args):
        logger.info("--no_pbt active: using raw cross-modal prototype banks instead of translated PBT banks")
    if getattr(args, "no_ira", False) and prototype_requested(args):
        logger.info(
            "--no_ira active: using global_%s prototype assignment instead of identity-restricted assignment",
            getattr(args, "no_ira_mode", "hard"),
        )

    logger.info("Prototype branch: %s", "enabled" if args.prototype_enabled else "disabled")
    logger.info("PBT translated banks: %s", "enabled" if args.pbt_enabled else "disabled")
    logger.info("IRA identity-restricted assignment: %s", "enabled" if args.ira_enabled else "disabled")
    logger.info("Prototype identity loss: %s", "enabled" if args.proto_id_loss_enabled else "disabled")
    logger.info(
        "Effective loss weights: prototype_id_weight=%s (configured=%s)",
        args.prototype_id_weight_effective,
        getattr(args, "prototype_id_weight", 0.2),
    )
