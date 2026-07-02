def prototype_requested(args):
    return bool(
        getattr(args, "prototype", False)
        or getattr(args, "use_loss_id", False)
    )


def pbt_enabled(args):
    return prototype_requested(args) and not bool(getattr(args, "no_pbt", False))


def ira_enabled(args):
    return (
        pbt_enabled(args)
        and bool(getattr(args, "use_loss_id", False))
        and not bool(getattr(args, "no_ira", False))
    )


def effective_prototype_id_weight(args):
    return float(getattr(args, "prototype_id_weight", 0.2)) if ira_enabled(args) else 0.0


def ablation_suffix(args):
    tags = []
    if getattr(args, "no_pbt", False):
        tags.append("no_pbt")
    if getattr(args, "no_ira", False):
        tags.append("no_ira")
    return "" if not tags else "_" + "_".join(tags)


def finalize_ablation_args(args):
    args.pbt_enabled = pbt_enabled(args)
    args.ira_enabled = ira_enabled(args)
    args.prototype_id_weight_effective = effective_prototype_id_weight(args)
    return args


def log_ablation_config(args, logger):
    finalize_ablation_args(args)

    if getattr(args, "no_pbt", False) and prototype_requested(args):
        logger.warning(
            "--no_pbt disables the prototype/PBT branch; prototype initialization, "
            "memory updates, and prototype losses are inactive."
        )
        if not getattr(args, "no_ira", False) and getattr(args, "use_loss_id", False):
            logger.warning(
                "IRA/proto_id_loss depends on the prototype branch and is disabled "
                "effectively because --no_pbt is active."
            )

    logger.info("PBT: %s", "enabled" if args.pbt_enabled else "disabled")
    logger.info("IRA/proto_id_loss: %s", "enabled" if args.ira_enabled else "disabled")
    logger.info(
        "Effective loss weights: prototype_id_weight=%s (configured=%s)",
        args.prototype_id_weight_effective,
        getattr(args, "prototype_id_weight", 0.2),
    )
