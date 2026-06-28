import logging
import os
import time
import torch
from torch.utils.data import DataLoader
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from utils.train_diagnostics import compute_train_diagnostics
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
from pylab import xticks,yticks,np
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from datasets.bases import ImageTextDataset
from datasets.build import build_transforms, collate


################### CODE FOR THE BETA MODEL  ########################

import scipy.stats as stats
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _prototype_requested(args):
    return (
        getattr(args, "prototype", False)
        or getattr(args, "use_loss_id", False)
    )


def _prototype_ready(model):
    branch = getattr(_unwrap_model(model), "prototype_branch", None)
    return branch is not None and branch.is_ready()


def _build_prototype_init_loader(train_loader, args):
    train_set = getattr(train_loader, "dataset", None)
    source_dataset = getattr(train_set, "dataset", None)
    if train_set is None or source_dataset is None:
        return None

    prototype_set = ImageTextDataset(
        source_dataset,
        args,
        transform=build_transforms(img_size=args.img_size, aug=False, is_train=False),
        text_length=getattr(train_set, "text_length", args.text_length),
        truncate=getattr(train_set, "truncate", True),
        inject_noise=False,
    )
    prototype_set.txt_aug = False
    prototype_set.img_aug = False

    return DataLoader(
        prototype_set,
        batch_size=getattr(args, "test_batch_size", args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )


@torch.no_grad()
def _project_prototype_feature_bank(branch, image_features, text_features, batch_size):
    image_projected, text_projected = [], []
    batch_size = max(int(batch_size), 1)
    for start in range(0, image_features.shape[0], batch_size):
        end = start + batch_size
        image_batch, text_batch = branch.project_for_memory(
            image_features[start:end],
            text_features[start:end],
        )
        image_projected.append(image_batch.cpu())
        text_projected.append(text_batch.cpu())
    return torch.cat(image_projected, dim=0), torch.cat(text_projected, dim=0)


@torch.no_grad()
def maybe_initialize_prototypes(model, train_loader, args, device, logger):
    model_without_ddp = _unwrap_model(model)
    branch = getattr(model_without_ddp, "prototype_branch", None)
    if branch is None or branch.is_ready():
        return

    logger.info(
        "Initializing PBT prototypes from full train embeddings (feature=%s)",
        getattr(model_without_ddp, "prototype_feature_source", "auto"),
    )
    was_training = model_without_ddp.training
    prototype_loader = _build_prototype_init_loader(train_loader, args)
    if prototype_loader is None:
        logger.warning("Falling back to the training loader for prototype initialization")
        prototype_loader = train_loader
    else:
        logger.info("Using a dedicated no-augmentation loader for prototype initialization")

    image_features, text_features, pids = [], [], []
    try:
        model_without_ddp.eval()
        for batch in prototype_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            image_feat, text_feat = model_without_ddp.extract_prototype_features(batch)
            image_features.append(image_feat.cpu())
            text_features.append(text_feat.cpu())
            pids.append(batch["pids"].cpu())
    finally:
        model_without_ddp.train(was_training)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    pids = torch.cat(pids, dim=0)

    if pids.numel() != len(getattr(prototype_loader, "dataset", [])):
        raise RuntimeError("prototype initialization did not scan the full train dataset")

    if hasattr(branch, "needs_pca_init") and branch.needs_pca_init():
        logger.info("Initializing prototype projector from raw train embeddings")
        branch.initialize_projector_from_features(image_features, text_features)

    image_features, text_features = _project_prototype_feature_bank(
        branch,
        image_features,
        text_features,
        getattr(args, "test_batch_size", getattr(args, "batch_size", 512)),
    )
    branch.initialize_projected(image_features, text_features, pids)
    logger.info("Prototype banks initialized with %d samples", pids.numel())
    synchronize()


def _loss_components(ret):
    return {
        key: value
        for key, value in ret.items()
        if "loss" in key and torch.is_tensor(value)
    }


def _to_float(value):
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() == 0:
            return None
        return value.float().mean().item()
    if value is None:
        return None
    return float(value)


def _update_meter(meters, key, value, batch_size):
    value = _to_float(value)
    if value is None:
        return
    meters.setdefault(key, AverageMeter()).update(value, batch_size)


def _grad_norm_by_loss(losses, model):
    params = [p for p in _unwrap_model(model).parameters() if p.requires_grad]
    norms = {}
    if not params:
        return norms

    for name, loss in losses.items():
        if not loss.requires_grad:
            norms[f"{name}_grad_norm"] = 0.0
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        grad_sq_sum = loss.new_zeros(())
        has_grad = False
        for grad in grads:
            if grad is None:
                continue
            has_grad = True
            grad_sq_sum = grad_sq_sum + grad.detach().float().pow(2).sum()
        norms[f"{name}_grad_norm"] = grad_sq_sum.sqrt().item() if has_grad else 0.0
    return norms


def get_loss(model, data_loader):
    logger = logging.getLogger("RDE.train")
    model.eval()
    model_without_ddp = _unwrap_model(model)
    device = "cuda"
    data_size = data_loader.dataset.__len__()
    real_labels = data_loader.dataset.real_correspondences
    lossA, lossB, simsA,simsB = torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size),torch.zeros(data_size)
    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        index = batch['index']
        with torch.no_grad(): 
            la, lb, sa, sb = model_without_ddp.compute_per_loss(batch)
            for b in range(la.size(0)):
                lossA[index[b]]= la[b]
                lossB[index[b]]= lb[b]
                simsA[index[b]]= sa[b]
                simsB[index[b]]= sb[b]
            if i % 100 == 0:
                logger.info(f'compute loss batch {i}')

    losses_A = (lossA-lossA.min())/(lossA.max()-lossA.min())    
    losses_B = (lossB-lossB.min())/(lossB.max()-lossB.min())
    
    input_loss_A = losses_A.reshape(-1,1) 
    input_loss_B = losses_B.reshape(-1,1)
 
    logger.info('\nFitting GMM ...') 
 
    if model_without_ddp.args.noisy_rate > 0.4 or model_without_ddp.args.dataset_name=='RSTPReid':
        # should have a better fit 
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
        gmm_B = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]
 
 
    pred_A = split_prob(prob_A, 0.5)
    pred_B = split_prob(prob_B, 0.5)
  
    return torch.Tensor(pred_A), torch.Tensor(pred_B)




def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0
    arguments["epoch"] = start_epoch - 1

    logger = logging.getLogger("RDE.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "bge_loss": AverageMeter(),
        "tse_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    # evaluator.eval(model.eval())
    # train
    sims = []
    train_diag_state = {"assignments": {}}
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        # model.train()
        model.epoch = epoch
        # data_size = train_loader.dataset.__len__()
        # pred_A, pred_B  =  torch.ones(data_size), torch.ones(data_size)
        if _prototype_requested(args):
            if epoch > getattr(args, "prototype_warmup_epochs", 0) and not _prototype_ready(model):
                maybe_initialize_prototypes(model, train_loader, args, device, logger)
    
        pred_A, pred_B = get_loss(model, train_loader)
    
        consensus_division = pred_A + pred_B # 0,1,2 
        consensus_division[consensus_division==1] += torch.randint(0, 2, size=(((consensus_division==1)+0).sum(),))
        label_hat = consensus_division.clone()
        label_hat[consensus_division>1] = 1
        label_hat[consensus_division<=1] = 0 
        
        model.train() 
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']
            
            batch['label_hat'] = label_hat[index.cpu()]
 
            ret = model(batch)
            loss_components = _loss_components(ret)
            total_loss = sum(loss_components.values())

            batch_size = batch['images'].shape[0]
            _update_meter(meters, 'loss', total_loss, batch_size)
            for loss_key, loss_value in loss_components.items():
                _update_meter(meters, loss_key, loss_value, batch_size)
            if (n_iter + 1) % log_period == 0:
                grad_norms = _grad_norm_by_loss(loss_components, model)
                for grad_key, grad_norm in grad_norms.items():
                    _update_meter(meters, grad_key, grad_norm, batch_size)
                train_diag_metrics = compute_train_diagnostics(model, ret, args, train_diag_state)
                for diag_key, diag_value in train_diag_metrics.items():
                    _update_meter(meters, diag_key, diag_value, batch_size)
         
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.count > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
 
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
 
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)
                    
def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("RDE.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
