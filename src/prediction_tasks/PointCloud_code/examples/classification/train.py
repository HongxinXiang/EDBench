import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score


def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()


# def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
#     accs_table = [f'{item:.2f}' for item in accs]
#     header = ['method', 'OA', 'mAcc'] + \
#         cfg.classes + ['best_epoch', 'log_path', 'wandb link']
#     data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
#         str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
#     with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow(header)
#         writer.writerow(data)
#         f.close()


def print_cls_results(avg_auc, result_dict, epoch):
    s = (
        f"[Epoch {epoch}] Avg ROC_AUC: {avg_auc:.6f} | "
        f"ACC: {result_dict['ACC']:.6f} | "
        f"ROC_AUC: {result_dict['ROC_AUC']:.6f} | "
        f"AUPR: {result_dict['AUPR']:.6f} | "
        f"F1: {result_dict['F1']:.6f}"
    )
    logging.info(s)



def main(gpu, cfg, profile=False):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    # if cfg.rank == 0 :
    #     Wandb.launch(cfg, cfg.wandb.use_wandb)
    #     writer = SummaryWriter(log_dir=cfg.run_dir)
    # else:
    writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )

    # num_classes = val_loader.dataset.num_classes if hasattr(
    #     val_loader.dataset, 'num_classes') else None
    # num_points = val_loader.dataset.num_points if hasattr(
    #     val_loader.dataset, 'num_points') else None
    # if num_classes is not None:
    #     assert cfg.num_classes == num_classes
    # logging.info(f"number of classes of the dataset: {num_classes}, "
    #              f"number of points sampled from dataset: {num_points}, "
    #              f"number of points as model input: {cfg.num_points}")

    # cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(
    #     val_loader.dataset, 'classes') else None or np.range(num_classes)
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            val_result_dict, val_avg_auc = validate_fn(model, val_loader, cfg)
            print_cls_results(val_avg_auc, val_result_dict, cfg.start_epoch)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                test_result_dict, test_avg_auc = validate_fn(model, test_loader, cfg)
                print_cls_results(test_avg_auc, test_result_dict, epoch)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                val_result_dict, val_avg_auc = validate_fn(model, val_loader, cfg)
                print_cls_results(val_avg_auc, val_result_dict, epoch)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder_inv':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                # load_checkpoint_inv(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    # ===> start training
    best_val, best_epoch = -np.inf, 0
    model.zero_grad()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_result_dict, val_avg_auc = validate_fn(model, val_loader, cfg)
            is_best = val_avg_auc > best_val
            if is_best:
                best_val = val_avg_auc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
                print_cls_results(val_avg_auc, val_result_dict, epoch)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_loss {train_loss:.6f}, val_avg_auc {val_avg_auc:.6f}, best val oa {best_val:.6f}')

        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_avg_auc, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
    # test the last epoch
    test_result_dict, test_avg_auc = validate(model, test_loader, cfg)
    print_cls_results(test_avg_auc, test_result_dict, best_epoch)
    if writer is not None:
        writer.add_scalar('test_avg_auc', test_avg_auc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_result_dict, test_avg_auc = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_avg_auc', test_avg_auc, epoch)
    print_cls_results(test_avg_auc, test_result_dict, best_epoch)

    if writer is not None:
        writer.close()
    dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        points = data['x']
        target = data['y']

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

        # print("data['x'] dtype:", data['x'].dtype)
        # print("data['y'] dtype:", data['y'].dtype)
        # print("data[pos'] dtype:", data['pos'].dtype)
        # print("target dtype:", target.dtype)
        # print("loss dtype:", loss.dtype)
        # with torch.autograd.set_detect_anomaly(True):
        logits, loss = model.get_logits_loss(data, target) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target)
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

#         logits = logits.detach().cpu().numpy()
#         target = target.detach().cpu().numpy()
        logits = logits.detach()
        target = target.detach()
        result_dict,  Avg_AUC= metric_cls_multitask(logits, target)

        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] Loss {loss_meter.val:.6f} Avg_AUC {Avg_AUC:.6f}")

    return loss_meter.avg


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode

    y_scores, y_true, y_pred, y_prob = [], [], [], []

    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        points = data['x']
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)

        if cfg.distributed:
            pred = torch.cat(dist.all_reduce(pred), dim=0)
            labels = torch.cat(dist.all_reduce(labels), dim=0)

        y_true.append(target)
        y_scores.append(logits)


    # if cfg.distributed:
    #     dist.all_reduce(tp), dist.all_reduce(count)
    
#     y_true = torch.cat(y_true, dim=0).cpu().numpy()
#     y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    
    y_true = torch.cat(y_true, dim=0)
    y_scores = torch.cat(y_scores, dim=0)

    result_dict = metric_cls_multitask(y_scores, y_true)

    return result_dict


def metric_cls_multitask(logits, targets):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu()
    
        
    probs = F.softmax(logits, dim=1).numpy()  # shape [batch_size, 2]
    preds = np.argmax(probs, axis=1)          # shape [batch_size]
    targets = targets.numpy()

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)

    try:
        roc_auc = roc_auc_score(targets, probs[:, 1])  # 正类的概率
    except ValueError:
        roc_auc = 0.0

    try:
        aupr = average_precision_score(targets, probs[:, 1])
    except ValueError:
        aupr = 0.0

    return {
        "ACC": acc,
        "ROC_AUC": roc_auc,
        "AUPR": aupr,
        "F1": f1
    }, float(roc_auc)  # 返回 ROC_AUC 作为主评估
