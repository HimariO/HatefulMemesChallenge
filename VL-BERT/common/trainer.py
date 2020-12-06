import os
import time
from functools import partial
from collections import namedtuple

import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel
from loguru import logger as my_logger

try:
    from apex import amp
    from apex.amp import _amp_state
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    Apex_DDP = DistributedDataParallel
    pass
    #raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'rank',
                            'add_step',
                            'data_in_time',
                            'data_transfer_time',
                            'forward_time',
                            'backward_time',
                            'optimizer_time',
                            'metric_time',
                            'eval_metric',
                            'locals'])


def _multiple_callbacks(callbacks, *args, **kwargs):
    """Sends args and kwargs to any configured callbacks.
    This handles the cases where the 'callbacks' variable
    is ``None``, a single function, or a list.
    """
    if isinstance(callbacks, list):
        for cb in callbacks:
            cb(*args, **kwargs)
        return
    if callbacks:
        callbacks(*args, **kwargs)


def to_cuda(batch):
    batch = list(batch)

    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking=True)
        elif isinstance(batch[i], list):
            for j, o in enumerate(batch[i]):
                if isinstance(batch[i], torch.Tensor):
                    batch[i][j] = o.cuda(non_blocking=True)

    return batch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model) 
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it 
        is assumed that :meth:`model.forward()` should be called on the first 
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        # if isinstance(input, (list, tuple)):
        #     input = input[0]
        # if device is not None:
        #     input = input.to(device)
        batch = to_cuda(input)
        model(*batch)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def get_base_model(model):
    # my_logger.info(f"[get_base_model] {type(model)}")
    if isinstance(model, torch.nn.DataParallel):
        # my_logger.info(f"[get_base_model] 1")
        return model.module
    elif isinstance(model, (DistributedDataParallel, Apex_DDP)):
        # my_logger.info(f"[get_base_model] 2")
        return model.module
    else:
        # my_logger.info(f"[get_base_model] 3")
        return model


def multi_loss(y_pred, y_true, loss_fn_a=None, loss_fn_b=None):
    loss_a = loss_fn_a(y_pred, y_true) * 0.1 # {(b, 1), (b, 1)} -> (b,)
    loss_b = loss_fn_b(y_pred, y_true) * 0.9
    return loss_a + loss_b


def train(net,
          optimizer,
          lr_scheduler,
          train_loader,
          train_sampler,
          metrics,
          begin_epoch,
          end_epoch,
          logger,
          rank=None,
          batch_end_callbacks=None,
          epoch_end_callbacks=None,
          writer=None,
          validation_monitor=None,
          fp16=False,
          clip_grad_norm=-1,
          gradient_accumulate_steps=1,
          debug=False,
          ckpt_path=None,
          swa_scheduler=None,
          swa_start_epoch=None,
          swa_cycle_epoch=None,
          swa_use_scheduler=True,
          roc_star=None,
          roc_star_start_epoch=None,
          roc_interleave=False):

    assert isinstance(gradient_accumulate_steps, int) and gradient_accumulate_steps >= 1
    if swa_scheduler:
        swa_model = AveragedModel(get_base_model(net).cpu())
        net.cuda()
    else:
        swa_model = None

    last_epoch = 0
    for epoch in range(begin_epoch, end_epoch):
        print('PROGRESS: %.2f%%' % (100.0 * epoch / end_epoch))
        last_epoch = epoch

        # set epoch as random seed of sampler while distributed training
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        # reset metrics
        metrics.reset()

        # set net to train mode
        net.train()

        # clear the paramter gradients
        # optimizer.zero_grad()

        # init end time
        end_time = time.time()

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            name, value = validation_monitor.metrics.get()
            val = value[name.index(validation_monitor.host_metric_name)]
            lr_scheduler.step(val, epoch)
        
        # validation_monitor(epoch, net, optimizer, writer)

        # training
        for nbatch, _batch in enumerate(train_loader):
            global_steps = len(train_loader) * epoch + nbatch
            os.environ['global_steps'] = str(global_steps)

            if debug:
                if nbatch >= 50 * gradient_accumulate_steps:
                    break

            # record time
            data_in_time = time.time() - end_time

            # transfer data to GPU
            data_transfer_time = time.time()
            batch = to_cuda(_batch)
            data_transfer_time = time.time() - data_transfer_time

            # forward
            forward_time = time.time()
            add_kwargs = {}
            if roc_star:
                if epoch >= roc_star_start_epoch:
                    if roc_interleave:
                        # if (epoch - roc_star_start_epoch) % 2 == 0:
                        #     add_kwargs['loss_fn'] = roc_star
                        add_kwargs['loss_fn'] = partial(
                            multi_loss,
                            loss_fn_a=torch.nn.functional.binary_cross_entropy_with_logits,
                            loss_fn_b=roc_star,
                        )
                    else:
                        add_kwargs['loss_fn'] = roc_star
            if not get_base_model(net).config.NETWORK.CLASSIFIER_SIGMOID:
                add_kwargs['loss_fn'] = torch.nn.functional.cross_entropy

            outputs, loss = net(*batch, **add_kwargs)
            
            if roc_star:
                if epoch < roc_star_start_epoch:
                    roc_star(outputs['label_logits'], outputs['label'])
            
            loss = loss.mean()
            if gradient_accumulate_steps > 1:
                loss = loss / gradient_accumulate_steps
            forward_time = time.time() - forward_time

            # backward
            backward_time = time.time()
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            backward_time = time.time() - backward_time

            optimizer_time = time.time()
            if (global_steps + 1) % gradient_accumulate_steps == 0:

                # clip gradient
                if clip_grad_norm > 0:
                    if fp16:
                        total_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                                    clip_grad_norm)
                    else:
                        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                                    clip_grad_norm)
                    if writer is not None:
                        writer.add_scalar(tag='grad-para/Total-Norm',
                                          scalar_value=float(total_norm),
                                          global_step=global_steps)

                optimizer.step()
                # clear the parameter gradients
                optimizer.zero_grad()
                
                # step LR scheduler
                if swa_scheduler is None:
                    if lr_scheduler is not None and not isinstance(lr_scheduler,
                                                                torch.optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step()
                else:
                    if epoch < swa_start_epoch or not swa_use_scheduler:
                        lr_scheduler.step()
                    else:
                        swa_scheduler.step()
            optimizer_time = time.time() - optimizer_time

            # update metric
            metric_time = time.time()
            metrics.update(outputs)
            if writer is not None:
                with torch.no_grad():
                    for group_i, param_group in enumerate(optimizer.param_groups):
                        writer.add_scalar(tag='Initial-LR/Group_{}'.format(group_i),
                                          scalar_value=param_group['initial_lr'],
                                          global_step=global_steps)
                        writer.add_scalar(tag='LR/Group_{}'.format(group_i),
                                          scalar_value=param_group['lr'],
                                          global_step=global_steps)
                    writer.add_scalar(tag='Train-Loss',
                                      scalar_value=float(loss.item()),
                                      global_step=global_steps)
                    name, value = metrics.get()
                    for n, v in zip(name, value):
                        writer.add_scalar(tag='Train-' + n,
                                          scalar_value=v,
                                          global_step=global_steps)

            metric_time = time.time() - metric_time

            # execute batch_end_callbacks
            if batch_end_callbacks is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, add_step=True, rank=rank,
                                                 data_in_time=data_in_time, data_transfer_time=data_transfer_time,
                                                 forward_time=forward_time, backward_time=backward_time,
                                                 optimizer_time=optimizer_time, metric_time=metric_time,
                                                 eval_metric=metrics, locals=locals())
                _multiple_callbacks(batch_end_callbacks, batch_end_params)
            
            if nbatch % 20 == 0:
                my_logger.warning(f"Batch [{nbatch}] get lr {get_lr(optimizer)}")

            # update end time
            end_time = time.time()
        
        if ckpt_path:
            os.makedirs(ckpt_path, exist_ok=True)
            model_param = net.state_dict()
            ckpt_name = f"{net.__class__.__name__}.{epoch}.pth"
            ckpt_file = os.path.join(ckpt_path, ckpt_name)
            torch.save(model_param, ckpt_file)

        # excute epoch_end_callbacks
        if validation_monitor is not None:
            validation_monitor(epoch, net, optimizer, writer)
        if epoch_end_callbacks is not None:
            _multiple_callbacks(epoch_end_callbacks, epoch, net, optimizer, writer, validation_monitor=validation_monitor)
        
        if swa_scheduler:
            if epoch + 1 >= swa_start_epoch:
                swa_progress = epoch - swa_start_epoch + 1
                if swa_progress % swa_cycle_epoch == 0:
                    my_logger.info(f'[SWA] update_parameters at end of epoch: {epoch}')
                    swa_model.update_parameters(get_base_model(net).cpu())
                    net.cuda()
        
        # NOTE: epoch end
    if swa_scheduler:
        swa_model = swa_model.cuda()
        my_logger.info('[SWA] update BN')
        update_bn(train_loader, swa_model)
        
        last_epoch += 1
        if validation_monitor is not None:
            validation_monitor(last_epoch, swa_model, optimizer, writer)
        if epoch_end_callbacks is not None:
            _multiple_callbacks(epoch_end_callbacks, last_epoch, swa_model, optimizer, writer, validation_monitor=validation_monitor)
        return swa_model.module
    else:
        return net

