import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import os, sys
import logging
import random
import utils as ut
import models
import wandb
import gc
from torch.nn.parallel import DistributedDataParallel as DDP
from prototype_dataset import RealFakeDataset

logger: logging.Logger = ut.logger

    
def train(
        args=None,
        logger=None,
):
    # initialize distributed training
    rank, local_rank, world_size = ut.dist_setup()

    # Set random seed
    torch.manual_seed(args.seed+rank)
    np.random.seed(args.seed+rank)
    random.seed(args.seed+rank)

    # Set device
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size

    #Val or not
    is_val = True

    if rank==0 and args.wandb_token != "":
        ut.init_wandb(args)

    # Load data
    if args.stageone:
        ds = RealFakeDataset()
        val_frac = 0.05
        train_len = int((1 - val_frac) * len(ds))
        val_len = len(ds) - train_len
        train_set, val_set = torch.utils.data.random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
        train_sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if val_len > 0 else None
        trainLoader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.cpus_per_gpu, pin_memory=True)
        valLoader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.cpus_per_gpu, pin_memory=True) if val_len > 0 else None
        trainLoaderLen = len(trainLoader)
        print(f"Length {len(train_set)} (per-process {len(trainLoader)})")
    else:
        trainLoader, valLoader = ut.get_dataloader(args, mode='train')
        args = ut.get_epochs_for_itrs(args, len(trainLoader))
        trainLoaderLen = len(trainLoader)

    # Load model
    try:
        if args.stageone:
            model = models.ClipModel(num_classes=1, freeze_backbone=True, device=local_rank).to(args.local_rank)
        else:
            model = models.GAPLModel(num_classes=1, fe_path=args.fe_path, proto_path=args.prototype_path, freeze_backbone=False, device=local_rank).to(args.local_rank)
    except Exception as e:
        logger.error(f"Error loading model. rank={rank}: {e}")
        sys.exit(1)

    # Set optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp) if args.use_amp else None
    # Set scheduler
    if args.warmup_frac > 0:
        warmup_steps=round(args.warmup_frac*ut.get_total_itrs(args, trainLoaderLen))
    else:
        warmup_steps = round(args.warmup_epochs * trainLoaderLen)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*trainLoaderLen-warmup_steps, eta_min=ut.get_min_lr(args)) # set min_lr = lr if args.no_lr_schedule.

    # Set loss function
    criterion = nn.BCEWithLogitsLoss()

    # Load checkpoint if set
    if args.ckpt_path != '': # note that if this is set, it overrides the `--hf_model_repo` argument.
        if args.only_load_model_weights:
            model, epoch_start = ut.load_only_weights(model, args.ckpt_path, rank)
            epoch_start = 0
            total_itr = 0
        else:
            model, optimizer, scheduler, epoch_start, total_itr = ut.load_checkpoint(model, optimizer, scheduler, scaler, args.ckpt_path, rank)
            epoch_start = epoch_start+1 # Since it saves current epoch for ckpt, not next.
    elif args.hf_model_repo != '':
        model = ut.load_ckpt_from_huggingface(model, args.hf_model_repo, rank)
        epoch_start = 0
        total_itr = 0
    else:
        epoch_start = 0
        total_itr = 0

    # try compiling the model
    if args.compile:
        print("Running compile models")
        model = torch.compile(model, dynamic=True)

    # Set DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) #DDP
    torch.cuda.empty_cache()
    dist.barrier()
    logger.info(f"Model loaded and DDP set. rank={rank}")

    # Train
    local_window_loss=ut.LocalWindow(100)
    for epoch in range(epoch_start, args.epochs):
        gc.collect() # run garbage collection
        avgTrainLoss, total_itr = ut.train_one_epoch(
            args=args,
            epoch=epoch,
            model=model,
            train_loader=trainLoader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            local_window_loss=local_window_loss,
            warmup_steps=warmup_steps,
            rank=rank,
            itr=total_itr,
        )
        if valLoader is not None and is_val:
            valLoss, valAcc, valAP = ut.evaluate_one_epoch(
                args=args,
                epoch=epoch,
                model=model,
                dataloader=valLoader,
                criterion=nn.BCEWithLogitsLoss(),
                rank=rank,
                evalName="Val",
                separate_eval=False,
                add_sigmoid=(not args.dont_add_sigmoid),
            )
            wandb_log_dict = {"epoch": epoch+1, "Loss/Train": avgTrainLoss, "Loss/Val": valLoss, "Acc/Val": valAcc, "AP/Val": valAP}
        else:
            valLoss, valAcc, valAP = -1, -1, -1
            wandb_log_dict = {"epoch": epoch+1, "Loss/Train": avgTrainLoss}
        scheduler.step()
        
        if rank<=0 and args.wandb_token != "":
            # log wandb
            wandb.log(
                wandb_log_dict, commit=False
            )
            wandb.finish()
        gc.collect() # run garbage collection

def main():
    args = ut.parse_args()

    args.random_port_offset = np.random.randint(-1000,1000) # randomize to avoid port conflict in same device
    
    if args.debug_port > 0:
        import debugpy
        debugpy.listen(('localhost', args.debug_port))
        logger.info(f"Waiting for debugger to attach on port {args.debug_port}...")
        debugpy.wait_for_client()
        debugpy.breakpoint()

    if args.gpus_list != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_list
        logger.info(f"Setting CUDA_VISIBLE_DEVICES to {args.gpus_list}.")
        args.gpus = len(args.gpus_list.split(','))

    assert args.gpus <= torch.cuda.device_count(), f'Not enough GPUs! {torch.cuda.device_count()} available, {args.gpus} required.'
    assert args.gpus > 0, f'Number of GPUs must be greater than 0!'
    assert args.cpus_per_gpu > 0, f'Number of CPUs per GPU must be greater than 0!'

    if args.ckpt_save_path == '':
        args.ckpt_save_path = args.save_path

    logger.info(f"Spawning processes on {args.gpus} GPUs.")
    logger.info(f"Verbosity: {args.verbose} (0: None, 1: Every epoch, 2: Every iteration)")

    logger.info(f"Model save name: {os.path.basename(args.save_path)}")

    train(
        args=args,
        logger=logger,
    )

if __name__ == "__main__":
    main()
    # down()