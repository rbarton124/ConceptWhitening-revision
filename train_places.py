import argparse
import os
import sys
import gc
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from MODELS.model_resnet import *
from plot_functions import *
from PIL import ImageFile, Image

##############################################################################
# GLOBAL CONFIGURABLE CONSTANTS
##############################################################################

# Hard-coded schedule for learning rate decay: LR = LR * (LR_DECAY_FACTOR^(epoch // LR_DECAY_EPOCH))
LR_DECAY_EPOCH = 30         # every 30 epochs
LR_DECAY_FACTOR = 0.1       # multiply LR by 0.1

# Frequency of concept alignment updates
CW_ALIGN_FREQ = 30          # every 30 iterations in train()
BASELINE_ALIGN_FREQ = 20    # every 20 iterations in train_baseline()

# Image transformation sizes
CROP_SIZE = 224
RESIZE_SIZE = 256

# TQDM ProgressBar Settings
TQDM_MINITERS = 10
TQDM_SMOOTHING = 0.02

# If "one-batch-per-concept" should be used when doing concept alignment
ONE_BATCH_PER_CONCEPT = True

# Pin memory usage in DataLoaders for performance improvement if enough system memory is available
PIN_MEMORY = True

RANDOM_SUBSET_SIZE = 300
TOP_K_IMAGES = 4

##############################################################################
# END GLOBAL CONSTANTS
##############################################################################

ImageFile.LOAD_TRUNCATED_IMAGES = True

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description='PyTorch Training Script (Places / CUB / Etc.)')

############################
# ARGUMENT DEFINITIONS
############################

parser.add_argument('--main_data', required=True,
                    help='Path to main dataset containing train/val/test subdirs.')
parser.add_argument('--concept_data', required=True,
                    help='Path to concept dataset containing concept_train/concept_test subdirs.')
parser.add_argument('--arch', '-a', default='resnet_cw',
                    help='Model architecture identifier (e.g. resnet_cw, resnet_original, densenet_cw, etc.)')
parser.add_argument('--whitened_layers', default='5',
                    help='Comma-separated indices of BN layers to replace with CW layers (e.g. "5" or "2,5,8").')
parser.add_argument('--act_mode', default='pool_max',
                    help='Activation mode for CW layers ("mean","max","pos_mean","pool_max").')
parser.add_argument('--depth', default=18, type=int,
                    help='Depth of the ResNet (18 or 50). Also used for some densenet arches.')
parser.add_argument('--ngpu', default=1, type=int,
                    help='Number of GPUs for DataParallel.')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='Number of data loading workers.')
parser.add_argument('--epochs', default=100, type=int,
                    help='Total epochs to run for training.')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='Override start epoch if resuming (useful for manual restarts).')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='Mini-batch size.')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='Initial learning rate.')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='SGD momentum.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='Weight decay factor (L2 regularization).')
parser.add_argument('--concepts', type=str, required=True,
                    help='Comma-separated list of concept folder names.')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='Frequency for printing mini-batch logs.')
parser.add_argument('--resume', default='', type=str,
                    help='Path to checkpoint to resume from.')
parser.add_argument('--only-load-weights', action='store_true',
                    help='Only load model weights from checkpoint (ignore epoch/optimizer state).')
parser.add_argument("--seed", type=int, default=1234,
                    help='Random seed for reproducibility.')
parser.add_argument("--prefix", type=str, required=True,
                    help='Prefix for logging & checkpoint saving. Often appended with whitened_layers.')
parser.add_argument('--evaluate', type=str, default=None,
                    help='Evaluation mode (e.g. "plot_top50" or "plot_auc"). If set, no training occurs.')

best_prec1 = 0
writer = None

def main():
    """
    1) Parse arguments and set seeds.
    2) Initialize TensorBoard writer.
    3) Build model and load checkpoint if requested.
    4) Construct DataLoaders for main and concept data.
    5) Run either training loop or evaluation/plotting logic.
    """
    global args, best_prec1, writer
    args = parser.parse_args()

    print("=============== ARGUMENTS ===============")
    print("Args:", args)
    print("=========================================")

    # Optional check for concept folders before running
    for c_name in args.concepts.split(','):
        concept_path = os.path.join(args.concept_data, 'concept_train', c_name)
        if not os.path.isdir(concept_path):
            print(f"[Warning] Concept folder '{c_name}' not found at: {concept_path}. Please double-check.")

    # Set manual seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # Incorporate the whitened_layers into prefix for clarity
    args.prefix += '_' + '_'.join(args.whitened_layers.split(','))

    # Initialize TensorBoard logging
    current_time = str(int(time.time()))
    writer = SummaryWriter(log_dir=os.path.join('runs', f"{args.prefix}_{current_time}"))

    # Build the model (does not load checkpoint here, just constructs the architecture)
    model = build_model(args)

    # Create an SGD optimizer for the model
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Possibly resume from checkpoint
    epoch, best_prec1_ckpt = 0, 0
    if args.resume and os.path.isfile(args.resume):
        # Load checkpoint: model weights and (optionally) optimizer, epoch, best_prec
        epoch, best_prec1_ckpt = load_checkpoint(
            model, optimizer, args.resume,
            only_weights=args.only_load_weights
        )
        # If user also provided --start-epoch, override the loaded epoch
        if args.start_epoch > 0:
            print(f"[Info] Overriding checkpoint epoch {epoch} with --start-epoch {args.start_epoch}")
            epoch = args.start_epoch
    else:
        # Start from scratch
        epoch = args.start_epoch

    # Record the best accuracy from checkpoint (if any)
    global best_prec1
    best_prec1 = max(0, best_prec1_ckpt)

    # Wrap model in DataParallel for multi-GPU usage, then move to GPU
    model = nn.DataParallel(model, device_ids=list(range(args.ngpu))).cuda()

    # Summarize model size
    total_params = sum(p.data.nelement() for p in model.parameters())
    print(f"[Model Setup] Architecture: {args.arch}")
    print(f"[Model Setup] Number of parameters: {total_params}")
    cudnn.benchmark = True  # enable cudnn autotuner

    # Build DataLoaders for main training, concept sets, val, test
    (train_loader,
     concept_loaders,
     val_loader,
     test_loader,
     test_loader_with_path) = setup_dataloaders(args)

    # Basic info about dataset sizes
    print(f"[Data] Train dataset size: {len(train_loader.dataset)}")
    c_names = args.concepts.split(',')
    for i, c_loader in enumerate(concept_loaders):
        print(f"[Data] Concept '{c_names[i]}' size: {len(c_loader.dataset)}")
    print(f"[Data] Validation dataset size: {len(val_loader.dataset)}")
    print(f"[Data] Test dataset size: {len(test_loader.dataset)}")

    # If --evaluate is set, skip training
    if args.evaluate is not None:
        print(f"[Evaluate] Running evaluation mode: {args.evaluate}")
        plot_figures(args, model, test_loader_with_path,
                     train_loader, concept_loaders,
                     os.path.join(args.concept_data, 'concept_test'))
        writer.close()
        return

    # ======================
    # MAIN TRAINING LOOP
    # ======================
    print(f"[Training] Starting from epoch {epoch} for {args.epochs} total epochs.")
    for current_epoch in range(epoch, epoch + args.epochs):
        # Adjust learning rate based on epoch schedule
        lr_now = adjust_learning_rate(optimizer, current_epoch, args)
        writer.add_scalar('LR', lr_now, current_epoch)

        # Branch to appropriate training function based on arch
        # For resnet_cw, use train with concept alignment
        # For resnet_baseline, use train_baseline
        # Otherwise, normal train with no concept alignment
        if args.arch == 'resnet_cw':
            train(train_loader, concept_loaders, model, nn.CrossEntropyLoss().cuda(), optimizer, current_epoch)
        elif args.arch == 'resnet_baseline':
            train_baseline(train_loader, concept_loaders, model, nn.CrossEntropyLoss().cuda(), optimizer, current_epoch)
        else:
            # e.g. for resnet_original or densenet_original, no concept alignment
            train(train_loader, [], model, nn.CrossEntropyLoss().cuda(), optimizer, current_epoch)

        # Evaluate on validation set
        val_top1 = validate(val_loader, model, nn.CrossEntropyLoss().cuda(), current_epoch)
        writer.add_scalar('Val/Top1_Accuracy', val_top1, current_epoch)

        # Check if this is the best so far
        is_best = (val_top1 > best_prec1)
        best_prec1 = max(val_top1, best_prec1)

        # Save checkpoint
        save_checkpoint({
            'epoch': current_epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),  # DataParallel unwrap
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.prefix)

    # Final test pass
    print(f"[Training] Completed. Best top-1 accuracy was {best_prec1:.2f}")
    final_test_acc = validate(test_loader, model, nn.CrossEntropyLoss().cuda(), args.epochs)
    print(f"[Testing] Final test accuracy: {final_test_acc:.2f}")

    # Close the TensorBoard logger
    writer.close()


def build_model(args):
    """
    Construct an unwrapped (no DataParallel) model instance
    based on user-specified architecture and depth.
    Checkpoint loading is handled separately in load_checkpoint().

    :param args: Parsed command-line arguments
    :return: A torch.nn.Module (ResNet, DenseNet, or VGG) with or without CW
    """
    arch = args.arch.lower()
    depth = args.depth
    wh_layers = [int(x) for x in args.whitened_layers.split(',')]

    if arch == 'resnet_cw':
        print("[build_model] Using build_resnet_cw()")
        return build_resnet_cw(num_classes=365, depth=depth,
                               whitened_layers=wh_layers,
                               act_mode=args.act_mode)
    elif arch == 'resnet_original' or arch == 'resnet_baseline':
        print("[build_model] Using build_resnet_bn()")
        return build_resnet_bn(num_classes=365, depth=depth)
    elif arch == 'densenet_cw':
        print("[build_model] Using build_densenet_cw()")
        return build_densenet_cw(num_classes=365, depth='161',
                                 whitened_layers=wh_layers,
                                 act_mode=args.act_mode)
    elif arch == 'densenet_original':
        print("[build_model] Using build_densenet_bn()")
        return build_densenet_bn(num_classes=365, depth='161')
    elif arch == 'vgg16_cw':
        print("[build_model] Using build_vgg_cw()")
        return build_vgg_cw(num_classes=365, whitened_layers=wh_layers,
                            act_mode=args.act_mode)
    elif arch == 'vgg16_bn_original':
        print("[build_model] Using build_vgg_bn()")
        return build_vgg_bn(num_classes=365)
    else:
        print(f"[Warning] Unrecognized arch '{arch}', defaulting to build_resnet_bn(18).")
        return build_resnet_bn(num_classes=365, depth=18)


def setup_dataloaders(args):
    """
    Build DataLoaders for:
     - Main train dataset (args.main_data/train)
     - Concept datasets (args.concept_data/concept_train/[concept folders])
     - Validation dataset (args.main_data/val)
     - Test dataset (args.main_data/test)
     - Test dataset with path logging

    :param args: Argparse namespace with data paths, batch size, etc.
    :return: Tuple of (train_loader, concept_loaders, val_loader, test_loader, test_loader_with_path)
    """
    print("[setup_dataloaders] Creating transforms & DataLoaders...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    traindir = os.path.join(args.main_data, 'train')
    valdir = os.path.join(args.main_data, 'val')
    testdir = os.path.join(args.main_data, 'test')
    conceptdir_train = os.path.join(args.concept_data, 'concept_train')

    # Main train dataset
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    # Concept datasets (one DataLoader per concept)
    concept_loaders = []
    for concept in args.concepts.split(','):
        c_dataset = datasets.ImageFolder(
            os.path.join(conceptdir_train, concept),
            transforms.Compose([
                transforms.RandomResizedCrop(CROP_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        c_loader = torch.utils.data.DataLoader(
            c_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=PIN_MEMORY
        )
        concept_loaders.append(c_loader)

    # Validation dataset
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    # Test dataset
    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    # Test dataset with path logging
    test_dataset_with_path = ImageFolderWithPaths(
        testdir,
        transforms.Compose([
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    )
    test_loader_with_path = torch.utils.data.DataLoader(
        test_dataset_with_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    return train_loader, concept_loaders, val_loader, test_loader, test_loader_with_path


def get_cw_layer(resnet_model, target_layer):
    """
    A helper to locate the BN/CW layer for hooking,
    based on an integer 'target_layer'.
    This method simply enumerates layer1..layer4 blocks in sequence.

    If your layer indexing differs (e.g., you want bn2 in each block),
    you should adapt this logic.
    """
    blocks = [resnet_model.layer1, resnet_model.layer2,
              resnet_model.layer3, resnet_model.layer4]
    counts = [len(b) for b in blocks]

    # Flatten indexing: if target_layer <= counts[0], it's in layer1, else subtract counts[0] and move on
    layer_index = target_layer
    for l in range(4):
        if layer_index <= counts[l]:
            block = blocks[l][layer_index - 1]
            return block.bn1
        else:
            layer_index -= counts[l]

    # Fallback: if something goes wrong, return the last BN
    print(f"[Warning] get_cw_layer fallback triggered, returning last BN in layer4.")
    return blocks[-1][-1].bn1


def train(train_loader, concept_loaders, model, criterion, optimizer, epoch):
    """
    Standard training function for one epoch, with optional concept alignment
    for resnet_cw. Uses TQDM progress bar and logs metrics to TensorBoard.

    :param train_loader: DataLoader for the main training dataset
    :param concept_loaders: list of DataLoaders for each concept (could be empty)
    :param model: DataParallel model
    :param criterion: e.g. CrossEntropyLoss
    :param optimizer: e.g. SGD
    :param epoch: Current epoch index
    """
    global writer
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    cw_align_meter = AverageMeter()

    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}] (Train)", smoothing=TQDM_SMOOTHING, miniters=TQDM_MINITERS)

    for i, (input, target) in pbar:
        iteration = epoch * len(train_loader) + i

        # Periodically run concept alignment if arch is resnet_cw
        if args.arch == "resnet_cw" and concept_loaders:
            if (i + 1) % CW_ALIGN_FREQ == 0:
                _run_concept_alignment(
                    model, concept_loaders, cw_align_meter, iteration, args
                )
        
                _log_main_data_topk(
                        model=model,
                        dataset=train_loader.dataset,
                        subset_size=RANDOM_SUBSET_SIZE,
                        top_k=TOP_K_IMAGES,
                        iteration=iteration,
                        whitened_layer=int(args.whitened_layers),
                        concept_count=len(concept_loaders),
                        writer=writer
                    )

        # MAIN CLASSIFICATION
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1_acc.update(prec1[0].item(), input.size(0))
        top5_acc.update(prec5[0].item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # Derived metrics
        misclassification_rate = 100.0 - top1_acc.val

        # TensorBoard logging
        writer.add_scalar('Train/CrossEntropyLoss', losses.val, iteration)
        writer.add_scalar('Train/Top-1_Accuracy(%)', top1_acc.val, iteration)
        writer.add_scalar('Train/Misclassification(%)', misclassification_rate, iteration)

        # Update TQDM bar
        pbar.set_postfix({
            "CE Loss": f"{losses.val:.3f}",
            "Top-1 Acc (%)": f"{top1_acc.val:.2f}",
            "Top-5 Acc (%)": f"{top5_acc.val:.2f}",
            "Concept Align (%)": f"{cw_align_meter.val:.2f}"
        })


def _run_concept_alignment(model, concept_loaders, cw_align_meter, iteration, args):
    """
    A helper to handle concept alignment for resnet_cw architectures.
    1) Switch model to eval mode (no gradient needed).
    2) Hook the final BN/CW layer to retrieve activations.
    3) For each concept, feed one batch (if available).
    4) Update rotation matrix + revert model to training mode.
    5) Compute alignment accuracy + log to TensorBoard.
    """
    global writer

    model.eval()
    with torch.no_grad():
        concept_alignment_correct = 0
        concept_alignment_total = 0

        concept_images_list = []
        concept_activations_list = []
        labels_list = []
        cw_outputs = []

        def cw_hook(module, in_t, out_t):
            cw_outputs.append(out_t)

        # Identify BN/CW layer to hook
        last_bn = get_cw_layer(model.module.model, int(args.whitened_layers))
        hook_handle = last_bn.register_forward_hook(cw_hook)

        # Feed each concept once
        for concept_idx, loader_cpt in enumerate(concept_loaders):
            model.module.change_mode(concept_idx)
            for j, (Xc, _) in enumerate(loader_cpt):
                Xc_var = torch.autograd.Variable(Xc).cuda()
                model(Xc_var)

                # Only handle first batch if ONE_BATCH_PER_CONCEPT is True
                if ONE_BATCH_PER_CONCEPT:
                    pass  # We'll just break after hooking once

                if len(cw_outputs) == 0:
                    break

                rep = cw_outputs[0]  # [B, C, H, W]
                B, C, H, W = rep.shape
                rep_avg = rep.mean(dim=(2,3))  # [B, C]
                predicted_axis = rep_avg.argmax(dim=1)  # [B]
                correct_mask = (predicted_axis == concept_idx)
                concept_alignment_correct += correct_mask.sum().item()
                concept_alignment_total += B

                concept_images_list.append(Xc.cpu())
                concept_activations_list.append(rep_avg[:, concept_idx].cpu())
                labels_list.extend([concept_idx]*B)

                cw_outputs.clear()
                if ONE_BATCH_PER_CONCEPT:
                    break

        # Remove hook & update rotation
        hook_handle.remove()
        model.module.update_rotation_matrix()
        model.module.change_mode(-1)

    # measure alignment & log
    if concept_alignment_total > 0:
        concept_acc = 100.0 * concept_alignment_correct / concept_alignment_total
        cw_align_meter.update(concept_acc, 1)  # accumulate in the meter

        writer.add_scalar('CW/ConceptAlignment(%)', concept_acc, iteration)

    model.train()  # revert to train mode


def validate(val_loader, model, criterion, epoch):
    """
    Validation pass on the val set with a TQDM progress bar. Returns top-1 accuracy.

    :param val_loader: DataLoader for validation
    :param model: DataParallel model (eval mode inside)
    :param criterion: e.g. CrossEntropyLoss
    :param epoch: current epoch index
    :return: average top-1 accuracy (float)
    """
    global writer
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()

    end = time.time()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch [{epoch+1}] (Val)")

    with torch.no_grad():
        for i, (input, target) in pbar:
            iteration = epoch * len(val_loader) + i
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1_acc.update(prec1[0].item(), input.size(0))
            top5_acc.update(prec5[0].item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            misclassification_rate = 100.0 - top1_acc.val
            pbar.set_postfix({
                "CE Loss": f"{losses.val:.3f}",
                "Top-1 Acc (%)": f"{top1_acc.val:.2f}",
                "Misclass (%)": f"{misclassification_rate:.2f}"
            })

    print(f"[Validation] Epoch {epoch+1}: "
          f"Top-1 Acc = {top1_acc.avg:.2f}%, "
          f"Top-5 Acc = {top5_acc.avg:.2f}%, "
          f"Loss = {losses.avg:.4f}")

    return top1_acc.avg


def train_baseline(train_loader, concept_loaders, model, criterion, optimizer, epoch, activation_mode='pool_max'):
    """
    Baseline training function that still incorporates a concept alignment step,
    but using a different approach than resnet_cw. Also logs to TensorBoard.

    :param train_loader: main dataset loader
    :param concept_loaders: concept dataset loaders
    :param model: DataParallel model
    :param criterion: e.g. CrossEntropyLoss
    :param optimizer: e.g. SGD
    :param epoch: current epoch index
    :param activation_mode: how to compute concept activation ('mean','max','pos_mean','pool_max')
    """
    global writer
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()

    loss_aux = AverageMeter()
    concept_acc = AverageMeter()

    n_cpt = len(concept_loaders)
    inter_feature = []

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    def hookf(module, input, output):
        # Capture the first 'n_cpt' feature maps for concept alignment
        inter_feature.append(output[:, :n_cpt, :, :])

    end = time.time()
    concept_names = args.concepts.split(',')

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch [{epoch+1}] (TrainBaseline)",
                smoothing=TQDM_SMOOTHING, miniters=TQDM_MINITERS)

    for i, (input, target) in pbar:
        iteration = epoch * len(train_loader) + i
        data_time.update(time.time() - end)

        # Concept alignment step every BASELINE_ALIGN_FREQ batches
        if (i + 1) % BASELINE_ALIGN_FREQ == 0 and n_cpt > 0:
            layer = int(args.whitened_layers)
            layers = model.module.layers

            # Register hook in the corresponding BN
            if layer <= layers[0]:
                hook = model.module.model.layer1[layer - 1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1]:
                hook = model.module.model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1] + layers[2]:
                hook = model.module.model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hookf)
            else:
                hook = model.module.model.layer4[layer - (layers[0]+layers[1]+layers[2]) - 1].bn1.register_forward_hook(hookf)

            inter_feature.clear()
            y = []
            concept_batch_images = []

            # Take one batch from each concept
            for concept_index, c_loader in enumerate(concept_loaders):
                for j, (X, _) in enumerate(c_loader):
                    y += [concept_index] * X.size(0)
                    concept_batch_images.append(X)
                    X_var = torch.autograd.Variable(X).cuda()
                    model(X_var)
                    break  # one batch per concept

            # Combine all captured activations
            inter_feat = torch.cat(inter_feature, 0)
            y_var = torch.Tensor(y).long().cuda()

            # Pooling approach
            if activation_mode == 'mean':
                y_pred = F.avg_pool2d(inter_feat, inter_feat.size()[2:]).squeeze()
            elif activation_mode == 'max':
                y_pred = F.max_pool2d(inter_feat, inter_feat.size()[2:]).squeeze()
            elif activation_mode == 'pos_mean':
                y_pred = F.avg_pool2d(F.relu(inter_feat), inter_feat.size()[2:]).squeeze()
            elif activation_mode == 'pool_max':
                kernel_size = 3
                y_pred = F.max_pool2d(inter_feat, kernel_size)
                y_pred = F.avg_pool2d(y_pred, y_pred.size()[2:]).squeeze()

            cpt_loss = 10 * criterion(y_pred, y_var)
            [cpt_acc_percent] = accuracy(y_pred.data, y_var, topk=(1,))
            loss_aux.update(cpt_loss.item(), inter_feat.size(0))
            concept_acc.update(cpt_acc_percent[0].item(), inter_feat.size(0))

            optimizer.zero_grad()
            cpt_loss.backward()
            optimizer.step()

            hook.remove()

            # Log top-4 images in TensorBoard
            all_images = torch.cat(concept_batch_images, dim=0)
            for c_idx in range(n_cpt):
                concept_scores = y_pred[:, c_idx]
                top_vals, top_inds = torch.sort(concept_scores, descending=True)
                top_inds = top_inds[:4]
                best_imgs = all_images[top_inds]

                unnorm_list = []
                for bimg in best_imgs:
                    unnorm_list.append(inv_normalize(bimg).unsqueeze(0))
                unnorm_tensor = torch.cat(unnorm_list, dim=0)

                concept_name = concept_names[c_idx]
                writer.add_images(f"ConceptActivations/{concept_name}", unnorm_tensor, iteration)

        # STANDARD CLASSIFICATION
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1_acc.update(prec1[0].item(), input.size(0))
        top5_acc.update(prec5[0].item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        iteration = epoch * len(train_loader) + i
        misclass_rate = 100.0 - top1_acc.val

        # Log to TensorBoard
        writer.add_scalar('TrainBaseline/CE_Loss', losses.val, iteration)
        writer.add_scalar('TrainBaseline/Concept_Loss', loss_aux.val, iteration)
        writer.add_scalar('TrainBaseline/Top-1_Acc(%)', top1_acc.val, iteration)
        writer.add_scalar('TrainBaseline/Concept_Acc(%)', concept_acc.val, iteration)
        writer.add_scalar('TrainBaseline/Misclassification(%)', misclass_rate, iteration)

        pbar.set_postfix({
            "CE Loss": f"{losses.val:.3f}",
            "Concept Loss": f"{loss_aux.val:.3f}",
            "Top-1 Acc(%)": f"{top1_acc.val:.2f}",
            "Concept Acc(%)": f"{concept_acc.val:.2f}"
        })


def plot_figures(args, model, test_loader_with_path, train_loader, concept_loaders, conceptdir):
    """
    Plotting/evaluation routines. 
    This function is invoked if --evaluate is set.
    Currently supports 'plot_top50' or 'plot_auc' as examples.
    """
    concept_name = args.concepts.split(',')
    out_dir = './plot/' + '_'.join(concept_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if args.evaluate == 'plot_top50':
        print("[plot_figures] Plotting top50 activated images for each concept axis.")
        # The function load_resnet_model is presumably in plot_functions
        model = load_resnet_model(args, arch='resnet_cw', depth=18, whitened_layer='8')
        plot_concept_top50(args, test_loader_with_path, model, '8', activation_mode=args.act_mode)
    elif args.evaluate == 'plot_auc':
        print("[plot_figures] Plot AUC-based concept purity metrics.")
        print("[plot_figures] Note: requires multiple models with different whitened layers for best results.")
        aucs_cw = plot_auc_cw(args, conceptdir, '1,2,3,4,5,6,7,8',
                              plot_cpt=concept_name, activation_mode=args.act_mode)
        print("[plot_figures] Running AUCs using SVM classifier.")
        model = load_resnet_model(args, arch='resnet_original', depth=18)
        aucs_svm = plot_auc_lm(args, model, concept_loaders, train_loader, conceptdir,
                               '1,2,3,4,5,6,7,8', plot_cpt=concept_name, model_type='svm')
        print("[plot_figures] Running AUCs using logistic regression.")
        model = load_resnet_model(args, arch='resnet_original', depth=18)
        aucs_lr = plot_auc_lm(args, model, concept_loaders, train_loader, conceptdir,
                              '1,2,3,4,5,6,7,8', plot_cpt=concept_name, model_type='lr')
        print("[plot_figures] Running best filter approach.")
        model = load_resnet_model(args, arch='resnet_original', depth=18)
        aucs_filter = plot_auc_filter(args, model, conceptdir,
                                      '1,2,3,4,5,6,7,8', plot_cpt=concept_name)
        print("[plot_figures] Finished AUC plotting.")


def save_checkpoint(state, is_best, prefix, checkpoint_folder='./checkpoints'):
    """
    Save checkpoint to disk. If architecture is CW, store in subfolder named after concepts.
    Otherwise store in checkpoints/ prefix. Also copy file to *model_best.pth.tar if is_best=True.
    """
    if args.arch in ["resnet_cw", "densenet_cw", "vgg16_cw"]:
        concept_name = '_'.join(args.concepts.split(','))
        cpt_dir = os.path.join(checkpoint_folder, concept_name)
        if not os.path.exists(cpt_dir):
            os.mkdir(cpt_dir)
        filename = os.path.join(cpt_dir, f'{prefix}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            best_fn = os.path.join(cpt_dir, f'{prefix}_model_best.pth.tar')
            shutil.copyfile(filename, best_fn)
        print(f"[Checkpoint] Saved to {filename}. Best? {is_best}")
    elif args.arch in ["resnet_original", "densenet_original", "vgg16_original"]:
        filename = os.path.join(checkpoint_folder, f'{prefix}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            best_fn = os.path.join(checkpoint_folder, f'{prefix}_model_best.pth.tar')
            shutil.copyfile(filename, best_fn)
        print(f"[Checkpoint] Saved to {filename}. Best? {is_best}")


def load_checkpoint(model, optimizer, checkpoint_path, only_weights=False):
    """
    Load a checkpoint from 'checkpoint_path' into 'model' and (optionally) 'optimizer'.
    Returns (loaded_epoch, loaded_best_prec).

    If only_weights=True, we only load the model's param weights
    (ignoring 'epoch', 'best_prec1', and 'optimizer' states).

    1) Parse checkpoint dict for 'state_dict' or assume raw.
    2) Convert "module.*" -> "model.*" if needed.
    3) Filter out any shape mismatches (common if BN->CW changed shapes).
    4) Load with strict=False into model.
    5) Optionally load epoch/best_prec/optimizer state if not only_weights.
    """
    print(f"[load_checkpoint] Loading checkpoint from {checkpoint_path}, only_weights={only_weights}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    raw_sd = checkpoint.get('state_dict', checkpoint)
    new_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("module."):
            k_core = k[len("module."):]
        else:
            k_core = k
        # This assumes your model param keys have "model.xxxx"
        final_key = f"model.{k_core}"
        new_sd[final_key] = v

    # Filter out shape mismatches
    filtered_sd = {}
    model_sd = model.state_dict()
    for ckpt_key, ckpt_val in new_sd.items():
        if ckpt_key in model_sd:
            model_val = model_sd[ckpt_key]
            if ckpt_val.shape == model_val.shape:
                filtered_sd[ckpt_key] = ckpt_val
            else:
                print(f"[Info] Skipping param '{ckpt_key}' due to shape mismatch: {ckpt_val.shape} vs {model_val.shape}")
        else:
            print(f"[Info] Dropping param '{ckpt_key}' not found in model.")

    model.load_state_dict(filtered_sd, strict=False)

    loaded_epoch, loaded_best_prec = 0, 0
    if not only_weights:
        if 'epoch' in checkpoint:
            loaded_epoch = checkpoint['epoch']
        if 'best_prec1' in checkpoint:
            loaded_best_prec = checkpoint['best_prec1']
        if 'optimizer' in checkpoint and hasattr(optimizer, 'load_state_dict'):
            opt_sd = checkpoint['optimizer']
            try:
                optimizer.load_state_dict(opt_sd)
            except Exception as e:
                print(f"[Warning] Could not load optimizer state due to: {e}")

    print(f"[load_checkpoint] Done. (Epoch={loaded_epoch}, BestPrec={loaded_best_prec:.2f})")
    return loaded_epoch, loaded_best_prec


class AverageMeter(object):
    """
    Computes and stores both the average and current value for a metric.
    Useful for tracking stats like loss or accuracy across mini-batches.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """
    Decay the learning rate by LR_DECAY_FACTOR every LR_DECAY_EPOCH epochs.
    Example: lr = initial_lr * (0.1 ^ (epoch // 30)) by default.
    """
    # global LR_DECAY_EPOCH, LR_DECAY_FACTOR
    new_lr = args.lr * (LR_DECAY_FACTOR ** (epoch // LR_DECAY_EPOCH))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy for the specified values of k.
    Returns a list of tensor scalars (floats).
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # Flatten the first k rows
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def _log_main_data_topk(model, dataset, subset_size, top_k, iteration, whitened_layer, concept_count, writer):
    """
    Sample a random subset of images from the main dataset,
    run them through the CW layer to get concept-axis activations,
    pick top-K for each concept axis, and log to TensorBoard.

    :param model: DataParallel model (eval mode temporarily)
    :param dataset: The main training dataset (an ImageFolder or similar).
    :param subset_size: How many random images to sample from the dataset.
    :param top_k: How many top-activating images to log per concept axis.
    :param iteration: Current global iteration (for TB logging).
    :param whitened_layer: Which BN/CW layer index to hook.
    :param concept_count: Number of concept axes (i.e., len(concept_loaders)).
    :param writer: TensorBoard SummaryWriter.
    """
    try:
        if concept_count == 0 or subset_size == 0:
            return

        # Ensure subset_tensor is preloaded
        if not hasattr(dataset, 'subset_tensor'):
            fixed_indices = random.sample(range(len(dataset)), subset_size)
            subset_images = [dataset[i][0] for i in fixed_indices]
            dataset.subset_tensor = torch.stack(subset_images).cuda()

        batch_tensor = dataset.subset_tensor  # [subset_size, C, H, W]

        # Process in chunks to avoid memory overload
        chunk_size = 100
        rep_avg_list = []

        model.eval()
        cw_outputs = []

        def cw_hook(module, in_t, out_t):
            cw_outputs.append(out_t)

        last_bn = get_cw_layer(model.module.model, whitened_layer)
        hook_handle = last_bn.register_forward_hook(cw_hook)

        with torch.no_grad():
            for i in range(0, subset_size, chunk_size):
                chunk = batch_tensor[i:i + chunk_size]
                model(chunk)
                if not cw_outputs:
                    continue
                rep = cw_outputs[0]  # [chunk_size, C, H, W]
                rep_avg = rep.mean(dim=(2, 3))  # [chunk_size, C]
                rep_avg_list.append(rep_avg)
                cw_outputs.clear()

            rep_avg = torch.cat(rep_avg_list, dim=0)  # [subset_size, C]

            # Log top-k images for each concept axis
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            resize_transform = transforms.Resize((128, 128))

            for c_idx in range(concept_count):
                axis_activations = rep_avg[:, c_idx]
                top_vals, top_inds = torch.topk(axis_activations, k=top_k, dim=0)
                best_images = batch_tensor[top_inds]  # [top_k, C, H, W]

                # Resize and normalize
                resized_images = [resize_transform(img.cpu()) for img in best_images]
                unnorm_list = [inv_normalize(img).unsqueeze(0) for img in resized_images]
                unnorm_tensor = torch.cat(unnorm_list, dim=0)  # [top_k, C, 128, 128]

                # Log to TensorBoard
                concept_tag = f"MainData_ConceptAxis/Concept_{c_idx}"
                writer.add_images(concept_tag, unnorm_tensor, iteration)

        # Cleanup
        hook_handle.remove()
        model.train()
    except Exception as e:
        print(f"[Error] _log_main_data_topk failed: {e}")
        if 'hook_handle' in locals():
            hook_handle.remove()
        model.train()


if __name__ == '__main__':
    main()
