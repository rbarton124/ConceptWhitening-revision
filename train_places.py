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
from tqdm import tqdm  # <--- TQDM for modern progress bars

from MODELS.model_resnet import *
from plot_functions import *
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('--main_data', required=True,
                    help='path to main dataset (train/val/test)')
parser.add_argument('--concept_data', required=True,
                    help='path to concept dataset (concept_train/concept_test)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--whitened_layers', default='8')
parser.add_argument('--act_mode', default='pool_max')
parser.add_argument('--depth', default=18, type=int, metavar='D',
                    help='model depth')
parser.add_argument('--ngpu', default=4, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--concepts', type=str, required=True,
                    help='comma-separated list of concepts')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS',
                    help='random seed for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX',
                    help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', type=str, default=None,
                    help='type of evaluation')
best_prec1 = 0

# We use a global SummaryWriter (unchanged function signatures)
writer = None

def main():
    global args, best_prec1, writer
    args = parser.parse_args()

    print("args", args)

    # Optional checks for concept folders
    for c_name in args.concepts.split(','):
        concept_path = os.path.join(args.concept_data, 'concept_train', c_name)
        if not os.path.isdir(concept_path):
            print(f"Warning: concept folder '{c_name}' not found at {concept_path}.")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    args.prefix += '_' + '_'.join(args.whitened_layers.split(','))

    # Create TensorBoard writer
    current_time = str(int(time.time()))
    writer = SummaryWriter(log_dir=os.path.join('runs', f"{args.prefix}_{current_time}"))

    # =============== Create Model ===============
    model = build_model(args)

    # =============== Define Loss & Optimizer ===============
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()

    print("Created model:")
    print(model)
    total_params = sum(p.data.nelement() for p in model.parameters())
    print(f"Number of model parameters: {total_params}")

    cudnn.benchmark = True

    # =============== Build Dataloaders ===============
    train_loader, concept_loaders, val_loader, test_loader, test_loader_with_path = setup_dataloaders(args)

    # Print dataset info
    print(f"Main train dataset size: {len(train_loader.dataset)} images")
    concept_names = args.concepts.split(',')
    for idx, c_loader in enumerate(concept_loaders):
        print(f"Concept '{concept_names[idx]}' dataset size: {len(c_loader.dataset)} images")
    print(f"Validation dataset size: {len(val_loader.dataset)} images")
    print(f"Test dataset size: {len(test_loader.dataset)} images")

    # =============== Main Flow ===============
    if args.evaluate is None:
        print("Starting training process...")
        best_prec1 = 0

        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            # Adjust and log LR
            current_lr = adjust_learning_rate(optimizer, epoch)
            writer.add_scalar('LR', current_lr, epoch)

            # Train for one epoch
            if args.arch == "resnet_cw":
                train(train_loader, concept_loaders, model, criterion, optimizer, epoch)
            elif args.arch == "resnet_baseline":
                train_baseline(train_loader, concept_loaders, model, criterion, optimizer, epoch)
            else:
                train(train_loader, [], model, criterion, optimizer, epoch)

            # Evaluate on validation set
            val_top1 = validate(val_loader, model, criterion, epoch)
            writer.add_scalar('Val/Top1_Accuracy', val_top1, epoch)

            # Check if best
            is_best = val_top1 > best_prec1
            best_prec1 = max(val_top1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.prefix)

        # Show final results
        print(f"Training complete. Best Top-1 Accuracy: {best_prec1:.2f}%")
        final_test_acc = validate(test_loader, model, criterion, epoch)
        print(f"Final Test Top-1 Accuracy: {final_test_acc:.2f}%")

        writer.close()
    else:
        # Evaluate only
        plot_figures(args, model, test_loader_with_path,
                     train_loader, concept_loaders, 
                     os.path.join(args.concept_data, 'concept_test'))
        writer.close()


def build_model(args):
    """Build the requested model architecture (CW or original) without changing function signatures."""
    # Simplifies main() readability
    if args.arch == "resnet_cw":
        if args.depth == 50:
            return ResidualNetTransfer(365, args, 
                                       [int(x) for x in args.whitened_layers.split(',')],
                                       arch='resnet50', 
                                       layers=[3, 4, 6, 3],
                                       model_file='./checkpoints/resnet50_places365.pth.tar')
        elif args.depth == 18:
            return ResidualNetTransfer(365, args, 
                                       [int(x) for x in args.whitened_layers.split(',')],
                                       arch='resnet18', 
                                       layers=[2, 2, 2, 2],
                                       model_file='./checkpoints/resnet18_places365.pth.tar')
    elif args.arch == "resnet_original" or args.arch == "resnet_baseline":
        if args.depth == 50:
            return ResidualNetBN(365, args, 
                                 arch='resnet50',
                                 layers=[3, 4, 6, 3],
                                 model_file='./checkpoints/resnet50_places365.pth.tar')
        elif args.depth == 18:
            return ResidualNetBN(365, args,
                                 arch='resnet18',
                                 layers=[2, 2, 2, 2],
                                 model_file='./checkpoints/resnet18_places365.pth.tar')
    elif args.arch == "densenet_cw":
        return DenseNetTransfer(365, args,
                                [int(x) for x in args.whitened_layers.split(',')],
                                arch='densenet161',
                                model_file='./checkpoints/densenet161_places365.pth.tar')
    elif args.arch == 'densenet_original':
        return DenseNetBN(365, args, arch='densenet161',
                          model_file='./checkpoints/densenet161_places365.pth.tar')
    elif args.arch == "vgg16_cw":
        return VGGBNTransfer(365, args,
                             [int(x) for x in args.whitened_layers.split(',')],
                             arch='vgg16_bn',
                             model_file='./checkpoints/vgg16_bn_places365_12_model_best.pth.tar')
    elif args.arch == "vgg16_bn_original":
        return VGGBN(365, args,
                     arch='vgg16_bn',
                     model_file='./checkpoints/vgg16_bn_places365_12_model_best.pth.tar')
    else:
        print(f"Unrecognized arch {args.arch}, defaulting to resnet18 baseline.")
        return ResidualNetBN(365, args,
                             arch='resnet18',
                             layers=[2,2,2,2],
                             model_file='./checkpoints/resnet18_places365.pth.tar')


def setup_dataloaders(args):
    """Set up train_loader, concept_loaders, val_loader, test_loader, test_loader_with_path."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    traindir = os.path.join(args.main_data, 'train')
    valdir = os.path.join(args.main_data, 'val')
    testdir = os.path.join(args.main_data, 'test')
    conceptdir_train = os.path.join(args.concept_data, 'concept_train')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False
    )

    concept_loaders = []
    for concept in args.concepts.split(','):
        c_dataset = datasets.ImageFolder(
            os.path.join(conceptdir_train, concept),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        c_loader = torch.utils.data.DataLoader(
            c_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=False
        )
        concept_loaders.append(c_loader)

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False
    )

    test_dataset = datasets.ImageFolder(
        testdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    test_dataset_with_path = ImageFolderWithPaths(
        testdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    test_loader_with_path = torch.utils.data.DataLoader(
        test_dataset_with_path,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False
    )

    return train_loader, concept_loaders, val_loader, test_loader, test_loader_with_path


def train(train_loader, concept_loaders, model, criterion, optimizer, epoch):
    """Train for one epoch using TQDM for progress, with improved metric naming/logging."""
    global writer
    model.train()

    # Meters for timing and metrics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()   # Top-1 accuracy
    top5_acc = AverageMeter()   # Top-5 accuracy

    end = time.time()

    # Wrap in a TQDM progress bar for clarity
    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch [{epoch+1}] (Train)")

    for i, (input, target) in pbar:
        iteration = epoch * len(train_loader) + i

        # If we have concept whitening, update the concept rotation matrix every 30 steps
        if args.arch == "resnet_cw" and (i + 1) % 30 == 0 and len(concept_loaders) > 0:
            model.eval()
            with torch.no_grad():
                for concept_index, loader_cpt in enumerate(concept_loaders):
                    model.module.change_mode(concept_index)
                    for j, (X, _) in enumerate(loader_cpt):
                        X_var = torch.autograd.Variable(X).cuda()
                        model(X_var)
                        break
                model.module.update_rotation_matrix()
                model.module.change_mode(-1)
            model.train()

        data_time.update(time.time() - end)
        
        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Forward & loss
        output = model(input_var)
        loss = criterion(output, target_var)

        # Compute top-1 and top-5 accuracy
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1_acc.update(prec1[0].item(), input.size(0))
        top5_acc.update(prec5[0].item(), input.size(0))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # Log to TensorBoard
        misclassification_rate = 100.0 - top1_acc.val
        writer.add_scalar('Train/CrossEntropyLoss', losses.val, iteration)
        writer.add_scalar('Train/Top-1_Accuracy(%)', top1_acc.val, iteration)
        writer.add_scalar('Train/Misclassification(%)', misclassification_rate, iteration)

        # Update TQDM description with better naming
        pbar.set_postfix({
            "CE Loss": f"{losses.val:.3f}",
            "Top-1 Acc (%)": f"{top1_acc.val:.2f}",
            "Top-5 Acc (%)": f"{top5_acc.val:.2f}",
            "BatchTime": f"{batch_time.val:.3f}s",
            "DataTime": f"{data_time.val:.3f}s"
        })


def validate(val_loader, model, criterion, epoch):
    """Validate model using TQDM for neat logs, returning top-1 accuracy."""
    global writer
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()

    end = time.time()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                desc=f"Epoch [{epoch+1}] (Val)")

    with torch.no_grad():
        for i, (input, target) in pbar:
            iteration = epoch * len(val_loader) + i
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            # compute accuracy
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
          f"Top-1 Acc = {top1_acc.avg:.2f}%, Top-5 Acc = {top5_acc.avg:.2f}%, "
          f"Loss = {losses.avg:.4f}")
    return top1_acc.avg


def train_baseline(train_loader, concept_loaders, model, criterion, optimizer, epoch, activation_mode='pool_max'):
    """Train baseline with concept alignment, using TQDM and clearer logging of concept metrics."""
    global writer
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()

    # Additional concept loss + concept accuracy
    loss_aux = AverageMeter()
    concept_acc = AverageMeter()  # This is "Prec_cpt@1" in original

    n_cpt = len(concept_loaders)
    inter_feature = []
    def hookf(module, input, output):
        inter_feature.append(output[:, :n_cpt, :, :])

    end = time.time()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch [{epoch+1}] (TrainBaseline)")

    for i, (input, target) in pbar:
        iteration = epoch * len(train_loader) + i
        data_time.update(time.time() - end)

        if (i + 1) % 20 == 0 and n_cpt > 0:
            # Attach forward hook to get concept feature
            layer = int(args.whitened_layers)
            layers = model.module.layers
            if layer <= layers[0]:
                hook = model.module.model.layer1[layer - 1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1]:
                hook = model.module.model.layer2[layer - layers[0] - 1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1] + layers[2]:
                hook = model.module.model.layer3[layer - layers[0] - layers[1] - 1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                hook = model.module.model.layer4[layer - layers[0] - layers[1] - layers[2] - 1].bn1.register_forward_hook(hookf)

            # Collect feature for each concept
            inter_feature.clear()
            y = []
            for concept_index, c_loader in enumerate(concept_loaders):
                for j, (X, _) in enumerate(c_loader):
                    y += [concept_index] * X.size(0)
                    X_var = torch.autograd.Variable(X).cuda()
                    model(X_var)
                    break

            # Now compute concept classification metric
            inter_feat = torch.cat(inter_feature, 0)  # shape: [total_cpt_images, n_cpt, h, w]
            y_var = torch.Tensor(y).long().cuda()

            # Activation aggregator
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

            # remove the hook
            hook.remove()

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Normal forward
        output = model(input_var)
        loss = criterion(output, target_var)

        # Evaluate accuracy
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1_acc.update(prec1[0].item(), input.size(0))
        top5_acc.update(prec5[0].item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # Log to TensorBoard: normal classification + concept alignment
        misclass_rate = 100.0 - top1_acc.val
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
    """Plot or do additional evaluation. Not changing signature or internal logic."""
    concept_name = args.concepts.split(',')
    out_dir = './plot/' + '_'.join(concept_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if args.evaluate == 'plot_top50':
        print("Plot top50 activated images")
        model = load_resnet_model(args, arch='resnet_cw', depth=18, whitened_layer='8')
        plot_concept_top50(args, test_loader_with_path, model, '8', activation_mode=args.act_mode)
    elif args.evaluate == 'plot_auc':
        print("Plot AUC-concept_purity")
        print("Note: multiple models with different whitened layers are needed.")
        aucs_cw = plot_auc_cw(args, conceptdir, '1,2,3,4,5,6,7,8',
                              plot_cpt=concept_name, activation_mode=args.act_mode)
        print("Running AUCs SVM")
        model = load_resnet_model(args, arch='resnet_original', depth=18)
        aucs_svm = plot_auc_lm(args, model, concept_loaders, train_loader, conceptdir,
                               '1,2,3,4,5,6,7,8', plot_cpt=concept_name, model_type='svm')
        print("Running AUCs LR")
        model = load_resnet_model(args, arch='resnet_original', depth=18)
        aucs_lr = plot_auc_lm(args, model, concept_loaders, train_loader, conceptdir,
                              '1,2,3,4,5,6,7,8', plot_cpt=concept_name, model_type='lr')
        print("Running best filter approach")
        model = load_resnet_model(args, arch='resnet_original', depth=18)
        aucs_filter = plot_auc_filter(args, model, conceptdir,
                                      '1,2,3,4,5,6,7,8', plot_cpt=concept_name)
        print("Finished AUC plotting.")


def save_checkpoint(state, is_best, prefix, checkpoint_folder='./checkpoints'):
    """Unmodified logic except improved readability."""
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
    elif args.arch in ["resnet_original", "densenet_original", "vgg16_original"]:
        filename = os.path.join(checkpoint_folder, f'{prefix}_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            best_fn = os.path.join(checkpoint_folder, f'{prefix}_model_best.pth.tar')
            shutil.copyfile(filename, best_fn)


class AverageMeter(object):
    """Unchanged logic: computes and stores average & current value"""
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


def adjust_learning_rate(optimizer, epoch):
    """Sets LR = initial LR * (0.1^(epoch//30)). Returns the new LR for logging."""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy. Returns list of accuracies (floats)."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # shape [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
