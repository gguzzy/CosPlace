import os
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T
from adapt import DATrainDataset
import test
import util
import parserCustom
import commons
import cosface_loss
import augmentations
from cosplace_model import cosplace_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

from pytorch_adapt.containers import Models, Optimizers
from pytorch_adapt.datasets import DataloaderCreator, get_mnist_mnistm
from pytorch_adapt.hooks import DANNHook
from pytorch_adapt.models import Discriminator, mnistC, mnistG
from pytorch_adapt.utils.common_functions import batch_to_device
from pytorch_adapt.validators import IMValidator

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parserCustom.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Model
if args.domain_adapt == 'True':
    model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim,
                                                alpha=0.05, domain_adapt=args.domain_adapt)
else:
    model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
# model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# for vit best
model_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

#### Datasets
groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]

# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
# classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]
classifiers_optimizers = [torch.optim.AdamW(classifier.parameters(), lr=args.classifiers_lr) for classifier in
                          classifiers]

# Dataset and dataloader for domain adaptation

# How many classes and images for the class label prediction
logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(
    f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
test_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                      positive_dist_threshold=args.positive_dist_threshold)
logging.info(f"Validation set: {val_ds}")
logging.info(f"Test set: {test_ds}")

# NIGHT DATASET
groups_night = [
    DATrainDataset(args, "/kaggle/working/data/tokyo_xs/night_database", M=args.M,
                   alpha=args.alpha, N=args.N, L=args.L,
                   current_group=n, min_images_per_class=args.min_images_per_class, night=True) for n in
    range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers_night = [cosface_loss.MarginCosineProduct(2, len(group)) for group in groups_night]
classifiers_optimizers_night = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in
                                classifiers_night]

# How many classes and images for the night domain label prediction
logging.info(f"Using {len(groups_night)} groups")
logging.info(
    f"The {len(groups_night)} groups have respectively the following number of classes {[len(g) for g in groups_night]}")
logging.info(
    f"The {len(groups_night)} groups have respectively the following number of images {[g.get_images_num() for g in groups_night]}")

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(
        f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")

# Adding two new types of Augmentations: GaussianBlur and AutoAugment
if args.augmentation_device == "cuda":
    # Parse the kernel_size and sigma values from strings to tuples
    # Check if kernel_size and sigma are already tuples, if not convert them to tuples
    if isinstance(args.kernel_size, int):
        kernel_size = (args.kernel_size, args.kernel_size)
    elif isinstance(args.kernel_size, str):
        kernel_size = tuple(map(int, args.kernel_size.split(',')))
    else:
        kernel_size = args.kernel_size

    if isinstance(args.sigma, float):
        sigma = (args.sigma, args.sigma)
    elif isinstance(args.sigma, str):
        sigma = tuple(map(float, args.sigma.split(',')))
    else:
        sigma = args.sigma

    gpu_augmentation = T.Compose([
        augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                contrast=args.contrast,
                                                saturation=args.saturation,
                                                hue=args.hue),
        augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                      scale=[1 - args.random_resized_crop, 1]),
        augmentations.DeviceAgnosticAdjustBrightnessContrastSaturation(brightness_factor=args.brightness_factor,
                                                                       contrast_factor=args.contrast_factor,
                                                                       saturation_factor=args.saturation_factor),
        augmentations.DeviceAgnosticRandomPerspective(),
        augmentations.DeviceAgnosticAdjustGamma(gamma=args.gamma, gain=1.2),
        augmentations.DeviceAgnosticGaussianBlur(kernel_size,
                                                 sigma),
        # Initialize DeviceAgnosticAutoAugment for each policy
        augmentations.DeviceAgnosticAutoAugment(policy=T.AutoAugmentPolicy.IMAGENET,
                                                interpolation=T.InterpolationMode.NEAREST),
        augmentations.DeviceAgnosticAutoAugment(policy=T.AutoAugmentPolicy.CIFAR10,
                                                interpolation=T.InterpolationMode.NEAREST),
        augmentations.DeviceAgnosticAutoAugment(policy=T.AutoAugmentPolicy.SVHN,
                                                interpolation=T.InterpolationMode.NEAREST),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

# If domain adaption is set to True we train
if args.domain_adapt == 'True':
    # we are inside domain adaptation
    for epoch_num in range(start_epoch_num, args.epochs_num):
        # Training the model
        epoch_start_time = datetime.now()  # set epoch start time
        # Select classifier and dataloader according to epoch
        current_group_num = epoch_num % args.groups_num  # rest of

        # class label classifier
        classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
        util.move_to_device(classifiers_optimizers[current_group_num], args.device)

        # domain classifiers


        # normal dataloader
        dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                                batch_size=args.batch_size, shuffle=True,
                                                pin_memory=(args.device == "cuda"), drop_last=True)

        dataloader_iterator = iter(dataloader)
        model = model.train()

        # Domain dataloaders
        """
        Using pytorch adapt we create using DataloaderCreator
        """

        logging.info(f"Dataloader CLASSIC: {len(dataloader)}")
        logging.info(f"Dataloader DAY: {len(dataloader_day)}")
        logging.info(f"Dataloader NIGHT: {len(dataloader_night)}")

        epoch_losses = np.zeros((0, 1), dtype=np.float32)
        for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
            # images, target UTM labels
            images, targets, _ = next(dataloader_iterator)
            images, targets = images.to(args.device), targets.to(args.device)

            # images, target day domain labels


            # images,target night domain labels


            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)

            model_optimizer.zero_grad()
            classifiers_optimizers[current_group_num].zero_grad()

            if not args.use_amp16:
                # normal loss
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
                loss.backward()  # allows

                # # loss night
                # descriptors_night = model(images_night, alpha=0.05)
                # output_night = classifiers_night[current_group_num](descriptors_night, targets_night)
                # loss_night = criterion(output_night, targets_night)
                #
                # # loss domain
                # loss_domain = loss_night + loss_day
                # loss_domain.backward()

                epoch_losses = np.append(epoch_losses, loss.item())
                del loss, output, images
                model_optimizer.step()
                classifiers_optimizers[current_group_num].step()
            else:  # Use AMP 16
                with torch.cuda.amp.autocast():
                    # normal loss
                    descriptors = model(images)
                    output = classifiers[current_group_num](descriptors, targets)
                    loss = criterion(output, targets)

                    # # loss day
                    # descriptors_day = model(images_day, alpha=0.05)
                    # output_day = classifiers_day[current_group_num](descriptors_day, targets_day)
                    # loss_day = criterion(output_day, targets_day)
                    #
                    # # loss night
                    # descriptors_night = model(images_night, alpha=0.05)
                    # output_night = classifiers_night[current_group_num](descriptors_night, targets_night)
                    # loss_night = criterion(output_night, targets_night)
                    #
                    # # loss domain
                    # loss_domain = loss_night + loss_day

                scaler.scale(loss).backward()
                # scaler.scale(loss_domain).backward()

                epoch_losses = np.append(epoch_losses, loss.item())
                epoch_losses = np.append(epoch_losses, loss_domain.item())
                # Adversarial training
                target_images, _, _ = next(dataloader_iterator)
                target_images = target_images.to(args.device)
                target_descriptors = model(target_images)
                domain_labels = torch.ones(images.size(0)).to(args.device)  # Domain labels for source domain
                target_domain_labels = torch.zeros(target_images.size(0)).to(
                    args.device)  # Domain labels for target domain

                del loss, output, images
                scaler.step(model_optimizer)
                scaler.step(classifiers_optimizers[current_group_num])
                scaler.update()

        classifiers[current_group_num] = classifiers[current_group_num].cpu()
        util.move_to_device(classifiers_optimizers[current_group_num], "cpu")

        logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                      f"loss = {epoch_losses.mean():.4f}")

        #### Evaluation
        recalls, recalls_str = test.test(args, val_ds, model)
        logging.info(
            f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
        is_best = recalls[0] > best_val_recall1
        best_val_recall1 = max(recalls[0], best_val_recall1)
        # Save checkpoint, which contains all training parameters
        util.save_checkpoint({
            "epoch_num": epoch_num + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_optimizer.state_dict(),
            "classifiers_state_dict": [c.state_dict() for c in classifiers],
            "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
            "best_val_recall1": best_val_recall1
        }, is_best, args.output_folder)
else:
    pass

logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set v1
best_model_state_dict = torch.load(f"{args.output_folder}/best_model.pth")
model.load_state_dict(best_model_state_dict)

logging.info(f"Now testing on the test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model, args.num_preds_to_save)
logging.info(f"{test_ds}: {recalls_str}")

logging.info("Experiment finished (without any errors)")