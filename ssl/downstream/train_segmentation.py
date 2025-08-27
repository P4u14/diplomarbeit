import argparse
import os
import sys
import PIL.Image as PILImage
from tqdm import tqdm

from preprocessing.square_image_preprocessor import SquareImagePreprocessor
from preprocessing.torso_roi_preprocessor import TorsoRoiPreprocessor
from ssl.downstream.dice_loss import DiceLoss
from ssl.downstream.preprocessed_segmentation_dataset import PreprocessedSegmentationDataset
from ssl.downstream.segmentation_dataset import SegmentationDataset
from ssl.downstream.simple_classifier.segmentation_head import SegmentationHead
from ssl.downstream.unet.jigsaw_absolut_position.unet_decoder_jigsaw_abs_pos import UnetDecoderJigsawAbsPos
from ssl.downstream.unet.jigsaw_permutation.alex_net_attention_unet_decoder import AlexNetAttentionUNetDecoder
from ssl.pretext.jigsaw_absolute_position.JigsawAbsPosNetwork import JigsawAbsPosNetwork
from ssl.pretext.jigsaw_permutation.JigsawPermNetwork import JigsawPermNetwork

# ensure project root is on the import path
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import glob
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import ImageDraw, Image
import numpy as np
import time
import datetime
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Adapt pre-trained Jigsaw model for segmentation task')
parser.add_argument('--config', type=str, default='ssl/downstream/config.yaml', help='Path to config file')
args = parser.parse_args()


def load_encoder(pretext_classes, pretrained_model_path, backbone):
    if backbone == 'jigsaw_abs_pos':
        model = JigsawAbsPosNetwork(pretext_classes)
        model.load(pretrained_model_path)
        # Keep only the feature extraction backbone
        encoder = model.features
        return encoder
    else:
        model = JigsawPermNetwork(pretext_classes)
        model.load(pretrained_model_path)
        # Keep only the convolutional backbone
        encoder = model.conv
        return encoder


def main():
    config = yaml.safe_load(open(args.config, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = os.path.join(config['io']['output'],
                              config['model']['backbone'],
                              config['model']['downstream_arch'],
                              config['io']['name'])
    os.makedirs(output_dir, exist_ok=True)

    total_start_time = time.time()
    log_print('Start training: %s' % datetime.datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S'), log_dir=output_dir)

    preprocessing_steps = [
        TorsoRoiPreprocessor(target_ratio=5 / 7),
        SquareImagePreprocessor(),
    ]

    train_data_path = os.path.join(config['io']['data'], "ILSVRC2012_img_train")
    val_data_path = os.path.join(config['io']['data'], "ILSVRC2012_img_val")
    # train_dl = create_train_dl(config, train_data_path)
    train_dl = create_train_dl_with_pp(config, train_data_path, preprocessing_steps)
    # val_dl, val_imgs = create_val_dl(config, val_data_path)
    val_dl, val_imgs = create_val_dl_with_pp(config, val_data_path, preprocessing_steps)

    backbone, encoder, model, model_type = create_model(config, device, output_dir)

    num_freeze_layers = freeze_layers(backbone, config, encoder, output_dir)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr_head"]
    )

    train_dices, train_losses, val_dices, val_losses = train(backbone, config, device, model, model_type,
                                                             num_freeze_layers, optimizer, output_dir, train_dl, val_dl,
                                                             val_imgs)

    # Print total training time
    total_end_time = time.time()
    log_print('End training: %s' % datetime.datetime.fromtimestamp(total_end_time).strftime('%Y-%m-%d %H:%M:%S'),log_dir=output_dir)
    duration_seconds = total_end_time - total_start_time
    m, s = divmod(duration_seconds, 60)
    h, m = divmod(m, 60)
    log_print("Training duration: %d hours, %d minutes und %d seconds" % (h, m, s), log_dir=output_dir)

    plot_train_and_val_metrics(config, output_dir, train_dices, train_losses, val_dices, val_dl, val_losses)


def plot_train_and_val_metrics(config, output_dir, train_dices, train_losses, val_dices, val_dl, val_losses):
    epochs_range = range(1, config["training"]["epochs"] + 1)
    plt.figure(figsize=(12, 5))
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    if val_dl:
        plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Plot Dice Score
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_dices, label='Train Dice')
    if val_dl:
        plt.plot(epochs_range, val_dices, label='Val Dice')
    plt.title('Dice Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics_plot.png')
    plt.savefig(plot_path)
    log_print(f"Training metrics plot saved to {plot_path}", log_dir=output_dir)
    plt.close()


def train(backbone, config, device, model, model_type, num_freeze_layers, optimizer, output_dir, train_dl, val_dl,
          val_imgs):
    # Add a weight to the positive class to combat class imbalance.
    # This tells the loss function to penalize false negatives more heavily.
    pos_weight_value = config["training"].get("pos_weight", 20.0)
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_dice = DiceLoss()
    dice_coeff = lambda p, t: (2 * (p * t).sum()) / (p.sum() + t.sum() + 1e-6)
    train_losses, train_dices, val_losses, val_dices = [], [], [], []
    best_val_loss = float('inf')
    is_first_epoch = True
    # Training loop
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        for batch in train_dl:
            imgs, masks = batch[:2]
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            # The model outputs a different size than the masks, so we need to resize it
            # before computing the loss.
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            # Combined loss with weighting to balance their magnitudes
            loss_bce = criterion_bce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = loss_bce + (loss_dice * 20)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            dice = dice_coeff(preds, masks).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_dice += dice * imgs.size(0)
        avg_loss = total_loss / len(train_dl.dataset)
        avg_dice = total_dice / len(train_dl.dataset)
        log_print(f"Epoch {epoch + 1}/{config['training']['epochs']} - Train Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}", log_dir=output_dir)
        train_losses.append(avg_loss)
        train_dices.append(avg_dice)

        # Validation step
        if val_dl:
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            with torch.no_grad():
                for i, batch in enumerate(val_dl):
                    if isinstance(batch, dict):
                        imgs = batch.get("image") or batch.get("images")
                        masks = batch.get("mask") or batch.get("masks")
                    else:
                        imgs, masks = batch[0], batch[1]

                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    logits = model(imgs)
                    logits = F.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                    # Combined loss for validation with weighting
                    loss_bce = criterion_bce(logits, masks)
                    loss_dice = criterion_dice(logits, masks)
                    loss = loss_bce + (loss_dice * 20)

                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()

                    # Debugging output for the first batch of each validation run
                    if i == 0:
                        log_print("\n--- Debugging Model Output (First Validation Batch) ---", log_dir=output_dir)
                        log_print(
                            f"Logits -> min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}", log_dir=output_dir)
                        log_print(
                            f"Probs  -> min: {probs.min().item():.4f}, max: {probs.max().item():.4f}, mean: {probs.mean().item():.4f}", log_dir=output_dir)
                        log_print(f"Preds  -> sum: {preds.sum().item()}, non-zero pixels: {(preds > 0).sum().item()}", log_dir=output_dir)
                        log_print(f"Masks  -> sum: {masks.sum().item()}, non-zero pixels: {(masks > 0).sum().item()}", log_dir=output_dir)
                        log_print("-----------------------------------------------------\n", log_dir=output_dir)

                    dice = dice_coeff(preds, masks).item()

                    val_loss += loss.item() * imgs.size(0)
                    val_dice += dice * imgs.size(0)
            avg_val_loss = val_loss / len(val_dl.dataset)
            avg_val_dice = val_dice / len(val_dl.dataset)
            log_print(
                f"Epoch {epoch + 1}/{config['training']['epochs']} - Val Loss:   {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}", log_dir=output_dir)
            val_losses.append(avg_val_loss)
            val_dices.append(avg_val_dice)

            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss or is_first_epoch:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    log_print(f"Saved new best model with validation loss: {best_val_loss:.4f}", log_dir=output_dir)
                else:
                    log_print(f"Saved model from first epoch with validation loss: {avg_val_loss:.4f}", log_dir=output_dir)

                is_first_epoch = False
                ckpt = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                }
                ckpt_name = "best_model.pth"
                torch.save(ckpt, os.path.join(output_dir, ckpt_name))

            # Create a directory for the current epoch's visualizations
            vis_dir = os.path.join(output_dir, "visualizations", f"epoch_{epoch + 1:03d}")
            os.makedirs(vis_dir, exist_ok=True)

            # --- Visualize all validation images ---
            model.eval()
            val_img_counter = 0
            with torch.no_grad():
                # Get a single batch from the validation loader for visualization
                try:
                    vis_batch = next(iter(val_dl))

                    if isinstance(vis_batch, dict):
                        vis_batch_imgs = (vis_batch.get("image") or vis_batch.get("images")).to(device)
                        vis_batch_masks = (vis_batch.get("mask") or vis_batch.get("masks")).to(device)
                        vis_paths = vis_batch.get("image_path") or vis_batch.get("paths")
                    else:
                        vis_batch_imgs = vis_batch[0].to(device)
                        vis_batch_masks = vis_batch[1].to(device)
                        vis_paths = vis_batch[2] if len(vis_batch) > 2 else None
                except StopIteration:
                    # Handle case where val_dl is empty
                    continue

                logits = model(vis_batch_imgs)
                logits = F.interpolate(
                    logits,
                    size=vis_batch_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                for i in range(vis_batch_imgs.size(0)):
                    # Get the original image path.
                    # This assumes shuffle=False for the validation loader.
                    original_img_path = vis_paths[i] if vis_paths is not None else val_imgs[i]

                    # Load the original image
                    img_pil = Image.open(original_img_path).convert("RGB")
                    # Resize to match the model's input/output size for accurate contour drawing
                    img_pil = img_pil.resize(tuple(config.get("img_size", (256, 256))))
                    draw = ImageDraw.Draw(img_pil)

                    mask_tensor = vis_batch_masks[i].cpu()
                    pred_tensor = preds[i].cpu()

                    # Convert masks to numpy and find contours
                    mask_np = mask_tensor.squeeze().numpy().astype(np.uint8)
                    pred_np = pred_tensor.squeeze().numpy().astype(np.uint8)

                    # Find contours using a simple method (checking for non-zero neighbors)
                    def find_contours(mask):
                        contours = []
                        for y in range(1, mask.shape[0] - 1):
                            for x in range(1, mask.shape[1] - 1):
                                if mask[y, x] > 0:
                                    # Check if it's a border pixel
                                    if (mask[y - 1, x] == 0 or mask[y + 1, x] == 0 or
                                            mask[y, x - 1] == 0 or mask[y, x + 1] == 0):
                                        contours.append((x, y))
                        return contours

                    # Draw ground truth contours (pink)
                    gt_contours = find_contours(mask_np)
                    if gt_contours:
                        draw.point(gt_contours, fill=(255, 105, 180))  # Pink

                    # Draw prediction contours (orange)
                    pred_contours = find_contours(pred_np)
                    if pred_contours:
                        draw.point(pred_contours, fill=(255, 165, 0))  # Orange

                    # Save the visualized image
                    img_pil.save(os.path.join(vis_dir, f"val_img_{val_img_counter:04d}.png"))
                    val_img_counter += 1

        # Unfreeze encoder after specified epochs
        if (epoch + 1) == config["training"].get("freeze_epochs", 5) and num_freeze_layers > 0:
            log_print(f"Unfreezing the first {num_freeze_layers} blocks/layers for fine-tuning.", log_dir=output_dir)

            target_encoder = None
            if model_type == "attention_unet":
                target_encoder = model.encoder
            elif isinstance(model, nn.Sequential):
                target_encoder = model[0]  # Encoder is the first part of the sequential model

            if target_encoder:
                if backbone == 'mobilenet_v2':
                    # For MobileNetV2, the 'encoder' variable is already the correct wrapper
                    for i, child in enumerate(target_encoder.children()):
                        if i < num_freeze_layers:
                            for params in child.parameters():
                                params.requires_grad = True
                else:
                    # For the original Jigsaw or ResNet-based encoders
                    for name, child in target_encoder.named_children():
                        # Unfreeze based on layer number for Jigsaw
                        try:
                            layer_num = int(name.split('_')[0][-1])
                            if layer_num <= num_freeze_layers:
                                for params in child.parameters():
                                    params.requires_grad = True
                        except (ValueError, IndexError):
                            # Unfreeze based on layer name for ResNet-style encoders
                            if name.startswith('layer') and int(name[5:]) <= num_freeze_layers:
                                for param in child.parameters():
                                    param.requires_grad = True
                            elif name in ['conv1', 'bn1', 'relu', 'maxpool'] and num_freeze_layers >= 1:
                                for param in child.parameters():
                                    param.requires_grad = True

                # Adjust learning rate for fine-tuning
                optimizer.param_groups[0]['lr'] = config["training"]["lr_finetune"]
            else:
                log_print("WARNING: Could not find the encoder for defrosting.", log_dir=output_dir)
    return train_dices, train_losses, val_dices, val_losses


def freeze_layers(backbone, config, encoder, output_dir):
    num_freeze_layers = config['training']['freeze_layers']
    if num_freeze_layers > 0:
        log_print(f"Freezing the first {num_freeze_layers} blocks/layers of the encoder.", log_dir=output_dir)
        if backbone == 'mobilenet_v2':
            # For MobileNetV2, freeze the first N feature blocks
            for i, child in enumerate(encoder.children()):
                if i < num_freeze_layers:
                    for params in child.parameters():
                        params.requires_grad = False
        else:
            # Original freezing logic for JigsawNet
            for name, child in encoder.named_children():
                # Layer names are like 'conv1_s1', 'pool2_s1', etc.
                # We extract the number to decide if we should freeze it.
                try:
                    layer_num = int(name.split('_')[0][-1])
                    if layer_num <= num_freeze_layers:
                        for params in child.parameters():
                            params.requires_grad = False
                except (ValueError, IndexError):
                    # This handles layers that don't match the expected name format,
                    # ensuring they are not accidentally frozen.
                    log_print(f"  - Skipping layer with unexpected name: {name}", log_dir=output_dir)
    return num_freeze_layers


def create_model(config, device, output_dir):
    pretrained_model_path = config['model']['pretrained_model']
    log_print(f"Loading pretrained encoder from: {pretrained_model_path}", log_dir=output_dir)
    model_type = config["model"]['downstream_arch']
    backbone = config["model"]['backbone']
    if model_type == "attention_unet" and backbone == "jigsaw_abs_pos":
        log_print("Using Attention U-Net with MobileNetV2 backbone (JigsawAbsPos).", log_dir=output_dir)
        model = UnetDecoderJigsawAbsPos(
            pretext_model_path=pretrained_model_path,
            num_classes=config["model"]["downstream_classes"],
            pretext_classes=config['model']['pretext_classes']
        ).to(device)
        encoder = model.encoder
    elif model_type == "attention_unet" and backbone == "jigsaw_perm":
        log_print("Using Attention U-Net with AlexNet backbone (JigsawPerm).", log_dir=output_dir )
        encoder = load_encoder(config['model']['pretext_classes'], pretrained_model_path, backbone='jigsaw').to(device)
        model = AlexNetAttentionUNetDecoder(
            encoder=encoder,
            num_classes=config["model"]["downstream_classes"]
        ).to(device)
    else:
        log_print(f"Using standard Segmentation Head model with {backbone} backbone.", log_dir=output_dir)
        encoder = load_encoder(config['model']['pretext_classes'], pretrained_model_path, backbone=backbone).to(device)
        head = SegmentationHead(
            # in_channels=config["model"]["in_channels"],
            num_classes=config["model"]["downstream_classes"],
            # upsample_factor=config["model"]["upsample_factor"]
        ).to(device)
        model = nn.Sequential(encoder, head).to(device)
    return backbone, encoder, model, model_type


def create_val_dl(config, val_data_path):
    val_dl = None
    val_imgs = None
    if val_data_path is not None:
        all_val_files = sorted(glob.glob(os.path.join(val_data_path, "*.Gauss.png")))
        val_imgs = [f for f in all_val_files if "-mask" not in os.path.basename(f)]
        val_masks = sorted(glob.glob(os.path.join(val_data_path, "*-mask.Gauss.png")))

        val_ds = SegmentationDataset(
            val_imgs,
            val_masks,
            img_size=tuple(config.get("img_size", (256, 256)))
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=4
        )
    return val_dl, val_imgs

def create_val_dl_with_pp(config, val_data_path, preprocessing_steps):
    use_cache   = config["training"].get("preprocess_cache", True)
    cache_suf   = config["training"].get("cache_suffix", "_pp")
    pp_steps_suffix = ''

    for step in config['training'].get("preprocessing", []):
        pp_steps_suffix += f'_{step}'

    cache_dir   = val_data_path + cache_suf + pp_steps_suffix

    if use_cache:
        _materialize_preprocessed_cache_seg(val_data_path, cache_dir, preprocessing_steps)
        val_imgs, val_masks = _load_cached_pairs_as_lists(cache_dir)
        ds = PreprocessedSegmentationDataset(
            image_paths=val_imgs,
            mask_paths=val_masks,
            preprocessing_steps=[],
        )
    else:
        all_val = sorted(glob.glob(os.path.join(val_data_path, "*.Gauss.png")))
        val_imgs  = [f for f in all_val if "-mask" not in os.path.basename(f)]
        val_masks = [f for f in all_val if "-mask" in os.path.basename(f)]
        ds = PreprocessedSegmentationDataset(
            image_paths=val_imgs,
            mask_paths=val_masks,
            preprocessing_steps=preprocessing_steps,
        )

    dl = DataLoader(
        ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(not torch.backends.mps.is_available() and torch.cuda.is_available()),
    )
    return dl, val_imgs if not use_cache else (dl, [os.path.basename(p) for p in val_imgs])

def create_train_dl(config, train_data_path):
    all_train_files = sorted(glob.glob(os.path.join(train_data_path, "*.Gauss.png")))
    train_imgs = [f for f in all_train_files if "-mask" not in os.path.basename(f)]
    train_masks = sorted(glob.glob(os.path.join(train_data_path, "*-mask.Gauss.png")))
    train_ds = SegmentationDataset(
        train_imgs,
        train_masks,
        img_size=(256, 256)
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    return train_dl

def create_train_dl_with_pp(config, train_data_path, preprocessing_steps):
    use_cache   = config["training"].get("preprocess_cache", True)
    cache_suf   = config["training"].get("cache_suffix", "_pp")
    pp_steps_suffix = ''

    for step in config['training'].get("preprocessing", []):
        pp_steps_suffix += f'_{step}'

    cache_dir = train_data_path + cache_suf + pp_steps_suffix

    if use_cache:
        _materialize_preprocessed_cache_seg(train_data_path, cache_dir, preprocessing_steps)
        train_imgs, train_masks = _load_cached_pairs_as_lists(cache_dir)
        ds = PreprocessedSegmentationDataset(
            image_paths=train_imgs,
            mask_paths=train_masks,
            preprocessing_steps=[],  # important - otherwise duplicated preprocessing
        )
    else:
        all_train = sorted(glob.glob(os.path.join(train_data_path, "*.Gauss.png")))
        train_imgs  = [f for f in all_train if "-mask" not in os.path.basename(f)]
        train_masks = [f for f in all_train if "-mask" in os.path.basename(f)]
        ds = PreprocessedSegmentationDataset(
            image_paths=train_imgs,
            mask_paths=train_masks,
            preprocessing_steps=preprocessing_steps,
        )

    dl = DataLoader(
        ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(not torch.backends.mps.is_available() and torch.cuda.is_available()),
    )
    return dl


def log_print(*text, log_dir, sep=' ', end='\n', file=None, flush=False):
    # standard output
    print(*text, sep=sep, end=end, file=file, flush=flush)
    # write in log file in checkpoint folder
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'logs.txt')
    with open(log_path, 'a', encoding='utf-8') as f:
        print(*text, sep=sep, end=end, file=f)

def build_preprocessing_steps_from_config(config):
    steps = []
    pp_steps = config["training"].get("preprocessing", [])
    torso_target_ratio = config["training"].get("torso_target_ratio", 5 / 7)
    square_resize = config["training"].get("square_resize", 256)
    square_crop = config["training"].get("square_crop", 224)
    print('Preprocessing steps: ', pp_steps)
    for name in pp_steps:
        if name == 'torso':
            steps.append(TorsoRoiPreprocessor(target_ratio=torso_target_ratio))
        elif name == 'square':
            steps.append(SquareImagePreprocessor(resize_size=square_resize,
                                                 crop_size=square_crop))
        else:
            raise ValueError(f"Unknown preprocessing step: {name}")

    # Sanity check for Jigsaw: last stage should deliver 255x255
    if 'square' in pp_steps:
        if square_crop != 255:
            print("[Warning] For Jigsaw, square_crop=255 is common.")
        if square_resize < square_crop:
            raise ValueError("--square-resize must be >= --square-crop.")
    return steps

def _list_image_mask_pairs(root_dir):
    """Find (*.Gauss.png, *-mask.Gauss.png) pairs in the directory (recursively not necessary if flat)."""
    all_pngs = sorted(glob.glob(os.path.join(root_dir, "*.Gauss.png")))
    imgs  = [p for p in all_pngs if "-mask" not in os.path.basename(p)]
    masks = [p for p in all_pngs if "-mask" in os.path.basename(p)]
    # map by stem without '-mask'
    def _stem(p):
        b = os.path.basename(p)
        if b.endswith("-mask.Gauss.png"):
            return b[:-len("-mask.Gauss.png")]
        return b[:-len(".Gauss.png")]
    mask_by_stem = { _stem(m): m for m in masks }
    pairs = []
    for ip in imgs:
        s = _stem(ip)
        mp = mask_by_stem.get(s)
        if mp is not None:
            pairs.append((ip, mp))
    return pairs  # list of (img_path, mask_path)

def _materialize_preprocessed_cache_seg(in_dir, out_dir, preprocessing_steps):
    """
    Applies the same preprocessing_steps to the image and mask
    and saves the results in out_dir with identical file names.
    """
    os.makedirs(out_dir, exist_ok=True)
    pairs = _list_image_mask_pairs(in_dir)

    for img_path, mask_path in tqdm(pairs, desc="Writing segmentation preprocess cache"):
        dst_img = os.path.join(out_dir, os.path.basename(img_path))
        dst_msk = os.path.join(out_dir, os.path.basename(mask_path))
        if os.path.exists(dst_img) and os.path.exists(dst_msk):
            continue

        img  = np.asarray(PILImage.open(img_path).convert("RGB"))
        mask = np.asarray(PILImage.open(mask_path))

        x, m = img, mask
        for step in (preprocessing_steps or []):
            x, params = step.preprocess_image(x)
            m = step.preprocess_mask(m, params)

            # ---------- normalize image: -> (H,W,3), uint8 ----------
            x = np.asarray(x)
            x = np.squeeze(x)
            if x.ndim == 2:
                x = np.repeat(x[..., None], 3, axis=2)  # gray levels -> RGB
            elif x.ndim == 3 and x.shape[2] == 1:
                x = np.repeat(x, 3, axis=2)  # (H,W,1) -> (H,W,3)
            elif x.ndim != 3 or x.shape[2] not in (3,):
                raise ValueError(f"Unexpected image shape after preprocessing: {x.shape}")
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 255).astype(np.uint8)

            # ---------- normalize mask: -> (H,W), binary 0/255, uint8 ----------
            m = np.asarray(m)
            m = np.squeeze(m)
            if m.ndim == 3:
                if m.shape[2] == 1:
                    m = m[..., 0]  # (H,W,1) -> (H,W)
                elif m.shape[2] == 3:
                    # reduce any 3 channel mask to 2D
                    m = (m > 0).any(axis=2).astype(np.uint8)
                else:
                    # Fallback: use only the first channel
                    m = m[..., 0]
            elif m.ndim != 2:
                raise ValueError(f"Unexpected mask shape after preprocessing: {m.shape}")

            # binary & uint8
            mb = (m > 0).astype(np.uint8) * 255

            # ---------- Save ----------
            PILImage.fromarray(x).save(dst_img)  # RGB
            PILImage.fromarray(mb).save(dst_msk)  # L (gray, 0/255)

def _load_cached_pairs_as_lists(cache_dir):
    """Build the lists for the dataset constructor from the cache folder."""
    all_pngs = sorted(glob.glob(os.path.join(cache_dir, "*.Gauss.png")))
    imgs  = [p for p in all_pngs if "-mask" not in os.path.basename(p)]
    masks = [p for p in all_pngs if "-mask" in os.path.basename(p)]
    # mate again (same name stem)
    def stem(p):
        b = os.path.basename(p)
        if b.endswith("-mask.Gauss.png"):
            return b[:-len("-mask.Gauss.png")]
        return b[:-len(".Gauss.png")]
    m_by = { stem(m): m for m in masks }
    img_list, mask_list = [], []
    for ip in imgs:
        s = stem(ip)
        mp = m_by.get(s)
        if mp is not None:
            img_list.append(ip); mask_list.append(mp)
    return img_list, mask_list

if __name__ == "__main__":
    main()
