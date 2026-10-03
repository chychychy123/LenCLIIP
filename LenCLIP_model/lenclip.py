"""LenCLIP: frozen CLIP + LPG + VA + online segmentation training."""

import torch
from torch import nn
from torch.nn import functional as F
import clip
from clip.clip_text import class_names, class_names_coco
from .segformer_head import SegFormerHead
from .Decoder.TransDecoder import DecoderTransformer
from .PAR import PAR
from .prompts import LearnablePrompts
from .lpg import MaskGenerator, region_contrastive_losses
from .visual_alignment import VisualAlignment


class LenCLIP(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = cfg
        names = class_names if cfg.dataset.name == "voc" else class_names_coco
        self.num_classes = len(names) + 1
        if int(cfg.dataset.num_classes) != self.num_classes:
            raise ValueError("Dataset name and num_classes do not match")
        self.encoder, _ = clip.load(str(cfg.model.clip_pretrain_path), device=device, jit=False)
        self.encoder.float().requires_grad_(False).eval()
        if not hasattr(self.encoder.visual, "patch_size"):
            raise ValueError("LenCLIP requires a CLIP ViT backbone")
        self.patch_size = int(self.encoder.visual.patch_size)
        if self.patch_size != 16:
            raise ValueError("These configurations use ViT-B/16")
        for transformer in (self.encoder.visual.transformer, self.encoder.transformer):
            transformer.gradient_checkpointing = bool(cfg.model.gradient_checkpointing)
        width = self.encoder.visual.conv1.out_channels
        self.fusion_layers = int(cfg.model.fusion_layers)
        if not 1 <= self.fusion_layers <= self.encoder.visual.transformer.layers:
            raise ValueError("Invalid number of fusion layers")
        dim = int(cfg.model.embedding_dim)
        self.decoder_fts_fuse = SegFormerHead(
            in_channels=[width] * 4, embedding_dim=dim,
            num_classes=self.num_classes, index=self.fusion_layers,
        )
        self.decoder = DecoderTransformer(dim, int(cfg.model.decoder_layers), 8, self.num_classes)
        self.prompts = LearnablePrompts(self.encoder, names, cfg.prompt.n_ctx, cfg.prompt.class_specific)
        self.mask_generator = MaskGenerator(dim, self.num_classes - 1)
        self.visual_alignment = VisualAlignment(**dict(cfg.va))
        self.par = PAR(dilations=list(cfg.pseudo.par_dilations), num_iter=int(cfg.pseudo.par_iterations))
        self.register_buffer("image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def normalize(self, images):
        return (images - self.image_mean) / self.image_std

    def extract_features(self, images):
        h, w = images.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("Input height and width must be multiples of 16")
        # Only the unmasked image branch is detached. Masked images and prompts
        # pass through the same frozen encoder with their input gradients intact.
        with torch.no_grad():
            tokens, attention = self.encoder.encode_image(self.normalize(images), h, w, require_all_fts=True)
            maps = torch.stack([
                t[1:].permute(1, 2, 0).reshape(images.shape[0], -1, h // 16, w // 16)
                for t in tokens[:self.fusion_layers]
            ])
            last = self.encoder.visual.ln_post(tokens[-1].permute(1, 0, 2))
            projected = (last[:, 1:] @ self.encoder.visual.proj).transpose(1, 2)
            dense = projected.reshape(images.shape[0], -1, h // 16, w // 16)
        return self.decoder_fts_fuse(maps), dense, attention

    def encode_masked_regions(self, images, masks, image_labels):
        pairs = torch.nonzero(image_labels > 0, as_tuple=False)
        if pairs.shape[0] == 0:
            return None
        size = int(self.cfg.model.region_size)
        if size <= 0 or size % self.patch_size:
            raise ValueError("region_size must be a positive multiple of 16")
        chunk = max(1, int(self.cfg.model.region_batch_size))
        fg_features, bg_features = [], []
        for selected in pairs.split(chunk):
            batch, category = selected[:, 0], selected[:, 1]
            image = images[batch]
            mask = masks[batch, category].unsqueeze(1)
            # Mask raw RGB before normalization, as X_f=M*X, X_b=(1-M)*X.
            regions = torch.cat([image * mask, image * (1 - mask)], dim=0)
            regions = F.interpolate(regions, (size, size), mode="bilinear", align_corners=False)
            tokens, _ = self.encoder.encode_image(self.normalize(regions), size, size)
            visual = self.encoder.visual.ln_post(tokens[0]) @ self.encoder.visual.proj
            foreground, background = F.normalize(visual.float(), dim=-1).chunk(2)
            fg_features.append(foreground)
            bg_features.append(background)
        return torch.cat(fg_features), torch.cat(bg_features), pairs[:, 1]

    @torch.no_grad()
    def pseudo_labels(self, images, cams, labels, attention, affinity, step, valid_mask):
        if bool(self.cfg.pseudo.use_clip_attention):
            layers = min(int(self.cfg.pseudo.attention_layers), len(attention))
            transition = torch.stack(attention[-layers:]).mean(0)[:, 1:, 1:].float()
            if step >= int(self.cfg.pseudo.affinity_start):
                transition = transition * affinity.detach()
            transition = transition / transition.sum(-1, keepdim=True).clamp_min(1e-6)
            flat = cams.flatten(2)
            cams = torch.bmm(flat, transition.transpose(1, 2)).view_as(cams)
        cams = cams / cams.flatten(2).amax(-1)[..., None, None].clamp_min(1e-6)
        cams = cams * labels[:, :, None, None]
        cams = F.interpolate(cams, images.shape[-2:], mode="bilinear", align_corners=False)
        # Refine only the present channels to bound PAR memory on COCO.
        predictions = []
        for batch in range(images.shape[0]):
            keys = torch.nonzero(labels[batch] > 0, as_tuple=False).flatten()
            if keys.numel() == 0:
                predictions.append(torch.zeros_like(cams[batch, 0], dtype=torch.long))
                continue
            foreground = cams[batch:batch + 1, keys]
            background = (1 - foreground.amax(1, keepdim=True)).clamp(0, 1).pow(float(self.cfg.pseudo.background_power))
            scores = torch.cat((background, foreground), dim=1)
            if int(self.cfg.pseudo.par_iterations) > 0:
                # Rescaling PAR is configurable; full crop resolution by default.
                scale = float(self.cfg.pseudo.par_scale)
                if scale != 1:
                    shape = tuple(max(1, round(v * scale)) for v in images.shape[-2:])
                    scores = F.interpolate(scores, shape, mode="bilinear", align_corners=False)
                scores = self.par(images[batch:batch + 1], scores)
                scores = F.interpolate(scores, images.shape[-2:], mode="bilinear", align_corners=False)
            ids = torch.cat((keys.new_zeros(1), keys + 1))
            predictions.append(ids[scores.argmax(1)[0]])
        target = torch.stack(predictions)
        if valid_mask is not None:
            target = target.masked_fill(~valid_mask.bool(), int(self.cfg.dataset.ignore_index))
        return target

    def forward(self, images, class_labels=None, step=0, valid_mask=None, generate_pseudo=False):
        features, dense, attention = self.extract_features(images)
        logits, _ = self.decoder(features)
        output = {"seg": logits}
        if class_labels is None:
            return output
        if class_labels.shape != (images.shape[0], self.num_classes - 1):
            raise ValueError("class_labels must be [batch, foreground_classes]")
        texts = {branch: self.prompts(self.encoder, branch) for branch in ("mg", "foreground", "background")}
        masks = self.mask_generator(features, images.shape[-2:])
        if valid_mask is not None:
            masks = masks * valid_mask[:, None].to(masks.dtype)
        flattened = features.flatten(2)
        affinity = (flattened.transpose(1, 2) @ flattened).sigmoid()
        if self.training:
            regions = self.encode_masked_regions(images, masks, class_labels)
            if regions is None:
                zero = features.sum() * 0 + sum(t.sum() * 0 for t in texts.values()) + masks.sum() * 0
                losses = {key: zero for key in ("match", "foreground", "background_suppression", "lp_paper", "background_completion", "lp", "lpg")}
            else:
                losses = region_contrastive_losses(*regions, texts, self.cfg.loss)
            output["losses"] = losses
        if self.training or generate_pseudo:
            # Pseudo-label creation has no gradient; LPG gradients come from the
            # masked-image/text contrastive objective, not from argmax labels.
            with torch.no_grad():
                cams = self.visual_alignment(dense, texts["foreground"], texts["background"], class_labels)
                target = self.pseudo_labels(images, cams, class_labels, attention, affinity, step, valid_mask)
            output.update(pseudo=target, cams=cams, affinity=affinity, masks=masks)
        return output
