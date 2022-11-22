import os

import clip
import cv2
import networkx as nx
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
import torch
from torchvision import transforms


def match_text_to_imgs(language_instr, images_list):
    """img_feats: (Nxself.clip_feat_dim), text_feats: (1xself.clip_feat_dim)"""
    imgs_feats = get_imgs_feats(images_list)
    text_feats = get_text_feats([language_instr])
    scores = imgs_feats @ text_feats.T
    scores = scores.squeeze()
    return scores, imgs_feats, text_feats


def get_nn_img(raw_imgs, text_feats, img_feats):
    """img_feats: (Nxself.clip_feat_dim), text_feats: (1xself.clip_feat_dim)"""
    scores = img_feats @ text_feats.T
    scores = scores.squeeze()
    high_to_low_ids = np.argsort(scores).squeeze()[::-1]
    high_to_low_imgs = [raw_imgs[i] for i in high_to_low_ids]
    high_to_low_scores = np.sort(scores).squeeze()[::-1]
    return high_to_low_ids, high_to_low_imgs, high_to_low_scores


def get_img_feats(img, preprocess, clip_model):
    img_pil = Image.fromarray(np.uint8(img))
    img_in = preprocess(img_pil)[None, ...]
    with torch.no_grad():
        img_feats = clip_model.encode_image(img_in.cuda()).float()
    img_feats /= img_feats.norm(dim=-1, keepdim=True)
    img_feats = np.float32(img_feats.cpu())
    return img_feats


def get_imgs_feats(raw_imgs, preprocess, clip_model, clip_feat_dim):
    imgs_feats = np.zeros((len(raw_imgs), clip_feat_dim))
    for img_id, img in enumerate(raw_imgs):
        imgs_feats[img_id, :] = get_img_feats(img, preprocess, clip_model)
    return imgs_feats


def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    text_tokens = clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats
