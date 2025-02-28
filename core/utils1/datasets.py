import os
from io import BytesIO
import random 
from random import randint
import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data.sampler import WeightedRandomSampler
from collections import defaultdict

from .config import CONFIGCLASS

ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(root: str, cfg: CONFIGCLASS):
    if cfg.mode == "binary":
        return VideoDataset(root, cfg)  # Use the VideoDataset instead of binary_dataset
    if cfg.mode == "filename":
        return FileNameDataset(root, cfg)
    raise ValueError("cfg.mode needs to be binary or filename.")


class VideoDataset(datasets.ImageFolder):
    def __init__(self, root, cfg):
        super().__init__(root)
        self.cfg = cfg

        # Map frames to their respective video
        self.video_to_frames = defaultdict(list)
        for idx, (path, label) in enumerate(self.samples):
            video_id = self._extract_video_id(path)
            self.video_to_frames[video_id].append(idx)

        # Assign a single random augmentation per video
        self.video_augmentations = {}
        self._assign_augmentations()

        # Create a dataset with both original and augmented versions
        self.augmented_samples = self._duplicate_samples()

        # Define the resize transform
        self.resize_transform = transforms.Resize((self.cfg.cropSize, self.cfg.cropSize))

    def _extract_video_id(self, path):
        """ Extracts video ID from file path """
        return os.path.basename(os.path.dirname(path))  # Assumes directories per video

    def _assign_augmentations(self):
        """ Assigns a single random augmentation to each video """
        possible_augmentations = [
            lambda img: transforms.RandomResizedCrop(self.cfg.cropSize, scale=(0.8, 1.0))(img),
            lambda img: transforms.RandomHorizontalFlip(p=1.0)(img) if self.cfg.aug_flip else img,
            lambda img: transforms.RandomVerticalFlip(p=1.0)(img) if self.cfg.aug_flip else img,
            lambda img: transforms.RandomRotation(degrees=(-10, 10))(img) if self.cfg.aug_rotation else img,
            lambda img: blur_jpg_augment(img, self.cfg)
        ]
        
        for video_id in self.video_to_frames:
            self.video_augmentations[video_id] = random.choice(possible_augmentations)

    def _duplicate_samples(self):
        """ Creates a dataset with both original and augmented versions of each video """
        augmented_samples = []
        for video_id, indices in self.video_to_frames.items():
            for idx in indices:
                # Original version (resized)
                original = (self.samples[idx][0], self.samples[idx][1], None)  # None = no augmentation
                augmented = (self.samples[idx][0], self.samples[idx][1], self.video_augmentations[video_id])  # Augmented version
                
                augmented_samples.append(original)
                augmented_samples.append(augmented)
        
        return augmented_samples

    def __getitem__(self, index):
        """ Loads a frame and applies resizing + augmentation if needed """
        path, target, augmentation = self.augmented_samples[index]
        img = self.loader(path)  # Load image

        # Apply resizing to ensure all images are of size (cropSize, cropSize)
        img = self.resize_transform(img)

        # Apply augmentation only if it's assigned
        if augmentation is not None:
            img = augmentation(img)

        # Convert to tensor and normalize
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.cfg.aug_norm else transforms.Lambda(lambda x: x),
        ])(img)

        return img, target

class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return path


def blur_jpg_augment(img: Image.Image, cfg):
    img: np.ndarray = np.array(img)  # Convert to NumPy array

    if cfg.isTrain:
        if random.random() < cfg.blur_prob:
            sig = sample_continuous(cfg.blur_sig)
            img = gaussian_blur(img, sig)  # Ensure function returns an image
            if img is None:
                raise ValueError("gaussian_blur returned None")

        if random.random() < cfg.jpg_prob:
            method = sample_discrete(cfg.jpg_method)
            qual = randint(50, 100)  # Random quality factor between 50 and 100
            img = jpeg_from_key(img, qual, method)  # Ensure function returns an image
            if img is None:
                raise ValueError("jpeg_from_key returned None")

    return Image.fromarray(img)  # Convert back to PIL Image


def sample_continuous(s: list):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s: list):
    return s[0] if len(s) == 1 else random.choice(s)


def gaussian_blur(img: np.ndarray, sigma: float):
    img[:, :, 0] = gaussian_filter(img[:, :, 0], sigma=sigma)
    img[:, :, 1] = gaussian_filter(img[:, :, 1], sigma=sigma)
    img[:, :, 2] = gaussian_filter(img[:, :, 2], sigma=sigma)
    return img  # Ensure it returns the modified image


def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img: np.ndarray, compress_val: int):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img: np.ndarray, compress_val: int, key: str) -> np.ndarray:
    method = jpeg_dict[key]
    return method(img, compress_val)  # Ensure it returns an image

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

def custom_resize(img: Image.Image, cfg: CONFIGCLASS) -> Image.Image:
    interp = sample_discrete(cfg.rz_interp)
    return TF.resize(img, cfg.loadSize, interpolation=rz_dict[interp])


def get_dataset(cfg: CONFIGCLASS):
    dset_lst = []
    for dataset in cfg.datasets:
        root = os.path.join(cfg.dataset_root, dataset)
        print("-------> root: ", root)
        dset = dataset_folder(root, cfg)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset: torch.utils.data.ConcatDataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataloader(cfg: CONFIGCLASS):
    shuffle = not cfg.serial_batches if (cfg.isTrain and not cfg.class_bal) else False
    dataset = get_dataset(cfg)
    sampler = get_bal_sampler(dataset) if cfg.class_bal else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
    )
