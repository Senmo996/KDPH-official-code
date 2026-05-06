from torch.utils.data import Dataset
import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize, 
    RandomHorizontalFlip, RandomResizedCrop, ColorJitter,
    RandomGrayscale, RandomApply, RandomAffine,
    RandomPerspective, RandomErasing, InterpolationMode
)
from common.register import registry
from .base import BaseDataset


# ==================== 自定义图像增强 ====================

class RandomGaussianBlur:
    """随机高斯模糊"""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class RandomSolarization:
    """随机曝光效果"""
    def __init__(self, p=0.2, threshold=128):
        self.p = p
        self.threshold = threshold

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img, self.threshold)
        return img


class RandomAdjustSharpness:
    """随机锐度调整"""
    def __init__(self, p=0.3, sharpness_range=(0.5, 2.0)):
        self.p = p
        self.sharpness_range = sharpness_range

    def __call__(self, img):
        if random.random() < self.p:
            enhancer = ImageEnhance.Sharpness(img)
            factor = random.uniform(*self.sharpness_range)
            return enhancer.enhance(factor)
        return img


class RandomAutoContrast:
    """随机自动对比度"""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.autocontrast(img)
        return img


class RandomEqualize:
    """随机直方图均衡化"""
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.equalize(img)
        return img


# ==================== 文本增强策略 ====================

class TextAugmentor:
    """文本数据增强类"""
    
    def __init__(self, p=0.3):
        self.p = p
        
    def random_word_dropout(self, words, p=0.1):
        """随机删除词"""
        if len(words) <= 3:
            return words
        return [w for w in words if random.random() > p]
    
    def random_word_shuffle(self, words):
        """随机打乱相邻词序"""
        if len(words) <= 3:
            return words
        words = words.copy()
        n_swaps = max(1, int(len(words) * 0.15))
        for _ in range(n_swaps):
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return words
    
    def random_word_repeat(self, words):
        """随机重复某个词"""
        if len(words) <= 1:
            return words
        words = words.copy()
        idx = random.randint(0, len(words) - 1)
        words.insert(idx, words[idx])
        return words
    
    def augment(self, text):
        """综合文本增强"""
        if random.random() > self.p:
            return text
        
        words = text.split()
        if len(words) <= 2:
            return text
        
        # 随机选择一种增强方法
        aug_type = random.choice(['dropout', 'shuffle', 'repeat'])
        
        if aug_type == 'dropout':
            words = self.random_word_dropout(words, p=0.15)
        elif aug_type == 'shuffle':
            words = self.random_word_shuffle(words)
        elif aug_type == 'repeat':
            words = self.random_word_repeat(words)
        
        return ' '.join(words) if words else text


# ==================== 主数据集类 ====================

@registry.register_dataset("transformer_dataset")
class Transformer_Dataset(BaseDataset):

    def __init__(self, 
            captions: dict,
            indexs: dict,
            labels: dict,
            is_train=True,
            imageResolution=224,
            tokenizer=None,
            maxWords=77,
            npy=False,
            # 新增增强参数
            aug_level='light',      # 'light', 'medium', 'strong'
            use_text_aug=True,       # 是否使用文本增强
            text_aug_prob=0.3,       # 文本增强概率
            **kwargs):
        
        super().__init__()
        self.captions = captions
        self.indexs = indexs
        self.labels = labels
        self.is_train = is_train
        self.__length = len(self.indexs)
        
        # 文本增强器
        self.use_text_aug = use_text_aug and is_train
        self.text_augmentor = TextAugmentor(p=text_aug_prob) if self.use_text_aug else None
        
        # 构建图像变换
        self.transform = self._build_transform(imageResolution, aug_level)
        
        self.npy = npy
        self.maxWords = maxWords
        self.tokenizer = tokenizer
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>", 
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]", 
            "UNK_TOKEN": "[UNK]", 
            "PAD_TOKEN": "[PAD]"
        }

    def _build_transform(self, imageResolution, aug_level='medium'):
        """构建不同强度的数据增强"""
        
        normalize = Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        )
        
        # 验证/测试时的变换
        if not self.is_train:
            return Compose([
                Resize((imageResolution, imageResolution), 
                       interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                normalize,
            ])
        
        # ===== 训练时的增强策略 =====
        
        if aug_level == 'light':
            # 轻度增强：基础几何变换
            return Compose([
                RandomResizedCrop(
                    imageResolution, 
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=InterpolationMode.BICUBIC
                ),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                normalize,
            ])
        
        elif aug_level == 'medium':
            # 中度增强：几何 + 颜色
            return Compose([
                RandomResizedCrop(
                    imageResolution, 
                    scale=(0.6, 1.0),
                    ratio=(0.75, 1.33),
                    interpolation=InterpolationMode.BICUBIC
                ),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(
                    brightness=0.3, 
                    contrast=0.3, 
                    saturation=0.3, 
                    hue=0.1
                ),
                RandomGrayscale(p=0.1),
                RandomGaussianBlur(p=0.2),
                ToTensor(),
                normalize,
            ])
        
        else:  # strong
            # 强度增强：几何 + 颜色 + 高级效果
            return Compose([
                # 几何变换
                RandomResizedCrop(
                    imageResolution, 
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.33),
                    interpolation=InterpolationMode.BICUBIC
                ),
                RandomHorizontalFlip(p=0.5),
                RandomApply([
                    RandomAffine(
                        degrees=15, 
                        translate=(0.1, 0.1), 
                        scale=(0.9, 1.1),
                        shear=5
                    )
                ], p=0.3),
                RandomApply([
                    RandomPerspective(distortion_scale=0.2)
                ], p=0.2),
                
                # 颜色变换
                ColorJitter(
                    brightness=0.4, 
                    contrast=0.4, 
                    saturation=0.4, 
                    hue=0.15
                ),
                RandomGrayscale(p=0.2),
                
                # 高级效果
                RandomGaussianBlur(p=0.3),
                RandomSolarization(p=0.1),
                RandomAdjustSharpness(p=0.2),
                RandomAutoContrast(p=0.1),
                
                ToTensor(),
                normalize,
                
                # 随机擦除（在tensor上操作）
                RandomErasing(p=0.15, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])

    def __len__(self):
        return self.__length

    def _load_image(self, index: int) -> torch.Tensor:
        if not self.npy:
            image_path = self.indexs[index].strip().replace('/images', '/images')
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        else:
            image = self.transform(Image.fromarray(self.indexs[index], mode="RGB"))

        return image

    def _load_text(self, index: int):
        captions = self.captions[index]
        
        if self.tokenizer is not None:
            # 随机选择一个caption
            use_cap = captions[random.randint(0, len(captions) - 1)]
            
            # 文本增强（仅训练时）
            if self.use_text_aug and self.text_augmentor is not None:
                use_cap = self.text_augmentor.augment(use_cap)
            
            words = self.tokenizer.tokenize(use_cap)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.maxWords - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            caption = self.tokenizer.convert_tokens_to_ids(words)

            while len(caption) < self.maxWords:
                caption.append(0)

        caption = torch.tensor(caption)
        key_padding_mask = (caption == 0)
        return caption, key_padding_mask
    
    def _load_label(self, index: int) -> torch.Tensor:
        label = self.labels[index]
        label = torch.from_numpy(label)
        return label

    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.float32)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels

    # 返回格式保持不变！
    def __getitem__(self, index):
        image = self._load_image(index)
        caption, key_padding_mask = self._load_text(index)
        label = self._load_label(index)

        return image, caption, key_padding_mask, label, index