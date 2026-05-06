import os
from tqdm import tqdm
import torch
import scipy.io as scio
from common.calc_utils import calc_map_k
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from dataset.builder import build_dataloader
from utils.logger import get_color_logger
from common.register import registry

# @registry.register_runner("BaseTrainer")
class BaseTrainer():

    def __init__(self, 
                cfg, 
                is_train=True, 
                device=None, 
                world_size=torch.cuda.device_count(), 
                output_dim=16, 
                train_num=10000,
                query_num=5000,
                epochs=100, 
                save_dir="./result", 
                display_step=20,
                top_k=5000,
                model_state="",
                batch_size=128,
                distributed=False, 
                # used to open the softmax hash method.
                **kwags) -> None:
        
        self.cfg = cfg

        self.is_train = is_train
        if distributed:
            # world_size=2
            # self.gpus = self.cfg.run.device
            self.logger = get_color_logger(cfg.run.log_dir, cfg.dataset.name + "-" + str(device), display=(device==0))
            self._init_distribution(rank=device, world_size=world_size)
            # self.logger.info(device==self.gpus[0])
            # self.logger.info("distribution inited!")
        else:
            self.logger = get_color_logger(cfg.run.log_dir, cfg.dataset.name + "-" + str(device))
        # self.logger.info(f"display: {device==self.gpus[0]} {device} {self.gpus}")
        self.logger.info(f"parameters: {cfg}")
        self.device = device
        self.output_dim = output_dim
        self.train_num = train_num
        self.query_num = query_num
        self.epochs = epochs
        self.display_step = display_step
        self.top_k=top_k
        self.model_state = model_state
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.distributed = True if distributed else False
        os.makedirs(save_dir, exist_ok=True)

        # Initialize wandb
        if not self.distributed or (self.distributed and self.device == 0):
            try:
                # Convert config to dict for wandb
                config_dict = OmegaConf.to_container(cfg, resolve=True)
                
                wandb.init(
                    project="clip-hash",
                    name=f"{cfg.dataset.name}-{cfg.model.arch}",
                    dir=save_dir,
                    config=config_dict,
                    reinit=True
                )
                self.wandb_enabled = True
                self.logger.info("W&B initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
                self.wandb_enabled = False
        else:
            self.wandb_enabled = False

        # used to display the middle step.
        self.global_step = 0

        # used to record the best result.
        self.max_mapi2t = 0
        self.max_mapt2i = 0
        self.best_epoch_i = 0
        self.best_epoch_t = 0

        self.calc_map_k = calc_map_k
        self.distributed = distributed
        self.world_size = world_size
    
    def _init_distribution(self, rank=0, world_size=4):
        self.rank = rank
        self.world_size = world_size
        self.logger.info("Initializing distributed")
        os.environ['MASTER_ADDR'] = self.cfg.run.distributed_addr
        os.environ['MASTER_PORT'] = str(self.cfg.run.distributed_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    def build_model(self, cfg_model, output_dim=16, **kwags):

        arch = cfg_model.get("arch", "DCMHT")
        # print(arch)
        self.model = registry.get_model_class(arch).from_config(cfg_model, output_dim=output_dim, train_num=self.train_num)
        if os.path.isfile(self.model_state):
            self.logger.info("loading model...")
            self.model.load_state_dict(torch.load(self.model_state, map_location=f"cuda:{self.device}"))
        self.model.float()
        self.model.to(self.device)

        if self.distributed:
            self.logger.info("use distribution mode.")
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model_ddp = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=True)
        else:
            self.model_ddp = None

        self.logger.info("Building model!")
        # print(self.model)
        self.logger.info(f"Output dim: {self.output_dim}")
    
    def build_optimizer(self, cfg_optimizer, parameters=None):

        lr_schedu = None
        arch = cfg_optimizer.get("arch", "BertAdam")
        backbone_lr = cfg_optimizer.get("clip_lr", 0.00001)
        lr = cfg_optimizer.get("lr", 0.001)
        warmup_proportion = cfg_optimizer.get("warmup_proportion", 0.1)
        schedule = cfg_optimizer.get("schedule", "warmup_cosine")
        b1 = cfg_optimizer.get("b1", 0.9)
        b2 = cfg_optimizer.get("b2", 0.98) 
        e = cfg_optimizer.get("e", 0.000001) 
        max_grad_norm = cfg_optimizer.get("max_grad_norm", 1.0)  
        weight_decay = cfg_optimizer.get("weight_decay",  0.2)
        self.logger.info(f"weight_decay: {weight_decay}")

        if parameters is None:
            parameters = [{'params': self.model.backbone.parameters(), 'lr': backbone_lr},
                        {'params': self.model.hash.parameters(), 'lr': lr}]
            
        optimizer = registry.get_optimizer_class(arch)(parameters, lr=lr, warmup=warmup_proportion, 
                                                            schedule=schedule, b1=b1, b2=b2, e=e, t_total=len(self.train_loader) * self.epochs,
                                                            weight_decay=weight_decay, max_grad_norm=max_grad_norm)
        self.logger.info("Building optimizer!")
        return optimizer, lr_schedu
        
    
    def build_dataset(self, cfg, train_num=10000, query_num=5000, batch_size=128, num_workers=4, pin_memory=True, shuffle=True):
        dataname = cfg.get("name", "mirflickr25k")
        path = cfg.get("path", "./data")
        self.logger.info(f"Using {dataname} dataset.")
        image_file = os.path.join(path, dataname, cfg.get("img_file", "index.mat"))
        text_file = os.path.join(path, dataname, cfg.get("txt_file", "caption.mat"))
        label_file = os.path.join(path, dataname, cfg.get("label_file", "caption.mat"))
        max_word = cfg.get("max_word", 32)
        image_resolution = cfg.get("image_resolution", 224)
        dataset_cls = cfg.get("arch", "transformer_dataset")

        train_data, query_data, retrieval_data = build_dataloader(
            captionFile=text_file, indexFile=image_file, labelFile=label_file, imageResolution=image_resolution, 
            maxWords=max_word, query_num=query_num, train_num=train_num, dataset_cls=dataset_cls, tokenizer=registry.get_tokenizer_class(cfg.get("tokenizer_arch", "clip_tokenizer"))()
        )
        self.build_loader(train_data=train_data, query_data=query_data, retrieval_data=retrieval_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
    
    def build_loader(self, train_data, query_data, retrieval_data, batch_size, num_workers, pin_memory, shuffle, drop_last=False):

        self.train_labels = train_data.get_all_label()
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"train shape: {self.train_labels.shape}")
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

        if self.distributed:
            train_data_sampler = DistributedSampler(
                dataset=train_data,
                rank=self.rank,
                num_replicas=self.world_size,
                shuffle=True
            )
            batch_size = batch_size // self.world_size
        else:
            train_data_sampler = None

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False
        )
    
    def run(self):
        if self.is_train:
            self.train()
        else:
            self.test()

    def generate_hash(self, image, text, key_padding_mask=None):
        image_hash = self.model.encode_image(image)
        text_hash = self.model.encode_text(text)

        return image_hash, text_hash

    def get_code(self, data_loader, length: int):
        self.change_state(mode="valid")
        img_buffer = torch.empty(length, self.output_dim, dtype=torch.float).to(self.device)
        text_buffer = torch.empty(length, self.output_dim, dtype=torch.float).to(self.device)

        for image, text, key_padding_mask, label, index in tqdm(data_loader):
            image = image.to(self.device, non_blocking=True)
            text = text.to(self.device, non_blocking=True)
            index = index.numpy()
            image_hash, text_hash = self.generate_hash(image=image, text=text, key_padding_mask=key_padding_mask)
            
            # Handle different return types from generate_hash
            def process_hash(hash_value):
                # Handle tuple return values (e.g., from HPHNet)
                if isinstance(hash_value, tuple):
                    hash_value = hash_value[0]
                
                # Check if already binary (contains only -1, 0, 1)
                if torch.all(torch.abs(hash_value) <= 1.0 + 1e-6):
                    return hash_value  # Already binary
                else:
                    return self.make_hash_code(hash_value.data)  # Needs binarization
            
            img_buffer[index, :] = process_hash(image_hash)
            text_buffer[index, :] = process_hash(text_hash)
         
        return img_buffer, text_buffer
    
    def validate_hash_codes(self, img_buffer, text_buffer):
        """Validate that hash codes are properly binary (-1, 0, 1)."""
        def check_binary(tensor, name):
            unique_vals = torch.unique(tensor)
            if not torch.all(torch.abs(unique_vals) <= 1.0 + 1e-6):
                self.logger.warning(f"Invalid {name} hash values detected: {unique_vals}")
                return False
            return True
        
        img_valid = check_binary(img_buffer, "image")
        txt_valid = check_binary(text_buffer, "text")
        
        if img_valid and txt_valid:
            self.logger.info("Hash codes validated successfully")
        else:
            self.logger.warning("Hash code validation failed - this may affect mAP calculation")
        
        return img_valid and txt_valid
    
    def change_state(self, mode):
        """
        This method need to be rewrote if the self.model is not the single model, splited as the image_model and the text model.
        """
        if self.model_ddp is None:
            if mode == "train":
                self.model.train()
                self.model.unfreezen()
            else:
                self.model.eval()
                self.model.freezen()
        else:
            if mode == "train":
                self.model_ddp.train()
                self.model.train()
            else:
                self.model_ddp.eval()
                self.model.eval()
    
    def train(self):

        for epoch in range(self.epochs):
            self.train_epoch(epoch=epoch)
            self.valid(epoch, k=self.top_k)
        
        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")
        
        # Close wandb
        if self.wandb_enabled:
            try:
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish W&B: {e}")
    
    def train_epoch(self, epoch: int):
        raise NotImplementedError()

    def compute_loss(self, image_embed, text_embed, label, index, epoch=0, times=0, global_step=0, **kwags):
        """
        This method work with computing loss and displaying the loss result.

        To adapt the projection method, it is separated from the train_epoch code.
        """
        raise NotImplementedError()
    
    def calc_map_k_matrix_fast(self, qB: torch.Tensor, rB: torch.Tensor, query_L: torch.Tensor, retrieval_L: torch.Tensor, k=None, rank=1):
        """
        计算 mAP@k 的高效向量化版本。
        - qB: 查询二进制码 (num_query, bit_length)
        - rB: 检索库二进制码 (num_retrieval, bit_length)
        - query_L: 查询标签 (num_query, num_classes)
        - retrieval_L: 检索库标签 (num_retrieval, num_classes)
        - k: Top-k
        """
        num_query = query_L.shape[0]
        num_retrieval = retrieval_L.shape[0]
        if k is None:
            k = num_retrieval

        # 1. 批量计算 Ground Truth 矩阵
        # gnd[i, j] = 1 表示查询 i 与检索 j 相关，否则为 0
        gnd = (query_L @ retrieval_L.t() > 0).float()

        # 2. 批量计算汉明距离
        # 假设二进制码为{-1, 1}, bit_length 为码长
        # Hamming_distance = 0.5 * (bit_length - qB @ rB.T)
        bit_length = qB.shape[1]
        hamm_dist = 0.5 * (bit_length - qB @ rB.t())

        # 3. 批量排序，获取 Top-k 索引
        # 对每个查询的距离进行排序
        _, ind = torch.sort(hamm_dist, dim=1)
        ind = ind[:, :k] # 只取前 k 个

        # 4. 根据排序后的索引，获取对应的 Ground Truth
        # 使用 torch.gather 高效地索引
        gnd_sorted = torch.gather(gnd, 1, ind)

        # 5. 批量计算 Average Precision (AP)
        # tsum 是每个查询相关的总样本数
        tsum = torch.sum(gnd, dim=1)
        # 避免除以 0 的情况
        tsum[tsum == 0] = 1e-6 

        # 计算每个位置的精度
        # cumsum 是排序后相关样本的累积数量
        precision_at_k = gnd_sorted.cumsum(dim=1) / (torch.arange(k, device=gnd.device) + 1)

        # AP 是所有相关位置精度的平均值
        # (precision_at_k * gnd_sorted) 只保留相关位置的精度
        ap = (precision_at_k * gnd_sorted).sum(dim=1) / tsum

        # mAP 是所有查询 AP 的平均值
        return ap.mean()
    
    def valid(self, epoch, k=None):
        assert self.query_loader is not None and self.retrieval_loader is not None
        save_dir = os.path.join(self.save_dir, "mat_files")
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info("Valid.")
        # self.change_state(mode="valid")

        with torch.no_grad():
            query_img, query_txt = self.get_code(self.query_loader, self.query_num)
            retrieval_img, retrieval_txt = self.get_code(self.retrieval_loader, self.retrieval_num)
        device = query_img.device
        query_labels_on_device = self.query_labels.float().to(device)
        retrieval_labels_on_device = self.retrieval_labels.float().to(device)
        mAPi2t = self.calc_map_k(query_img, retrieval_txt, query_labels_on_device, retrieval_labels_on_device, k)
        # print("map map")
        mAPt2i = self.calc_map_k(query_txt, retrieval_img, query_labels_on_device, retrieval_labels_on_device, k)
        mAPi2i = self.calc_map_k(query_img, retrieval_img, query_labels_on_device, retrieval_labels_on_device, k)
        mAPt2t = self.calc_map_k(query_txt, retrieval_txt, query_labels_on_device,retrieval_labels_on_device, k)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            if not self.distributed or (self.distributed and self.device == 0):
                self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "i2t-best.mat"))
                self.save_model(save_dir=self.save_dir, epoch=epoch)
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            if not self.distributed or (self.distributed and self.device == 0):
                self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "t2i-best.mat"))
                self.save_model(save_dir=self.save_dir, epoch=epoch)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)

        if not self.distributed or (self.distributed and self.device == 0):
            self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "last.mat"))

        if hasattr(self.model, 'hyp') and hasattr(self.model.hyp, 'save_proxies') and callable(self.model.hyp.save_proxies):
            self.model.hyp.save_proxies(os.path.join(save_dir, "proxies.t"))
            
        self.logger.info(f">>>>>> [{epoch}/{self.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, "                    f"MAX MAP(i->t): {self.max_mapi2t}, epoch: {self.best_epoch_i}, MAX MAP(t->i): {self.max_mapt2i}, epoch: {self.best_epoch_t}")
        
        # Log evaluation metrics to wandb
        if self.wandb_enabled:
            try:
                eval_metrics = {
                    "evaluation/MAP_i2t": mAPi2t,
                    "evaluation/MAP_t2i": mAPt2i,
                    "evaluation/MAP_t2t": mAPt2t,
                    "evaluation/MAP_i2i": mAPi2i,
                    "evaluation/MAX_MAP_i2t": self.max_mapi2t,
                    "evaluation/MAX_MAP_t2i": self.max_mapt2i
                }
                wandb.log(eval_metrics, step=self.global_step)
            except Exception as e:
                self.logger.warning(f"Failed to log evaluation metrics to W&B: {e}")
    
    def test(self):
        assert not self.model_state == "", "test step must provide the model file!"
        self.logger.info("Test.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.save_dir, "mat_files")
        os.makedirs(save_dir, exist_ok=True)

        query_img, query_txt = self.get_code(self.query_loader, self.query_num)
        retrieval_img, retrieval_txt = self.get_code(self.retrieval_loader, self.retrieval_num)

        mAPi2t = self.calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, self.top_k)
        # print("map map")
        mAPt2i = self.calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, self.top_k)
        mAPi2i = self.calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, self.top_k)
        mAPt2t = self.calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, self.top_k)
        self.save_mat(query_img, query_txt, self.query_labels, retrieval_img, retrieval_txt, self.retrieval_labels, save_file=os.path.join(save_dir, "test.mat"))
        self.logger.info(f">>>>>> TEST, MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")
    
    def print_loss_dict(self, loss_dict, bits=16, epoch=0, times=0):

        print_str = f">>>>>> Display ({self.loss_type} loss-{bits}) >>>>>> [{epoch}/{self.epochs}], [{times}/{len(self.train_loader)}]: "

        def leaf_str(dict_, key, print_str):
            
            print_str += f"{key}: "
            if isinstance(dict_[key], dict):
                for k in dict_[key]:
                    print_str = leaf_str(dict_[key], k, print_str)
            else:
                print_str += f"{dict_[key]}, "
            return print_str
        
        for key in loss_dict.keys():
            print_str += leaf_str(loss_dict, key=key, print_str="")
        
        print_str += f"lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}"
        self.logger.info(print_str)
        
        # Log losses to wandb
        if self.wandb_enabled:
            try:
                def log_losses_to_wandb(dict_, prefix=""):
                    log_dict = {}
                    for key, value in dict_.items():
                        if isinstance(value, dict):
                            nested_dict = log_losses_to_wandb(value, f"{prefix}{key}/")
                            log_dict.update(nested_dict)
                        else:
                            log_dict[f"{prefix}{key}"] = value
                    return log_dict
                
                loss_log_dict = log_losses_to_wandb(loss_dict)
                
                # Log learning rates
                lrs = sorted(list(set(self.optimizer.get_lr())))
                for i, lr in enumerate(lrs):
                    loss_log_dict[f"learning_rate/lr_{i}"] = lr
                
                wandb.log(loss_log_dict, step=self.global_step)
            except Exception as e:
                self.logger.warning(f"Failed to log losses to W&B: {e}")
    
    def save_model(self, save_dir, epoch, other=""):
        """
        I haven't write the load_state() code. It will be added in the furture. 2023/10/19
        """
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model-" + other + str(epoch) + ".pth"))
        self.logger.info("save mode to {}".format(os.path.join(save_dir, "model-" + other + str(epoch) + ".pth")))
    
    @classmethod
    def save_mat(cls, query_img, query_txt, query_labels, retrieval_img, retrieval_txt, retrieval_labels, save_file="i2t"):

        if isinstance(query_img, torch.Tensor):
            query_img = query_img.cpu().detach().numpy()
            query_txt = query_txt.cpu().detach().numpy()
            retrieval_img = retrieval_img.cpu().detach().numpy()
            retrieval_txt = retrieval_txt.cpu().detach().numpy()
            query_labels = query_labels.numpy()
            retrieval_labels = retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_file), result_dict)
    
    @classmethod
    def make_hash_code(cls, code):
        # print("aaaaaaaa")
        return code.sign_()
    
    @classmethod
    def from_config(cls, cfg, logger=None):
        raise NotImplementedError()

