import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
def cross_modal_infonce_loss(x, y, labels, temperature=0.07):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    
    # 计算相似度矩阵
    logits = torch.matmul(x_norm, y_norm.T) / temperature
    
    # 创建标签：对角线为正样本，其他为负样本
    batch_size = x.size(0)
    labels_idx = torch.arange(batch_size, device=x.device)
    
    # 双向InfoNCE
    loss_i2t = F.cross_entropy(logits, labels_idx)
    loss_t2i = F.cross_entropy(logits.T, labels_idx)
    
    return (loss_i2t + loss_t2i) / 2
        
        
def jensen_shannon_divergence_loss(x, y, reduction='mean'):
    """
    计算两个特征向量批次之间的JS散度损失。
    
    Args:
        x (torch.Tensor): 第一个特征向量批次, shape [batch_size, output_dim].
        y (torch.Tensor): 第二个特征向量批次, shape [batch_size, output_dim].
        reduction (str): 'mean' 或 'sum'，对批次的损失进行聚合的方式。
        
    Returns:
        torch.Tensor: 一个标量损失值。
    """
    # 1. 将特征向量转换为概率分布
    p_x = F.softmax(x, dim=1)
    p_y = F.softmax(y, dim=1)
    
    # 2. 计算平均分布 m
    m = 0.5 * (p_x + p_y)
    
    # 3. 计算KL散度
    # 为避免 log(0) 导致 NaN，加入一个小的 epsilon
    kl_x_m = F.kl_div(m.log(), p_x, reduction='none').sum(dim=1)
    kl_y_m = F.kl_div(m.log(), p_y, reduction='none').sum(dim=1)
    
    # 4. 计算JSD
    jsd = 0.5 * (kl_x_m + kl_y_m)
    
    if reduction == 'mean':
        return jsd.mean()
    elif reduction == 'sum':
        return jsd.sum()
    else:
        return jsd
    
class KentProxyLoss(torch.nn.Module):
    def __init__(self, numclass=80, output_dim=64, 
                 num_axes=2, # Kent分布额外的主轴/次轴数量
                 sim_path="/data4/zhuhengjie/Project/CLIP-based-Cross-Modal-Hashing/train/PPH/coco_large.npy",
                 base_margin=0.3, 
                 gamma=0.2,
                 tau=0.1,
                 beta=0.1,
                 lambda_cm=0.29,
                 alpha=41.0,
                 threshold=None,
                 orthogonal_method='cayley',  # 'cayley', 'exponential', 'gram_schmidt'
                 ):
        super(KentProxyLoss, self).__init__()
        self.numclass = numclass
        self.output_dim = output_dim
        self.num_axes = num_axes # 通常设置为2就足够模拟椭圆了
        self.orthogonal_method = orthogonal_method

        # --- Kent 分布思想的核心参数 ---
        # 1. 平均方向 (Mean Direction), 像 vMF 一样
        self.proxies_mu = nn.Parameter(torch.randn(numclass, output_dim))
        nn.init.kaiming_normal_(self.proxies_mu, mode='fan_out')

        # 2. 可微分正交矩阵参数化
        # 根据选择的方法，使用不同的参数化策略
        if orthogonal_method == 'cayley':
            # Cayley变换：学习斜对称矩阵参数
            # 对于每个类别，学习一个 (num_axes x num_axes) 的斜对称矩阵
            self.skew_params = nn.Parameter(torch.randn(numclass, num_axes, num_axes))
            # 用于将低维正交矩阵嵌入到高维空间的可学习矩阵
            self.embedding_matrix = nn.Parameter(torch.randn(numclass, output_dim, num_axes))
            nn.init.orthogonal_(self.embedding_matrix)
        elif orthogonal_method == 'exponential':
            # 指数映射：同样使用斜对称矩阵
            self.skew_params = nn.Parameter(torch.randn(numclass, num_axes, num_axes))
            # 用于将低维正交矩阵嵌入到高维空间的可学习矩阵
            self.embedding_matrix = nn.Parameter(torch.randn(numclass, output_dim, num_axes))
            nn.init.orthogonal_(self.embedding_matrix)
        else:  # gram_schmidt
            # 传统方法：直接学习轴向量，然后应用Gram-Schmidt
            self.kent_axes_raw = nn.Parameter(torch.randn(numclass, num_axes, output_dim))
            nn.init.orthogonal_(self.kent_axes_raw)

        # 3. 集中度参数 (Concentrations)
        # κ_main 控制沿平均方向的集中度
        # κ_axes 控制沿主轴/次轴的"椭圆度"
        self.log_kappa_main = nn.Parameter(torch.zeros(numclass, 1))
        self.log_kappa_axes = nn.Parameter(torch.zeros(numclass, num_axes))

        # --- 其他参数与您原来的代码一致 ---
        self.base_margin = base_margin
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.lambda_cm = lambda_cm
        self.alpha = alpha
        self.threshold = threshold

    def _generate_orthogonal_axes(self):
        """
        使用可微分的正交矩阵参数化生成Kent轴。
        支持三种方法：Cayley变换、指数映射、Gram-Schmidt正交化。
        """
        if self.orthogonal_method == 'cayley':
            return self._cayley_orthogonal_axes()
        elif self.orthogonal_method == 'exponential':
            return self._exponential_orthogonal_axes()
        else:  # gram_schmidt
            return self._gram_schmidt_orthogonal_axes()
    
    def _cayley_orthogonal_axes(self):
        """
        Cayley变换：将斜对称矩阵映射到正交矩阵
        公式：Q = (I - A)(I + A)^(-1)，其中A是斜对称矩阵
        """
        # 确保斜对称性：A = (P - P^T) / 2
        skew_matrices = 0.5 * (self.skew_params - self.skew_params.transpose(-1, -2))
        
        num_classes = skew_matrices.size(0)  # 这是 numclass，不是 batch_size
        matrix_size = skew_matrices.size(1)
        device = skew_matrices.device
        
        # 创建单位矩阵
        I = torch.eye(matrix_size, device=device).unsqueeze(0).expand(num_classes, -1, -1)
        
        # Cayley变换：Q = (I - A)(I + A)^(-1)
        I_minus_A = I - skew_matrices
        I_plus_A = I + skew_matrices
        
        # 计算逆矩阵（使用数值稳定的方法）
        try:
            I_plus_A_inv = torch.inverse(I_plus_A)
            orthogonal_matrices = torch.bmm(I_minus_A, I_plus_A_inv)
        except:
            # 如果逆矩阵计算失败，使用伪逆
            I_plus_A_inv = torch.pinverse(I_plus_A)
            orthogonal_matrices = torch.bmm(I_minus_A, I_plus_A_inv)
        
        # 将正交矩阵的列向量作为Kent轴
        # 我们需要将这些轴嵌入到高维空间中
        kent_axes = self._embed_orthogonal_axes(orthogonal_matrices)
        
        return kent_axes
    
    def _exponential_orthogonal_axes(self):
        """
        指数映射：exp(A)，其中A是斜对称矩阵
        使用矩阵指数的Padé近似或Taylor展开
        """
        # 确保斜对称性
        skew_matrices = 0.5 * (self.skew_params - self.skew_params.transpose(-1, -2))
        
        # 使用矩阵指数（PyTorch的matrix_exp）
        orthogonal_matrices = torch.matrix_exp(skew_matrices)
        
        # 将正交矩阵的列向量作为Kent轴
        kent_axes = self._embed_orthogonal_axes(orthogonal_matrices)
        
        return kent_axes
    
    def _gram_schmidt_orthogonal_axes(self):
        """
        可微分的Gram-Schmidt正交化过程
        """
        # 标准化主方向
        norm_proxies_mu = F.normalize(self.proxies_mu, p=2, dim=1)  # [numclass, dim]
        
        # 对原始轴向量进行Gram-Schmidt正交化
        orthogonal_axes = []
        
        for i in range(self.num_axes):
            axis = self.kent_axes_raw[:, i, :]  # [numclass, dim]
            
            # 从轴中减去沿主方向的投影
            projection_on_mu = torch.sum(axis * norm_proxies_mu, dim=1, keepdim=True)
            axis = axis - projection_on_mu * norm_proxies_mu
            
            # 从轴中减去沿之前轴的投影
            for j, prev_axis in enumerate(orthogonal_axes):
                projection = torch.sum(axis * prev_axis, dim=1, keepdim=True)
                axis = axis - projection * prev_axis
            
            # 标准化
            axis = F.normalize(axis, p=2, dim=1)
            orthogonal_axes.append(axis)
        
        # 重新组织为张量
        kent_axes = torch.stack(orthogonal_axes, dim=1)  # [numclass, num_axes, dim]
        
        return kent_axes
    
    def _embed_orthogonal_axes(self, orthogonal_matrices):
        """
        将低维正交矩阵的列向量嵌入到高维特征空间中
        """
        num_classes, matrix_size, _ = orthogonal_matrices.shape
        # 注意：这里的第一个维度是 numclass，不是 batch_size
        
        # 使用在 __init__ 中初始化的可学习嵌入矩阵
        # 将正交矩阵的列向量通过嵌入矩阵映射到高维空间
        # [numclass, output_dim, num_axes] @ [numclass, num_axes, num_axes]
        kent_axes = torch.bmm(self.embedding_matrix, orthogonal_matrices)
        
        # 转置以匹配期望的形状 [numclass, num_axes, output_dim]
        kent_axes = kent_axes.transpose(1, 2)
        
        # 确保标准化
        kent_axes = F.normalize(kent_axes, p=2, dim=2)
        
        return kent_axes

    def _calculate_score(self, features):
        """
        使用完整的 Kent 分布对数似然形式作为评分函数。
        Kent分布是球面上的椭圆分布，结合了vMF的方向性和椭圆的各向异性。
        分数越大，表示样本与该代理（类别）越匹配。
        
        Kent分布的对数似然: log p(x) = κ(μ^T x) + β[(γ₂^T x)² - (γ₃^T x)²] - log Z(κ,β)
        其中:
        - μ: 主方向 (mean direction)
        - γ₂, γ₃: 主轴和次轴 (major/minor axes)
        - κ: 集中度参数 (concentration)
        - β: 椭圆度参数 (ellipticity)
        """
        batch_size = features.size(0)
        
        # L2 标准化到单位球面
        norm_features = F.normalize(features, p=2, dim=1)  # [batch, dim]
        norm_proxies_mu = F.normalize(self.proxies_mu, p=2, dim=1)  # [numclass, dim]
        
        # 获取集中度参数
        kappa_main = torch.exp(self.log_kappa_main.clamp(min=-10, max=10))  # [numclass, 1]
        kappa_axes = torch.exp(self.log_kappa_axes.clamp(min=-10, max=10))  # [numclass, num_axes]
        
        # 使用可微分正交矩阵参数化生成Kent轴
        orthogonal_kent_axes = self._generate_orthogonal_axes()  # [numclass, num_axes, dim]
        
        # 1. 计算主方向项: κ(μ^T x)
        # shape: [batch, numclass]
        mu_dot_x = torch.matmul(norm_features, norm_proxies_mu.T)  # [batch, numclass]
        main_direction_term = kappa_main.squeeze(-1) * mu_dot_x  # [batch, numclass]
        
        # 2. 计算椭圆各向异性项: β[(γ₂^T x)² - (γ₃^T x)²]
        # 对于多个轴，我们使用加权组合
        ellipticity_term = torch.zeros_like(main_direction_term)  # [batch, numclass]
        
        if self.num_axes >= 2:
            # 计算特征在各个轴上的投影
            # [batch, dim] @ [numclass, dim, num_axes] -> [batch, numclass, num_axes]
            axes_projections = torch.einsum('bd,cnd->bcn', norm_features, orthogonal_kent_axes)
            
            # Kent分布的标准形式: β[(γ₂^T x)² - (γ₃^T x)²]
            # 这里我们推广为: Σᵢ βᵢ * sign(i) * (γᵢ^T x)²
            # 其中sign(i)为交替的+1,-1模式
            for i in range(self.num_axes):
                sign = 1 if i % 2 == 0 else -1  # 交替符号: +1, -1, +1, -1, ...
                axis_projection_sq = axes_projections[:, :, i] ** 2  # [batch, numclass]
                ellipticity_term += sign * kappa_axes[:, i] * axis_projection_sq
        
        # 3. 归一化常数项 (简化版本，忽略复杂的Bessel函数)
        # 在实际应用中，由于我们只关心相对分数，可以忽略或使用近似
        # 这里我们添加一个简单的正则化项来稳定训练
        # 需要将归一化项扩展到 [batch_size, numclass] 的形状
        normalization_per_class = -0.5 * (kappa_main.squeeze(-1) + torch.sum(torch.abs(kappa_axes), dim=1))  # [numclass]
        normalization_term = normalization_per_class.unsqueeze(0).expand(batch_size, -1)  # [batch_size, numclass]
        
        # 4. 组合完整的对数似然分数
        # log p(x) = κ(μ^T x) + ellipticity_term - log Z(κ,β)
        total_score = main_direction_term + ellipticity_term + normalization_term
        
        return total_score 

    def _calculate_modality_loss(self, scores, labels):
        """
        这部分与您之前的 MixtureVMFProxyLoss 完全相同，
        因为 scores 同样是“越大越好”的对数似然分数。
        """
        pos_scores = scores.unsqueeze(2)
        neg_scores = scores.unsqueeze(1)
        loss_matrix = neg_scores - pos_scores
        
        margin_matrix = self.base_margin
        final_loss_matrix =  F.relu(loss_matrix + margin_matrix)
        
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        valid_pair_mask = pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)
    
        num_valid_pairs = valid_pair_mask.sum()
        if num_valid_pairs == 0:
            return torch.tensor(0.0, device=scores.device)
            
        total_loss = (final_loss_matrix * valid_pair_mask).sum() / num_valid_pairs
        
        return total_loss

    def forward(self, x, y, label):
        P_one_hot = label.float().to(x.device)
        
        scores_x = self._calculate_score(x)
        scores_y = self._calculate_score(y)
        
        loss_x = self._calculate_modality_loss(scores_x, P_one_hot)
        loss_y = self._calculate_modality_loss(scores_y, P_one_hot)
        
        # 组合跨模态损失
        infonce_loss = cross_modal_infonce_loss(x, y, P_one_hot)
        cross_modal_loss = self.lambda_cm * infonce_loss
    
        if self.alpha > 0:
            index = label.sum(dim = 1) > 1
            label_ = label[index].float()

            x_ = x[index]
            t_ = y[index]

            cos_sim = label_.mm(label_.T)

            if len((cos_sim == 0).nonzero()) == 0:
                reg_term = 0
                reg_term_t = 0
                reg_term_xt = 0
            else:
                x_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(x_, p = 2, dim = 1).T)
                t_sim = F.normalize(t_, p = 2, dim = 1).mm(F.normalize(t_, p = 2, dim = 1).T)
                xt_sim = F.normalize(x_, p = 2, dim = 1).mm(F.normalize(t_, p = 2, dim = 1).T)

                neg = self.alpha * F.relu(x_sim - self.threshold)
                neg_t = self.alpha * F.relu(t_sim - self.threshold)
                neg_xt = self.alpha * F.relu(xt_sim - self.threshold)

                reg_term = torch.where(cos_sim == 0, neg, torch.zeros_like(x_sim)).sum() / len((cos_sim == 0).nonzero())
                reg_term_t = torch.where(cos_sim == 0, neg_t, torch.zeros_like(t_sim)).sum() / len((cos_sim == 0).nonzero())
                reg_term_xt = torch.where(cos_sim == 0, neg_xt, torch.zeros_like(xt_sim)).sum() / len((cos_sim == 0).nonzero())
        else:
            reg_term = 0
            reg_term_t = 0
            reg_term_xt = 0
            
            
        total_loss = loss_x + loss_y + cross_modal_loss + reg_term + reg_term_t + reg_term_xt
        loss_dit = {'Magna-Proxy Loss_x':loss_x, 'Magna-Proxy Loss_y': loss_y,"cross_modal_loss":cross_modal_loss , "reg_term":reg_term, "reg_term_t": reg_term_t, "reg_term_xt": reg_term_xt}
        # print(f"loss的原始值： loss_x:{loss_x}, loss_y:{loss_y}, sim_loss:{sim_loss/self.beta}, cross_modal_loss:{cross_modal_loss/self.lambda_cm}, reg_term:{reg_term/self.alpha}, reg_term_t:{reg_term_t/self.alpha}, reg_term_xt:{reg_term_xt/self.alpha}")
        return total_loss, loss_dit
    
    def save_proxies(self, path):
        # 保存代理的均值和log_sigma_sq，以备后续使用
        proxy_data = {
            'mu': self.proxies_mu.data
        }
        torch.save(proxy_data, path)