
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    [batch_size, 1, 1, 1]
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

# register_buffer 是 PyTorch 中 torch.nn.Module 提供的一个方法, 允许用户将某些张量注册为模块的一部分,
# 但不会被视为可训练参数. 这些张量会随模型保存和加载, 但在反向传播过程中不会更新
class GaussianDiffusionTrainer(nn.Module):
    # 1e-4, 0.02, 1000
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        # 对输入张量沿指定维度进行逐元素累积乘积操作, 返回一个与输入张量大小相同的张量, 其中每个元素的值等于其在输入张量及之前所有元素的乘积
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    # 训练的时候并没有执行 1000 步, 而是随机执行 1000 步中的某一步, 然后计算和噪声的误差
    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # 均方误差损失: MSE Loss = (1/n) * Σ(y_pred - y_actual)^2
        # reduction='none'：求所有对应位置的差的平方，返回的仍然是一个和原来形状一样的矩阵。
        # reduction='mean'：求所有对应位置差的平方的均值，返回的是一个标量。
        # reduction='sum'：求所有对应位置差的平方的和，返回的是一个标量。
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        # self.posterior_var[1:2] 是一个固定值, 放在这里的意义是什么 ???
        # var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = self.posterior_var
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

class GaussianDDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, sample_step = 20):
        super().__init__()

        self.model = model
        self.T = T
        self.sample_step = sample_step

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [self.sample_step, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(alphas_bar_prev) / torch.sqrt(alphas_bar))
        self.register_buffer('coeff2', torch.sqrt(1. - alphas_bar_prev) - self.coeff1 * torch.sqrt(1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t +
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean(self, x_t, t):
        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(0, self.T, self.sample_step)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean = self.p_mean(x_t=x_t, t=t)
            x_t = mean
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
