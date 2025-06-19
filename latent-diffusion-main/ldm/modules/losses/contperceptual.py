import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?

# 该类用于计算 VAE 的损失. 损失由四部分组成:
#   (1) 真实图 - 生成图 像素级别的 L1 损失
#   (2) 真实图 - 生成图 特征级别的相似度损失
#   (3) VAE 的 KL 损失
#   (4) 生成器和鉴别器的损失
'''
    disc_start: 用于开始应用鉴别器损失的迭代次数，影响 GAN 损失的权重
    logvar_init: 对数方差的初始值，用于衡量重构损失和正则损失
    kl_weight: KL 损失的权重 (VAE 的预测高斯分布和标准高斯分布的KL损失, 这一损失也被认为是VAE中的一个正则损失)
    pixelloss_weight: 像素损失的权重. 但这个参数在代码中完全没有用到 (像素损失: 真实的图像和生成的图像之间的 L1 损失)
    disc_num_layers: 鉴别器的层数
    disc_in_channels: 鉴别器的输入通道数
    disc_factor: 控制 GAN 损失的因子. 它和 disc_start, disc_weight 共同最终决定 GAN 损失的权重
    disc_weight: 生成器/鉴别器损失的权重
    perceptual_weight: 感知相似损失的权重 (感知相似损失: 和像素损失类似, 保证真实图像和生成图像相似.
                                        感知损失是把图像放入VGG中, 计算各层的特征, 并计算特征之间的相似性)
    use_actnorm: 是否在 GAN 中使用激活归一化
    disc_conditional: 鉴别器是否为有条件的
    disc_loss: 鉴别器损失函数的类型
'''
class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval() # LPIPS 类用于计算两个图像的感知相似度
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    # 计算自适应权重以平衡真实图 - 生成图的损失和生成/鉴别的损失
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    # 前向过程, 计算损失.
    # input. 真实的输入图像.
    # reconstructions. VAE 重构的图像.
    # posteriors. VAE 中间层预测的均值和方差的分布.
    # optimizer_idx. 一个指示器, 当其为 0 时优化生成器, 1 时优化鉴别器.
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        # rec_loss 为原图和生成图的 L1 距离
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            # p_loss 是 LPIPS 损失, 由图像的每一层 vgg 特征之间的相似度计算得来
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            # 乘一个因子 self.perceptual_weight 来衡量不同损失的重要程度 重构损失 = L1距离 + w * LPIPS损失
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # 计算非负对数似然
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # 计算后验分布和标准高斯分布之间的距离
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        # optimizer_idx 有两个取值, 0或1, 0时更新生成器, 1时更新鉴别器
        if optimizer_idx == 0:
            # generator update
            if cond is None:  # cond表示是否有条件判别
                assert not self.disc_conditional  # 无条件判别
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            # logits_fake 是判别器的输出
            # 注意我们的输入是 reconstructions, 这是假数据, 当前正在训练生成器, 目标是欺骗鉴别器
            # 鉴别器: 真数据 ---> 0;  假数据 ---> 1
            g_loss = -torch.mean(logits_fake)  # 生成器损失

            # 下面是给生成器损失乘一个权重, 目的是加强训练生成器
            # 当生成器权重<=0.0时, 不再使用生成器
            # 生成器只在训练VAE阶段用, 在训练Diffusion阶段不用
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            # 损失 = 重构损失(weighted_nll_oss) + 正则KL损失(kl_loss) + 生成器损失(g_loss)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:  # 同上, 是否有条件鉴别
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            # 同上, 鉴别器权重
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

