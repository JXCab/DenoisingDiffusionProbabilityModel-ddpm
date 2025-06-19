from DiffusionFreeGuidence.TrainCondition import train, eval

'''
条件生成在生成对抗网络 GAN 中就有多种实现方式, 包括 CGAN,ACGAN/PCGAN,InfoGAN 等, 通过语义标签生成对应内容, 训练都需要将标签 label 作为输入.

大体可以分为三种方法:
    强监督: 模型输出标签的预判, 通过添加多标签分类损失训练
    弱监督: 仅添加标签作为特征输入模型, 嵌入图像特征（相加）, 但不改变训练过程（不额外输出、不添加损失函数）
    无监督: 通过输入假标签(pseudo label), 这种标签是按某一特征规律自动生成的, 且通常需要对应的损失函数(分类、聚类、对比学习等)

Classifier-Free 是用的弱监督方法, 即仅将标签作为特征嵌入进图像即可, 这里用了最简单的方式, 类似t的嵌入,
直接将 label_embedding 注入到 U-Net 的 middle 层, 再广播到每一个元素即可, 即:
x = x + t_embedding + label_embedding
'''

def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_69_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
