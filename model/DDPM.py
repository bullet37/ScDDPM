import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
import torch
import torch.nn as nn
import io
from PIL import Image

class MLPDiffusion(nn.Module):
    def __init__(self, input_dim, n_steps=100, num_units=128):
        super(MLPDiffusion, self).__init__()
        self.n_steps = n_steps
        self.input_dim = input_dim
        self.n_steps = n_steps

        # Constants
        self.betas = torch.sigmoid(torch.linspace(-6, 6, self.n_steps)) * (0.5e-2 - 1e-5) + 1e-5
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)

        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert self.alphas.shape == self.alphas_prod.shape == self.alphas_prod_p.shape == \
               self.alphas_bar_sqrt.shape == self.one_minus_alphas_bar_log.shape \
               == self.one_minus_alphas_bar_sqrt.shape

        self.encoder = nn.ModuleList([
            nn.Linear(self.input_dim, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, num_units),
            nn.ReLU(),
            nn.Linear(num_units, self.input_dim) ]
        )

        self.step_embeddings = nn.ModuleList([
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units)
        ])

    def forward(self, x, t):
        # 扩散
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.encoder[2 * idx](x)
            x += t_embedding
            x = self.encoder[2 * idx + 1](x)
        x_reconstructed = self.encoder[-1](x)
        return x_reconstructed

    def sample(self, num_samples, t):
        # 生成数据样本，类似于VAE的解码过程
        x = torch.randn(num_samples, self.input_dim)
        x_reconstructed = self.forward(x, t)
        return x_reconstructed

    def q_x(self, x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = self.alphas_bar_sqrt[t]
        alphas_1_m_t = self.one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    # 编写训练误差函数
    def diffusion_loss_fn(self, x_0):
        """对任意时刻t进行采样计算loss"""
        batch_size = x_0.shape[0]

        # 对一个batchsize样本生成随机的时刻t，t变得随机分散一些，一个batch size里面覆盖更多的t
        t = torch.randint(0, self.n_steps, size=(batch_size // 2,))
        t = torch.cat([t, self.n_steps - 1 - t], dim=0)  # t的形状（bz）
        t = t.unsqueeze(-1)  # t的形状（bz,1）

        a = self.alphas_bar_sqrt[t]
        aml = self.one_minus_alphas_bar_sqrt[t]
        # 生成随机噪音eps
        e = torch.randn_like(x_0)
        # 构造模型的输入
        x = x_0 * a + e * aml
        # 送入模型，得到t时刻的随机噪声预测值
        output = self.forward(x, t.squeeze(-1))
        # 与真实噪声一起计算误差，求平均值
        return (e - output).square().mean()

    def p_sample_loop(self, shape):
        cur_x = torch.randn(shape)
        x_seq = [cur_x]
        for i in reversed(range(self.n_steps)):
            cur_x = self.p_sample(cur_x, i)
            x_seq.append(cur_x)
        return x_seq

    def p_sample(self,  x, t):
        t = torch.tensor([t])
        coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]
        eps_theta = self.forward(x, t)
        mean = (1 / (1 - self.betas[t]).sqrt()) * (x - (coeff * eps_theta))
        z = torch.randn_like(x)
        sigma_t = self.betas[t].sqrt()
        return (mean + sigma_t * z)

# Test Data
s_curve,_ = make_s_curve(10**4,noise=0.1)
s_curve = s_curve[:,[0,2]]/10.0
data = s_curve.T #[2,10000]
dataset = torch.Tensor(s_curve).float()

print('Training model...')
batch_size = 128
# dataset放到dataloader中
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epoch = 4000
plt.rc('text', color='blue')
#实例化模型，传入一个数
n_steps = 100

model = MLPDiffusion(2, n_steps)  # 输出维度是2，输入是x和step
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# epoch遍历
for t in range(num_epoch):
    # dataloader遍历
    for idx, batch_x in enumerate(dataloader):
        # 得到loss
        loss = model.diffusion_loss_fn(batch_x)
        optimizer.zero_grad()
        loss.backward()
        #梯度clip，保持稳定性
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    #每100步打印效果
    if (t % 100 == 0):
        print(loss)
        #根据参数采样一百个步骤的x，每隔十步画出来，迭代了4000个周期，逐渐更接近于原始
        x_seq =  model.p_sample_loop(dataset.shape)
        fig, axs = plt.subplots(1, 10, figsize=(28, 3))
        for i in range(1, 11):
            cur_x = x_seq[i * 10].detach()
            axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white')
            axs[i - 1].set_axis_off()
            axs[i - 1].set_title('$q(\mathbf{x}_{' + str(i * 10) + '})$')
