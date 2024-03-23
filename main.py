#
from pytorch_msssim import ssim
from torchvision.datasets import MNIST
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms
from kornia import augmentation
import torch
import numpy as np
from models.disciminator import get_model
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F
import os 
from models.generator import Generator

from calculate_ssim import calculate_ssim

client_prt_path = "models_pth/ours_client_model_ac5.pth" 
server_prt_path = "models_pth/ours_server_model_ac5.pth"
#client_prt_path = "models_pth/SFL_client_model.pth" 
#server_prt_path = "models_pth/SFL_server_model.pth"
data_path = "/home/whang1234/Downloads/data/mnist"
output = "run_separately/ac5/"
iter = 400

class CombinedModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(CombinedModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        return x

def get_discriminator():
    client_net = get_model(name="LeNetClientNetwork") 
    server_net = get_model(name="LeNetServerNetwork")
    client_net.load_state_dict(torch.load(client_prt_path))
    server_net.load_state_dict(torch.load(server_prt_path))
    net = CombinedModel(client_net, server_net)
    return net 

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

##没用到
class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):  # hook_fn(module, input, output) -> None
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()

##没用到
class MultiTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)




if __name__ == '__main__':
    # Load the generator model
    #targets = torch.randint(low=0, high=10, size=(100,))
    #targets = targets.sort()[0]
    #targets = targets.cuda()
    sequence = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    targets = torch.cat([sequence] * 1)
    targets = targets.cuda()
    hooks = []
    net = get_discriminator()
    net.eval()
    print(net)
    z = torch.randn(size=(10,256)).cuda()
    #z = np.random.normal(0, 1, (10, 100))
    z.requires_grad = True
    generator = Generator()
    generator.to('cuda').train()
    reset_model(generator)    
    print(generator)
    img_size = [1,28,28]

    optimizer_G = torch.optim.Adam([{"params": generator.parameters()}, {"params": [z]}],1e-3,betas=[0.5, 0.999],)
    aug = MultiTransform([
        # global view
        transforms.Compose([
            augmentation.RandomCrop(size=[img_size[-2], img_size[-1]], padding=4),
            augmentation.RandomHorizontalFlip(),
        ]),
        # local view
        transforms.Compose([
            augmentation.RandomResizedCrop(size=[img_size[-2], img_size[-1]], scale=[0.25, 1.0]),
            augmentation.RandomHorizontalFlip(),
        ]),
    ])

    ##MNIST不用调整，调整后生成结果更加糟糕
    aug = MultiTransform([
        # global view
        transforms.Compose([
            augmentation.RandomCrop(size=[img_size[-2], img_size[-1]], padding=4),
            augmentation.RandomHorizontalFlip(),
        ]),
        # local view
        transforms.Compose([
            augmentation.RandomResizedCrop(size=[img_size[-2], img_size[-1]], scale=[0.25, 1.0]),
            augmentation.RandomHorizontalFlip(),
        ]),
    ])

    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            hooks.append(DeepInversionHook(m))

    for i in range (iter):
        # Generate a random input
        optimizer_G.zero_grad()
        if not os.path.exists('images/'):
            os.makedirs('images/')

        img = generator(z)

        global_view, _ = aug(img)

        t_out = net(img)
        #t_out = net(img)
        loss_bn = sum([h.r_feature for h in hooks])
        loss_oh = F.cross_entropy(t_out,targets)
        ##s_out=
        loss = loss_bn+loss_oh
        if i % 10 == 0:
            save_image(img.data.clone(),"images/img_%d.png" % i,nrow=10,normalize= True)
            print("iter = {},loss_bn={},loss_oh={:.4},loss={:.4}".format(i,loss_bn,loss_oh,loss))
        
        #保存单个图片
        if i == iter-1:
            if isinstance(img, torch.Tensor):
                img = (img.detach().clamp(0, 1).cpu().numpy() * 255).astype('uint8')
            base_dir = os.path.dirname(output)
            if base_dir != '':
                os.makedirs(base_dir, exist_ok=True)

            output_filename = output.strip('.png')
            for idx, img in enumerate(img):
                if img.shape[0] == 1:
                    img = Image.fromarray(img[0])
                else:
                    img = Image.fromarray(img.transpose(1, 2, 0))
                img.save(output_filename + '-%d.png' % (idx))
 
        ##TODO:是否使用其他损失
        loss_oh.backward()
        optimizer_G.step()
    

    calculate_ssim(data_path,output)
