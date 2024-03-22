#
import torch
import torchvision
from models.disciminator import get_model
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F
import os 
from models.generator import Generator

client_prt_path = "models_pth/ours_client_model_ac3.pth" 
server_prt_path = "models_pth/ours_server_model_ac3.pth"

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




if __name__ == '__main__':
    # Load the generator model
    targets = torch.randint(low=0, high=10, size=(100,))
    targets = targets.sort()[0]
    targets = targets.cuda()
    hooks = []
    net = get_discriminator()
    net.eval()
    print(net)
    iter = 100
    z = torch.randn(size=(100,256)).cuda()
    z.requires_grad = True
    generator = Generator()
    generator.to('cuda').train()
    reset_model(generator)    
    print(generator)

    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            hooks.append(DeepInversionHook(m))

    for i in range (iter):
        # Generate a random input
        img = generator(z)
        if not os.path.exists('images/'):
            os.makedirs('images/')

        optimizer_G = torch.optim.Adam([{"params": generator.parameters()}, {"params": [z]}],1e-3,betas=[0.5, 0.999],)
        optimizer_G.zero_grad()

        t_out = net(img)
        loss_bn = sum([h.r_feature for h in hooks])
        loss_oh = F.cross_entropy(t_out,targets)
        ##s_out=
        loss = loss_bn+loss_oh
        if i % 10 == 0:
            save_image(img.data.clone(),"images/img_%d.png" % i,nrow=10,normalize= True)
            print("iter = {},loss_bn={},loss_oh={:.4},loss={:.4}".format(i,loss_bn,loss_oh,loss))
        ##TODO:是否使用其他损失
        loss_oh.backward()
        optimizer_G.step()


