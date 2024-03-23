from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from pytorch_msssim import ssim
class_num = 10

#data_path = "/home/whang1234/Downloads/data/mnist"
#separately_path = "run_separately/"

def calculate_ssim(data_path,separately_path):
    imgs = []
    for i in range(class_num):
        img = Image.open(separately_path+"-"+str(i)+".png")
        img = ToTensor()(img).unsqueeze(0)
        imgs.append(img)

    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.ToTensor()])


    dataset = MNIST(data_path, train=False, download=False, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    raws = [[] for _ in range(class_num)]#不同数汇总
    data_num_sum = 0
    for data,label in dataloader:
        for i in range (len(label)):
            data_num_sum+=1
            #if label[i]==0:
            #    raw.append(data[i])
            raws[label[i]].append(data[i])

    ssim_val_sum = 0

    for i in range (len(raws)): #对应不同数，i=0,1,2,3,4,5,6,7,8,9 
        for j in range (len(raws[i])):#对应同一数
            raws[i][j] = raws[i][j].unsqueeze(0)
            #ssim_val = ssim(imgs[0],torch.stack(raws),data_range=1.0, size_average=True)
            ssim_val = ssim(imgs[i],raws[i][j],data_range=1.0, size_average=True)
            ssim_val_sum += ssim_val
    print("ssim_val={}".format(ssim_val_sum/data_num_sum))



