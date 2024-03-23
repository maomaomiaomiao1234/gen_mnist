from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from pytorch_msssim import ssim
class_num = 10
# 前两个与最后一个的ssim值
output = ["run_separately/ac5/", "run_separately/ac3/", "run_separately/SFL/"]


def calculate_ssim_mutual():
    imgs = []
    for separately_path in output:
        img_tmp=[]
        for i in range(class_num):
            img = Image.open(separately_path+"-"+str(i)+".png")
            img = ToTensor()(img).unsqueeze(0)
            img_tmp.append(img)
        imgs.append(img_tmp)

    for i in range (len(imgs)-1): 
        #ssim_val = ssim(imgs[0],torch.stack(raws),data_range=1.0, size_average=True)
        ssim_val_sum = 0    
        for j in range (len(imgs[i])):
            ssim_val = ssim(imgs[i][j],imgs[len(imgs)-1][j],data_range=1.0, size_average=True)
            print("第{}个文件图像库的第{}个图像的ssim值={}".format(i,j,ssim_val))
            ssim_val_sum += ssim_val
        print("ssim_val={}".format(ssim_val_sum/class_num))
calculate_ssim_mutual()


