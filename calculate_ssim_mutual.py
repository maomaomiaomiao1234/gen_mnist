from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from pytorch_msssim import ssim
class_num = 10
output = ["run_separately/ac5/", "run_separately/ac3/", "run_separately/SFL/"]


def calculate_ssim_mutual():
    imgs = [[] for _ in range(3)]
    for separately_path in output:
        img_tmp=[]
        for i in range(class_num):
            img = Image.open(separately_path+"-"+str(i)+".png")
            img = ToTensor()(img).unsqueeze(0)
            img_tmp.append(img)
        imgs.append(img_tmp)

    for i in range (len(imgs)-1): #对应不同数，i=0,1,2,3,4,5,6,7,8,9 
        #ssim_val = ssim(imgs[0],torch.stack(raws),data_range=1.0, size_average=True)
        ssim_val_sum = 0    
        for j in range (len(imgs[i])):
            ssim_val = ssim(imgs[i][j],imgs[len(imgs)-1][j],data_range=1.0, size_average=True)
            ssim_val_sum += ssim_val
        print("ssim_val={}".format(ssim_val_sum/class_num))
calculate_ssim_mutual()


