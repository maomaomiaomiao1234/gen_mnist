from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from pytorch_msssim import ssim

data_path = "/home/whang1234/Downloads/data/mnist"

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


dataset = MNIST(data_path, train=False, download=False, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
image = Image.open("/home/whang1234/Downloads/githubFiles/gen_mnist/images_SFL/img_490.png")
image = ToTensor()(image).unsqueeze(0)
first_column_images = image.view(-1, 10, 1, 28, 28)[:, 0]

print(image.shape)

i = 0
for data,label in dataloader:
    i= i+1
print("i={}".format(i))
#real_images, real_labels = next(iter(dataloader))
#print("len(real_images)={},len(real_labels)={}".format(len(real_images), len(real_labels)))
