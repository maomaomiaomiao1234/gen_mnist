#
import torch
import torchvision
from models.generator import Generator

client_prt_path = "models_pth/ours_client_model_ac3_pth" 
server_prt_path = "models_pth/ours_server_model_ac3_pth"

def get_discriminator():
    return 



if __name__ == '__main__':
    # Load the generator model
    generator = Generator()
    ##TODO:修改gen模型加载为discriminator模型加载
    generator.load_state_dict(torch.load(client_prt_path))
    generator.eval()
    print(generator)

    # Generate a random input
    z = torch.randn(1, 100)
    img = generator(z)
    print(img.shape)
    print(img.min(), img.max())

    # Save the generated image
    torchvision.utils.save_image(img, 'generated.png', nrow=1, normalize=True)
