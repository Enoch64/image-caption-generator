import sys
import torch 
import matplotlib.pyplot as plt
from PIL import Image
from dataset import FlickrDataset
from model import EncoderDecoder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


#Create a sequence of transformations to be applied on each image
transform = v2.Compose([
    v2.Resize((224,224)),  #Input images are 224 x 224
    v2.PILToTensor(),      #Convert the image to a pytorch tensor
    v2.ToDtype(torch.float32, scale=True)
])

device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
dataset = FlickrDataset(transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

def show_image(inp, title=None):
    # """Image for Tensor"""
    inp = inp.numpy().transpose((1,2,0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig('output.png')
    # plt.show()


def generate_caption(img_path):
    model = EncoderDecoder(
    embed_size=300,
    vocab_size = len(dataset.vocab),
    attention_dim=256,
    encoder_dim=2048,
    decoder_dim=512
    ).to(device)
    

    model.load_state_dict(torch.load("../Models/caption_generator.pth",map_location=torch.device('cpu')))
    model.eval()

    sample_image = Image.open(img_path)
    sample_image = transform(sample_image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        features = model.encoder(sample_image)
        caption, _ = model.decoder.generate_caption(features, vocab=dataset)

    # Print the generated caption
    print('Generated Caption:', ' '.join(caption))

    # Show the image with the generated caption
    show_image(sample_image.cpu().squeeze(), title=' '.join(caption))

if __name__ == '__main__':
    img_path = sys.argv[1]
    generate_caption(img_path)