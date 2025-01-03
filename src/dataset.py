import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class FlickrDataset(Dataset):
    def __init__(self, data_transform):
        self.dataset = pd.read_csv("../Dataset/captions.txt")
        self.transform = data_transform
        self.vocab = {"<START>","<END>", "<PAD>"}
        self.max_line = 0
        self.itos = dict()
        self.stoi = dict()
        self.__load_vocab()

    def __len__(self):
        return len(self.dataset)
    
    def __load_vocab(self):
        for i in range(len(self.dataset)):
            line = self.dataset.iloc[i]['caption']
            self.max_line = max(self.max_line, len(line.split()))
            for word in line.split():
                self.vocab.add(word.lower())
        self.vocab = list(self.vocab)
        self.max_line += 2
        for i in range(len(self.vocab)):
            self.itos[i] = self.vocab[i]
            self.stoi[self.vocab[i]] = i
        
    def __getitem__(self, index):
        img_path = self.dataset.iloc[index]['image']
        img = Image.open('../Dataset/Images/'+img_path)
        img = self.transform(img)
        caption = self.dataset.iloc[index]['caption']
        caption = caption.split()
        caption = [self.vocab.index(word.lower()) for word in caption]
        caption = [self.vocab.index("<START>")] + caption + [self.vocab.index("<END>")]
    
        if (len(caption) < self.max_line):
            caption = caption + [self.vocab.index("<PAD>")] * (self.max_line - len(caption))
        
        caption = torch.tensor(caption)
        return img, caption