import torch
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

class OOF_dataset(Dataset):
    def __init__(self, root, type, transforms) -> None:
        super().__init__()
        if type == 'eval':
            self.root_clear = os.path.join(root,'{}-clear'.format('val'))
            self.root_blurred = os.path.join(root,'{}-blurred'.format('val'))
        else:
            self.root_clear = os.path.join(root,'{}-clear'.format(type))
            self.root_blurred = os.path.join(root,'{}-blurred'.format(type))
            
        self.type = type
        self.transforms = transforms
        self.image_names = os.listdir(self.root_clear)
        self.len = len(self.image_names)
        
    def __getitem__(self, index):
        img_clear = Image.open(os.path.join(self.root_clear, self.image_names[index]))
        img_blurred = Image.open(os.path.join(self.root_blurred, self.image_names[index]))
        
        seed = np.random.randint(2147483647)
        
        torch.random.manual_seed(seed)
        img_clear = self.transforms(img_clear)
        
        torch.random.manual_seed(seed)
        img_blurred = self.transforms(img_blurred)
        
        if self.type == 'eval':
            return  transforms.Resize((224,224))(img_blurred),\
                    transforms.Resize((112,112))(img_blurred),\
                    transforms.Resize((56 ,56 ))(img_blurred),\
                    transforms.Resize((28 ,28 ))(img_blurred),\
                    transforms.Resize((224,224))(img_clear),\
                    transforms.Resize((112,112))(img_clear),\
                    transforms.Resize((56 ,56 ))(img_clear),\
                    transforms.Resize((28 ,28 ))(img_clear),\
                    self.image_names[index]
        else:
            return  transforms.Resize((224,224))(img_blurred),\
                    transforms.Resize((112,112))(img_blurred),\
                    transforms.Resize((56 ,56 ))(img_blurred),\
                    transforms.Resize((28 ,28 ))(img_blurred),\
                    transforms.Resize((224,224))(img_clear),\
                    transforms.Resize((112,112))(img_clear),\
                    transforms.Resize((56 ,56 ))(img_clear),\
                    transforms.Resize((28 ,28 ))(img_clear)
        
    def __len__(self):
        return self.len
    
def get_dataloader(args):
    
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        # transforms.Normalize([0.5,0.5,0.5],[1,1,1]),
        # transforms.RandomRotation(180),
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5],[1,1,1]),
    ])
    
    train_dataset = OOF_dataset(
        root = args.data_root,
        type = 'train',
        transforms = train_transforms
    )
    
    val_dataset = OOF_dataset(
        root = args.data_root,
        type = 'val',
        transforms = val_transforms
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
    )
    
    return train_dataloader, val_dataloader

def get_eval_dataloader(args):

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    eval_dataset = OOF_dataset(
        root = args.data_root,
        type = 'eval',
        transforms = eval_transforms
    )
    
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True,
    )
    
    return eval_dataloader

if __name__ == '__main__':
    # img_clear = np.array(Image.open(r"D:\desktop\de-OOF\data\CRC-224\CRC-1-21\train-clear\ADI-TCGA-AEDALKHL.png"))
    # img_blurred = np.array(Image.open(r"D:\desktop\de-OOF\data\CRC-224\CRC-1-21\train-blurred\ADI-TCGA-AEDALKHL.png"))
    
    # print(img_clear.shape)
    # img_concat = np.concatenate((img_clear, img_blurred), axis=2)
    # print(img_concat.shape)
    

    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str,default='./data/CRC-224/CRC-1-21')
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument('--num_workers',type = int, default=0)
    

    args = parser.parse_args()
    
    train_loader, val_loader = get_dataloader(args=args)
    
    print(len(train_loader),len(val_loader))
    for img, label in train_loader:
        print(img.shape, label.shape)
        img = np.array(transforms.ToPILImage()(img[0]))
        label = np.array(transforms.ToPILImage()(label[0]))
        plt.subplot(121),plt.imshow(img),plt.axis('off')
        plt.subplot(122),plt.imshow(label),plt.axis('off')
        plt.show()
        break
    
    for img, label in val_loader:
        print(img.shape, label.shape)
        img = np.array(transforms.ToPILImage()(img[0]))
        label = np.array(transforms.ToPILImage()(label[0]))
        plt.subplot(121),plt.imshow(img),plt.axis('off')
        plt.subplot(122),plt.imshow(label),plt.axis('off')
        plt.show()
        break
    
    
    ...
     