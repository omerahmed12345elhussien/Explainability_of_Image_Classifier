import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
from matplotlib import pyplot as plt
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class RISE(nn.Module):
    def __init__(self, model:nn.Module, input_size:int=224) -> None:
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size

    def generate_masks(self, Num_mask:int , mask_size:int , prob:float) -> None:
        self.Num_mask = Num_mask
        self.prob = prob
        cell_size = np.ceil(self.input_size/mask_size)
        up_size = (mask_size + 1 )* cell_size
        #Sample Num_mask binary masks of size mask_size*mask_size.
        #Set each element to 1 with probability p and to 0 with probability 1-p
        grid = np.random.rand(Num_mask,mask_size,mask_size) < prob
        grid = grid.astype('float32')
        #Inialize the upsampled masks.
        self.masks = np.empty((Num_mask,self.input_size,self.input_size))

        #Upsample all masks.
        for idx in tqdm(range(Num_mask)):
            # Random shifts
            shift_x = np.random.randint(0,cell_size)
            shift_y = np.random.randint(0,cell_size)
            self.masks[idx] = resize(grid[idx],(up_size,up_size) , order= 1, anti_aliasing= False)[shift_x:shift_x+self.input_size , shift_y:shift_y+self.input_size]
        self.masks = self.masks.reshape(-1,1,self.input_size,self.input_size)
        self.masks = torch.from_numpy(self.masks).float().to(device)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        _,Height,Width = x.size()
        x = x[None]
        #Apply the masks on the image
        stack = self.masks * x
        result_list = []
        for idx in range(0,self.Num_mask , 100):
            result_list.append(self.model(stack[idx:min(idx+100,self.Num_mask)]))
        result_list = torch.cat(result_list)
        class_num = result_list.size(1)
        salience_map = result_list.transpose(0,1)@self.masks.reshape(self.Num_mask,Height*Width)
        return (salience_map.reshape(class_num,Height,Width))/(self.Num_mask*self.prob)

def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)