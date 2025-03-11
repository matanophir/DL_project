import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse

import nets

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=32, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    return parser.parse_args()
    

if __name__ == "__main__":

    args = get_args()
    freeze_seeds(args.seed)
                
                                           
    if args.mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  
        ])
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        
    #Data
    # When you create your dataloader you should split train_dataset or test_dataset to leave some aside for validation
    ds_val, ds_train = torch.utils.data.random_split(train_dataset, [0.2, 0.8])

    ds_train = torch.utils.data.Subset(ds_train, range(100))
    

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    img_shape = train_dataset[0][0].shape

    #Model
    # encoder_model = torch.nn.Linear(32*32*3,args.latent_dim).to(args.device)
    # decoder_model = torch.nn.Linear(args.latent_dim,32*32*3 if args.self_supervised else NUM_CLASSES).to(args.device) 
    model = nets.AE(img_shape, args.latent_dim).to(args.device)

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-1, weight_decay = 1e-8)

    #Loss
    loss_function = torch.nn.MSELoss()

    #Trainer
 
    epochs = 20
    outputs = []
    losses = []
    for epoch in range(epochs):
        for i, (x, _) in enumerate(dl_train):
            x = x.to(args.device)
            optimizer.zero_grad()
            y = model(x)
            loss = loss_function(y, x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        outputs.append((epochs, x, y))
        print(f'Epoch: {epoch}, Loss: {loss.item()}')


    
    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    # Plotting the last 100 values
    plt.plot(losses[-100:])


 
    for i, item in enumerate(x):
        # Reshape the array for plotting
        item = item.reshape(-1, 32, 32)
        plt.imshow(item[0].detach().cpu().numpy())
        
    for i, item in enumerate(y):
        item = item.reshape(-1, 32, 32)
        plt.imshow(item[0].detach().cpu().numpy())


