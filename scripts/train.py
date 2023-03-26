import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.image import torch_to_numpy
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from scripts.validate import validate

def train(net, dataset, val_dataset, batch_size=6, val_pad=nn.ZeroPad2d((24,24, 9, 9))):
    """Train the model on dataset and evaluate on val_dataset for each epoch

    Args:
        net (nn.Module): model to train
        dataset (MedicalDataset): train dataset
        val_dataset (MedicalDataset): Validation dataset
        batch_size (int, optional): batch size. Defaults to 6.
        val_pad : validation padding (if network requires cropping inputs)
    """
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True)
   
    # defining losses for train
    criterion = nn.L1Loss()

    # defining the training strategy
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  

    for epoch in range(100):  # loop over the dataset multiple times
        # Training step
        running_loss = 0.0
        print("epoch", epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            sample = data
            inputs = sample["input"]
            labels = sample["label"]
            inputs = [inputs[0].cuda(),inputs[1].cuda()]
            labels = labels.cuda()
            
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (len(trainloader)):.3f}')
        running_loss = 0.0
        
        # Validation step
        net.eval()
        with torch.no_grad():
            val_loss = validate(net, val_dataset, pad=val_pad)
            print("val loss : ", val_loss)
        
        net.train()
        scheduler.step()
    print('Finished Training')
