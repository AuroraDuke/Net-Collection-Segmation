#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, random_split,Dataset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.datasets import VOCSegmentation
from PIL import Image
import os
import cv2

#iuo
import torch.nn.functional as F
from sklearn.metrics import jaccard_score

# # Data load

# In[2]:


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')] #jpg or png etc.
        self.mask_filenames = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        

        assert len(self.image_filenames) == len(self.mask_filenames), "Image and mask counts do not match"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')  # 'L' grayscale mask
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask


# # train test & visulaziton 

# In[3]:


# Data converters
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# dataset
dataset = CustomDataset(image_dir='DataSet/images', mask_dir='DataSet/masks', transform=transform)
dataset = torch.utils.data.Subset(dataset, range(400))

# In[4]:


idx =50
image, mask = dataset[idx]

# Convert tensors back to numpy arrays
image = image.numpy().transpose(1, 2, 0)  # CHW to HWC
mask = mask.numpy().squeeze()             # Remove channel dimension

# Denormalize the image (assuming normalization was done)
image = image * 255.0
image = image.astype(np.uint8)

# Plot the image and mask
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(image)
axs[0].set_title('Image')
axs[0].axis('off')

axs[1].imshow(mask, cmap='gray')
axs[1].set_title('Mask')
axs[1].axis('off')

plt.show()

# In[5]:


# Separate training and testing dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# In[6]:


# Print dimensions of training and testing dataset
print(f'Total dataset size: {len(dataset)}')
print(f'Training dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

# Print data sample count of data loaders
print(f'Training data loader total number of samples: {len(train_loader.dataset)}')
print(f'Test data loader total number of samples: {len(test_loader.dataset)}')

# Print batch number in training and test data loaders
print(f'Total batch number of training data loader: {len(train_loader)}')
print(f'Test data loader total batch number: {len(test_loader)}')

# Print a sample batch size (from the training data loader)
for images, masks in train_loader:
 print(f'Size of a training batch - Image: {images.size()}, Mask: {masks.size()}')
 break # check first batch only

# In[7]:


import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

# Get indexes of test dataset
test_indices = np.arange(len(test_dataset))

# Select 5 random indexes
random_indices = np.random.choice(test_indices, size=5, replace=False)

# Create subset
ran_test_subset = Subset(test_dataset, random_indices)

# Create DataLoader
ran_test_loader = DataLoader(ran_test_subset, batch_size=1, shuffle=False)

# Sample data output
for images, masks in ran_test_loader:
    print(f"Images batch shape: {images.shape}")
    print(f"Masks batch shape: {masks.shape}")
    break  # Show only the first example

# # criterion ve deeplab- modeli yükleme

# In[8]:


from torchvision.models.segmentation import deeplabv3_resnet50

# Create the model and take it to training
model_deeplab = deeplabv3_resnet50(pretrained=True)
model_deeplab.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)  # Single channel (grayscale)

criterion= torch.nn.BCEWithLogitsLoss()  # Binary Cross-Entropy for binary segmentation
optimizer_deeplab = torch.optim.Adam(model_deeplab.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_deeplab.to(device)


# # UnetPlusPlus 

# In[9]:


from segmentation_models_pytorch import UnetPlusPlus

model_unetpp = UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)

optimizer_unetpp = torch.optim.Adam(model_unetpp.parameters(), lr=1e-4)
model_unetpp.to(device)


# # Unet

# In[10]:


from segmentation_models_pytorch import Unet

# Create the UNet model
model_unet = Unet(
    encoder_name="resnet34",       # Encoder type
    encoder_weights="imagenet",    # Encoder weights pre-trained with ImageNet
    in_channels=3,                 # Number of channels in the input image (e.g. 3 for RGB)
    classes=1                      # Number of output channels (1 for segmentation mask)
)
# Set the model to run on the device
model_unet = model_unet.to(device)

# Define the optimizer
optimizer_unet = torch.optim.Adam(model_unet.parameters(), lr=1e-4)

# # UsquareNet

# In[11]:


!git clone https://github.com/AuroraDuke/Net-Collection-Segmation

# In[12]:


print(os.listdir('Net-Collection-Segmation'))
#print(os.listdir('U-2-Net/model'))
#with open('U-2-Net/model/u2net.py', 'r') as file:
   # print(file.read())

# In[13]:


import sys
sys.path.append(os.path.join(os.getcwd(), 'Net-Collection-Segmation'))
from pytorch.u2net import U2NET # File containing the U2Net model
# Create the model and set it to run on the device
model_u2net = U2NET(3, 1)  # 3 channels RGB and 1 channel output
optimizer_u2net = torch.optim.Adam(model_u2net.parameters(), lr=1e-4)

model_u2net.to(device)

# # SegNet

# In[14]:


from pytorch.segnet import SegNet

# In[15]:


model_segnet = SegNet(input_nbr=3, label_nbr=1) #1 can be adjusted according to the number of classes
optimizer_segnet = torch.optim.Adam(model_segnet.parameters(), lr=1e-4)
model_segnet.to(device)

# # IoU Metric

# In[16]:


def iou_metric(pred, target, threshold=0.5):
    # Apply sigmoid to model outputs to get probabilities
    pred = torch.sigmoid(pred)
    
    # Apply threshold to get binary predictions
    pred = (pred > threshold).float()
    
    # Invert target if necessary (0 -> background, 1 -> object)
    target = 1 - target
    
    # Compute intersection and union
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    
    if union == 0:
        return float('nan')  # Handle edge case when union is zero
    
    iou = intersection / (union + 1e-6)  # Adding epsilon to avoid division by zero
    return iou.item()

# # Model save

# In[17]:


# Create a folder called 'models', don't create it if it exists
os.makedirs('models', exist_ok=True)

# # eğitme ve görüntüleme

# In[18]:


def train_model(model, criterion, optimizer, train_loader, num_epochs=5, is_deeplab=False):
    model.train()
    print("Starting Net...")
    print(f'Epoch[ 0/{num_epochs}],')
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_iou = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            if is_deeplab:
                outputs = outputs['out']
                
           # if isinstance(outputs, tuple):
              #  outputs = outputs[0]  # Adjust this if your model returns multiple outputs

            if isinstance(outputs, tuple):
                # Assuming the first output is the main prediction
                primary_output = outputs[0]
            else:
                primary_output = outputs
            
            loss = criterion(primary_output.squeeze(1), masks.squeeze(1))
            loss.backward()
            optimizer.step()

            # Calculate IoU for the batch
            iou = iou_metric(primary_output.squeeze(1), masks)
            running_iou += iou * images.size(0)
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_iou = running_iou / len(train_loader.dataset)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f},IoU: {epoch_iou:.4f}')

        # Define save path and save the model
    model_name = f'{model.__class__.__name__}.pth'
    save_path = os.path.join('models', model_name)
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

# In[19]:


import matplotlib.pyplot as plt

def visualize_results(model, test_loader, is_deeplab=False):
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # If Deeplab model, get main output
            if is_deeplab:
                primary_output = outputs['out']
            else:
                primary_output = outputs
            
            # If outputs is a tuple, usually some of the model's output is the main output.
            #for U2Net
            if isinstance(primary_output, tuple):
                primary_output = primary_output[0]
            
            # Take appropriate action
            primary_output = torch.sigmoid(primary_output).squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            masks = masks.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            
            # calculate IoU
            iou = iou_metric(primary_output, masks)
            print(f'IoU: {iou:.4f}')
            
            # Visualization
            for i in range(1):
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(images[i].cpu().numpy().transpose(1, 2, 0))  # (B, C, H, W) -> (H, W, C)
                plt.title('Input Image')

                plt.subplot(1, 3, 2)
                plt.imshow(masks[i].cpu().numpy(), cmap='gray')  # (H, W)
                plt.title('Ground Truth')

                plt.subplot(1, 3, 3)
                plt.imshow(primary_output[i].cpu().numpy(), cmap='gray')  # (H, W)
                plt.title('Predicted Mask')

                plt.show()
                break


# # Deeplab 

# In[20]:


# Modeli eğit
train_model(model_deeplab, criterion, optimizer_deeplab, train_loader, num_epochs=5, is_deeplab=True)

# In[21]:


# Sonuçları görselleştir
visualize_results(model_deeplab, ran_test_loader,is_deeplab=True)

# # unetpp 

# In[22]:


train_model(model_unetpp, criterion, optimizer_unetpp, train_loader, num_epochs=5)

# In[23]:


visualize_results(model_unetpp, ran_test_loader)

# # Unet

# In[24]:


train_model(model_unet, criterion, optimizer_unet, train_loader, num_epochs=5)

# In[25]:


visualize_results(model_unetpp, ran_test_loader)

# # UsquareNet

# In[26]:


train_model(model_u2net, criterion, optimizer_u2net, train_loader, num_epochs=5)

# In[27]:


visualize_results(model_u2net, ran_test_loader)

# # Segnet

# In[28]:


train_model(model_segnet, criterion, optimizer_segnet, train_loader, num_epochs=5)

# In[29]:


visualize_results(model_segnet, test_loader)
