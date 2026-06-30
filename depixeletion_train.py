import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# Define a custom dataset
class ImagePairDataset(Dataset):
    def __init__(self, pixelated_dir, high_res_dir, transform=None):
        self.pixelated_dir = pixelated_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        
        # List all image filenames in both directories
        self.pixelated_images = os.listdir(pixelated_dir)
        self.high_res_images = os.listdir(high_res_dir)

    def __len__(self):
        return len(self.pixelated_images)

    def __getitem__(self, idx):
        # Load pixelated image
        pixelated_img_name = self.pixelated_images[idx]
        pixelated_img_path = os.path.join(self.pixelated_dir, pixelated_img_name)
        pixelated_img = Image.open(pixelated_img_path).convert('RGB')
        
        # Load corresponding high-resolution image
        high_res_img_name = self.high_res_images[idx]
        high_res_img_path = os.path.join(self.high_res_dir, high_res_img_name)
        high_res_img = Image.open(high_res_img_path).convert('RGB')
        
        # Apply transformation if specified
        if self.transform:
            pixelated_img = self.transform(pixelated_img)
            high_res_img = self.transform(high_res_img)
        
        return pixelated_img, high_res_img

# Define the EnhancedUNet model
class EnhancedUNet(nn.Module):
    def __init__(self):
        super(EnhancedUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load datasets
pixelated_dir = r"C:\Users\Sanjit\OneDrive\Desktop\Intel-Depixelation\mrithul-20240618T050023Z-001\25 300"
high_res_dir = r"C:\Users\Sanjit\OneDrive\Desktop\Intel-Depixelation\mrithul-20240618T050023Z-001\DIV2K_train_HR"

dataset = ImagePairDataset(pixelated_dir, high_res_dir, transform=transform)

# DataLoader setup
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, and Optimizer
model = EnhancedUNet().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, (pixelated_imgs, high_res_imgs) in enumerate(data_loader):
        pixelated_imgs = pixelated_imgs.to(device)
        high_res_imgs = high_res_imgs.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pixelated_imgs)
        
        loss = criterion(outputs, high_res_imgs)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(data_loader)}')

# Save the trained model
torch.save(model.state_dict(), 'enhanced_unet_model.pth')
print("Model saved as 'enhanced_unet_model.pth'")

# Evaluation and visualization on a test image
model.eval()
with torch.no_grad():
    test_pixelated_img_path = r"C:\Users\Sanjit\OneDrive\Desktop\Intel-Depixelation\mrithul-20240618T050023Z-001\10\0027.png"
    test_pixelated_img = Image.open(test_pixelated_img_path).convert('RGB')
    test_pixelated_tensor = transform(test_pixelated_img).unsqueeze(0).to(device)
    
    output = model(test_pixelated_tensor)
    
    output = torch.clamp(output, 0, 1)
    
    pixelated_img = test_pixelated_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    output_img = output.squeeze().cpu().permute(1, 2, 0).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(pixelated_img)
    ax1.set_title('Input (Pixelated)')
    ax1.axis('off')

    ax2.imshow(output_img)
    ax2.set_title('Output (Depixelated)')
    ax2.axis('off')

    plt.savefig('comparison.png')
    plt.show()

print("Process completed. Check 'comparison.png' for results.")
