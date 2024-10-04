# # Import necessary libraries
# from Diffusion_tutorial import Trainer  # Already implemented Trainer class
# from Diffusion_tutorial import GaussianDiffusion  # Implemented Diffusion model class
# from Diffusion_tutorial import Unet
# from torchvision import transforms
# import torch
# import matplotlib.pyplot as plt
# import torchvision
# import pyarrow.dataset as ds
# from torch.utils.data import Dataset, DataLoader

# # Arrow file-based PyTorch Dataset class definition
# class ArrowDataset(Dataset):
#     def __init__(self, arrow_file, transform=None):
#         self.dataset = ds.dataset(arrow_file, format='arrow')
#         self.transform = transform
#         self.table = self.dataset.to_table()
#         self.images = self.table['image_column']  # Adjust 'image_column' to the correct column name in your dataset
#         # If labels are available, adjust the following line:
#         # self.labels = self.table['label_column']

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img = self.images[idx].as_py()  # Convert image to numpy
#         if self.transform:
#             img = self.transform(img)
#         return img

# # Define the path to the Arrow file
# arrow_file_path = '/home/fall/exercise/dataset/train/data-00000-of-00001.arrow'

# # Define transformations
# preprocess = transforms.Compose([
#     transforms.Resize((128, 128)),  # Resize images to 128x128
#     transforms.RandomHorizontalFlip(),  # Random horizontal flip
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
# ])

# # Load the ArrowDataset with transformations
# dataset = ArrowDataset(arrow_file=arrow_file_path, transform=preprocess)

# # Apply DataLoader to the dataset
# train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# # Create the diffusion model
# diffusion_model = GaussianDiffusion(
#     model=Unet(),          # Use the U-Net model for training
#     image_size=128,        # Image size
#     timesteps=1000,        # Number of timesteps
#     objective='pred_noise' # Training objective
# )

# # Initialize the Trainer
# trainer = Trainer(
#     diffusion_model=diffusion_model,
#     train_dataloader=train_dataloader,  # Use DataLoader for the dataset
#     train_batch_size=16,  # Batch size
#     train_lr=1e-4,        # Learning rate
#     train_num_steps=100000,  # Number of training steps
#     ema_update_every=10,  # EMA update frequency
#     ema_decay=0.995,      # EMA decay rate
#     save_and_sample_every=1000,  # Save and sample every 1000 steps
#     num_samples=25,       # Number of sample images
#     results_folder='./results'  # Results folder
# )

# # Start training
# trainer.train()

# # After training, generate samples
# samples = trainer.sample(batch_size=16)

# # Display the sampled images
# grid_img = torchvision.utils.make_grid(samples, nrow=4)
# plt.imshow(grid_img.permute(1, 2, 0))
# plt.show()
