import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import os

# Transforms for data preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),  # Data augmentation - horizontal flip
        transforms.RandomRotation(10),      # Data augmentation - random rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

def load_data(data_dir, batch_size=32, num_workers=4):
    
    # Loads training and test datasets.
    
    # Load training dataset (expects labeled folders)
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    
    # Get class names
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Training set size: {len(train_dataset)}")
    
    # 20% of the training data is split as test data.
    print("No labeled test folder found. Using 20% of the training set for testing.")
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    # Update transform for test split
    test_dataset.dataset.transform = data_transforms['test']
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, class_names

def visualize_data_samples(data_loader, class_names, num_samples=8):
    
    # Visualizes sample images from the dataset.
    
    # Get a mini-batch
    images, labels = next(iter(data_loader))
    
    # Undo normalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Display the sample images
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        img = images_denorm[i].permute(1, 2, 0).numpy()
        
        # Handle labels if using random_split
        if isinstance(labels[i], torch.Tensor):
            label = labels[i].item()
            class_name = class_names[label] if label < len(class_names) else "Unknown"
        else:
            class_name = "Unknown"
        
        ax.imshow(img)
        ax.set_title(f"{class_name}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('data_samples.png')
    plt.show()
    
    return fig

def get_class_distribution(dataset, class_names):
    
    # Calculates the class distribution in the dataset.

    # Initialize class counts
    class_counts = {class_name: 0 for class_name in class_names}
    
    # For ImageFolder datasets
    if hasattr(dataset, 'imgs'):
        for _, label in dataset.imgs:
            class_counts[class_names[label]] += 1
    # For datasets created with random_split
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'imgs'):
        indices = dataset.indices
        for idx in indices:
            _, label = dataset.dataset.imgs[idx]
            class_counts[class_names[label]] += 1
    
    return class_counts

def plot_class_distribution(class_counts):
    
    # Plots the class distribution as a bar chart.
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    
    # Display values on top of the bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                 str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()
    
    return plt.gcf()
