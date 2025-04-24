import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from PIL import Image

def visualize_features(model, layer_num, img_path, device, data_transform=None):
    """
    Visualizes feature maps from a specific CNN layer.
    
    Parameters:
        model (nn.Module): CNN model
        layer_num (int): Layer number to visualize
        img_path (str): Path to the image file
        device: Device to use (CPU/GPU)
        data_transform: Image transformation (default: None)
        
    Returns:
        tuple: (feature_grid_fig, detailed_features_fig) Two matplotlib figures
    """
    model.eval()
    
    # Default transformation
    if data_transform is None:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')
    img_tensor = data_transform(img).unsqueeze(0).to(device)
    
    # Get feature maps for custom CNN
    if hasattr(model, 'get_features'):
        with torch.no_grad():
            features = model.get_features(img_tensor, layer_num)
    # Get feature maps for VGG16
    else:
        from src.models import get_vgg_features
        
        # VGG16 layer names
        layer_names = {
            1: 'features.0',  # First conv layer
            3: 'features.10',  # Middle conv layer
            5: 'features.30'   # Last conv layer
        }
        
        with torch.no_grad():
            features = get_vgg_features(model, img_tensor, layer_names[layer_num])
    
    # Visualization
    features = features.cpu().squeeze().detach()
    num_features = features.size(0)
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    plt.figure(figsize=(15, 15))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Show feature maps
    plt.subplot(1, 2, 2)
    feature_grid = make_grid(features.unsqueeze(1), nrow=grid_size, normalize=True, padding=1)
    plt.imshow(feature_grid.permute(1, 2, 0))
    plt.title(f'Layer {layer_num} Feature Maps')
    plt.axis('off')
    
    plt.savefig(f'features_layer_{layer_num}.png')
    feature_grid_fig = plt.gcf()
    plt.close()
    
    # Show detailed first 16 feature maps
    plt.figure(figsize=(15, 8))
    for i in range(min(16, num_features)):
        plt.subplot(4, 4, i+1)
        plt.imshow(features[i], cmap='viridis')
        plt.title(f'Filter {i+1}')
        plt.axis('off')
    
    plt.savefig(f'detailed_features_layer_{layer_num}.png')
    detailed_features_fig = plt.gcf()
    plt.close()
    
    return feature_grid_fig, detailed_features_fig

def visualize_predictions(model, data_loader, class_names, device, num_samples=8):
    """
    Visualizes model predictions.
    
    Parameters:
        model (nn.Module): CNN model
        data_loader (DataLoader): Data loader
        class_names (list): Class names
        device: Device to use (CPU/GPU)
        num_samples (int): Number of samples to visualize
        
    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    model.eval()
    
    # Get a mini-batch
    images, labels = next(iter(data_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
    
    # De-normalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Visualization
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        img = images_denorm[i].permute(1, 2, 0).numpy()
        label = labels[i].item()
        pred = preds[i]
        
        ax.imshow(img)
        title_color = 'green' if label == pred else 'red'
        ax.set_title(f'True: {class_names[label]}\nPred: {class_names[pred]}', 
                    color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """
    Visualizes the confusion matrix.
    
    Parameters:
        y_true (array): Ground truth labels
        y_pred (array): Predicted labels
        class_names (list): Class names
        model_name (str): Model name (used in filename)
        
    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    
    return plt.gcf()

def plot_comparison_chart(results_dict, metric_name='accuracy'):
    """
    Draws a comparison chart of different model performances.
    
    Parameters:
        results_dict (dict): Dictionary of model performance metrics
            Format: {'model_name': {'accuracy': 0.xx, 'precision': 0.xx, ...}}
        metric_name (str): Metric to visualize
        
    Returns:
        matplotlib.figure.Figure: Generated figure
    """
    model_names = list(results_dict.keys())
    metric_values = [results_dict[model][metric_name] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title(f'Model Comparison - {metric_name.capitalize()}')
    plt.xlabel('Model')
    plt.ylabel(metric_name.capitalize())
    plt.ylim(0, 1.0)
    
    # Display metric values above bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{metric_name}.png')
    
    return plt.gcf()

def visualize_model_architecture(model, input_size=(3, 224, 224)):
    """
    Visualizes the model architecture.
    
    Note: This function requires the torchviz library.
    
    Parameters:
        model (nn.Module): Model to visualize
        input_size (tuple): Input tensor shape (channels, height, width)
        
    Returns:
        graphviz.Digraph: Generated graph
    """
    try:
        from torchviz import make_dot
        
        # Create dummy input
        x = torch.randn(1, *input_size).requires_grad_(True)
        
        # Forward pass
        y = model(x)
        
        # Create graph
        graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
        graph.render('model_architecture', format='png', cleanup=True)
        
        return graph
    except ImportError:
        print("torchviz library not found. You can install it via 'pip install torchviz'.")
        return None
