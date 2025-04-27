import torch
import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):
    """
    Custom CNN model
    
    Includes 5 convolutional layers and fully connected layers.
    """
    def __init__(self, num_classes=5):
    
        #Initializes the custom CNN model.
        
        super(CustomCNN, self).__init__()
        # Conv Layer 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 5
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc_relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        
        # Forward pass
        
        # Conv Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Conv Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Conv Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Conv Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Conv Layer 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        # Fully connected layers
        x = x.view(-1, 256 * 7 * 7)
        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x, layer_num):
        
        # Gets feature maps from a specific layer.
        
        if layer_num >= 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            if layer_num == 1:
                return x
        
        if layer_num >= 2:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            if layer_num == 2:
                return x
        
        if layer_num >= 3:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.pool3(x)
            if layer_num == 3:
                return x
        
        if layer_num >= 4:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.pool4(x)
            if layer_num == 4:
                return x
        
        if layer_num >= 5:
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            x = self.pool5(x)
            if layer_num == 5:
                return x
        
        return x

def create_vgg16_feature_extractor(num_classes=5):
    
    # Configures the VGG16 model as a feature extractor.
    
    model = models.vgg16(pretrained=True)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model

def create_vgg16_fine_tuned(num_classes=5):
    
    # Configures the VGG16 model for fine-tuning.
    # The first convolutional block is frozen; the rest is fine-tuned.
    
    model = models.vgg16(pretrained=True)
    
    # Freeze the first convolutional block (5 layers: 2 conv + 3 others)
    for i, param in enumerate(model.features.parameters()):
        if i < 10:
            param.requires_grad = False
    
    # Replace the final layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model

def get_vgg_features(model, x, layer_name):
    
    # Retrieves feature maps from a specific layer of the VGG16 model.
    
    features = {}
    
    def get_features_hook(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # Register hook to the relevant layer
    layer_parts = layer_name.split('.')
    if len(layer_parts) == 2:
        if layer_parts[0] == 'features':
            layer_idx = int(layer_parts[1])
            handle = model.features[layer_idx].register_forward_hook(get_features_hook(layer_name))
        elif layer_parts[0] == 'classifier':
            layer_idx = int(layer_parts[1])
            handle = model.classifier[layer_idx].register_forward_hook(get_features_hook(layer_name))
    
    # Forward pass
    model(x)
    
    # Remove the hook
    handle.remove()
    
    return features[layer_name]
