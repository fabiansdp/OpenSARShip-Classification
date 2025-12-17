import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50WithFeatures(nn.Module):
    """
    ResNet50 with scale-variant features
    Architecture as shown in Fig. 1 of the paper
    """
    
    def __init__(self, num_features=14, num_classes=4, pretrained=True):
        super(ResNet50WithFeatures, self).__init__()
        
        # 1x1 Conv to transform 2 channels (VV, VH) to 3 channels (RGB)
        self.channel_adapter = nn.Conv2d(2, 3, kernel_size=1, stride=1, padding=0)
        
        # Load pretrained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.resnet = resnet50(weights=weights)
        else:
            self.resnet = resnet50(weights=None)
        
        # Remove final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Freeze early layers (fine-tune last 20 layers only)
        for i, param in enumerate(self.resnet.parameters()):
            if i < len(list(self.resnet.parameters())) - 20:
                param.requires_grad = False
        
        # Image feature dimension from ResNet50
        resnet_out_features = 2048
        
        # 3-layer dense network for scale-variant features
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined feature dimension
        combined_features = resnet_out_features + 16
        
        # Final classification layers with L1 regularization
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, image, features):
        """
        Forward pass
        
        Args:
            image: (B, 2, 64, 64) - VV and VH stacked
            features: (B, 14) - scale-variant features
        
        Returns:
            logits: (B, num_classes)
        """
        # Convert 2 channels to 3 channels
        x = self.channel_adapter(image)
        
        # Extract image features with ResNet50
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # (B, 2048)
        
        # Process scale-variant features
        f = self.feature_net(features)  # (B, 16)
        
        # Concatenate features
        combined = torch.cat([x, f], dim=1)  # (B, 2048 + 16)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits


class BaselineModel(nn.Module):
    """
    Baseline model from the paper (8 conv layers)
    """
    
    def __init__(self, num_features=14, num_classes=4):
        super(BaselineModel, self).__init__()
        
        # Convolutional block (8 layers with 3 max pooling)
        self.conv_block = nn.Sequential(
            # Conv block 1
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened size: 64x64 -> 8x8 after 3 pooling layers
        conv_out_size = 256 * 8 * 8
        
        # Feature network
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_size + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, image, features):
        # Process image
        x = self.conv_block(image)
        x = torch.flatten(x, 1)
        
        # Process features
        f = self.feature_net(features)
        
        # Combine and classify
        combined = torch.cat([x, f], dim=1)
        logits = self.classifier(combined)
        
        return logits