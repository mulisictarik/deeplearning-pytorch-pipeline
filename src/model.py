import torch.nn as nn

class CIFAR10InferenceModel(nn.Module):
   
    def __init__(self, dropout_prob=0.3):
        super(CIFAR10InferenceModel, self).__init__()
        
        # Feature extraction layers (Convolutional Base)
        self.feature_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classification head (Dense Layers)
        self.output_head = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob), 
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # Pass input through the feature extractor
        x = self.feature_layers(x)
        
        # Flatten the output for the linear layers
        x = x.view(x.size(0), -1) 
        
        # Generate class scores
        return self.output_head(x)

def get_model():
    return CIFAR10InferenceModel()