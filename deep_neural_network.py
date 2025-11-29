import torch
import torch.nn as nn

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        """
        Deep Neural Network for multi-label classification
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output labels (5 in this case)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(DeepNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.clamp(x, min=-1e6, max=1e6)
        x = self.network(x)
        output = self.sigmoid(x)
        output = torch.clamp(output, min=1e-7, max=1-1e-7)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual   
        out = self.relu(out)
        return out


class DeepResidualNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_blocks=3, dropout_rate=0.3):
        """
        Deep Residual Network for capturing non-linear relationships
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output labels
            hidden_dim: Dimension of hidden layers
            num_blocks: Number of residual blocks
            dropout_rate: Dropout rate for regularization
        """
        super(DeepResidualNetwork, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.clamp(x, min=-1e6, max=1e6)
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.output_layer(x)
        output = self.sigmoid(x)
        output = torch.clamp(output, min=1e-7, max=1-1e-7)
        return output