import flwr as fl              # Flower 
import torch                   # PyTorch 
import torch.nn as nn          # NN
import torch.nn.functional as F # NN functions
import torch.optim as optim    # Optimization algorithms
from torchvision import datasets, transforms  # For handling image datasets
from collections import OrderedDict
import numpy as np
import logging
from datetime import datetime
import time
import os

# Set up logging to track what the client is doing
logging.basicConfig(
    level=logging.INFO,                     # Set logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.StreamHandler(),            # Print logs to console
        logging.FileHandler('federated_learning_client.log')  # Save logs to file
    ]
)#this makes a log file AND GIVES ALL THE DETAILS OF THE CLIENT .ACCURACY LOSS  
logger = logging.getLogger(__name__)

# NN model
class MOONCNN(nn.Module):
    def __init__(self):
        super(MOONCNN, self).__init__()
        # Convolutional layers for processing images
        self.conv1 = nn.Conv2d(1, 32, 3, 1)    # First conv layer: 1 input channel, 32 output channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)   # Second conv layer: 32 input channels, 64 output channels
        
        # Dropout layers to prevent overfitting
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(9216, 128)        # First fully connected layer
        
        # Projection head for MOON contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.fc2 = nn.Linear(128, 10)          # Output layer: 10 classes for MNIST

    def forward(self, x):
        # Define the forward pass of the network
        x = F.relu(self.conv1(x))              # Apply first conv layer and ReLU
        x = F.relu(self.conv2(x))              # Apply second conv layer and ReLU
        x = F.max_pool2d(x, 2)                 # Max pooling to reduce dimensions
        x = self.dropout1(x)                   # Apply dropout
        x = torch.flatten(x, 1)                # Flatten the tensor
        x = self.fc1(x)                        # First fully connected layer
        representation = F.relu(x)             # Get representation for MOON
        projection = self.projection(representation)  # Get projection for MOON
        output = self.fc2(representation)      # Get final output
        return output, projection

# Define the MOON loss function
def moon_loss(local_rep, global_rep, prev_rep, temperature=0.5):
    # Reshape representations
    local_rep = local_rep.view(local_rep.size(0), -1)
    global_rep = global_rep.view(global_rep.size(0), -1)
    prev_rep = prev_rep.view(prev_rep.size(0), -1)
    
    # Calculate similarities between representations
    sim_local_global = F.cosine_similarity(local_rep, global_rep, dim=1)
    sim_local_prev = F.cosine_similarity(local_rep, prev_rep, dim=1)
    
    # Prepare logits and labels for contrastive loss
    logits = torch.cat([sim_local_global.unsqueeze(1), sim_local_prev.unsqueeze(1)], dim=1)
    logits /= temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    
    return F.cross_entropy(logits, labels)

# Define the Federated Learning Client
class MOONClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid                     # Client ID
        # Initialize metrics dictionary to track progress
        self.metrics = {
            'client_id': self.cid,
            'training_history': [],
            'evaluation_history': []
        }
        self.setup_device()                # Set up CPU or GPU
        self.setup_model()                 # Initialize the model
        self.setup_data()                  # Prepare the dataset
        self.prev_model = None             # Previous model for MOON

    def setup_device(self):
        # Determine if GPU is available, otherwise use CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Client {self.cid} using device: {self.device}")

    def setup_model(self):
        # Initialize model, optimizer, and learning rate scheduler
        self.model = MOONCNN().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.7)
        self.criterion = nn.CrossEntropyLoss()

    def setup_data(self):
        # Prepare data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load training dataset
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        
        # Load validation dataset
        val_dataset = datasets.MNIST(
            root='./data', train=False, transform=transform
        )
        self.valloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32
        )

    # Get model parameters as numpy arrays
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # Update model with new parameters from server
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # Train the model
    def fit(self, parameters, config):
        try:
            # Set model parameters received from server
            self.set_parameters(parameters)
            
            # Initialize previous model for MOON if not exists
            if self.prev_model is None:
                self.prev_model = MOONCNN().to(self.device)
            self.prev_model.load_state_dict(self.model.state_dict())
            
            # Initialize metrics for this training round
            round_metrics = {
                'round': config.get('current_round', 0),
                'epoch_metrics': [],
                'total_training_time': 0,
                'total_samples': len(self.trainloader.dataset)
            }
            
            start_time = time.time()
            
            # Get training parameters from config
            epochs = config.get("local_epochs", 1)
            mu = config.get("mu", 1)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                epoch_metrics = {
                    'epoch': epoch,
                    'batch_metrics': [],
                    'epoch_loss': 0,
                    'epoch_accuracy': 0
                }
                
                correct = 0
                total = 0
                
                # Batch training loop
                for batch_idx, (data, target) in enumerate(self.trainloader):
                    batch_start = time.time()
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    output, local_rep = self.model(data)
                    _, global_rep = self.prev_model(data)
                    
                    # Calculate losses
                    loss = self.criterion(output, target)
                    moon_loss_value = moon_loss(local_rep, global_rep, local_rep)
                    total_loss = loss + mu * moon_loss_value
                    
                    # Backward pass and optimization
                    total_loss.backward()
                    self.optimizer.step()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    
                    # Record batch metrics
                    batch_metrics = {
                        'batch': batch_idx,
                        'loss': total_loss.item(),
                        'accuracy': 100 * correct / total,
                        'batch_time': time.time() - batch_start
                    }
                    epoch_metrics['batch_metrics'].append(batch_metrics)
                    
                    # Log progress every 100 batches
                    if batch_idx % 100 == 0:
                        logger.info(f"Client {self.cid} - Epoch {epoch} - Batch {batch_idx}: "
                                   f"Loss: {batch_metrics['loss']:.4f}, "
                                   f"Accuracy: {batch_metrics['accuracy']:.2f}%")
                
                # Calculate and record epoch metrics
                epoch_metrics['epoch_loss'] = sum(b['loss'] for b in epoch_metrics['batch_metrics']) / len(epoch_metrics['batch_metrics'])
                epoch_metrics['epoch_accuracy'] = 100 * correct / total
                round_metrics['epoch_metrics'].append(epoch_metrics)
                
                logger.info(f"Client {self.cid} - Epoch {epoch} completed: "
                           f"Loss: {epoch_metrics['epoch_loss']:.4f}, "
                           f"Accuracy: {epoch_metrics['epoch_accuracy']:.2f}%")
                
                self.scheduler.step()
            
            # Record total training time
            round_metrics['total_training_time'] = time.time() - start_time
            self.metrics['training_history'].append(round_metrics)
            
            # Return updated model parameters and metrics
            return self.get_parameters(config={}), len(self.trainloader.dataset), {
                'client_id': self.cid,
                'round': config.get('current_round', 0),
                'training_time': round_metrics['total_training_time'],
                'final_loss': round_metrics['epoch_metrics'][-1]['epoch_loss'],
                'final_accuracy': round_metrics['epoch_metrics'][-1]['epoch_accuracy']
            }
        
        except Exception as e:
            logger.error(f"Client {self.cid} - Error during training: {e}")
            raise

    # Evaluate the model
    def evaluate(self, parameters, config):
        try:
            # Set model parameters received from server
            self.set_parameters(parameters)
            self.model.eval()
            
            # Initialize evaluation metrics
            eval_metrics = {
                'round': config.get('current_round', 0),
                'loss': 0,
                'accuracy': 0,
                'total_samples': len(self.valloader.dataset)
            }
            
            correct = 0
            # Evaluation loop
            with torch.no_grad():
                for data, target in self.valloader:
                    data, target = data.to(self.device), target.to(self.device)
                    output, _ = self.model(data)
                    eval_metrics['loss'] += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Calculate final metrics
            eval_metrics['accuracy'] = 100 * correct / len(self.valloader.dataset)
            eval_metrics['loss'] /= len(self.valloader)
            
            self.metrics['evaluation_history'].append(eval_metrics)
            
            logger.info(f"Client {self.cid} - Evaluation completed: "
                       f"Loss: {eval_metrics['loss']:.4f}, "
                       f"Accuracy: {eval_metrics['accuracy']:.2f}%")
            
            return eval_metrics['loss'], len(self.valloader.dataset), {
                'client_id': self.cid,
                'accuracy': eval_metrics['accuracy']
            }
        
        except Exception as e:
            logger.error(f"Client {self.cid} - Error during evaluation: {e}")
            raise

# Main function to start the client
def main():
    try:
        # Get client ID from environment variable or use default (0)
        client_id = int(os.getenv('CLIENT_ID', 0))
        logger.info(f"Starting client with ID: {client_id}")
        
        # Start Flower client
        fl.client.start_client(
            server_address="127.0.0.1:8081",
            client=MOONClient(cid=client_id)
        )
    except Exception as e:
        logger.error(f"Error starting client: {e}")

if __name__ == "__main__":
    main()