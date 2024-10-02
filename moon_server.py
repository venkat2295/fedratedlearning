import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import logging
from datetime import datetime
import time
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging for the server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('federated_learning_server.log')
    ]
)
logger = logging.getLogger(__name__)

# MongoDB setup for storing results
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'federated_learning')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'training_rounds')

# Try to connect to MongoDB
try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Test MongoDB connection
    mongo_client.admin.command('ismaster')
    logger.info("Successfully connected to MongoDB")
    
    # Log server start event
    collection.insert_one({
        "event": "server_start",
        "timestamp": datetime.now(),
        "status": "initialized"
    })
except Exception as e:
    logger.error(f"MongoDB setup failed: {e}")
    mongo_client = None
    db = None
    collection = None
    # SERVER CODE (continued)

class MOONStrategy(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,          # Fraction of clients to use for training
        fraction_evaluate: float = 1.0,     # Fraction of clients to use for evaluation
        min_fit_clients: int = 2,           # Minimum number of clients for training
        min_evaluate_clients: int = 2,      # Minimum number of clients for evaluation
        min_available_clients: int = 2,     # Minimum number of clients that need to be available
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        self.round_history = []             # Keep track of training history

    def aggregate_fit(
        self,
        server_round: int,                  # Current round number
        results: List[Tuple[ClientProxy, FitRes]], # Results from clients
        failures: List[BaseException],      # Any failures that occurred
    ) -> Optional[Parameters]:
        try:
            # Check if we received any results
            if not results:
                logger.warning(f"Round {server_round}: No results received from clients")
                return None
            
            # Aggregate parameters using parent class method
            aggregated_parameters = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_parameters is not None:
                # Process and record the round results
                round_metrics = self._process_round_results(server_round, results, failures)
                self.round_history.append(round_metrics)
                
                # Log the round summary
                logger.info(f"Round {server_round} completed:\n"
                           f"Number of clients: {round_metrics['num_clients']}\n"
                           f"Average loss: {round_metrics['average_loss']:.4f}\n"
                           f"Average accuracy: {round_metrics['average_accuracy']:.2f}%\n"
                           f"Round duration: {round_metrics['round_duration']:.2f} seconds")
            
            return aggregated_parameters
        
        except Exception as e:
            logger.error(f"Error in aggregate_fit: {e}")
            return None

    def _process_round_results(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], 
                              failures: List[BaseException]) -> Dict:
        start_time = time.time()
        
        # Initialize metrics tracking
        client_metrics = []
        total_examples = 0
        weighted_loss = 0
        weighted_accuracy = 0
        
        # Process results from each client
        for client_proxy, fit_res in results:
            client_metric = {
                'client_id': fit_res.metrics.get('client_id', 'unknown'),
                'num_examples': fit_res.num_examples,
                'loss': fit_res.metrics.get('final_loss', 0),
                'accuracy': fit_res.metrics.get('final_accuracy', 0),
                'training_time': fit_res.metrics.get('training_time', 0)
            }
            client_metrics.append(client_metric)
            
            # Calculate weighted metrics
            total_examples += fit_res.num_examples
            weighted_loss += client_metric['loss'] * fit_res.num_examples
            weighted_accuracy += client_metric['accuracy'] * fit_res.num_examples
        
        # Compile round metrics
        round_metrics = {
            'round': server_round,
            'timestamp': datetime.now().isoformat(),
            'num_clients': len(results),
            'num_failures': len(failures),
            'total_examples': total_examples,
            'average_loss': weighted_loss / total_examples if total_examples > 0 else 0,
            'average_accuracy': weighted_accuracy / total_examples if total_examples > 0 else 0,
            'round_duration': time.time() - start_time,
            'client_metrics': client_metrics
        }
        
        # Log metrics to MongoDB if available
        if collection is not None:
            try:
                collection.insert_one(round_metrics)
                logger.info(f"Round {server_round} results successfully logged to MongoDB")
            except Exception as e:
                logger.error(f"Error logging round results to MongoDB: {e}")
        
        return round_metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        # Configure the training parameters for this round
        config = {
            "local_epochs": int(os.getenv('LOCAL_EPOCHS', '1')),  # Number of local training epochs
            "mu": float(os.getenv('MU', '1.0')),                  # MOON hyperparameter
            "current_round": server_round
        }
        
        # Create FitIns object with parameters and config
        fit_ins = fl.common.FitIns(parameters, config)

        # Sample clients for this round
        sample_size = int(self.fraction_fit * len(client_manager.clients))
        clients = client_manager.sample(
            num_clients=max(sample_size, self.min_fit_clients),
            min_num_clients=self.min_available_clients
        )
        
        # Log round configuration
        logger.info(f"Round {server_round} configuration:\n"
                   f"Number of selected clients: {len(clients)}\n"
                   f"Local epochs: {config['local_epochs']}\n"
                   f"Mu parameter: {config['mu']}")
        
        return [(client, fit_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        # Check if we received any evaluation results
        if not results:
            logger.warning(f"Round {server_round}: No evaluation results received from clients")
            return None, {}
        
        try:
            # Calculate weighted averages of evaluation metrics
            total_examples = sum(r.num_examples for _, r in results)
            weighted_loss = sum(r.loss * r.num_examples for _, r in results)
            weighted_accuracy = sum(r.metrics['accuracy'] * r.num_examples for _, r in results)
            
            average_loss = weighted_loss / total_examples if total_examples > 0 else 0
            average_accuracy = weighted_accuracy / total_examples if total_examples > 0 else 0
            
            # Compile evaluation metrics
            eval_metrics = {
                'round': server_round,
                'timestamp': datetime.now().isoformat(),
                'num_clients': len(results),
                'num_failures': len(failures),
                'total_examples': total_examples,
                'average_loss': float(average_loss),
                'average_accuracy': float(average_accuracy),
                'client_metrics': [
                    {
                        'client_id': r.metrics.get('client_id', 'unknown'),
                        'num_examples': r.num_examples,
                        'loss': float(r.loss),
                        'accuracy': float(r.metrics['accuracy'])
                    }
                    for _, r in results
                ]
            }
            
            # Log evaluation metrics to MongoDB if available
            if collection is not None:
                try:
                    collection.insert_one({
                        'type': 'evaluation',
                        **eval_metrics
                    })
                except Exception as e:
                    logger.error(f"Error logging evaluation results to MongoDB: {e}")
            
            # Log evaluation summary
            logger.info(f"Round {server_round} evaluation completed:\n"
                       f"Number of clients: {len(results)}\n"
                       f"Average loss: {average_loss:.4f}\n"
                       f"Average accuracy: {average_accuracy:.2f}%")
            
            return float(average_loss), {"accuracy": float(average_accuracy)}
        except Exception as e:
            logger.error(f"Error in aggregate_evaluate: {e}")
            return None, {}

def main():
    try:
        # Server configuration
        server_config = fl.server.ServerConfig(
            num_rounds=int(os.getenv('NUM_ROUNDS', '3'))  # Number of federated learning rounds
        )
        
        # Strategy configuration
        strategy = MOONStrategy(
            min_fit_clients=int(os.getenv('MIN_FIT_CLIENTS', '2')),
            min_available_clients=int(os.getenv('MIN_AVAILABLE_CLIENTS', '2')),
            min_evaluate_clients=int(os.getenv('MIN_EVALUATE_CLIENTS', '2')),
            fraction_fit=float(os.getenv('FRACTION_FIT', '1.0')),
            fraction_evaluate=float(os.getenv('FRACTION_EVALUATE', '1.0'))
        )

        # Log server configuration
        logger.info(f"Starting server with configuration:\n"
                   f"Number of rounds: {server_config.num_rounds}\n"
                   f"Minimum fit clients: {strategy.min_fit_clients}\n"
                   f"Minimum available clients: {strategy.min_available_clients}\n"
                   f"Minimum evaluate clients: {strategy.min_evaluate_clients}\n"
                   f"Fraction fit: {strategy.fraction_fit}\n"
                   f"Fraction evaluate: {strategy.fraction_evaluate}")

        # Start Flower server
        fl.server.start_server(
            server_address=os.getenv('SERVER_ADDRESS', '0.0.0.0:8081'),
            strategy=strategy,
            config=server_config
        )
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()