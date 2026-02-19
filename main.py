import torch
import yaml
import sys
import os

sys.path.append(os.path.abspath("src"))

from dataset import get_cifar10_loaders
from model import get_model
from train import train_and_validate, export_metrics

def run_pipeline():
    #Loads the YAML configuration and sequentially runs all experiments
    config_path = 'params.yaml'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return
    
    # Read "r" all experiments from file

    with open(config_path, "r") as f:
        all_experiments = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System status: Executing on {device}")

    #Loop every experiment
    for exp_name, config in all_experiments.items():
        print(f"\n{'='*50}")
        print(f" STARTING: {exp_name.upper()}")
        print(f" Configuration: {config}")
        print(f"{'='*50}")
        
        #Load data based on the batch size
        train_loader, test_loader, classes = get_cifar10_loaders(batch_size=config['batch_size'])
        
        model = get_model().to(device)
        
         # Train and save results
        results = train_and_validate(model, train_loader, test_loader, config, device, exp_name)
        export_metrics(results, exp_name)

if __name__ == "__main__":
    run_pipeline()
