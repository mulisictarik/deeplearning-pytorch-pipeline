import torch
import torch.nn as nn
import json
import os
from tqdm import tqdm

def train_and_validate(model, train_loader, test_loader, config, device, run_name):
   
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss_log': [], 'accuracy_log': []}
    best_accuracy = 0.0

    print(f"\nInitializing training: {run_name} for {config['epochs']} epochs")
    print("-" * 50)

    for epoch in range(config['epochs']):
        # --- TRAINING PHASE ---
        model.train()
        current_loss = 0.0
        
        for batch_data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_data, labels = batch_data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            current_loss += loss.item()

        # --- VALIDATION PHASE ---
        model.eval()
        correct_preds = 0
        total_samples = 0
        epoch_test_loss = 0.0
        
        with torch.no_grad():
            for batch_data, labels in test_loader:
                batch_data, labels = batch_data.to(device), labels.to(device)
                output = model(batch_data)
                
                # Calculate validation loss
                t_loss = criterion(output, labels)
                epoch_test_loss += t_loss.item()
                
                # Calculate accuracy
                prediction = output.argmax(dim=1)
                total_samples += labels.size(0)
                correct_preds += (prediction == labels).sum().item()

        # Calculate metrics
        avg_train_loss = current_loss / len(train_loader)
        avg_test_loss = epoch_test_loss / len(test_loader)
        acc = 100. * correct_preds / total_samples
        
        history['loss_log'].append(avg_train_loss)
        history['accuracy_log'].append(acc)

        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | "
              f"Test Acc: {acc:.2f}%")

        # Automatically checkpoint best model
        if acc > best_accuracy:
            best_accuracy = acc
            save_folder = 'models'
            os.makedirs(save_folder, exist_ok=True)
            torch.save(model.state_dict(), f"{save_folder}/{run_name}_best.pth")

    return {"best_accuracy": best_accuracy, "history": history}

def export_metrics(metrics_data, experiment_id):
    """Saves session results to the metrics directory"""
    os.makedirs('metrics', exist_ok=True)
    with open(f"metrics/metrics_{experiment_id}.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

