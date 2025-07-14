from dataset import HDF5Dataset
from CNN_classifier import BasicCNNClassifier
from ViT_classifier import ViTClassifier # Vision Transformer-based classifier

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import argparse

def accuracy(outputs, y):
    y_hat = (torch.sigmoid(outputs) > .5).float()
    return (y_hat == y).float().mean()

def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    """
    Trains the model for one epoch over the provided data loader.
    - Computes binary cross-entropy loss and tracks average loss and accuracy.
    - Applies gradient clipping for stability.
    - Updates optimizer and learning rate scheduler at each step.
    Returns:
        Tuple of (average_loss, average_accuracy) over the epoch.
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.squeeze().float().to(device) 
        outputs = model(inputs).squeeze()             
        with torch.no_grad():
            acc = accuracy(outputs, labels)
            running_acc += acc.item()
        # Compute binary cross-entropy loss for this batch.
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        # Clip gradients to avoid exploding gradients and improve training stability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #added clip grad norm for better stability
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    return running_loss / len(data_loader), running_acc / len(data_loader)

def validate_one_epoch(model, data_loader, device):
    """ Returns accuracy for validation set in no_grad mode
    """
    model.eval()
    running_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.squeeze().float().to(device)
            outputs = model(inputs).squeeze()
            acc = accuracy(outputs, labels)
            running_acc += acc.item()
    print("Validation Accuracy: ", running_acc / len(data_loader))
    return running_acc / len(data_loader)

def write_test_predictions(model, data_loader, device):
    """
    Runs inference on the test dataset and writes predicted labels to './outputs/predictions.txt'.
    - Each prediction is written on a separate line, following the required submission format.
    - Assumes binary classification; outputs 0 or 1 per sample.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0] # Assumes DataLoader returns (inputs, ...) 
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_labels = (torch.sigmoid(outputs) > 0.5).long().squeeze().cpu().numpy()
            preds.extend(pred_labels.tolist())
    output_file = f"./outputs/predictions.txt"
    with open(output_file, 'w') as f:
        for p in preds:
            f.write(f"{p}\n")
    print(f"Predictions written to {output_file}")


if __name__ == "__main__":
    # Command line argument parsing for model and mode selection.
    """
    Supports two key arguments:
    - '--model': Choose between 'vit' (Vision Transformer, default) and 'cnn' (Convolutional Neural Network).
    - '--mode' : Choose 'train' (train a new model) or 'test' (run inference with a saved model), defaults to 'train'.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit', choices=['cnn', 'vit'],
                        help="Model architecture to use: 'vit' (default) or 'cnn'")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help="Execution mode: 'train' (default) to train a new model, or 'test' to run inference with a saved model")
    args = parser.parse_args()

    epochs = 100
    batch_size = 8
    learning_rate = 1e-5
    input_dim = 2
    num_classes = 1
    input_shape = 28
    dropout = 0.2
    weight_decay = 1e-3
  

    #Process input data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = HDF5Dataset("data/train.hdf5")
    val_dataset = HDF5Dataset("data/valid.hdf5")
    test_dataset = HDF5Dataset("data/test_no_labels.hdf5")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=4, pin_memory= True)
    test_loader= DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=4, pin_memory= True)
   
    # Model selection based on command line argument.
    # Instantiates either a CNN or Vision Transformer (ViT) model, and moves it to the target device.
    if args.model == 'cnn':
        # Initialize a basic convolutional neural network classifier.
        model = BasicCNNClassifier(input_dim=input_dim, num_classes=num_classes, input_shape=input_shape, dropout=dropout).to(device)
    else:
        # Initialize a Vision Transformer (ViT) classifier.
        model = ViTClassifier(img_size=28, patch_size=2, in_chans=input_dim, num_classes=num_classes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=len(train_loader) * epochs)
    best_val_acc = 0.0

    # Main execution block: supports both training and inference modes.
    """
    Two modes are available:
    - Training mode (`args.mode == 'train'`): trains a new model from scratch and checkpoints the best model.
    - Inference mode (`args.mode == 'test'`): loads the best saved model and generates predictions on the test set.
    """

    if args.mode =='train':
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            val_acc = validate_one_epoch(model, val_loader, device)
            # Save model checkpoint if current validation accuracy improves the previous best.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pt')
                print(f"Best model saved with validation acc: {val_acc}")
            print(f"Epoch {epoch} | Train Loss {train_loss} | Train Acc {train_acc} | Val acc {val_acc}")
    else:
        # Load the best saved model and run inference on the test dataset.
        model.load_state_dict(torch.load('best_model.pt'))
        write_test_predictions(model, test_loader, device)


