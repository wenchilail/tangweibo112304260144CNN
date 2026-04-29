import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


class MNISTDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data / 255.0
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].reshape(28, 28, 1)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        
        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(config, exp_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Experiment: {exp_name}, Using device: {device}')

    train_df = pd.read_csv('digit-recognizer/train.csv')
    X = train_df.drop('label', axis=1).values
    y = train_df['label'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    if config['data_augmentation']:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    train_dataset = MNISTDataset(X_train, y_train, transform=train_transform)
    val_dataset = MNISTDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    train_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        epoch_val_acc = correct_val / total_val

        train_losses.append(epoch_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f'{exp_name} - Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Acc: {epoch_val_acc:.4f}')

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'model_{exp_name}.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if config['early_stopping'] and patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    results = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'min_loss': min(train_losses),
        'final_train_acc': train_accs[-1],
    }
    
    return results


def main():
    configs = {
        'Exp1': {
            'optimizer': 'SGD',
            'lr': 0.01,
            'batch_size': 64,
            'data_augmentation': False,
            'early_stopping': False,
        },
        'Exp2': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 64,
            'data_augmentation': False,
            'early_stopping': False,
        },
        'Exp3': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 128,
            'data_augmentation': False,
            'early_stopping': True,
        },
        'Exp4': {
            'optimizer': 'Adam',
            'lr': 0.001,
            'batch_size': 64,
            'data_augmentation': True,
            'early_stopping': True,
        },
    }

    all_results = {}
    for exp_name, config in configs.items():
        results = train_model(config, exp_name)
        all_results[exp_name] = results

    plt.figure(figsize=(12, 6))
    for exp_name, results in all_results.items():
        plt.plot(results['train_losses'], label=f'{exp_name}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    print('Loss curves saved as loss_curves.png')

    print('\n' + '='*80)
    print('实验结果总结：')
    print('='*80)
    for exp_name, results in all_results.items():
        print(f'{exp_name}:')
        print(f'  最佳验证准确率: {results["best_val_acc"]:.4f}')
        print(f'  最终训练准确率: {results["final_train_acc"]:.4f}')
        print(f'  最低损失: {results["min_loss"]:.4f}')
        print(f'  收敛Epoch: {results["best_epoch"]}')
        print()


if __name__ == '__main__':
    main()
