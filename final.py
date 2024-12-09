import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from torch.utils.data import DataLoader, TensorDataset

# Check GPU availability
device = torch.device('cuda')
print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")


# Data Preprocessing Function
def preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # GC Content Function
    def gc_content(seq):
        sequ = seq.upper()
        gc_ratio = (sequ.count('G') + sequ.count('C')) / len(seq)
        return 1 if gc_ratio > 0.5 else 0

    # Preprocessing cell column
    data['cell'] = data['cell'].str.split('-').str[1].apply(gc_content)

    # Feature Selection based on correlation
    corr = data.corr()['cell']
    corr = corr.drop(['cell'])

    # Select features with moderate correlation
    selected_features = [
        i for i in corr.index
        if 0.04 < abs(corr[i]) < 0.7
    ]

    # Prepare X and y
    X = data[selected_features]
    X = np.log1p(X)
    y = data['cell']

    return X, y


# Neural Network Model
class CellClassificationNet(nn.Module):
    def __init__(self, input_size):
        super(CellClassificationNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.float())

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate training metrics
            total_train_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.float())

                total_val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        # Calculate metrics
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        train_accuracy = correct_train / total_train
        val_accuracy = correct_val / total_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Print epoch results
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, val_losses, train_accuracies, val_accuracies


# Main Execution
def main():
    # Load and preprocess data
    X, y = preprocess_data("./data/CellsKmers.csv")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=50)  # Adjust components based on explained variance
    X_train_scaled = pca.fit_transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)

    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_train_scaled = poly.fit_transform(X_train_scaled)
    X_test_scaled = poly.transform(X_test_scaled)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Split train into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model
    model = CellClassificationNet(input_size=X_train_scaled.shape[1]).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer
    )

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = (outputs.squeeze() > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    # Training History Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')


# Run the main function
if __name__ == '__main__':
    main()
