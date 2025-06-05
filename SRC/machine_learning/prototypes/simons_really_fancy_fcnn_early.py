import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model_on_preds(preds: np.ndarray, y_true: pd.Series) -> dict:
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    if preds.shape[1] > 1:
        preds = preds[:, 1]  # use positive class

    auc = roc_auc_score(y_true, preds)
    accuracy = accuracy_score(y_true, np.round(preds))
    f1 = f1_score(y_true, np.round(preds))
    return {'auc': auc, 'accuracy': accuracy, 'f1': f1}


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)  # For binary classification with softmax

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class Early_or_Single_Model:
    def __init__(self, epochs=30, lr=1e-3, batch_size=32):
        self.model = None
        self.fitted = False
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def data(self, *modalities: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y = modalities[0]['subtype']
        features = [mod.drop(columns=['subtype', 'submitter_id.samples']) for mod in modalities]
        X = pd.concat(features, axis=1)
        return X, y

    def fit(self, *train_modalities: pd.DataFrame):
        X_train_df, y_train_series = self.data(*train_modalities)
        X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
        y_train = torch.tensor(y_train_series.values, dtype=torch.long)

        input_dim = X_train.shape[1]
        self.model = SimpleNN(input_dim).to(self.device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.fitted = True

    def predict(self, *val_modalities: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X_val_df, _ = self.data(*val_modalities)
        X_val = torch.tensor(X_val_df.values, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_val)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        return probs

    def wrapper(self, train_modalities, val_modalities) -> np.ndarray:
        self.fit(*train_modalities)
        preds = self.predict(*val_modalities)
        return preds[:, 1]  # Probability of class 1


# --- Example usage ---

if __name__ == "__main__":
    train_rna = pd.read_csv("../../../processed_data/train_rna.csv")
    train_mirna = pd.read_csv("../../../processed_data/train_mir.csv")

    val_rna = pd.read_csv("../../../processed_data/val_rna.csv")
    val_mirna = pd.read_csv("../../../processed_data/val_mir.csv")

    model = Early_or_Single_Model()
    predictions = model.wrapper([train_rna, train_mirna], [val_rna, val_mirna])

    performance = evaluate_model_on_preds(predictions, val_rna['subtype'])
    print(performance)
