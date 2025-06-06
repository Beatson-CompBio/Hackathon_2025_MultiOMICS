import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def evaluate_model_on_preds(preds: np.ndarray, y_true: pd.Series) -> dict:
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    auc = roc_auc_score(y_true, preds)
    accuracy = accuracy_score(y_true, np.round(preds))
    f1 = f1_score(y_true, np.round(preds))
    return {'auc': auc, 'accuracy': accuracy, 'f1': f1}


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class IntermediateIntegrationModel(nn.Module):
    def __init__(self, input_dims, hidden_dim=128, encoded_dim=64, dropout=0.3):
        super().__init__()
        self.encoders = nn.ModuleList([
            ModalityEncoder(in_dim, hidden_dim, encoded_dim, dropout)
            for in_dim in input_dims
        ])

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(len(input_dims) * encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        feats = [encoder(x) for encoder, x in zip(self.encoders, inputs)]
        combined = torch.cat(feats, dim=-1)
        return self.classifier(combined).squeeze(-1)


class IntermediateIntegrationWrapper:
    def __init__(self, hidden_dim=128, encoded_dim=64, dropout=0.3, epochs=30, lr=1e-3, batch_size=32):
        self.model = None
        self.fitted = False
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.encoded_dim = encoded_dim
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def data(self, modalities: dict[str, pd.DataFrame]):
        """
        Extract features from each modality DataFrame.
        Assumes all share the same labels in a 'subtype' column.
        """
        y = next(iter(modalities.values()))['subtype']
        x_dfs = [
            df.drop(columns=['subtype', 'submitter_id.samples']) for df in modalities.values()
        ]
        return x_dfs, y

    def fit(self, train_modalities: dict[str, pd.DataFrame]):
        x_dfs, y_series = self.data(train_modalities)
        x_tensors = [torch.tensor(x.values, dtype=torch.float32) for x in x_dfs]
        y = torch.tensor(y_series.values, dtype=torch.float32)

        dataset = TensorDataset(*x_tensors, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dims = [x.shape[1] for x in x_tensors]
        self.model = IntermediateIntegrationModel(
            input_dims=input_dims,
            hidden_dim=self.hidden_dim,
            encoded_dim=self.encoded_dim,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for *batch_xs, batch_y in loader:
                batch_xs = [x.to(self.device) for x in batch_xs]
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_xs)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.fitted = True

    def predict(self, val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        x_dfs, _ = self.data(val_modalities)
        x_tensors = [torch.tensor(x.values, dtype=torch.float32).to(self.device) for x in x_dfs]

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_tensors)
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

    def wrapper(self, train_modalities: dict[str, pd.DataFrame], val_modalities: dict[str, pd.DataFrame]) -> np.ndarray:
        self.fit(train_modalities)
        return self.predict(val_modalities)


if __name__ == "__main__":
    train_modalities = {
        'rna': pd.read_csv("../../../processed_data/train_rna.csv"),
        'mirna': pd.read_csv("../../../processed_data/train_mir.csv"),
        'cnv': pd.read_csv("../../../processed_data/train_cnv.csv"),
    }

    val_modalities = {
        'rna': pd.read_csv("../../../processed_data/val_rna.csv"),
        'mirna': pd.read_csv("../../../processed_data/val_mir.csv"),
        'cnv': pd.read_csv("../../../processed_data/val_cnv.csv"),
    }

    model = IntermediateIntegrationWrapper()
    predictions = model.wrapper(train_modalities, val_modalities)
    performance = evaluate_model_on_preds(predictions, val_modalities['rna']['subtype'])
    print(performance)

