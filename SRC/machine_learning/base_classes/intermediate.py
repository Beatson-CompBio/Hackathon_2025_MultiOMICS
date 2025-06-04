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
        super(ModalityEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class IntermediateIntegrationModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim=128, encoded_dim=64, dropout=0.3):
        super(IntermediateIntegrationModel, self).__init__()
        self.encoder1 = ModalityEncoder(input_dim1, hidden_dim, encoded_dim, dropout)
        self.encoder2 = ModalityEncoder(input_dim2, hidden_dim, encoded_dim, dropout)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2):
        feat1 = self.encoder1(x1)
        feat2 = self.encoder2(x2)
        combined = torch.cat((feat1, feat2), dim=-1)
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

    def data(self, modality1, modality2):
        y = modality1['subtype']
        x1 = modality1.drop(columns=['subtype', 'submitter_id.samples'])
        x2 = modality2.drop(columns=['subtype', 'submitter_id.samples'])
        return x1, x2, y

    #this is hard coded to two modalities! big no no!
    def fit(self, train_mod1: pd.DataFrame, train_mod2: pd.DataFrame):
        x1_df, x2_df, y_series = self.data(train_mod1, train_mod2)
        x1 = torch.tensor(x1_df.values, dtype=torch.float32)
        x2 = torch.tensor(x2_df.values, dtype=torch.float32)
        y = torch.tensor(y_series.values, dtype=torch.float32)

        dataset = TensorDataset(x1, x2, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim1 = x1.shape[1]
        input_dim2 = x2.shape[1]
        self.model = IntermediateIntegrationModel(
            input_dim1, input_dim2,
            hidden_dim=self.hidden_dim,
            encoded_dim=self.encoded_dim,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x1, batch_x2, batch_y in loader:
                batch_x1, batch_x2, batch_y = batch_x1.to(self.device), batch_x2.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x1, batch_x2)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.fitted = True

    def predict(self, val_mod1: pd.DataFrame, val_mod2: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        x1_df, x2_df, _ = self.data(val_mod1, val_mod2)
        x1 = torch.tensor(x1_df.values, dtype=torch.float32).to(self.device)
        x2 = torch.tensor(x2_df.values, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x1, x2)
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs


if __name__ == "__main__":
    train_rna = pd.read_csv("../../../processed_data/train_rna.csv")
    train_mirna = pd.read_csv("../../../processed_data/train_mir.csv")
    val_rna = pd.read_csv("../../../processed_data/val_rna.csv")
    val_mirna = pd.read_csv("../../../processed_data/val_mir.csv")

    model = IntermediateIntegrationWrapper()
    model.fit(train_rna, train_mirna)
    predictions = model.predict(val_rna, val_mirna)

    performance = evaluate_model_on_preds(predictions, val_rna['subtype'])
    print(performance)
