import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from src.bias.bias_subspace import BiasSubspace
from tqdm import tqdm

class DeepSoftDebiasNetwork(nn.Module):
    """
    Neural Network for Deep Soft Debiasing.
    Uses an Autoencoder-like architecture to map original embeddings
    to debiased embeddings.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h)

class DeepSoftDebias:
    """
    Implementation of Deep Soft Debiasing (DSD) using Deep Neural Networks.
    
    This trains a network to map word embeddings to a new space where:
    1. The embeddings are close to the original (semantic preservation).
    2. The projection onto the bias subspace is minimized (bias removal).
    """
    
    def __init__(self, 
                 lambda_2: float = 1.0, 
                 k: int = 1, 
                 hidden_dim: int = 300,
                 lr: float = 1e-3, 
                 epochs: int = 100):
        self.lambda_2 = lambda_2
        self.k = k
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.bias_subspace = BiasSubspace(k=self.k)
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, embeddings: Dict[str, np.ndarray], gender_pairs: List[Tuple[str, str]]):
        """
        Trains the neural network to debias the embeddings.
        """
        # 1. Fit the standard bias subspace
        self.bias_subspace.fit(embeddings, gender_pairs)
        
        # Extract the bias projection matrix P_B (ensure it's a PyTorch tensor)
        P_B = torch.tensor(self.bias_subspace.P_B, dtype=torch.float32).to(self.device)
        
        # 2. Prepare data for PyTorch
        words = list(embeddings.keys())
        X = np.stack([embeddings[w] for w in words])
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        input_dim = X.shape[1]
        self.model = DeepSoftDebiasNetwork(input_dim, self.hidden_dim).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        mse_loss = nn.MSELoss()
        
        # 3. Train the network
        self.model.train()
        print(f"Training Deep Soft Debiasing network for {self.epochs} epochs...")
        for epoch in tqdm(range(self.epochs), desc="Training DSD", unit="epoch"):
            optimizer.zero_grad()
            
            # Forward pass
            X_hat = self.model(X_tensor)
            
            # Loss 1: Semantic preservation
            L_sem = mse_loss(X_hat, X_tensor)
            
            # Loss 2: Bias mitigation
            bias_projection = torch.matmul(X_hat, P_B.T)
            L_bias = torch.mean(torch.sum(bias_projection ** 2, dim=1))
            
            # Total Loss
            loss = L_sem + self.lambda_2 * L_bias
            
            loss.backward()
            optimizer.step()
            tqdm.write(f"Epoch {epoch+1}/{self.epochs} | "
               f"Loss: {loss.item():.6f} | "
               f"L_sem: {L_sem.item():.6f} | "
               f"L_bias: {L_bias.item():.6f}")
                
        self.model.eval()

    def transform(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Passes a dictionary of embeddings through the trained network.
        """
        if self.model is None:
            raise ValueError("DeepSoftDebias must be fitted before calling transform().")
            
        debiased_embeddings = {}
        with torch.no_grad():
            for word, vec in embeddings.items():
                v_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                v_hat = self.model(v_tensor).squeeze(0).cpu().numpy()
                debiased_embeddings[word] = v_hat
                
        return debiased_embeddings
        
    def transform_tensor(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Passes a raw numpy array of embeddings (N, D) through the trained network.
        """
        if self.model is None:
            raise ValueError("DeepSoftDebias must be fitted before calling transform_tensor().")
            
        with torch.no_grad():
            X_tensor = torch.tensor(embedding_matrix, dtype=torch.float32).to(self.device)
            X_hat = self.model(X_tensor).cpu().numpy()
            
        return X_hat
