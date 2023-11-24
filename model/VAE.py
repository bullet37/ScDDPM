from utils import *
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset

class VAE(nn.Module):
    def __init__(self, x_dimension, z_dimension=100, learning_rate=0.001, dropout_rate=0.2, alpha=0.001, model_path="./scgen"):
        super(VAE, self).__init__()
        self.x_dim = x_dimension
        self.z_dim = z_dimension
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model_path = model_path
        self.alpha = alpha
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim, 800, bias=False),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(800, 800, bias=False),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.mu = nn.Linear(800, self.z_dim)
        self.log_var = nn.Linear(800, self.z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 800, bias=False),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(800, 800, bias=False),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(800, self.x_dim),
            nn.ReLU()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        z = self._sample_z(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var
    def _sample_z(self, mu, log_var):  # Reparameterization trick.
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def loss_function(self, x_hat, x, mu, log_var):
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.alpha * kl_loss

    def to_latent(self, x):
        x = torch.tensor(x)
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return self._sample_z(mu, log_var)

    def reconstruct(self, z, use_data=False):
      self.eval()  # Set the model to evaluation mode
      with torch.no_grad():  # Do not calculate gradients
          if use_data:
              latent = z
          else:
              latent = self.to_latent(z)  # You need to define this method to transform data to latent space.
          rec_data = self.decoder(latent)  # Assuming the decoder is a method of the model
      return rec_data


    def _avg_vector(self, data):
        latent = self.to_latent(data)
        latent_avg = np.average(latent.detach().numpy(), axis=0)
        return torch.from_numpy(latent_avg)

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)

    def restore_model(self,load_path=None):
        if load_path== None:
          load_path = self.model_path
        self.load_state_dict(torch.load(load_path+"model.pth"))


    def train_vae(self, train_data, valid_data=None, use_validation=False,
                    n_epochs=25, batch_size=32, early_stop_limit=20, threshold=0.0025, shuffle=True, save_path=None, use_cuda=False ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        if use_validation and valid_data is None:
            raise Exception("valid_data is None but use_validation is True.")

        if use_validation:
            if sparse.issparse(valid_data.X):
                valid_dataset = TensorDataset(torch.tensor(valid_data.X.toarray(), dtype=torch.float32))
            else:
                valid_dataset = TensorDataset(torch.tensor(valid_data.X, dtype=torch.float32))

        #print(train_data.X.shape) # (55007, 6619)
        if sparse.issparse(train_data.X):
            train_dataset = TensorDataset(torch.tensor(train_data.X.toarray(), dtype=torch.float32))
        else:
            train_dataset = TensorDataset(torch.tensor(train_data.X, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        optimizer =  torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if save_path == None:
          save_path = self.model_path
        loss_hist = []
        patience_cnt = 0
        patience = early_stop_limit
        for epoch in range(n_epochs):
            train_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs = data[0].to(device)
                print("inputs:")
                print(inputs.shape)
                reconstruction, mu, logvar = self(inputs)
                loss = self.loss_function(reconstruction, inputs, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            if use_validation:
                valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
                valid_loss = 0.0
                for i, data in enumerate(valid_loader):
                    inputs = data[0].to(device)
                    with torch.no_grad():
                        reconstruction, mu, logvar = self(inputs)
                        loss = self.loss_function(reconstruction, inputs, mu, logvar)
                    valid_loss += loss.item()

                loss_hist.append(valid_loss / len(valid_data))

                if epoch > 0 and loss_hist[epoch - 1] - loss_hist[epoch] > threshold:
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt > patience:
                    if save_path:
                        torch.save(self.state_dict(), save_path)
                    break
            print(f"Epoch {epoch+1}: Train Loss: {train_loss / len(train_data)}")

        if save_path:
            if not os.path.exists(save_path):
              os.makedirs(save_path)
            torch.save(self.state_dict(), save_path+"model.pth")
            print(f"Model saved in file: {save_path}. Training finished")

    def predict(self,  adata, conditions, cell_type_key, condition_key, adata_to_predict=None, celltype_to_predict=None,
            obs_key="all", biased=False, use_cuda=False):
      self.eval()
      # 4542 × 7000
      if obs_key == "all":
          ctrl_x = adata[adata.obs[condition_key] == conditions["ctrl"], :] # 2736 × 7000
          stim_x = adata[adata.obs[condition_key] == conditions["stim"], :] # 1806 × 7000
          if not biased:
              ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key) # 5328 × 7000
              stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key) # 3444 × 7000
      else:
          key = list(obs_key.keys())[0]
          values = obs_key[key]
          subset = adata[adata.obs[key].isin(values)]
          ctrl_x = subset[subset.obs[condition_key] == conditions["ctrl"], :]
          stim_x = subset[subset.obs[condition_key] == conditions["stim"], :]
          if len(values) > 1 and not biased:
              ctrl_x = balancer(ctrl_x, cell_type_key=cell_type_key, condition_key=condition_key)
              stim_x = balancer(stim_x, cell_type_key=cell_type_key, condition_key=condition_key)
      if celltype_to_predict is not None and adata_to_predict is not None:
          raise Exception("Please provide either a cell type or adata not both!")
      if celltype_to_predict is None and adata_to_predict is None:
          raise Exception("Please provide a cell type name or adata for your unperturbed cells")
      if celltype_to_predict is not None:
          ctrl_pred = extractor(adata, celltype_to_predict, conditions, cell_type_key, condition_key)[1]
          # data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == conditions["ctrl"])]
      else:
          ctrl_pred = adata_to_predict
      if not biased:
          eq = min(ctrl_x.shape[0], stim_x.shape[0])  # 3444
          cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
          stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)

      else:
          cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=ctrl_x.shape[0], replace=False)
          stim_ind = np.random.choice(range(stim_x.shape[0]), size=stim_x.shape[0], replace=False)
      if sparse.issparse(ctrl_x.X) and sparse.issparse(stim_x.X):
            latent_ctrl = self._avg_vector(ctrl_x.X.A[cd_ind, :])
            latent_sim = self._avg_vector(stim_x.X.A[stim_ind, :])
      else:
            latent_ctrl = self._avg_vector(ctrl_x.X[cd_ind, :])
            latent_sim = self._avg_vector(stim_x.X[stim_ind, :])
      delta = latent_sim - latent_ctrl
      if sparse.issparse(ctrl_pred.X):
            latent_cd = self.to_latent(ctrl_pred.X.A)
      else:
          latent_cd = self.to_latent(ctrl_pred.X)

      stim_pred = delta + latent_cd
      predicted_cells = self.reconstruct(stim_pred, use_data=True)
      return predicted_cells, delta