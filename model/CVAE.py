from utils import *
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset
class CVAE(nn.Module):
    def __init__(self, x_dimension, y_dimension, z_dimension=100, learning_rate=0.001, dropout_rate=0.2, alpha=0.001, model_path="./scgen/CVAE"):
        super(CVAE, self).__init__()
        self.x_dim = x_dimension
        self.y_dim = y_dimension
        self.z_dim = z_dimension
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.model_path = model_path
        self.alpha = alpha
        self.encoder = nn.Sequential(
            nn.Linear(self.x_dim+self.y_dim, 800, bias=False),
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
            nn.Linear(self.z_dim+self.y_dim, 800, bias=False),
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

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = self.encoder(xy)
        mu = self.mu(h)
        log_var = self.log_var(h)
        z = self._sample_z(mu, log_var)
        zy = torch.cat([z, y], dim=1)
        x_hat = self.decoder(zy)
        return x_hat, mu, log_var

    def _sample_z(self, mu, log_var):  # Reparameterization trick.
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def loss_function(self, x_hat, x, mu, log_var):
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + self.alpha * kl_loss

    def to_latent(self, x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        xy = torch.cat([x, y], dim=1)
        h = self.encoder(xy)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return self._sample_z(mu, log_var)

    def reconstruct(self, z, y, use_data=False):
      self.eval()  # Set the model to evaluation mode
      with torch.no_grad():  # Do not calculate gradients
          if use_data:
              latent = z
          else:
              latent = self.to_latent(z, y)  # You need to define this method to transform data to latent space.
          rec_data = self.decoder(latent)  # Assuming the decoder is a method of the model
      return rec_data


    def _avg_vector(self, data, label):
        latent = self.to_latent(data, label)
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
                valid_data_dataset = TensorDataset(torch.tensor(valid_data.X.toarray(), dtype=torch.float32))
            else:
                valid_data_dataset = TensorDataset(torch.tensor(valid_data.X, dtype=torch.float32))
            valid_labels, _ = label_encoder(valid_data)
            if sparse.issparse(valid_labels.X):
                valid_labels_dataset = TensorDataset(torch.tensor(valid_labels.X.toarray(), dtype=torch.float32))
            else:
                valid_labels_dataset = TensorDataset(torch.tensor(valid_labels.X, dtype=torch.float32))

        #print(train_data.X.shape) # (55007, 6619)
        if sparse.issparse(train_data.X):
            train_data_dataset = TensorDataset(torch.tensor(train_data.X.toarray(), dtype=torch.float32))
        else:
            train_data_dataset = TensorDataset(torch.tensor(train_data.X, dtype=torch.float32))
        train_data_loader = DataLoader(train_data_dataset, batch_size=batch_size, shuffle=shuffle)
        
        train_labels, le = label_encoder(train_data)
        if sparse.issparse(train_labels.X):
            train_labels_dataset = TensorDataset(torch.tensor(train_labels.X.toarray(), dtype=torch.float32))
        else:
            train_labels_dataset = TensorDataset(torch.tensor(train_labels.X, dtype=torch.float32))
        train_labels_loader = DataLoader(train_labels_dataset, batch_size=batch_size, shuffle=shuffle)

        optimizer =  torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if save_path == None:
          save_path = self.model_path

        loss_hist = []
        patience_cnt = 0
        patience = early_stop_limit

        for epoch in range(n_epochs):
            train_loss = 0.0

            for (data, label) in zip(train_data_loader, train_labels_loader):
                inputs = torch.cat([data[0],label], dim=1).to(device)
                reconstruction, mu, logvar = self(inputs)
                loss = self.loss_function(reconstruction, inputs, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_data_loader)
            
            if use_validation:
                valid_data_loader = DataLoader(valid_data_dataset, batch_size=batch_size, shuffle=shuffle)
                valid_labels_loader = DataLoader(valid_labels_dataset, batch_size=batch_size, shuffle=shuffle)
                valid_loss = 0.0
                for (data, label) in zip(valid_data_loader, valid_labels_loader):
                    inputs = torch.cat([data[0],label], dim=1).to(device)
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
      # todo