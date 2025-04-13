import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, emb_dim=1024, num_points=1024, use_bn=True):
        super(Autoencoder, self).__init__()
        self.emb_dim = emb_dim
        self.num_points = num_points
        self.use_bn = use_bn

        # Encoder: Refine the global feature vector
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.BatchNorm1d(512) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, emb_dim)  # Refined global feature vector
        )

        # Decoder: Fully connected layers to reconstruct point cloud
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.BatchNorm1d(512) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)  # Output: (B, num_points * 3)
        )

    def forward(self, x):
        # x: (B, emb_dim)

        # Encoder: Refine the global feature vector
        x = self.encoder(x)  # (B, emb_dim)

        # Decoder: Reconstruct point cloud from refined global feature
        x = self.decoder(x)  # (B, num_points * 3)
        x = x.view(-1, 3, self.num_points)  # Reshape to (B, 3, num_points)

        return x
    
if __name__ =='__main__':
    x = torch.randn((10,1024))
    ae = Autoencoder(emb_dim=1024, num_points=1024, use_bn=True)
    print(x.shape)
    print(ae.forward(x).shape)