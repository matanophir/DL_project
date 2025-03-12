import torch
import torch.nn as nn
import torch.nn.functional as F

class PrintLayer(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 12, 4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           
            nn.LeakyReLU(),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           
            nn.LeakyReLU(),
            nn.BatchNorm2d(48),
            nn.Flatten(),
            nn.Linear(48*3*3, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Reg_Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 12, 4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(48),
            nn.Flatten(),
            nn.Linear(48*3*3, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 48*3*3),
            nn.LeakyReLU(),
            nn.Unflatten(1, (48, 3, 3)),
            nn.ConvTranspose2d(48, 24, 3, stride=2, padding=0),  # did some changes to get 28x28
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(12, out_channels, 4, stride=2, padding=1),  
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(x)

class AE(nn.Module):
    def __init__(self, in_size, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_size[0], latent_dim)
        self.decoder = Decoder(latent_dim, in_size[0])

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def sample(self, imgs):
        with torch.no_grad():
            recon = self(imgs)
        return recon

    def _n_features(self) -> int: # not in use for now
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            features = self.feature_extractor(torch.zeros(1, *self.in_size))
            self.feature_shape = features.shape[1:]

            return torch.prod(torch.tensor(features.shape[1:])).item()
            # ========================
        finally:
            torch.set_rng_state(rng_state)
    

class Classifier(nn.Module):
    def __init__(self, encoder, latent_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier_head(encoded)

class Reg_Classifier(nn.Module):
    def __init__(self, encoder, latent_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier_head(encoded)

