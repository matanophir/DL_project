import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_size, latent_dim):
        """
        Args:
            in_size (tuple): (C, H, W) - input image dimensions.
            latent_dim (int): latent space dimension.
        """
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_size[0], 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 4, 2, 1),
        )

    def forward(self, x):
        return self.cnn(x)

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            features = self.feature_extractor(torch.zeros(1, *self.in_size))
            return torch.prod(torch.tensor(features.shape[1:])).item()
        finally:
            torch.set_rng_state(rng_state)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1)
        )

    def forward(self, h):
        return torch.tanh(self.cnn(h))

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            features = self.feature_extractor(torch.zeros(1, *self.in_size))
            return torch.prod(torch.tensor(features.shape[1:])).item()
            # ========================
        finally:
            torch.set_rng_state(rng_state)


class AE(torch.nn.Module):
    def __init__(self, in_size, latent_dim):
        super().__init__()
        self.in_size = in_size
        self.latent_dim = latent_dim
        self.n_features = None
        self.feature_shape = None

        self.feature_extractor =  nn.Sequential(
            nn.Conv2d(self.in_size[0], 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 4, 2, 1)
        )

        self._make_encoder()
        self._make_decoder()

         
 

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def sample(self, imgs):
        with torch.no_grad():
            recon = self(imgs)

        return recon



    def _make_encoder(self):

        self.n_features = self._n_features()

        self.encoder = nn.Sequential(
            self.feature_extractor,
            nn.Flatten(),
            nn.Linear(self.n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.latent_dim)
        )

    def _make_decoder(self):
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.n_features),
            nn.Unflatten(1, self.feature_shape),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, self.in_size[0], 4, 2, 1),
            nn.Tanh()
        )


    def _n_features(self) -> int:
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
    
    
