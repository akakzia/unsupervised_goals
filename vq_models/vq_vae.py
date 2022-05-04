import torch
import torch.nn as nn
import torch.nn.functional as F



# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        try:
            torch.nn.init.constant_(m.bias, 0)
        except AttributeError:
            pass


class VQModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim, commitment_cost, output_dim):
        super(VQModel, self).__init__()
        self._encoder = Encoder(input_dim, hidden_dim, embedding_dim)

        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self._decoder = Decoder(embedding_dim, hidden_dim, output_dim)
        
    
    def forward(self, x):
        z = self._encoder(x)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Compute distances 
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) + torch.sum(self._embedding.weight**2, dim=1) 
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()

        # self._linear1 = nn.Linear(input_dim // 2, feature_dim, bias=False)
        # self._linear2 = nn.Linear(64, 64)
        # self._linear3 = nn.Linear(64, feature_dim)

        self._linear4 = nn.Linear(input_dim, hidden_dim)
        self._linear5 = nn.Linear(hidden_dim, hidden_dim)
        self._linear6 = nn.Linear(hidden_dim, embedding_dim)

        self.apply(weights_init_)

    def forward(self, object_states):
        # Preprocessing input
        # Only use geometric positions here and discard orientations for now useless
        x = torch.cat([object_states[:, 0, :3], object_states[:, 1, :3]], axis=-1)

        x = F.relu(self._linear4(x))
        x = F.relu(self._linear5(x))
        x = self._linear6(x)

        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self._linear1 = nn.Linear(embedding_dim, hidden_dim)
        self._linear2 = nn.Linear(hidden_dim, hidden_dim)
        self._linear3 = nn.Linear(hidden_dim, output_dim)

        self.apply(weights_init_)
    
    def forward(self, embeddings):
        x = F.relu(self._linear1(embeddings))
        x = F.relu(self._linear2(x))
        x = self._linear3(x)

        return x