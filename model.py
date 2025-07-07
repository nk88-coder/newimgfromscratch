# ✅ Complete Inference Script for VQ-VAE + Transformer (Ready for GitHub)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel

# 📦 Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================== VQ-VAE Components ==========================
class VQEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, hidden_dim, 4, 2, 1), nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

class VQDecoder(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

class EMAQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=512, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(embed.clone(), requires_grad=False)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embedding_dim)
        distances = (z_flat ** 2).sum(1, keepdim=True) - 2 * z_flat @ self.embedding.T + (self.embedding ** 2).sum(1)
        indices = distances.argmin(1)
        encodings = F.one_hot(indices, self.num_embeddings).type(z.dtype)
        quantized = encodings @ self.embedding
        quantized = quantized.view(z.shape).permute(0, 3, 1, 2).contiguous()
        return quantized, 0.0, indices.view(z.size(0), -1)

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VQEncoder()
        self.quantizer = EMAQuantizer()
        self.decoder = VQDecoder()

    def forward(self, x):
        z = self.encoder(x)
        z_q, _, _ = self.quantizer(z)
        return self.decoder(z_q)

# ========================== Transformer Components ==========================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        B, T, C = q.shape
        q = self.q_proj(q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask[None, None, :, :] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, context, mask):
        x = x + self.attn(self.norm1(x), context, context, mask)
        x = x + self.ff(self.norm2(x))
        return x

class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size=1024, embed_dim=512, num_heads=4, num_layers=6, height=14, width=14, text_embed_dim=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.learned_bos = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, text_embed):
        B, T = x.shape
        x = self.token_embedding(x)
        x = torch.cat([self.learned_bos.expand(B, 1, -1), x[:, :-1]], dim=1)
        context = self.text_proj(text_embed).unsqueeze(1)
        mask = torch.tril(torch.ones(T + 1, T + 1, device=x.device))
        for layer in self.layers:
            x = layer(x, context, mask)
        return self.output_proj(x)

# ========================== Decode & Generation ==========================
@torch.no_grad()
def decode_tokens(vqvae, tokens):
    B, T = tokens.shape
    H = W = int(T ** 0.5)
    emb = vqvae.quantizer.embedding[tokens].view(B, H, W, -1).permute(0, 3, 1, 2)
    return vqvae.decoder(emb)

@torch.no_grad()
def generate_image_from_text(vqvae, transformer, text_embed, seq_len=196, p=0.9, temperature=1.0):
    transformer.eval()
    vqvae.eval()
    device = text_embed.device
    B = text_embed.size(0)
    x = transformer.learned_bos.expand(B, 1, -1)
    context = transformer.text_proj(text_embed).unsqueeze(1)
    out_tokens = []

    for _ in range(seq_len):
        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=device))
        x_step = x
        for layer in transformer.layers:
            x_step = layer(x_step, context, mask)
        logits = transformer.output_proj(x_step)[:, -1]
        sorted_logits, sorted_idx = torch.sort(logits / temperature, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        sorted_mask = cumprobs > p
        sorted_mask[:, 0] = False
        sorted_logits[sorted_mask] = -float("inf")
        filtered_logits = torch.full_like(logits, -float("inf"))
        filtered_logits.scatter_(1, sorted_idx, sorted_logits)
        final_probs = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(final_probs, num_samples=1)
        out_tokens.append(next_token)
        next_token_embed = transformer.token_embedding(next_token.squeeze(1))
        x = torch.cat([x, next_token_embed.unsqueeze(1)], dim=1)

    tokens = torch.cat(out_tokens, dim=1)
    return decode_tokens(vqvae, tokens)

# ========================== Run Inference ==========================
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a dog standing on the park")
    args = parser.parse_args()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    inputs = tokenizer([args.prompt], return_tensors="pt").to(DEVICE)
    text_embed = text_model(**inputs).last_hidden_state.mean(1)

    checkpoint = torch.load("t2i_models.pth", map_location=DEVICE)
    vqvae = VQVAE().to(DEVICE)
    transformer = AutoregressiveTransformer().to(DEVICE)
    vqvae.load_state_dict(checkpoint["vqvae_state"])
    transformer.load_state_dict(checkpoint["transformer_state"])

    print("✅ Successfully loaded weights into VQ-VAE and Transformer")
    image = generate_image_from_text(vqvae, transformer, text_embed)[0].permute(1, 2, 0).cpu().numpy()
    image = (image + 1) / 2
    plt.imshow(image)
    plt.axis("off")
    plt.title(args.prompt)
    plt.show()

