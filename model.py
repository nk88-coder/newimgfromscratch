# ✅ Final Version — Your Architecture + GitHub Weight Loading (No Upload Prompts)
# 🔥 Uses your VQ-VAE + Transformer code exactly. Loads from repo directly.

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- VQ-VAE --------------------
class VQEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, hidden_dim, 4, 2, 1), nn.ReLU(),
        )
    def forward(self, x): return self.encoder(x)

class VQDecoder(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()
        )
    def forward(self, x): return self.decoder(x)

class EMAQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=512):
        super().__init__()
        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.ema_w = nn.Parameter(embed.clone(), requires_grad=False)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, z.size(-1))
        dist = (z_flat**2).sum(1, keepdim=True) - 2 * z_flat @ self.embedding.t() + (self.embedding**2).sum(1)
        indices = dist.argmin(1)
        quantized = self.embedding[indices].view(z.shape).permute(0, 3, 1, 2)
        return quantized, indices.view(z.shape[0], -1)

class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VQEncoder()
        self.quantizer = EMAQuantizer()
        self.decoder = VQDecoder()
    def forward(self, x):
        z = self.encoder(x)
        z_q, _ = self.quantizer(z)
        return self.decoder(z_q)

# -------------------- Transformer --------------------
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
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
    def forward(self, x, mask=None):
        return x + self.attn(self.norm(x), self.norm(x), self.norm(x), mask)

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
    def forward(self, x, context):
        return x + self.attn(self.norm(x), context, context)

class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(self.norm(x))
        return x + self.proj(lstm_out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_block = SelfAttentionBlock(embed_dim, num_heads)
        self.cross_block = CrossAttentionBlock(embed_dim, num_heads)
        self.ff_block = FeedForwardBlock(embed_dim)
    def forward(self, x, context, mask=None):
        x = self.self_block(x, mask)
        x = self.cross_block(x, context)
        x = self.ff_block(x)
        return x

class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size=1024, embed_dim=512, num_heads=4, num_layers=6, height=14, width=14, text_embed_dim=512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.learned_bos = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.row_embed = nn.Parameter(torch.randn(1, height, 1, embed_dim // 2))
        self.col_embed = nn.Parameter(torch.randn(1, 1, width, embed_dim // 2))
        self.text_proj = nn.Linear(text_embed_dim, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight
        self.H = height
        self.W = width

    def forward(self, img_tokens, text_embeds):
        B, T = img_tokens.shape
        tok_emb = self.token_embedding(img_tokens)
        tok_emb = tok_emb.view(B, self.H, self.W, -1)
        pos = torch.cat([
            self.row_embed.expand(B, self.H, self.W, -1),
            self.col_embed.expand(B, self.H, self.W, -1)
        ], dim=-1)
        x = (tok_emb + pos).view(B, -1, tok_emb.size(-1))
        bos = self.learned_bos.expand(B, 1, -1)
        x = torch.cat([bos, x[:, :-1]], dim=1)
        context = self.text_proj(text_embeds).unsqueeze(1)
        mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device))
        for layer in self.layers:
            x = layer(x, context, mask)
        return self.output_proj(x)

# -------------------- Inference --------------------
@torch.no_grad()
def decode_tokens(vqvae, tokens):
    B, T = tokens.shape
    H = W = int(T**0.5)
    emb = vqvae.quantizer.embedding[tokens].view(B, H, W, -1).permute(0, 3, 1, 2)
    return vqvae.decoder(emb)

@torch.no_grad()
def generate_image(vqvae, transformer, text_embed, seq_len=196, p=0.9, temperature=1.0):
    x = transformer.learned_bos.expand(text_embed.size(0), 1, -1)
    context = transformer.text_proj(text_embed).unsqueeze(1)
    out_tokens = []
    for _ in range(seq_len):
        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=x.device))
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

# -------------------- CLI Entry --------------------
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a dog standing on the park")
    args, _ = parser.parse_known_args()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    inputs = tokenizer([args.prompt], return_tensors="pt").to(device)
    text_embed = clip(**inputs).last_hidden_state.mean(1)

    # 🔽 Load model weights (already in GitHub repo folder)
    checkpoint = torch.load("t2i_models.pth", map_location=device)
    vqvae = VQVAE().to(device)
    transformer = AutoregressiveTransformer().to(device)
    vqvae.load_state_dict(checkpoint["vqvae_state"])
    transformer.load_state_dict(checkpoint["transformer_state"])

    from PIL import Image
    import numpy as np

    print("✅ Weights loaded!")

    image = generate_image(vqvae, transformer, text_embed)[0].permute(1, 2, 0).cpu().numpy()
    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save("output.png")
    print("📸 Image saved as output.png ✅")

