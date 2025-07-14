import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Patch Embedding module.
    Splits the input image into non-overlapping patches and projects each patch to a vector embedding using a Conv2d layer.
    """
    def __init__(self, img_size=28, patch_size=4, in_chans=2, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Conv2d projects each patch (across channels) into embedding dimension.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Apply patch projection: output shape [B, embed_dim, H/ps, W/ps]
        x = self.proj(x)  
        # Flatten spatial dimensions and transpose to [B, N, D], where N=num_patches
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT) based image classifier.
    - Applies patch embedding to input images.
    - Adds a [CLS] token and positional encodings.
    - Processes sequence with Transformer encoder layers.
    - Outputs class logits from the [CLS] token representation.
    """
    def __init__(self, img_size=28, patch_size=2, in_chans=2, num_classes=2, embed_dim=32, depth=2, num_heads=2):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token for sequence classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embeddings for patches + [CLS] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # Define transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    
        # LayerNorm before classification head
        self.norm = nn.LayerNorm(embed_dim)
        # Linear head to map [CLS] token to class logits
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize positional and cls tokens with truncated normal for stable training
        # values more than 2 standard deviations from the mean are "cut off"
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        # Convert image to patch embeddings: [B, N, D]
        x = self.patch_embed(x) 
        # Expand [CLS] token to batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # Prepend [CLS] token to patch embeddings: [B, N+1, D]
        x = torch.cat((cls_tokens, x), dim=1) 
        # Add positional encoding to sequence
        x = x + self.pos_embed
        # Pass through transformer encoder stack
        x = self.transformer(x)
        # Use [CLS] token output, normalize, then classify
        x = self.norm(x[:, 0])
        return self.head(x)
