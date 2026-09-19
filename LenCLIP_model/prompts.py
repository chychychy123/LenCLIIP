"""Learnable MG, foreground and background contexts (paper Section 3.2)."""

import torch
from torch import nn
from torch.nn import functional as F
import clip


class LearnablePrompts(nn.Module):
    def __init__(self, encoder, class_names, n_ctx=30, class_specific=False):
        super().__init__()
        self.n_ctx = int(n_ctx)
        self.num_classes = len(class_names)
        if not 1 <= self.n_ctx < encoder.context_length - 3:
            raise ValueError("n_ctx leaves no room for the class name and end token")
        # X is only a tokenizer placeholder: its embeddings are replaced entirely.
        texts = [("X " * self.n_ctx) + name + "." for name in class_names]
        tokens = clip.tokenize(texts, context_length=encoder.context_length)
        device = encoder.token_embedding.weight.device
        with torch.no_grad():
            embeddings = encoder.token_embedding(tokens.to(device)).float()
        self.register_buffer("token_ids", tokens.to(device))
        self.register_buffer("prefix", embeddings[:, :1])
        self.register_buffer("suffix", embeddings[:, 1 + self.n_ctx:])
        shape = (self.num_classes if class_specific else 1, self.n_ctx, embeddings.shape[-1])
        self.contexts = nn.ParameterDict({
            name: nn.Parameter(torch.empty(shape, device=device))
            for name in ("mg", "foreground", "background")
        })
        for context in self.contexts.values():
            nn.init.normal_(context, std=0.02)

    def forward(self, encoder, branch):
        context = self.contexts[branch].expand(self.num_classes, -1, -1)
        x = torch.cat((self.prefix, context, self.suffix), dim=1).to(encoder.dtype)
        x = x + encoder.positional_embedding.to(x.dtype)
        x, _ = encoder.transformer(x.permute(1, 0, 2))
        x = encoder.ln_final(x.permute(1, 0, 2))
        rows = torch.arange(self.num_classes, device=x.device)
        text = x[rows, self.token_ids.argmax(-1)] @ encoder.text_projection
        return F.normalize(text.float(), dim=-1)
