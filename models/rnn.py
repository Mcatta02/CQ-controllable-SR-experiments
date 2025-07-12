import torch
from torch import nn

from models import register, make

from .convs import CausalConv1d

@register('rnn')
class RNNBlock(nn.Module):
    def __init__(self, embed_dim, pos_encoding=None, **kwargs):
        super().__init__()

        self.pos_encoding = make({'name': pos_encoding, 'args': {'dim': embed_dim}}) \
            if pos_encoding is not None else None
        self.layer = nn.Conv1d(2*embed_dim, embed_dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.Tanh()

    def recurrent(self, x, pos=None):
        """
        Args:
            x (tensor) : [batch, 1, channel]
            pos (int) : position in the sequence
        """

        if self.pos_encoding is not None:
            pos = self.pos_encoding(pos, x.device)
            x = x + pos.unsqueeze(1)

        try:
            hidden = self.hidden
        except AttributeError:
            hidden = torch.zeros(x.shape[0], x.shape[2], 1, device=x.device, dtype=x.dtype)

        x = x.permute(0, 2, 1) # [batch, channel, seq_length]
        x = torch.cat([x, hidden], dim=1)  # Concatenate input and hidden state
        x = self.layer(x)
        x = self.act(x)

        self.hidden = x

        return x.permute(0, 2, 1)

    def flush(self):
        """
        Flush the hidden state of the RNN block.
        """
        try:
            del self.hidden
        except AttributeError:
            pass

@register('lstm')
class LSTMBlock(nn.Module):
    def __init__(self, embed_dim, pos_encoding=None, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim

        self.pos_encoding = make({'name': pos_encoding, 'args': {'dim': embed_dim}}) \
            if pos_encoding is not None else None

        self.layer = nn.Conv1d(2*embed_dim, 4*embed_dim, kernel_size=1, stride=1, padding=0)


    def recurrent(self, x, pos=None):
        """
        Args:
            x (tensor): [batch, 1, channel]
            pos (int): position in the sequence
        """

        if self.pos_encoding is not None:
            pos = self.pos_encoding(pos, x.device)
            x = x + pos.unsqueeze(1)

        B, _, C = x.shape

        # Initialize hidden and cell if first step
        try:
            h = self.hidden
            c = self.cell
        except AttributeError:
            h = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)

        # Concatenate input and hidden state along channel
        concat_input = torch.cat([x, h], dim=2)   # [B, 1, 2C]
        concat_input = concat_input.permute(0, 2, 1)  # [B, 2C, 1]

        gates = self.layer(concat_input)  # [B, 4C, 1]
        gates = gates.permute(0, 2, 1)          # [B, 1, 4C]

        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=2)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        c = f_gate * c + i_gate * g_gate
        h = o_gate * torch.tanh(c)

        # Save hidden state
        self.hidden = h
        self.cell = c

        return h

    def flush(self):
        """
        Flush the hidden and cell state.
        """
        for attr in ['hidden', 'cell']:
            if hasattr(self, attr):
                delattr(self, attr)


@register('gru')
class GRUBlock(nn.Module):
    def __init__(self, embed_dim, pos_encoding=None, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

        # optional positional encoding
        self.pos_encoding = make({'name': pos_encoding, 'args': {'dim': embed_dim}}) \
            if pos_encoding is not None else None

        # z, r gates: from [x ; h] -> 2 * embed_dim
        self.zr_layer = nn.Conv1d(embed_dim * 2, embed_dim * 2, kernel_size=1, stride=1, padding=0)

        # candidate hidden: from [x ; (r * h)] -> embed_dim
        self.n_layer = nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, padding=0)

    def recurrent(self, x, pos=None):
        """
        Args:
            x: [batch, 1, channel]
            pos: optional integer position for positional encoding
        Returns:
            [batch, 1, channel]: next hidden state
        """
        if self.pos_encoding is not None:
            pos_enc = self.pos_encoding(pos, x.device)
            x = x + pos_enc.unsqueeze(1)

        B, _, C = x.shape

        # initialize hidden state if first step
        if not hasattr(self, 'hidden'):
            h = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        else:
            h = self.hidden

        # 1️⃣ compute z, r gates
        concat_input = torch.cat([x, h], dim=2)  # [B, 1, 2C]
        concat_input = concat_input.permute(0, 2, 1)  # [B, 2C, 1]
        zr_proj = self.zr_layer(concat_input)         # [B, 2C, 1]
        zr_proj = zr_proj.permute(0, 2, 1)            # [B, 1, 2C]
        z_gate, r_gate = zr_proj.chunk(2, dim=2)
        z_gate = torch.sigmoid(z_gate)
        r_gate = torch.sigmoid(r_gate)

        # 2️⃣ compute candidate hidden
        r_hidden = r_gate * h
        n_input = torch.cat([x, r_hidden], dim=2)     # [B, 1, 2C]
        n_input = n_input.permute(0, 2, 1)            # [B, 2C, 1]
        n_proj = self.n_layer(n_input)                # [B, C, 1]
        n_proj = n_proj.permute(0, 2, 1)              # [B, 1, C]
        n_cand = torch.tanh(n_proj)

        # 3️⃣ update hidden state
        h_new = (1 - z_gate) * h + z_gate * n_cand
        self.hidden = h_new

        return h_new

    def flush(self):
        """Flush the hidden state."""
        if hasattr(self, 'hidden'):
            del self.hidden