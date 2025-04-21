import torch
import torch.nn.functional as F

class DelayEmbedder:
    """
    Delay embedding for time series → image representation.
    """

    def __init__(self, device, seq_len, delay, embedding):
        self.device = device
        self.seq_len = seq_len
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None  # used for unpadding

    def pad_to_square(self, x, mask = 0):
        _, _, h, w = x.shape
        size = max(h, w)
        padding = (0, size - w, 0, size - h)
        return F.pad(x, padding, mode='constant', value=mask)

    def unpad(self, x, original_shape):
        _, _, h, w = original_shape
        return x[:, :, :h, :w]

    def ts_to_img(self, signal, pad = True, mask = 0):
        """
        Convert a time series (B, L, C) to a 2D image (B, C, embedding, cols), optionally padded to square.

        Args:
            signal (Tensor): Time series tensor of shape (B, L, C)
            pad (bool): Whether to pad to a square shape
            mask (int): Padding fill value

        Returns:
            Tensor: Image representation of the time series
        """
        B, L, C = signal.shape
        if L != self.seq_len:
            self.seq_len = L
        
        max_cols = ((self.seq_len - 1) // self.delay) + 1
        x_img = torch.zeros((B, C, self.embedding, max_cols), device=self.device)
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            s, e = i * self.delay, i * self.delay + self.embedding
            x_img[:, :, :, i] = signal[:, s:e].permute(0, 2, 1)
            i += 1

        # Edge case
        if i * self.delay < self.seq_len:
            s = i * self.delay
            tail = signal[:, s:].permute(0, 2, 1)
            x_img[:, :, :tail.shape[-1], i] = tail
            i += 1

        self.img_shape = (B, C, self.embedding, i)
        x_img = x_img[:, :, :, :i]

        return self.pad_to_square(x_img, mask) if pad else x_img

    def img_to_ts(self, img):
        """
        Convert an image (B, C, H, W) back to a time series (B, L, C)

        Args:
            img (Tensor): Image tensor from ts_to_img

        Returns:
            Tensor: Reconstructed time series
        """
        img = self.unpad(img, self.img_shape)
        B, C, H, W = img.shape
        ts = torch.zeros((B, C, self.seq_len), device=self.device)

        for i in range(W - 1):
            s, e = i * self.delay, i * self.delay + self.embedding
            ts[:, :, s:e] = img[:, :, :, i]

        # Last column
        s = (W - 1) * self.delay
        tail_len = ts[:, :, s:].shape[-1]
        ts[:, :, s:] = img[:, :, :tail_len, W - 1]

        return ts.permute(0, 2, 1)  # (B, L, C)