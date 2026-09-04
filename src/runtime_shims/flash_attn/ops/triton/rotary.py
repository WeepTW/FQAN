import torch


def apply_rotary(x, cos, sin, interleaved=False, inplace=False, conjugate=False, **kwargs):
    if conjugate:
        sin = -sin
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if interleaved:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        output = torch.stack((o1, o2), dim=-1).flatten(-2)
    else:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        output = torch.cat((o1, o2), dim=-1)
    if inplace:
        x.copy_(output)
        return x
    return output
