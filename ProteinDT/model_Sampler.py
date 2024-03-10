import abc
import torch
from .model_SDE import VPSDE, VESDE


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.
        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.
        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.
        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, x_mask, condition, t):
        f, G = self.rsde.discretize(x, x_mask, condition, t)  # (B, max_seq_len, num_class), (B)
        z = torch.randn_like(x)  # (B, max_seq_len, num_class)
        x_mean = x - f  # (B, max_seq_len, num_class)
        x = x_mean + G[:, None, None] * z  # (B, max_seq_len, num_class)
        return x, x_mean


class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, x_mask, condition, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, x_mask, condition, t)  # (B, max_seq_len, num_class)
            noise = torch.randn_like(x)  # (B, max_seq_len, num_class)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()  # 1
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()  # 1
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha  # (B)
            x_mean = x + step_size[:, None, None] * grad  # (B, max_seq_len, num_class)
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise  # (B, max_seq_len, num_class)
        return x, x_mean
