import torch
from typing import Optional, Callable, Union

class OrthoGrad(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        base_optimizer_cls=torch.optim.SGD,
        grad_renormalization: bool = True,
        **base_optimizer_args
    ):
        """
        A wrapper optimizer that projects gradients to be orthogonal
        to the current parameters before performing an update.

        Args:
            params (iterable): Iterable of parameters to optimize.
            base_optimizer_cls (Optimizer class): The base optimizer class
                (e.g., torch.optim.SGD, torch.optim.AdamW).
            grad_renormalization (bool): Whether to rescale the orthogonalized gradient
                to have the same norm as the original gradient.
            **base_optimizer_args: Arguments for the base optimizer.
        """
        defaults = {
            'grad_renormalization': grad_renormalization,
        }
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_args)
        self.step_count = 0

    @staticmethod
    def _orthogonalize_gradients(params, grad_renormalization: bool):
        """
        Projects the gradient g to be orthogonal to the current weights w.

        g_orth = g - ( (w·g)/(w·w + eps) ) * w

        And then optionally re-scales g_orth to have the same norm as g.
        """
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    w = p.view(-1)
                    g = p.grad.view(-1)

                    w_norm_sq = torch.dot(w, w) + 1e-30
                    dot_product = torch.dot(w, g)
                    
                    proj = dot_product / w_norm_sq
                    g_orth = g - proj * w

                    if grad_renormalization:
                        g_norm = g.norm(2)
                        g_orth_norm = g_orth.norm(2) + 1e-30
                        g_orth = g_orth * (g_norm / g_orth_norm)

                    p.grad.copy_(g_orth.view_as(p.grad))

    def step(self, closure=None):
        # Track metrics before orthogonalization
        metrics = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    w = p.view(-1)
                    g = p.grad.view(-1)
                    metrics['cosine_similarity'] = torch.dot(w, g) / (w.norm() * g.norm() + 1e-30)
                    metrics['gradient_norm'] = g.norm()
                    metrics['weight_norm'] = w.norm()
        
        for group in self.param_groups:
            self._orthogonalize_gradients(
                group['params'],
                group['grad_renormalization']
            )

        self.step_count += 1
        return self.base_optimizer.step(closure)

