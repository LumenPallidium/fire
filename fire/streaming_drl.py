import torch
from torch.optim import Optimizer
from typing import cast, List, Optional, Union
from torch import Tensor

def _obgd(
    params: List[Tensor],
    eligbility_traces : List[Tensor],
    grads: List[Tensor],
    error: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    momentum: float,
    lr: float,
    scaling_factor : float,
    discount_factor: float,
    eligibility_trace_decay: float,
    maximize: bool,
):
    """
    Single step of the (adaptive) Overshooting-bounded Gradient Descent (ObGD) optimizer.
    """
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        eligibility_trace = eligbility_traces[i]
        if eligibility_trace is None:
            eligibility_trace = torch.clone(grad).detach().requires_grad_(False)
            eligbility_traces[i] = eligibility_trace
        else:
            eligibility_scaler = eligibility_trace_decay * discount_factor
            eligibility_trace.mul_(eligibility_scaler).add_(grad)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(eligibility_trace).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(eligibility_trace,
                                        alpha = 1 - momentum)

        scaled_error = torch.maximum(error, torch.tensor(1)).mean()
        et_scaled = eligibility_trace / torch.sqrt(torch.tensor(momentum) + 1e-8)
        #TODO : might not be sum here - maybe keep batch
        momentum_scale = scaling_factor * scaled_error * et_scaled.abs().sum()
        lr_ = min(lr, 1 / momentum_scale.item())

        param.data.add_(et_scaled, alpha=-lr_ * error.item())


class ObGD(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 momentum=0,
                 maximize=False,
                 scaling_factor = 2,
                 discount_factor = 0.99,
                 eligibility_trace_decay = 0.9,):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if scaling_factor <= 1:
            raise ValueError(f"Invalid scaling factor: {scaling_factor}")
        if discount_factor < 0 or discount_factor > 1:
            raise ValueError(f"Invalid discount factor: {discount_factor}")
        if eligibility_trace_decay < 0 or eligibility_trace_decay > 1:
            raise ValueError(f"Invalid eligibility trace decay: {eligibility_trace_decay}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            maximize=maximize,
        )
        super().__init__(params, defaults)
        self.scaling_factor = scaling_factor
        self.discount_factor = discount_factor
        self.eligibility_trace_decay = eligibility_trace_decay

    def __setstate__(self, state):
        super().__setstate__(state)

    # mostly following official PyTorch SGD implementation
    def step(self, td_error, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            eligbility_traces: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)
                    eligbility_traces.append(self.state[p].get("eligibility_trace", None))
                    momentum_buffer_list.append(self.state[p].get("momentum_buffer", None))

            _obgd(
                params,
                eligbility_traces,
                grads,
                td_error,
                momentum_buffer_list,
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                momentum = group["momentum"],
                lr=group["lr"],
                scaling_factor = self.scaling_factor,
                discount_factor = self.discount_factor,
                eligibility_trace_decay = self.eligibility_trace_decay,
                maximize=group["maximize"],
            )

            for p in params:
                self.state[p]["eligibility_trace"] = eligbility_traces
                self.state[p]["momentum_buffer"] = momentum_buffer_list

        return loss
    

if __name__ == "__main__":
    actor = torch.nn.Linear(4, 2)
    critic = torch.nn.Linear(4, 1)

    optimizer = ObGD(list(actor.parameters()) + list(critic.parameters()), lr=1e-3, momentum=0.9)
    optimizer.zero_grad()
    obs = torch.randn(256, 4)
    action = actor(obs)
    value = critic(obs)

    action_probs = torch.nn.functional.softmax(action, dim=-1)
    loss = -torch.log(action_probs).sum()

    loss.backward()
    optimizer.step(value.detach())