"""Scheduling of learning rate and hyperparameters."""
import abc
import math
import warnings
from typing import Callable

from torch.optim.lr_scheduler import _LRScheduler


def warmup_fn(step: int, warmup_steps: int) -> float:
    """Learning rate warmup.

    Maps the step to a factor for rescaling the learning rate.
    """
    if warmup_steps:
        return min(1.0, step / warmup_steps)
    else:
        return 1.0


def exp_decay_after_warmup_fn(
    step: int, decay_rate: float, decay_steps: int, warmup_steps: int
) -> float:
    """Decay function for exponential decay with learning rate warmup.

    Maps the step to a factor for rescaling the learning rate.
    """
    factor = warmup_fn(step, warmup_steps)
    if step < warmup_steps:
        return factor
    else:
        return factor * (decay_rate ** ((step - warmup_steps) / decay_steps))


def exp_decay_with_warmup_fn(
    step: int, decay_rate: float, decay_steps: int, warmup_steps: int
) -> float:
    """Decay function for exponential decay with learning rate warmup.

    Maps the step to a factor for rescaling the learning rate.
    """
    factor = warmup_fn(step, warmup_steps)
    return factor * (decay_rate ** (step / decay_steps))


class CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine annealing with warmup."""

    def __init__(
        self,
        optimizer,
        T_max: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        error_on_exceeding_steps: bool = True,
        verbose: bool = False,
    ):
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.error_on_exceeding_steps = error_on_exceeding_steps
        super().__init__(optimizer, last_epoch, verbose)

    def _linear_lr_warmup(self, base_lr, step_num):
        return base_lr * ((step_num + 0.5) / self.warmup_steps)

    def _cosine_annealing(self, base_lr, step_num):
        fraction_of_steps = (step_num - self.warmup_steps) / (self.T_max - self.warmup_steps - 1)
        return self.eta_min + 1 / 2 * (base_lr - self.eta_min) * (
            1 + math.cos(math.pi * fraction_of_steps)
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )
        step_num = self.last_epoch

        if step_num < self.warmup_steps:
            # Warmup.
            return [self._linear_lr_warmup(base_lr, step_num) for base_lr in self.base_lrs]
        elif step_num < self.T_max:
            # Cosine annealing.
            return [self._cosine_annealing(base_lr, step_num) for base_lr in self.base_lrs]
        else:
            if self.error_on_exceeding_steps:
                raise ValueError(
                    "Tried to step {} times. The specified number of total steps is {}".format(
                        step_num + 1, self.T_max
                    )
                )
            else:
                return [self.eta_min for _ in self.base_lrs]


HPSchedulerT = Callable[[int], float]  # Type for function signatures.


class HPScheduler(metaclass=abc.ABCMeta):
    """Base class for scheduling of scalar hyperparameters based on the number of training steps."""

    @abc.abstractmethod
    def __call__(self, step: int) -> float:
        """Return current value of hyperparameter based on global step."""
        pass


class LinearHPScheduler(HPScheduler):
    def __init__(
        self, end_value: float, end_step: int, start_value: float = 0.0, start_step: int = 0
    ):
        super().__init__()
        if start_step > end_step:
            raise ValueError("`start_step` needs to be smaller equal to `end_step`.")

        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, step: int) -> float:
        if step < self.start_step:
            return self.start_value
        elif step > self.end_step:
            return self.end_value
        else:
            t = step - self.start_step
            T = self.end_step - self.start_step
            return self.start_value + t * (self.end_value - self.start_value) / T


class StepHPScheduler(HPScheduler):
    def __init__(self, end_value: float, switch_step: int, start_value: float = 0.0):
        super().__init__()
        self.start_value = start_value
        self.end_value = end_value
        self.switch_step = switch_step

    def __call__(self, step: int) -> float:
        if step < self.switch_step:
            return self.start_value
        elif step >= self.switch_step:
            return self.end_value


class CosineAnnealingHPScheduler(HPScheduler):
    """Cosine annealing."""

    def __init__(self, start_value: float, end_value: float, start_step: int, end_step: int):
        super().__init__()
        assert start_value >= end_value
        assert start_step <= end_step
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, step: int) -> float:

        if step < self.start_step:
            value = self.start_value
        elif step >= self.end_step:
            value = self.end_value
        else:
            a = 0.5 * (self.start_value - self.end_value)
            b = 0.5 * (self.start_value + self.end_value)
            progress = (step - self.start_step) / (self.end_step - self.start_step)
            value = a * math.cos(math.pi * progress) + b

        return value
