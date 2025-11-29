import math


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    tw: int,
    tc: int,
) -> float:
    """Cosine learning rate schedule with warm-up.

    Args:
        it: Current training iteration.
        max_learning_rate: Maximum learning rate (alpha_max).
        min_learning_rate: Minimum learning rate (alpha_min).
        tw: Number of linear warm-up iterations (T_w).
        tc: Total number of cosine annealing iterations (T_c).

    Returns:
        The learning rate for the current step.
    """
    if tw > tc:
        raise ValueError("tw cannot be greater than tc.")

    if it < tw:
        if tw == 0:
            return max_learning_rate
        return it / tw * max_learning_rate

    elif tw <= it <= tc:
        progress = (it - tw) / (tc - tw)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)

    else:
        return min_learning_rate
