import torch
from typing import Union, List

def print_trainable_parameters(model: torch.nn.Module) -> None:
    trainable_count = count_parameters(model, only_trainable=True)
    all_counts = count_parameters(model, only_trainable=False)

    print(f"Total Trainable Parameters: {trainable_count:,}")
    print(f"Total All Parameters (Trainable + Fixed): {all_counts[1]:,}")
    print("-" * 30)

def count_parameters(model: torch.nn.Module, only_trainable: bool = True) -> Union[int, List[int]]:
    """
    Counts the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model instance (e.g., a loaded model or a custom nn.Module).
        only_trainable (bool): If True, counts only parameters that require gradients (trainable).
                               If False, returns a list: [trainable_params, total_params].

    Returns:
        Union[int, List[int]]: The total count of parameters (or a list if only_trainable is False).
    """

    # Generator expression to count trainable parameters
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    if only_trainable:
        return trainable_params
    else:
        # Generator expression to count ALL parameters (trainable + fixed)
        all_params = sum(p.numel() for p in model.parameters())
        return [trainable_params, all_params]