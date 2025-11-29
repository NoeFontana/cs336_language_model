import torch

from cs336_basics.optim.muon import Muon


def test_muon_initialization() -> None:
    params = [torch.randn(10, 10)]
    opt = Muon(params)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["use_muon"] is True
    assert opt.param_groups[1]["use_muon"] is False


def test_muon_step_shape() -> None:
    p = torch.randn(4, 4, requires_grad=True)
    opt = Muon([p], lr=0.1)
    loss = (p**2).sum()
    loss.backward()
    opt.step()
    # Just checking it runs without error
    assert p.grad is not None


def test_muon_mixed_step() -> None:
    p_muon = torch.randn(10, 10, requires_grad=True)
    p_adam = torch.randn(10, requires_grad=True)

    opt = Muon([p_muon], adamw_params=[p_adam], lr=0.1, adamw_lr=0.01)

    loss = (p_muon**2).sum() + (p_adam**2).sum()
    loss.backward()

    opt.step()

    assert p_muon.grad is not None
    assert p_adam.grad is not None
