import torch

from reflex.models.snn import LIF, ReflexSNN, SNNConfig, count_parameters


def test_lif_integrates_fires_and_resets_by_subtraction():
    lif = LIF(beta=1.0, threshold=1.0, slope=5.0)
    mem, fired = None, []
    for _ in range(5):
        spk, mem = lif(torch.tensor([0.4]), mem)
        fired.append(int(spk.item()))
    assert fired == [0, 0, 1, 0, 0]              # 0.4, 0.8, 1.2 -> spike, keeps 0.2, 0.6, 1.0
    assert abs(mem.item() - 1.0) < 1e-6


def test_forward_shape_and_stats():
    model = ReflexSNN()
    x = (torch.rand(3, 10, 2, 45, 80) < 0.05).to(torch.uint8)
    logits, rates = model(x, return_stats=True)
    assert logits.shape == (3, 4) and len(rates) == 3
    assert all(0.0 <= r <= 1.0 for r in rates)
    assert count_parameters(model) < 50_000      # small on purpose


def test_gradients_reach_the_first_layer():
    model = ReflexSNN()
    x = (torch.rand(4, 10, 2, 45, 80) < 0.2).float()
    model(x).sum().backward()
    assert model.convs[0].weight.grad.abs().sum() > 0


def test_it_can_learn_left_from_right():
    torch.manual_seed(0)
    model = ReflexSNN(SNNConfig(channels=(8, 16), n_classes=2))
    x = torch.zeros(32, 6, 2, 24, 32)
    y = torch.arange(32) % 2
    for i in range(32):                          # activity on the left or on the right half
        cols = slice(0, 14) if y[i] == 0 else slice(18, 32)
        x[i, :, :, :, cols] = (torch.rand(6, 2, 24, 14) < 0.3).float()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert (model(x).argmax(1) == y).float().mean() > 0.95
