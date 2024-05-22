import numpy as np
import torch
import tqdm
from sklearn.datasets import make_moons
from utils import KLdivergence

from stellarperceptron.ddpm import ConditionalDiffusionModel

device = "cpu"


def sampling_two_normal(x1, y1, s1, x2, y2, s2, n=1000):
    """
    Sampling from two normal distributions

    Args:
        x1 (float): mean of the first normal distribution
        y1 (float): mean of the first normal distribution
        s1 (float): standard deviation of the first normal distribution
        x2 (float): mean of the second normal distribution
        y2 (float): mean of the second normal distribution
        s2 (float): standard deviation of the second normal distribution
        n (int): number of samples

    Returns:
        np.ndarray: samples from the two normal distributions
    """
    x1 = np.concatenate([np.random.normal(x1, s1, n), np.random.normal(y1, s2, n)])
    x2 = np.concatenate([np.random.normal(x2, s1, n), np.random.normal(y2, s2, n)])
    r = np.random.choice([0, 1], 1)[0]
    if r == 0:
        return x1
    else:
        return x2


def test_moon():
    nn_model = ConditionalDiffusionModel(
        dim=2, cond_dim=0, dense_num=128, num_steps=100, beta_end=1.0e-2, device=device
    )
    moons, _ = make_moons(10**4, noise=0.05)
    dataset = torch.tensor(moons, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
    batch_size = 128

    for t in tqdm.tqdm(range(256)):
        permutation = torch.randperm(dataset.size()[0])
        for i in range(0, dataset.size()[0], batch_size):
            # Retrieve current batch
            indices = permutation[i : i + batch_size]
            batch_x = dataset[indices]
            # Compute the loss.
            loss = nn_model.noise_estimation_loss(batch_x)
            # Before the backward pass, zero all of the network gradients
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to parameters
            loss.backward()
            # Calling the step function to update the parameters
            optimizer.step()
    with torch.inference_mode():
        x_seq = nn_model.p_sample_loop(size=len(dataset), return_steps=False)

    assert KLdivergence(x_seq.cpu().numpy(), dataset.cpu().numpy()) < 0.20


def test_two_normal_conditional():
    nn_model = ConditionalDiffusionModel(
        dim=2, cond_dim=6, dense_num=128, num_steps=100, beta_end=5.0e-2, device=device
    )

    dataset = np.zeros((100000, 2))
    cond = np.column_stack(
        [
            np.random.uniform(-2.0, 2.0, (100000, 2)),
            np.random.uniform(0.1, 0.5, (100000)),
            np.random.uniform(-2.0, 2.0, (100000, 2)),
            np.random.uniform(0.1, 0.5, (100000)),
        ]
    )

    for i in range(len(dataset)):
        dataset[i] = sampling_two_normal(*cond[i], n=1)

    dataset = torch.tensor(dataset, dtype=torch.float32).to(device)
    cond = torch.tensor(cond, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
    batch_size = 128

    for t in tqdm.tqdm(range(256)):
        permutation = torch.randperm(dataset.size()[0])
        for i in range(0, dataset.size()[0], batch_size):
            # Retrieve current batch
            indices = permutation[i : i + batch_size]
            batch_x = dataset[indices]
            batch_cond = cond[indices]
            # Compute the loss.
            loss = nn_model.noise_estimation_loss(batch_x, batch_cond)
            # Before the backward pass, zero all of the network gradients
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to parameters
            loss.backward()
            # Calling the step function to update the parameters
            optimizer.step()

    # test a few conditions
    for ground_cond in [
        [-0.5, 0.0, 0.1, 0.5, 0.0, 0.1],
        [-1.0, 0.0, 0.1, 0.0, -1.0, 0.1],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.1],
    ]:
        with torch.inference_mode():
            x_seq = nn_model.p_sample_loop(
                size=1000,
                cond=torch.tensor([ground_cond]).to(device),
                return_steps=False,
            ).cpu()
        assert (
            KLdivergence(x_seq.numpy(), np.stack([sampling_two_normal(*ground_cond, n=1) for _ in range(1000)]))
            < 0.30
        )
