from stellarperceptron.ddpm import ConditionalDiffusionModel
import torch
from sklearn.datasets import make_moons
from utils import KLdivergence

device = "cpu"


def test_moon():
    nn_model = ConditionalDiffusionModel(
        dim=2, cond_dim=0, dense_num=128, num_steps=100, beta_end=1.0e-2, device=device
    )
    moons, _ = make_moons(10**4, noise=0.05)
    dataset = torch.tensor(moons, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
    batch_size = 128

    for t in range(256):
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
    
    assert KLdivergence(x_seq.cpu().numpy(), dataset.cpu().numpy()) < 0.1