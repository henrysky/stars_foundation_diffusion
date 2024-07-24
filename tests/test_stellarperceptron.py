import pathlib
import shutil
import pytest

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing

from stellarperceptron.model import StellarPerceptron


@pytest.fixture(scope="module")
def device():
    return "cpu"


@pytest.fixture(scope="module")
def housing():
    housing = fetch_california_housing()
    val_labels = np.column_stack([housing.data, housing.target])
    obs_names = housing.feature_names + ["HouseValue"]
    val_labels_df = pd.DataFrame(val_labels, columns=obs_names)
    return val_labels_df, obs_names


def test_training(device, housing):
    val_labels_df, obs_names = housing
    model_path = pathlib.Path("test_california_model")

    nn_model = StellarPerceptron(
        vocabs=obs_names,
        embedding_dim=32,
        embedding_activation="gelu",
        encoder_head_num=4,
        encoder_dense_num=128,
        encoder_dropout_rate=0.1,
        encoder_activation="gelu",
        encoder_n_layers=2,
        diffusion_dense_num=64,
        diffusion_num_steps=100,
        device=device,
        mixed_precision=False,
        compile_model=False,
        folder=model_path,
    )

    nn_model.optimizer = torch.optim.AdamW(nn_model.torch_model.parameters(), lr=5.0e-3)
    nn_model.fit(
        inputs=val_labels_df[obs_names].values,
        inputs_name=obs_names,
        outputs_name=obs_names,
        batch_size=128,
        val_batchsize_factor=10,
        epochs=16,
        lr_scheduler=None,
        length_range=(0, 10),
        terminate_on_nan=True,
        checkpoint_every_n_epochs=8,
    )
    assert model_path.exists()

    # make sure freshly created model (and their epochs) can be loaded
    nn_model = StellarPerceptron.load(
        model_path,
        checkpoint_epoch=8,
        device=device,
        mixed_precision=False,
        compile_model=False,
    )
    assert nn_model.current_epoch == 8
    nn_model = StellarPerceptron.load(
        model_path,
        checkpoint_epoch=16,
        device=device,
        mixed_precision=False,
        compile_model=False,
    )
    assert nn_model.current_epoch == 16
    StellarPerceptron.load(
        model_path,
        device=device,
        mixed_precision=False,
        compile_model=False,
    )
    assert nn_model.current_epoch == 16
    # delete the model in case test on local machine
    shutil.rmtree(model_path)


def test_inference(device):
    nn_model = StellarPerceptron.load(
        "./trained_california_model/",
        device=device,
        mixed_precision=False,
        compile_model=False,
    )
    # make sure the case of no input tokens work
    pred = nn_model.predict_samples(request_tokens="MedInc", size=1000)
    assert len(pred) == 1000

    nn_high_HouseValue_pred = nn_model.predict_samples(
        inputs=[7.0], input_tokens=["MedInc"], request_tokens="HouseValue", size=1000
    )
    nn_low_HouseValue_pred = nn_model.predict_samples(
        inputs=[1.0], input_tokens=["MedInc"], request_tokens="HouseValue", size=1000
    )
    assert np.median(nn_high_HouseValue_pred) > np.median(nn_low_HouseValue_pred)
