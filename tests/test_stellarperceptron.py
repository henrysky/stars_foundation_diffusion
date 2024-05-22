import pathlib
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing

from stellarperceptron.model import StellarPerceptron

housing = fetch_california_housing()
val_labels = np.column_stack([housing.data, housing.target])
obs_names = housing.feature_names + ["HouseValue"]
val_labels_df = pd.DataFrame(val_labels, columns=obs_names)
device = "cpu"


def test_training():
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

    # There is no output labels, this model only has one output node depending what information you request
    # Here we choose a set of labels from inputs as possible information request to quickly train this model
    # In principle, any labels in inputs can be requested in output
    nn_model.fit(
        # give all because some of them will be randomly chosen shuffled in random order for each stars in each epoch
        inputs=val_labels,
        inputs_name=obs_names,
        # during training, one of these will be randomly chosen for each stars in each epoch
        outputs_name=obs_names,
        batch_size=128,
        val_batchsize_factor=10,
        epochs=16,
        lr_scheduler=None,
        length_range=(0, 10),
        terminate_on_nan=True,
        checkpoint_every_n_epochs=0,
    )
    assert model_path.exists()
    # delete the model in case test on local machine
    shutil.rmtree(model_path)


def test_inference():
    nn_model = StellarPerceptron.load(
        "./trained_california_model/",
        device=device,
        mixed_precision=False,
        compile_model=False,
    )
    nn_model.predict_samples(request_tokens="MedInc", size=100)
