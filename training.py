import argparse

import h5py
import torch
import numpy as np

from stellarperceptron.model import StellarPerceptron


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the StellarPerceptron model")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run PyTorch on, e.g. 'cuda' for a GPU or 'cpu' for a CPU  (DO NOT USE 'mps' for Apple Silicon)",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Whether to use mixed precision training for CUDA",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Whether to compile the model using torch.compile",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=192,
        help="Model embedding dimension",
    )
    parser.add_argument(
        "--encoder_dense_num",
        type=int,
        default=2048,
        help="Number of dense neurons in the encoder layer",
    )
    parser.add_argument(
        "--encoder_dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate for the encoder layers",
    )
    parser.add_argument(
        "--encoder_n_layers",
        type=int,
        default=2,
        help="Number of transformer encoder layers",
    )
    parser.add_argument(
        "--diffusion_dense_num",
        type=int,
        default=192,
        help="Number of neurons in the diffusion dense layer",
    )
    parser.add_argument(
        "--diffusion_num_steps",
        type=int,
        default=100,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--save_model_to_folder",
        type=str,
        default="./trained_model/",
        help="Folder to save the model",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate for the optimizer",
    )
    parser.add_argument(
        "--learning_rate_min",
        type=float,
        default=1e-8,
        help="Minimum learning rate for the optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10240,
        help="Epochs to train the model for",
    )
    parser.add_argument(
        "--cosine_annealing_t0",
        type=int,
        default=1024,
        help="Cosine annealing restart length in epochs",
    )
    parser.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=512,
        help="Save a checkpoint every n epochs",
    )
    args = parser.parse_args()

    # load training data
    # xp_apogee = h5py.File("./data_files/training_set.h5", mode="r")
    xp_apogee = h5py.File("./data_files/training_set.h5", mode="r")
    xp_relevancy = xp_apogee["raw"]["xp_relevancy"][()]
    xp_coeffs_gnorm = xp_apogee["raw"]["xp_coeffs_gnorm"][()]
    xp_coeffs_err_gnorm = xp_apogee["raw"]["xp_coeffs_gnorm_err"][()]

    # propagate to deal with 53, 54, 108, 109 NaN issues
    xp_relevancy[:, 52:55] = xp_relevancy[:, 51:52]
    xp_relevancy[:, 107:110] = xp_relevancy[:, 106:107]
    xp_coeffs_gnorm[~xp_relevancy] = np.nan
    xp_coeffs_err_gnorm[~xp_relevancy] = np.nan

    training_labels = np.column_stack(
        [
            xp_coeffs_gnorm,
            xp_apogee["raw"]["bprp"][()],
            xp_apogee["raw"]["jh"][()],
            xp_apogee["raw"]["jk"][()],
            xp_apogee["raw"]["teff"][()],
            xp_apogee["raw"]["logg"][()],
            xp_apogee["raw"]["m_h"][()],
            xp_apogee["raw"]["logc19"][()],
            xp_apogee["raw"]["g_fakemag"][()],
        ]
    )

    print("Number of training stars: ", len(training_labels))

    training_labels_err = np.column_stack(
        [
            xp_coeffs_err_gnorm,
            xp_apogee["raw"]["bprp_err"][()],
            xp_apogee["raw"]["jh_err"][()],
            xp_apogee["raw"]["jk_err"][()],
            xp_apogee["raw"]["teff_err"][()],
            xp_apogee["raw"]["logg_err"][()],
            xp_apogee["raw"]["m_h_err"][()],
            xp_apogee["raw"]["logc19_err"][()],
            xp_apogee["raw"]["g_fakemag_err"][()],
        ]
    )
    xp_apogee.close()
    training_labels_err = np.where(np.isnan(training_labels_err), 0.0, training_labels_err)

    obs_names = [
        *[f"bp{i}" for i in range(55)],
        *[f"rp{i}" for i in range(55)],
        "bprp",
        "jh",
        "jk",
        "teff",
        "logg",
        "m_h",
        "logebv",
        "g_fakemag",
    ]

    nn_model = StellarPerceptron(
        vocabs=obs_names,
        embedding_dim=args.embedding_dim,
        embedding_activation="gelu",
        encoder_head_num=args.embedding_dim // 8,
        encoder_dense_num=args.encoder_dense_num,
        encoder_dropout_rate=args.encoder_dropout_rate,
        encoder_activation="gelu",
        encoder_n_layers=args.encoder_n_layers,
        diffusion_dense_num=args.diffusion_dense_num,
        diffusion_num_steps=args.diffusion_num_steps,
        device=args.device,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile_model,
        folder=args.save_model_to_folder,
    )

    nn_model.optimizer = torch.optim.AdamW(
        nn_model.torch_model.parameters(), lr=args.learning_rate
    )

    # There is no output labels, this model only has one output node depending what information you request
    # Here we choose a set of labels from inputs as possible information request to quickly train this model
    # In principle, any labels in inputs can be requested in output
    nn_model.fit(
        # give all because some of them will be randomly chosen shuffled in random order for each stars in each epoch
        inputs=training_labels,
        inputs_name=obs_names,
        inputs_err=training_labels_err,
        # during training, one of these will be randomly chosen for each stars in each epoch
        outputs_name=[
            *[f"bp{i}" for i in range(55)],
            *[f"rp{i}" for i in range(55)],
            "jh",
            "teff",
            "logg",
            "m_h",
            "logebv",
            "g_fakemag",
        ],
        batch_size=args.batch_size,
        val_batchsize_factor=10,
        epochs=args.epochs,
        lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cosine_annealing_t0,
            T_mult=1,
            eta_min=args.learning_rate_min,
            last_epoch=-1,  # means really the last epoch ever
        ),
        terminate_on_nan=True,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
    )
