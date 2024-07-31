import copy
import inspect
import json
import os
import pathlib
import re
import subprocess
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from astropy.stats import mad_std
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from .layers import NonLinearEmbedding, StellarPerceptronTorchModelwDDPM
from .nn_utils import TrainingGenerator


class StellarPerceptron:
    """
    StellarPerceptron
    """

    def __init__(
        self,
        vocabs: List[str],
        vocab_tokens: Optional[List[int]] = None,
        embedding_dim: int = 32,
        embedding_activation=None,
        encoder_head_num: int = 2,
        encoder_dense_num: int = 128,
        encoder_dropout_rate: float = 0.1,
        encoder_activation=None,
        encoder_n_layers: int = 2,
        diffusion_dense_num: int = 128,
        diffusion_n_layers: int = 3,
        diffusion_num_steps: int = 100,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        mixed_precision: bool = False,
        compile_model: bool = False,  # compile the model using torch.compile()
        folder: str = "trained_model",
        built: bool = False,  # do not use this arguement, it is for internal use only
    ):
        # ====================== Basic model information ======================
        self._built = built
        # only grab version, without cpu/cuda detail
        self.backend_framewoark = f"torch-{torch.__version__[:5]}"
        self.device = device
        self.dtype = dtype
        self.mixed_precision = mixed_precision
        self.compile_model = compile_model
        self.system_info = {}
        if "cuda" in self.device:
            self.device_type = "cuda"
        if (
            "mps" in self.device
        ):  # autocast is not well-supported on mps, so set to CPU for now
            self.device_type = "cpu"
        else:
            self.device_type = device
        self.torch_checklist()

        # ====================== Data Processing ======================
        self.vocabs = vocabs

        # check if vocabs are unique
        if len(self.vocabs) != len(set(self.vocabs)):
            raise ValueError("Vocabs are not unique!")

        self.vocab_size = len(self.vocabs)
        # remember always one (the first) for special padding token
        if vocab_tokens is None:
            self.vocab_tokens = [i for i in range(1, self.vocab_size + 1)]
        else:
            self.vocab_tokens = vocab_tokens

        self._input_mean = np.zeros(np.max(self.vocab_tokens) + 1, dtype=float)
        self._input_std = np.ones(np.max(self.vocab_tokens) + 1, dtype=float)
        self._input_standardized = np.zeros(np.max(self.vocab_tokens) + 1, dtype=bool)

        # ===================== Training parameters =====================
        # Training parameters storage
        self.epochs = None  # how many epochs the model is supposed to train
        self.current_epoch = 0  # last epoch trained
        self.current_learning_rate = None  # last learning rate used
        self.optimizer = None  # optimizer
        self.scheduler = None  # learning rate scheduler
        # always scale the gradients if using cuda
        self.gradient_scaler = torch.GradScaler(
            device=self.device, enabled=self.device_type == "cuda"
        )

        self.root_folder = pathlib.Path(folder).resolve()
        # only do when not loading trained model, prevent any overwriting to exisitng model folder
        if not self._built:
            self.root_folder.mkdir(parents=True, exist_ok=False)

        # ====================== Model parameters ======================
        self.embedding_dim = embedding_dim
        self.embedding_activation = embedding_activation
        self.encoder_head_num = encoder_head_num
        self.encoder_dense_num = encoder_dense_num
        self.encoder_dropout_rate = encoder_dropout_rate
        self.encoder_activation = encoder_activation
        self.encoder_n_layers = encoder_n_layers
        self.diffusion_dense_num = diffusion_dense_num
        self.diffusion_num_steps = diffusion_num_steps
        self.diffusion_n_layers = diffusion_n_layers

        self.embedding_layer = NonLinearEmbedding(
            input_dim=self.vocab_size + 1,  # plus 1 special padding token
            output_dim=self.embedding_dim,
            embeddings_initializer=torch.nn.init.xavier_uniform_,
            activation=self.embedding_activation,
            **self.factory_kwargs,
        )

        # ====================== Model initialization ======================
        self.torch_model = StellarPerceptronTorchModelwDDPM(
            self.embedding_layer,
            embedding_dim=self.embedding_dim,
            num_heads=self.encoder_head_num,
            dense_num=self.encoder_dense_num,
            dropout_rate=self.encoder_dropout_rate,
            activation=self.encoder_activation,
            num_layers=self.encoder_n_layers,
            diffusion_dense_num=self.diffusion_dense_num,
            diffusion_num_steps=self.diffusion_num_steps,
            diffusion_n_layers=self.diffusion_n_layers,
            built=self._built,
            **self.factory_kwargs,
        )
        if self.compile_model:
            self.torch_model.compile(fullgraph=True)
        # ====================== Model initialization ======================

        # ipython Auto-completion
        try:
            from IPython import get_ipython
        except ImportError:
            pass
        else:
            if (ipy := get_ipython()) is not None:

                def list_all_vocabs_completer(ipython, event):
                    out = self.vocabs
                    return out

                ipy.set_hook(
                    "complete_command",
                    list_all_vocabs_completer,
                    re_key=".*predict_samples",
                )
                ipy.set_hook(
                    "complete_command",
                    list_all_vocabs_completer,
                    re_key=".*predict_summary",
                )

    @property
    def factory_kwargs(self) -> dict:
        return {"device": self.device, "dtype": self.dtype}

    def torch_checklist(self):
        """
        Basic checklist for PyTorch
        """
        if "cpu" in self.factory_kwargs["device"]:
            if self.mixed_precision:
                warnings.warn(
                    "Setting mixed_precision=False because mixed precision is not supported on CPU"
                )
                self.mixed_precision = False
        if self.compile_model:
            try:
                torch.compile()
            except RuntimeError as e:
                warnings.warn(
                    f"Setting compile_model=False because torch.compile() is not supported due to {e.args[0]}"
                )
                self.compile_model = False

    def get_parameters_sum(self):
        """
        Function to count the tortal number of trainable parameters
        """
        model_parameters = filter(
            lambda p: p.requires_grad, self.torch_model.parameters()
        )
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def _built_only(self):
        """
        Check if the model is built, if not raise error. For functions that can only be called after model is built
        """
        if not self._built:
            raise NotImplementedError("This model is not trained")

    def _fit_checklist(
        self,
        inputs: NDArray,
        inputs_name: Union[NDArray, list],
        outputs_name: List[str],
        inputs_err: Optional[NDArray] = None,
    ):
        """
        Backend framework independent fit() checklist
        """
        inputs_name = np.asarray(inputs_name)
        if inputs_name.ndim == 2 and not np.all(
            [len(set(inputs_name[:, idx])) == 1 for idx in inputs_name.shape[1]]
        ):
            raise ValueError(
                "You need to give a orderly structure inputs to me, i.e. inputs cannot be pre-emptively randomized by you but I will handle it later. You can check source code"
            )

        data_length = len(inputs)
        inputs_token = self.tokenize(inputs_name, data_length=data_length)
        standardized_inputs, inputs_token, standardized_inputs_err = self.standardize(
            inputs, inputs_token, inputs_err
        )
        outputs_tokens = self.tokenize(outputs_name)[0]
        self._built = True

        return (
            standardized_inputs,
            inputs_token,
            outputs_tokens,
            standardized_inputs_err,
        )

    def _tokenize_core_logic(self, one_str: str) -> int:
        """
        Core logic of tokenization, used by both tokenize() and tokenize_name().
        Only tokenize one string at a time
        """
        if one_str not in self.vocabs:
            raise NameError(
                f"'{one_str}' is not one of the vocabs the model know which is {self.vocabs}"
            )
        if one_str == "[pad]":
            return 0
        else:
            return self.vocab_tokens[self.vocabs.index(one_str)]

    def _set_standardization(
        self, means: NDArray, stddev: NDArray, names: List[str]
    ) -> None:
        """
        Internal use to set standardization if they are pre-computed
        """
        tokens = self.tokenize(names, data_length=1)[0]
        for idx, i in enumerate(tokens):
            self._input_mean[i] = means[idx]
            self._input_std[i] = stddev[idx]
            self._input_standardized[i] = True

    def tokenize(
        self,
        names: Union[List[Union[str, int]], NDArray[Union[np.str_, np.integer]]],
        data_length: Optional[int] = None,
    ) -> NDArray[np.integer]:
        """
        Function to tokenize names

        Parameters
        ----------
        names : Union[List[Union[str, int]], NDArray[Union[np.str_, np.integer]]]
            List of names to be tokenized
        data_length : Optional[int], optional
            If provided, the output token array will be of this length, by default None so output token array will be of same length as input names array

        Returns
        -------
        NDArray[np.integer]
            Tokenized names
        """
        names = np.atleast_2d(names)
        if len(names) > 1 and data_length is not None:
            assert (
                len(names) == data_length
            ), f"'data_length' arguement (if provided) must match 'names' array length, they are {data_length} and {len(names)} now"
        need_tiling = True
        if data_length is None:
            # need to figure out the data_length if not provided
            data_length = len(names)
        need_tiling = data_length != len(names)
        out_tokens = np.zeros(names.shape, dtype=int)  # initialize output token array

        if np.issubdtype(
            names.dtype, np.integer
        ):  # in case already tokenized, then nothing to do
            out_tokens = names
            if need_tiling:
                out_tokens = np.tile(names, (data_length, 1))
        else:  # the case where names are strings
            # in case tokens are already tiled and in nice order for every row
            # OR in case only header names are given, so do the header and tiles
            nice_order = np.all([len(set(i)) == 1 for i in names.T])
            if nice_order or need_tiling:
                _temp_names = names[0] if (nice_order and not need_tiling) else names
                out_tokens = np.tile(
                    [
                        self._tokenize_core_logic(i)
                        for i in np.atleast_1d(np.squeeze(_temp_names))
                    ],
                    (data_length, 1),
                )
            # this is the case where tokens ordering for each row is different
            else:
                # do it the slow way
                for i in np.unique(names):
                    idx = names == i
                    out_tokens[idx] = self._tokenize_core_logic(i)
        return out_tokens

    def detokenize(self, tokens: NDArray) -> NDArray:
        # need to offset by 1 index because 0 is reserved for padding
        detokenized = [self.vocabs[t - 1] for t in tokens]
        return detokenized

    def standardize(
        self,
        inputs: NDArray[Union[np.float32, np.float64]],
        inputs_token: NDArray[np.int_],
        inputs_error: Optional[NDArray] = None,
        dtype: np.dtype = np.float32,
    ):
        """
        Standardize input data to mean=0, stddev=1. Also set NaN to 0 and as padding

        if "inputs_error" is given, this function assume inputs_error is in the same order of inputs (i.e. inputs_token also applies to inputs_error)
        """
        # prevent changing the original array
        _inputs = copy.deepcopy(inputs)
        _inputs_token = self.tokenize(copy.deepcopy(inputs_token))
        _inputs_error = copy.deepcopy(inputs_error)

        unique_tokens = np.unique(inputs_token)

        for i in unique_tokens:
            if i != 0:  # no standardize for padding token
                if not self._input_standardized[i]:
                    # if not already standardized before, calcaulate mean and stddev again
                    self._input_mean[i] = np.nanmedian(_inputs[inputs_token == i])
                    self._input_std[i] = mad_std(
                        _inputs[inputs_token == i], ignore_nan=True
                    )
                    self._input_standardized[i] = True
                # standardize data
                _inputs[inputs_token == i] -= self._input_mean[i]
                _inputs[inputs_token == i] /= self._input_std[i]
                if _inputs_error is not None:
                    _inputs_error[inputs_token == i] /= self._input_std[i]

        # mask NaN
        nan_idx = np.isnan(inputs)
        _inputs_token = np.where(nan_idx, 0, inputs_token)
        _inputs = np.where(nan_idx, 0, _inputs)
        if _inputs_error is None:
            return _inputs, _inputs_token
        else:
            _inputs_error = np.where(nan_idx, 0, _inputs_error)
            return (
                _inputs.astype(dtype=dtype),
                _inputs_token,
                _inputs_error.astype(dtype=dtype),
            )

    def inverse_standardize(
        self,
        inputs: NDArray[Union[np.float32, np.float64]],
        inputs_token: NDArray[np.int_],
        inputs_error: Optional[NDArray[Union[np.float32, np.float64]]] = None,
    ):
        # prevent changing the original array
        _inputs = copy.deepcopy(inputs)
        _inputs_token = self.tokenize(copy.deepcopy(inputs_token))
        _inputs_error = copy.deepcopy(inputs_error)

        unique_tokens = np.unique(_inputs_token)

        for i in unique_tokens:
            _inputs[inputs_token == i] *= self._input_std[i]
            _inputs[inputs_token == i] += self._input_mean[i]
            if _inputs_error is not None:
                _inputs_error[inputs_token == i] *= self._input_std[i]
        if _inputs_error is None:
            return _inputs
        else:
            return _inputs, _inputs_error

    def get_config(self):
        # get config of the network
        nn_config = {
            "backend_framewoark": self.backend_framewoark,
            "embedding_dim": self.embedding_dim,
            "embedding_activation": self.embedding_activation,
            "encoder_head_num": self.encoder_head_num,
            "encoder_dense_num": self.encoder_dense_num,
            "encoder_dropout_rate": self.encoder_dropout_rate,
            "encoder_activation": self.encoder_activation,
            "encoder_n_layers": self.encoder_n_layers,
        }
        # get config of the tokenizer
        tokenizer_config = {
            "vocabs": self.vocabs,
            "vocab_tokens": self.vocab_tokens,
        }

        # get config of normalization
        norm_config = {
            "_input_mean": self._input_mean.tolist(),
            "_input_std": self._input_std.tolist(),
            "_input_standardized": self._input_standardized.tolist(),
        }

        return {
            "nn_config": nn_config,
            "ddpm_config": self.torch_model.diffusion_head.get_config(),  # denosing diffusion probabilistic model
            "tokenizer_config": tokenizer_config,
            "norm_config": norm_config,
        }

    def save(self, folder_name: str = "model") -> None:
        """
        Backend framework independent save
        """
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
        file_name = f"{folder_name}/checkpoints/epoch_{self.current_epoch}.pt"
        file_name = pathlib.Path(file_name).resolve()
        if not file_name.exists():
            torch.save(
                {
                    "model_state_dict": self.torch_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "optimizer": self.optimizer.__class__.__name__,
                    "scheduler": self.scheduler.__class__.__name__,
                    "scheduler_state_dict": self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None,
                    "gradient_scaler_state_dict": self.gradient_scaler.state_dict(),
                    "current_epoch": self.current_epoch,
                },
                file_name,
            )
        else:
            raise FileExistsError(
                f"This folder at {file_name}seems to already has a model saved inside! Please use a different folder name."
            )

    @classmethod
    def load(
        cls,
        folder_name: str,
        checkpoint_epoch: int = -1,
        mixed_precision: bool = False,
        compile_model: bool = False,
        device: str = "cpu",
    ):
        """
        Backend framework independent loading

        Parameters
        ----------
        folder_name : str
            folder name of the model
        checkpoint_epoch : int
            checkpoint epoch to load, -1 for the last model
        mixed_precision : bool
            whether to use mixed precision
        compile_model : bool
            whether to compile the model
        device : str
            device to load the model to
        """
        if not os.path.exists(folder_name):
            raise FileNotFoundError(f"Model folder {folder_name} not found!")
        else:
            with open(f"{folder_name}/config.json", "r") as f:
                config = json.load(f)

        if checkpoint_epoch != -1:
            path_to_weights = pathlib.Path(
                f"{folder_name}/checkpoints/epoch_{checkpoint_epoch}.pt"
            ).resolve()
            if not path_to_weights.exists():
                raise FileNotFoundError(
                    f"Checkpoint at epoch {checkpoint_epoch} not found at {path_to_weights}!"
                )
        elif checkpoint_epoch == -1:
            # get the last checkpoint
            checkpoints = list(pathlib.Path(f"{folder_name}/checkpoints").glob("*.pt"))
            if len(checkpoints) == 0:
                raise FileNotFoundError("No checkpoint found!")
            path_to_weights = sorted(
                checkpoints,
                key=lambda filename: int(
                    re.search(r"(\d+)", str(filename.stem)).group(1)
                ),
            )[-1]

        nn = cls(
            vocabs=config["tokenizer_config"]["vocabs"],
            vocab_tokens=config["tokenizer_config"]["vocab_tokens"],
            embedding_dim=config["nn_config"]["embedding_dim"],
            embedding_activation=config["nn_config"]["embedding_activation"],
            encoder_head_num=config["nn_config"]["encoder_head_num"],
            encoder_dense_num=config["nn_config"]["encoder_dense_num"],
            encoder_dropout_rate=config["nn_config"]["encoder_dropout_rate"],
            encoder_activation=config["nn_config"]["encoder_activation"],
            encoder_n_layers=config["nn_config"]["encoder_n_layers"],
            # denosing diffusion probabilistic model
            diffusion_dense_num=config["ddpm_config"]["dense_num"],
            diffusion_n_layers=config["ddpm_config"]["n_layers"],
            diffusion_num_steps=config["ddpm_config"]["num_steps"],
            # other config
            device=device,
            mixed_precision=mixed_precision,
            compile_model=compile_model,
            folder=folder_name,
            built=True,
        )

        nn._input_mean = np.array(config["norm_config"]["_input_mean"])
        nn._input_std = np.array(config["norm_config"]["_input_std"])
        nn._input_standardized = np.array(config["norm_config"]["_input_standardized"])

        model_f = torch.load(path_to_weights, map_location=device, weights_only=True)
        nn.torch_model.load_state_dict(
            model_f["model_state_dict"],
            strict=True,
        )
        nn.current_epoch = model_f["current_epoch"]

        # lookup optimizer name in state dict and instantiate it
        optimizer_func = getattr(torch.optim, model_f["optimizer"])
        optimizer_func_keywords = list(
            inspect.signature(optimizer_func).parameters.keys()
        )
        optimizer_func_keywords.remove("params")
        nn.optimizer = optimizer_func(
            params=nn.torch_model.parameters(),
            **{
                k: v
                for k, v in model_f["optimizer_state_dict"]["param_groups"][0].items()
                if k in optimizer_func_keywords
            },
        )
        nn.optimizer.load_state_dict(state_dict=model_f["optimizer_state_dict"])
        if model_f["scheduler"] != "NoneType":
            scheduler_func = getattr(torch.optim.lr_scheduler, model_f["scheduler"])
            # get list of arguments of the scheduler function
            scheduler_func_keywords = list(
                inspect.signature(scheduler_func).parameters.keys()
            )
            try:
                scheduler_func_keywords.remove(
                    "verbose"
                )  # deprecated keyword for scheduler
            except ValueError:
                pass
            scheduler_func_args = model_f["scheduler_state_dict"]
            try:
                scheduler_func_args["last_epoch"] -= (
                    1  # PyTorch scheduler last_epoch need to -1 since there is an initial run
                )
            except ValueError:
                pass
            nn.scheduler = scheduler_func(
                nn.optimizer,
                **{
                    k: v
                    for k, v in scheduler_func_args.items()
                    if k in scheduler_func_keywords
                },
            )
            nn.scheduler.load_state_dict(model_f["scheduler_state_dict"])
        else:
            nn.scheduler = None
        nn.gradient_scaler.load_state_dict(model_f["gradient_scaler_state_dict"])

        return nn

    def fit(
        self,
        inputs: NDArray,
        inputs_name: NDArray,
        outputs_name: List[str],
        inputs_err: Optional[NDArray] = None,
        outputs: Optional[NDArray] = None,
        batch_size: int = 64,
        length_range: Tuple[int, int] = (0, 64),
        # batch size to use for validation compared to training batch size
        val_batchsize_factor: int = 5,
        epochs: int = 32,
        validation_split: float = 0.1,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
        checkpoint_every_n_epochs: int = 0,  # save checkpoint every n epochs, put 0 to disable
        terminate_on_nan: bool = True,
        terminate_on_checkpoint: bool = False,
    ) -> None:
        self.epochs = epochs
        if inputs_err is None:
            inputs_err = np.zeros_like(inputs)
        if outputs is None:
            outputs = np.array(inputs)

        # check checkpoint_every_n_epochs
        if checkpoint_every_n_epochs < 0:
            raise ValueError("checkpoint_every_n_epochs can not be less than zero")
        elif checkpoint_every_n_epochs == 0:
            checkpoint_every_n_epochs = epochs
        pathlib.Path(f"{self.root_folder}/checkpoints").mkdir(
            parents=True, exist_ok=True
        )

        training_log_path = f"{self.root_folder}/training.log"
        # check if the exists
        if not pathlib.Path(training_log_path).exists():
            training_log_f = open(training_log_path, "w")
            training_log_f.write(f"Batch Size: {batch_size}\n")
            training_log_f.write("====================================\n")
        else:
            training_log_f = open(training_log_path, "a+")

        training_csv_metrics_path = f"{self.root_folder}/training_metrics.csv"
        # check if the file exists
        if not pathlib.Path(training_csv_metrics_path).exists():
            training_csv_metrics_f = open(training_csv_metrics_path, "w")
            training_csv_metrics_f.write(
                "time,loss,mse_loss,val_loss,val_mse_loss,lr\n"
            )
        else:
            training_csv_metrics_f = open(training_csv_metrics_path, "a+")

        system_info_path = f"{self.root_folder}/training_system_info.log"
        # check if the file exists
        if not pathlib.Path(system_info_path).exists():
            with open(
                f"{self.root_folder}/training_system_info.log", "w"
            ) as system_info_f:
                system_info_f.write(
                    subprocess.run(
                        ["python", "-m", "torch.utils.collect_env"],
                        stdout=subprocess.PIPE,
                    ).stdout.decode("utf-8")
                )

        (
            standardized_inputs,
            inputs_token,
            outputs_tokens,
            standardized_inputs_err,
        ) = self._fit_checklist(
            inputs=inputs,
            inputs_name=inputs_name,
            outputs_name=outputs_name,
            inputs_err=inputs_err,
        )

        (
            X_train,
            X_val,
            X_train_token,
            X_val_token,
            X_Err_train,
            X_Err_val,
        ) = train_test_split(
            standardized_inputs,
            inputs_token,
            standardized_inputs_err,
            test_size=validation_split,
        )

        training_generator = TrainingGenerator(
            batch_size=batch_size,
            data={
                "input": X_train,
                "input_token": X_train_token,
                "output": X_train,
                "output_err": X_Err_train,
            },
            aggregate_nans=False,
            length_range=length_range,
            possible_output_tokens=outputs_tokens,
            factory_kwargs=self.factory_kwargs,
        )

        val_generator = TrainingGenerator(
            batch_size=batch_size * val_batchsize_factor,
            data={
                "input": X_val,
                "input_token": X_val_token,
                "output": X_val,
                "output_err": X_Err_val,
            },
            aggregate_nans=False,
            length_range=length_range,
            possible_output_tokens=outputs_tokens,
            shuffle=False,  # no need to shuffle for validation
            factory_kwargs=self.factory_kwargs,
        )

        if self.scheduler is None and lr_scheduler is not None:
            self.scheduler = lr_scheduler(self.optimizer)

        json_path = f"{self.root_folder}/config.json"
        if not pathlib.Path(json_path).exists():
            with open(json_path, "w") as f:
                json.dump(self.get_config(), f, indent=4)

        # ====================== Training logic ======================
        elapsed_time = 0
        with tqdm.tqdm(
            range(self.current_epoch, self.epochs),
            total=self.epochs,
            initial=self.current_epoch,
            unit="epoch",
        ) as pbar:
            for epoch in pbar:
                self.current_epoch = epoch + 1
                # print(f"Epoch {self.epoch}/{self.epochs}")
                training_log_f.write(f"Epoch {self.current_epoch}/{self.epochs}\n")

                self.torch_model.train()
                running_loss = 0.0
                running_mse_loss = 0.0
                last_loss = 0.0

                # order: input, input_token, label_token, label, label_err
                for batch_num, (
                    inputs,
                    input_token,
                    label_token,
                    label,
                    label_err,
                ) in enumerate(training_generator):
                    # reset gradient for every batch
                    self.optimizer.zero_grad()
                    with torch.autocast(
                        device_type=self.device_type,
                        enabled=self.mixed_precision,
                    ):
                        loss = self.torch_model.get_loss(
                            inputs,
                            input_token,
                            label_token,
                            label,
                        )
                    self.gradient_scaler.scale(loss).backward()
                    self.gradient_scaler.step(self.optimizer)
                    self.gradient_scaler.update()
                    running_loss += loss.item()
                    # running_mse_loss += loss_mse.item()

                last_loss = running_loss / (batch_num + 1)
                last_mse_loss = running_mse_loss / (batch_num + 1)
                training_generator.on_epoch_end(epoch=self.current_epoch)
                # ======== epoch level validation ========
                self.torch_model.eval()
                running_vloss = 0.0
                # running_vloss_mse = 0.0
                with torch.no_grad():
                    # order: input, input_token, label_token, label, label_err
                    for batch_num, (
                        inputs,
                        input_token,
                        label_token,
                        label,
                        label_err,
                    ) in enumerate(val_generator):
                        with torch.autocast(
                            device_type=self.device_type,
                            enabled=self.mixed_precision,
                        ):
                            vloss = self.torch_model.get_loss(
                                inputs,
                                input_token,
                                label_token,
                                label,
                            )
                        running_vloss += vloss.item()
                        # running_vloss_mse += vloss_mse.item()

                avg_vloss = running_vloss / (batch_num + 1)
                # avg_vloss_mse = running_vloss_mse / (batch_num + 1)

                # store loss, val_loss and learning rate
                self.loss = last_loss
                self.val_loss = avg_vloss
                self.current_learning_rate = self.optimizer.param_groups[-1]["lr"]
                val_generator.on_epoch_end(epoch=self.current_epoch)

                # ======== post-epoch activity ========
                if self.scheduler is not None:
                    self.scheduler.step()
                lr_fmt = np.format_float_scientific(
                    self.current_learning_rate, precision=4, unique=False
                )
                temp_time = pbar.format_dict["elapsed"] - elapsed_time
                elapsed_time = pbar.format_dict["elapsed"]
                # training_log_f.write(
                #     f"elapsed: {str(timedelta(seconds=elapsed_time))}s - rate: {temp_time:.2f}s - loss: {last_loss:.4f} - mse_loss: {last_mse_loss:.4f} val_loss {avg_vloss:.4f} - val_mse_loss {avg_vloss_mse:.4f} - lr: {lr_fmt}\n"
                # )
                training_log_f.write(
                    f"elapsed: {str(timedelta(seconds=elapsed_time))}s - rate: {temp_time:.2f}s - loss: {last_loss:.4f} - val_loss {avg_vloss:.4f} - lr: {lr_fmt}\n"
                )
                training_log_f.flush()
                # training_csv_metrics_f.write(
                #     f"{temp_time},{last_loss},{last_mse_loss},{avg_vloss},{avg_vloss_mse},{lr_fmt}\n"
                # )
                training_csv_metrics_f.write(
                    f"{temp_time},{last_loss},{last_mse_loss},{avg_vloss},{lr_fmt}\n"
                )
                training_csv_metrics_f.flush()

                if terminate_on_nan and np.isnan(last_loss):
                    raise ValueError("Loss is NaN, hence training terminated!")

                if checkpoint_every_n_epochs > 0:
                    if self.current_epoch % checkpoint_every_n_epochs == 0:
                        self.save(folder_name=self.root_folder)
                        if terminate_on_checkpoint and self.current_epoch != 1:
                            warnings.warn(
                                "Training terminated due to checkpoint has been reached!"
                            )
                            break
            # ====================== Training logic ======================
        training_log_f.close()
        training_csv_metrics_f.close()

    def predict_samples(
        self,
        *,
        inputs: Union[List[float], NDArray] = None,
        input_tokens: List[Union[int, str]] = None,
        request_tokens: List[Union[int, str]] = None,
        size: int = 10000,
        batch_size: int = 128,
    ):
        """
        This function to generate posterior samples from the model

        Args:
            inputs (Union[List[float], NDArray]): The input data to be used. The shape of the input data should be (n_samples, n_features).
                If it is pandas DataFrame, the column names should be vacobs understood by the model. The inputs should NOT be standardized.
            input_tokens (List[Union[int, str]]): Tokens or names of input data.
            request_tokens (List[Union[int, str]]): Tokens or names of requested data.
            size (int, optional): Number of samples to generate posterior. Defaults to 10000.
            batch_size (int, optional): Batch size for prediction. Defaults to 128.

        Returns:
            np.ndarray: The size of the array will be (size, n_samples, n_requested_data)

        Examples:
            >>> nn_model.predict_samples([4700, 2.5], ["teff", "logg"], ["teff"])
        """
        self._built_only()
        self.torch_model.eval()
        if isinstance(request_tokens, list) and len(request_tokens) > 1:
            raise ValueError(
                "Only one requested token is allowed so far limited by implementation"
            )
        if inputs is None:
            inputs = [-9999.99]
            input_tokens = [0]
        inputs = np.atleast_2d(inputs)
        input_tokens = np.atleast_2d(input_tokens)
        request_tokens = np.atleast_2d(request_tokens)

        # deal with length of data
        data_len = len(inputs)
        num_batch = data_len // batch_size
        num_batch_remainder = data_len % batch_size

        input_tokens = self.tokenize(input_tokens, data_length=len(inputs))
        inputs, input_tokens = self.standardize(inputs, input_tokens)
        self._last_padding_mask = input_tokens == 0

        if request_tokens.dtype.type == np.str_:
            request_tokens = self.tokenize(request_tokens, data_length=len(inputs))

        # ndarray to torch tensor
        input_tokens = torch.as_tensor(
            input_tokens, device=self.factory_kwargs["device"], dtype=torch.int32
        )
        inputs = torch.atleast_3d(torch.as_tensor(inputs, **self.factory_kwargs))
        request_tokens = torch.as_tensor(
            request_tokens, device=self.factory_kwargs["device"], dtype=torch.int32
        )
        # TODO: actually should be len(request_tokens))
        posterior = np.zeros((size, data_len))

        with torch.inference_mode():
            if num_batch == 0:  # if smaller than batch_size, then do all at once
                x_cond = self.torch_model(inputs, input_tokens, request_tokens).repeat(
                    size, 1
                )
                request_tokens = request_tokens.repeat(size, 1)

                posterior = self.torch_model.diffusion_head.p_sample_loop(
                    size=data_len * size,
                    cond=x_cond,
                    return_steps=False,
                )
                posterior = (
                    self.inverse_standardize(
                        posterior.cpu(),
                        request_tokens.cpu(),
                    )
                    .view(size, -1)
                    .numpy()
                )
            else:
                for i in range(num_batch):
                    request_tokens_batch = request_tokens[
                        i * batch_size : i * batch_size + batch_size
                    ]
                    x_cond = self.torch_model(
                        inputs[i * batch_size : i * batch_size + batch_size],
                        input_tokens[i * batch_size : i * batch_size + batch_size],
                        request_tokens_batch,
                    ).repeat(size, 1)
                    request_tokens_batch = request_tokens_batch.repeat(size, 1)
                    _temp = self.torch_model.diffusion_head.p_sample_loop(
                        batch_size * size,
                        cond=x_cond,
                        return_steps=False,
                    )
                    posterior[:, i * batch_size : i * batch_size + batch_size] = (
                        self.inverse_standardize(
                            _temp.cpu(),
                            request_tokens_batch.cpu(),
                        )
                        .view(size, -1)
                        .numpy()
                    )
                if num_batch_remainder > 0:
                    request_tokens_batch = request_tokens[num_batch * batch_size :]
                    # do the remainder
                    x_cond = self.torch_model(
                        inputs[num_batch * batch_size :],
                        input_tokens[num_batch * batch_size :],
                        request_tokens_batch,
                    ).repeat(size, 1)
                    request_tokens_batch = request_tokens_batch.repeat(size, 1)

                    _temp = self.torch_model.diffusion_head.p_sample_loop(
                        size=num_batch_remainder * size,
                        cond=x_cond,
                        return_steps=False,
                    )
                    posterior[:, num_batch * batch_size :] = (
                        self.inverse_standardize(
                            _temp.cpu(),
                            request_tokens_batch.cpu(),
                        )
                        .view(size, -1)
                        .numpy()
                    )
        return posterior

    def predict_summary(
        self,
        *,
        inputs: Union[List[float], NDArray] = None,
        input_tokens: List[Union[int, str]] = None,
        request_tokens: List[Union[int, str]] = None,
        batch_size=128,
        size: int = 10000,
    ):
        """
        This function to generate summary statistics of posterior samples from the model

        Args:
            inputs (Union[List[float], NDArray]): The input data to be used. The shape of the input data should be (n_samples, n_features).
                If it is pandas DataFrame, the column names should be vacobs understood by the model.  The inputs should NOT be standardized.
            input_tokens (List[Union[int, str]]): Tokens or names of input data.
            request_tokens (List[Union[int, str]]): Tokens or names of requested data.
            batch_size (int, optional): Batch size for prediction. Defaults to 128.
            size (int, optional): Number of samples to generate posterior. Defaults to 10000.

        Returns:
            pd.DataFrame: The size of the array will be (n_samples, 2 * n_requested_data), where the first half is the median and the second half is the MAD standard deviation

        Examples:
            >>> nn_model.predict_summary([4700, 2.5], ["teff", "logg"], "teff")
        """
        if inputs is None:
            inputs = [-9999.99]
            input_tokens = [0]
        inputs = np.atleast_2d(inputs)
        input_tokens = np.atleast_2d(input_tokens)
        input_tokens = self.tokenize(input_tokens, data_length=len(inputs))
        request_tokens = np.atleast_2d(request_tokens)
        request_tokens_num = request_tokens.shape[1]

        # deal with length of data
        data_len = len(inputs)
        num_batch = data_len // batch_size
        num_batch_remainder = data_len % batch_size

        median_ls = np.ones((data_len, request_tokens_num), dtype=float) * np.nan
        mad_std_ls = np.ones((data_len, request_tokens_num), dtype=float) * np.nan

        if num_batch == 0:  # if smaller than batch_size, then do all at once
            for request_idx in range(request_tokens_num):
                posterior = self.predict_samples(
                    inputs=inputs,
                    input_tokens=input_tokens,
                    request_tokens=request_tokens[:, request_idx],
                    size=size,
                )
                median_ls[:, request_idx] = np.median(posterior, axis=0)
                mad_std_ls[:, request_idx] = mad_std(posterior, axis=0)
        else:
            # tqdm progress bar, will update manually
            with tqdm.tqdm(
                total=request_tokens_num * data_len,
                unit="samples",
            ) as pbar:
                for request_idx in range(request_tokens_num):
                    for i in range(num_batch):
                        with torch.autocast(
                            device_type=self.device_type,
                            enabled=self.mixed_precision,
                        ):
                            posterior = self.predict_samples(
                                inputs=inputs[i * batch_size : i * batch_size + batch_size],
                                input_tokens=input_tokens[
                                    i * batch_size : i * batch_size + batch_size
                                ],
                                request_tokens=request_tokens[:, request_idx],
                                size=size,
                            )
                            median_ls[
                                i * batch_size : i * batch_size + batch_size, request_idx
                            ] = np.median(posterior, axis=0)
                            mad_std_ls[
                                i * batch_size : i * batch_size + batch_size, request_idx
                            ] = mad_std(posterior, axis=0)
                            pbar.update(batch_size)
                if num_batch_remainder > 0:
                    # do the remainder
                    for request_idx in range(request_tokens_num):
                        posterior = self.predict_samples(
                            inputs=inputs[num_batch * batch_size :],
                            input_tokens=input_tokens[num_batch * batch_size :],
                            request_tokens=request_tokens[:, request_idx],
                            size=size,
                        )
                        median_ls[num_batch * batch_size :, request_idx] = np.median(
                            posterior, axis=0
                        )
                        mad_std_ls[num_batch * batch_size :, request_idx] = mad_std(
                            posterior, axis=0
                        )
                        pbar.update(num_batch_remainder)

        all_pred = np.column_stack((median_ls, mad_std_ls))
        # detokenize instead of getting from arguement to make sure the name is english
        col_names = request_tokens.flatten().tolist()
        col_names.extend([f"{i}_error" for i in col_names])
        df = pd.DataFrame(all_pred, columns=col_names)
        return df
