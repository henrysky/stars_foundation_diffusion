import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

rng = np.random.default_rng()


def robust_mean_squared_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    variance: torch.Tensor,
    labels_err: torch.Tensor,
) -> torch.Tensor:
    # Neural Net is predicting log(var), so take exp, takes account the target variance, and take log back
    total_var = torch.exp(variance) + torch.square(labels_err)
    wrapper_output = 0.5 * (
        (torch.square(y_true - y_pred) / total_var) + torch.log(total_var)
    )

    losses = wrapper_output.sum() / y_true.shape[0]
    return losses


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    losses = (torch.square(y_true - y_pred)).sum() / y_true.shape[0]
    return losses


def shuffle_row(arrays_ls: List[NDArray], first_n: Optional[List[int]] = None) -> None:
    """
    Function to shuffle row-wise multiple 2D array of the same shape in the same way

    This function would change the original array in place!!!

    Args:
        arrays_ls (List[NDArray]): The array(s) to be shuffled in the same way
        first_n (Optional[List[int]], optional): If not None, the first ``first_n`` elements of each row will be shuffled, by default None to shuffle all elements within each row
    """
    array_shape = tuple(arrays_ls[0].shape)
    if not isinstance(arrays_ls, list):
        raise ValueError(
            "arrays_ls should be a list of arrays, even if there is only one array"
        )
    if not all([a.shape == array_shape for a in arrays_ls]):
        raise ValueError("All arrays should have the same shape")
    
    master_idx = np.tile(np.arange(array_shape[1]), (array_shape[0], 1))

    if first_n is None:  # if yes then use a faster implementation
        # shuffling row by row for master_idx
        [rng.shuffle(master_idx[i, :]) for i in np.arange(array_shape[0])]
        return [np.take_along_axis(a, master_idx, axis=1) for a in arrays_ls]
    else:
        [rng.shuffle(master_idx[i, :n]) for i, n in zip(np.arange(array_shape[0]), first_n)]
        return [np.take_along_axis(a, master_idx, axis=1) for a in arrays_ls]


# def shuffle_row(arrays_ls, first_n=None) -> None:
#     """
#     Function to shuffle row-wise multiple 2D array of the same shape in the same way

#     This function would change the original array in place!!!
#     """
#     array_shape = tuple(arrays_ls[0].shape)
#     if not isinstance(arrays_ls, list):
#         raise ValueError(
#             "arrays_ls should be a list of arrays, even if there is only one array"
#         )
#     if not all([a.shape == array_shape for a in arrays_ls]):
#         raise ValueError("All arrays should have the same shape")
    
#     indices = torch.argsort(torch.rand(*array_shape), dim=-1)
#     idx = torch.arange(array_shape[0]).unsqueeze(-1)
#     return [a[idx, indices] for a in arrays_ls]


def random_choice(items: NDArray, prob_matrix: Optional[NDArray] = None) -> NDArray:
    """
    This function is to randomly choose an item from each row of a 2D array based on the probability matrix (if provided)

    See https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix
    """
    if prob_matrix is None:  # if not provided, assume uniform distribution
        prob_matrix = np.tile(
            np.ones_like(items.shape[1], dtype=float), (items.shape[0], 1)
        ).T
    # making sure prob_matrix is normalized to 1
    prob_matrix /= prob_matrix.sum(axis=1, keepdims=True)

    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0], 1)
    k = (s < r).sum(axis=1)
    return np.take_along_axis(items, k[:, None], axis=1)


class TrainingGenerator(torch.utils.data.IterableDataset):
    def __init__(
        self,
        batch_size: int,
        data: dict,
        length_range: Tuple[int] = (1, 64),
        possible_output_tokens: List[int] = None,
        shuffle: bool = True,
        aggregate_nans: bool = True,
        factory_kwargs: dict = {"device": "cpu", "dtype": torch.float32},
    ):
        """
        Parameters
        ----------
        batch_size : int
            batch size
        data : dict
            data dictionary that contains the following keys: ["input", "input_token", "output", "output_err"]
        length_range : Tuple[int], optional
            additional padding for output, by default 0
        possible_output_tokens : np.ndarray, optional
            possible output tokens, by default None
        shuffle : bool, optional
            shuffle data, by default True
        aggregate_nans : bool, optional
            aggregate nans of every rows to the end of those rows, by default True
        """
        self.factory_kwargs = factory_kwargs
        self.input = copy.deepcopy(data["input"])
        self.input_idx = copy.deepcopy(data["input_token"])
        self.output = data["output"]
        self.output_err = copy.deepcopy(data["output_err"])
        self.length_range = length_range
        self.data_length = len(self.input)
        self.data_width = self.input.shape[1]
        self.shuffle = shuffle  # shuffle row ordering, star level column ordering shuffle is mandatory
        self.aggregate_nans = aggregate_nans

        # handle possible output tokens for star-by-star basis
        self.possible_output_tokens = possible_output_tokens
        prob_matrix = np.tile(
            np.ones_like(possible_output_tokens, dtype=float), (self.data_length, 1)
        )
        bad_idx = (
            self.input_idx[
                np.arange(self.data_length),
                np.expand_dims(self.possible_output_tokens - 1, -1),
            ]
            == 0
        ).T
        # only need to do this once, very time consuming
        if aggregate_nans:  # aggregate nans to the end of each row
            # partialsort_idx = np.argpartition(self.input_idx, np.sum(self.input_idx == 0, axis=1), axis=1)
            partialsort_idx = np.argsort(self.input_idx == 0, axis=1, kind="mergesort")
            self.input = np.take_along_axis(self.input, partialsort_idx, axis=1)
            self.input_idx = np.take_along_axis(self.input_idx, partialsort_idx, axis=1)
            self.first_n_shuffle = self.data_width - np.sum(self.input_idx == 0, axis=1)
        else:
            self.first_n_shuffle = None

        # prob_matrix[bad_idx] = (
        #     0.0  # don't sample those token which are missing (i.e. padding)
        # )
        self.output_prob_matrix = prob_matrix

        self.batch_size = batch_size
        self.steps_per_epoch = self.data_length // self.batch_size

        # placeholder to epoch level data
        self.epoch_input = None
        self.epoch_input_idx = None
        self.epoch_output = None
        self.epoch_output_idx = None
        self.batches_lengthes = None

        self.idx_list = np.arange(self.data_length)

        # we put every preparation in on_epoch_end()
        self.on_epoch_end(epoch=0)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for i in range(len(self)):
            _temp = self.epoch_input[i * self.batch_size : (i + 1) * self.batch_size]
            _temp_idx = self.epoch_input_idx[
                i * self.batch_size : (i + 1) * self.batch_size
            ]
            _temp[:, self.batches_lengthes[i] :] = 0.0
            _temp_idx[:, self.batches_lengthes[i] :] = 0
            yield (
                _temp,
                _temp_idx,
                self.epoch_output_idx[i * self.batch_size : (i + 1) * self.batch_size],
                self.epoch_output[i * self.batch_size : (i + 1) * self.batch_size],
                self.epoch_output_err[i * self.batch_size : (i + 1) * self.batch_size],
            )

    def __len__(self):
        return self.steps_per_epoch

    # def on_epoch_end(self, epoch=None):
    #     """
    #     Major functionality is to prepare the data for the next epoch

    #     Parameters
    #     ----------
    #     epoch : int, optional
    #         epoch number in case the behavior depends on epoch, by default None
    #     """
    #     # shuffle the row ordering list when an epoch ends to prepare for the next epoch
    #     if self.shuffle:
    #         rng.shuffle(self.idx_list)

    #     # choose one depending on output prob matrix
    #     output_idx = random_choice(
    #         np.tile(self.possible_output_tokens, (self.data_length, 1)),
    #         self.output_prob_matrix,
    #     )

    #     self.epoch_input = torch.tensor(self.input, dtype=self.factory_kwargs["dtype"], device=self.factory_kwargs["device"])
    #     self.epoch_input_idx = torch.tensor(self.input_idx, dtype=torch.int32, device=self.factory_kwargs["device"])
    #     self.epoch_input, self.epoch_input_idx = shuffle_row(
    #         [self.epoch_input, self.epoch_input_idx], first_n=self.first_n_shuffle
    #     )
    #     # crop
    #     self.epoch_input = self.epoch_input[:, : self.length_range[1]]
    #     self.epoch_input_idx = self.epoch_input_idx[:, : self.length_range[1]]

    #     self.epoch_input = torch.atleast_3d(self.epoch_input)[self.idx_list]
    #     self.epoch_input_idx = torch.tensor(self.epoch_input_idx)[self.idx_list]
    #     self.epoch_output_idx = torch.tensor(output_idx, dtype=torch.int32).to(
    #         self.factory_kwargs["device"], non_blocking=True
    #     )[self.idx_list]
    #     self.epoch_output = torch.tensor(
    #         np.take_along_axis(self.output, output_idx - 1, axis=1),
    #         dtype=self.factory_kwargs["dtype"],
    #     ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
    #     self.epoch_output_err = torch.tensor(
    #         np.take_along_axis(self.output_err, output_idx - 1, axis=1),
    #         dtype=self.factory_kwargs["dtype"],
    #     ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
    #     self.batches_lengthes = rng.integers(
    #         low=self.length_range[0], high=self.length_range[1], size=len(self)
    #     )

    def on_epoch_end(self, epoch=None):
        """
        Major functionality is to prepare the data for the next epoch

        Parameters
        ----------
        epoch : int, optional
            epoch number in case the behavior depends on epoch, by default None
        """
        # shuffle the row ordering list when an epoch ends to prepare for the next epoch
        if self.shuffle:
            rng.shuffle(self.idx_list)

        # choose one depending on output prob matrix
        output_idx = random_choice(
            np.tile(self.possible_output_tokens, (self.data_length, 1)),
            self.output_prob_matrix,
        )

        self.epoch_input = copy.deepcopy(self.input)
        self.epoch_input_idx = copy.deepcopy(self.input_idx)
        self.epoch_input, self.epoch_input_idx = shuffle_row(
            [self.epoch_input, self.epoch_input_idx], first_n=self.first_n_shuffle
        )
        # crop
        self.epoch_input = self.epoch_input[:, : self.length_range[1]]
        self.epoch_input_idx = self.epoch_input_idx[:, : self.length_range[1]]

        self.epoch_input = torch.atleast_3d(
            torch.tensor(self.epoch_input, dtype=self.factory_kwargs["dtype"])
        ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
        self.epoch_input_idx = torch.tensor(
            self.epoch_input_idx,
            dtype=torch.int32,
        ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
        self.epoch_output_idx = torch.tensor(output_idx, dtype=torch.int32).to(
            self.factory_kwargs["device"], non_blocking=True
        )[self.idx_list]
        self.epoch_output = torch.tensor(
            np.take_along_axis(self.output, output_idx - 1, axis=1),
            dtype=self.factory_kwargs["dtype"],
        ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
        self.epoch_output_err = torch.tensor(
            np.take_along_axis(self.output_err, output_idx - 1, axis=1),
            dtype=self.factory_kwargs["dtype"],
        ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
        self.batches_lengthes = rng.integers(
            low=self.length_range[0], high=self.length_range[1]+1, size=len(self)
        )


# class TrainingGenerator(torch.utils.data.IterableDataset):
#     def __init__(
#         self,
#         batch_size: int,
#         data: dict,
#         outputs_padding: int = 0,
#         possible_output_tokens: List[int] = None,
#         input_length: int = None,
#         shuffle: bool = True,
#         aggregate_nans: bool = True,
#         factory_kwargs: dict = {"device": "cpu", "dtype": torch.float32},
#     ):
#         """
#         Parameters
#         ----------
#         batch_size : int
#             batch size
#         data : dict
#             data dictionary that contains the following keys: ["input", "input_token", "output", "output_err"]
#         outputs_padding : int, optional
#             additional padding for output, by default 0
#         possible_output_tokens : np.ndarray, optional
#             possible output tokens, by default None
#         input_length : int, optional
#             input length, by default None
#         shuffle : bool, optional
#             shuffle data, by default True
#         aggregate_nans : bool, optional
#             aggregate nans of every rows to the end of those rows, by default True
#         """
#         self.factory_kwargs = factory_kwargs
#         self.input = copy.deepcopy(data["input"])
#         self.input_idx = copy.deepcopy(data["input_token"])
#         self.output = data["output"]
#         self.output_err = copy.deepcopy(data["output_err"])
#         self.outputs_padding = outputs_padding
#         self.data_length = len(self.input)
#         self.data_width = self.input.shape[1]
#         self.shuffle = shuffle  # shuffle row ordering, star level column ordering shuffle is mandatory
#         self.aggregate_nans = aggregate_nans

#         # handle possible output tokens for star-by-star basis
#         self.possible_output_tokens = possible_output_tokens
#         prob_matrix = np.tile(
#             np.ones_like(possible_output_tokens, dtype=float), (self.data_length, 1)
#         )
#         bad_idx = (
#             self.input_idx[
#                 np.arange(self.data_length),
#                 np.expand_dims(self.possible_output_tokens - 1, -1),
#             ]
#             == 0
#         ).T
#         # only need to do this once, very time consuming
#         if aggregate_nans:  # aggregate nans to the end of each row
#             # partialsort_idx = np.argpartition(self.input_idx, np.sum(self.input_idx == 0, axis=1), axis=1)
#             partialsort_idx = np.argsort(self.input_idx == 0, axis=1, kind="mergesort")
#             self.input = np.take_along_axis(self.input, partialsort_idx, axis=1)
#             self.input_idx = np.take_along_axis(self.input_idx, partialsort_idx, axis=1)
#             self.first_n_shuffle = self.data_width - np.sum(self.input_idx == 0, axis=1)
#         else:
#             self.first_n_shuffle = None

#         prob_matrix[bad_idx] = (
#             0.0  # don't sample those token which are missing (i.e. padding)
#         )
#         self.output_prob_matrix = prob_matrix

#         self.batch_size = batch_size
#         self.steps_per_epoch = self.data_length // self.batch_size

#         # placeholder to epoch level data
#         self.epoch_input = None
#         self.epoch_input_idx = None
#         self.epoch_output = None
#         self.epoch_output_idx = None

#         self.idx_list = np.arange(self.data_length)

#         self.input_length = input_length
#         if self.input_length is None:
#             self.input_length = data["input"].shape[1]

#         # we put every preparation in on_epoch_end()
#         self.on_epoch_end()

#     def __iter__(self):
#         """Create a generator that iterate over the Sequence."""
#         for i in range(len(self)):
#             _temp = self.epoch_input[i * self.batch_size : (i + 1) * self.batch_size]
#             _temp_idx = self.epoch_input_idx[
#                 i * self.batch_size : (i + 1) * self.batch_size
#             ]
#             _temp[:, self.batches_lengthes[i] :] = 0.0
#             _temp_idx[:, self.batches_lengthes[i] :] = 0
#             yield (
#                 _temp,
#                 _temp_idx,
#                 self.epoch_output_idx[i * self.batch_size : (i + 1) * self.batch_size],
#                 self.epoch_output[i * self.batch_size : (i + 1) * self.batch_size],
#                 self.epoch_output_err[i * self.batch_size : (i + 1) * self.batch_size],
#             )

#     def __len__(self):
#         return self.steps_per_epoch

#     def on_epoch_end(self, epoch=None):
#         """
#         Major functionality is to prepare the data for the next epoch

#         Parameters
#         ----------
#         epoch : int, optional
#             epoch number in case the behavior depends on epoch, by default None
#         """
#         # shuffle the row ordering list when an epoch ends to prepare for the next epoch
#         if self.shuffle:
#             rng.shuffle(self.idx_list)

#         shuffle_row(
#             [self.input, self.input_idx, self.output_prob_matrix], first_n=self.first_n_shuffle
#         )
#         # choose one depending on output prob matrix
#         output_idx = random_choice(np.tile(np.arange(self.data_width), (self.data_length, 1)), self.output_prob_matrix)

#         # crop
#         self.epoch_input = self.input[:, : self.length_range[1]]
#         self.epoch_input_idx = self.input_idx[:, : self.length_range[1]]

#         self.epoch_input = torch.atleast_3d(
#             torch.tensor(self.epoch_input, dtype=self.factory_kwargs["dtype"])
#         ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
#         self.epoch_input_idx = torch.tensor(
#             self.epoch_input_idx,
#             dtype=torch.int32,
#         ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
#         self.epoch_output_idx = torch.tensor(output_idx, dtype=torch.int32).to(
#             self.factory_kwargs["device"], non_blocking=True
#         )[self.idx_list]
#         self.epoch_output = torch.tensor(
#             np.take_along_axis(self.output, output_idx - 1, axis=1),
#             dtype=self.factory_kwargs["dtype"],
#         ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
#         self.epoch_output_err = torch.tensor(
#             np.take_along_axis(self.output_err, output_idx - 1, axis=1),
#             dtype=self.factory_kwargs["dtype"],
#         ).to(self.factory_kwargs["device"], non_blocking=True)[self.idx_list]
#         self.batches_lengthes = rng.integers(
#             low=self.length_range[0], high=self.length_range[1], size=len(self)
#         )
#         print(self.epoch_output)
