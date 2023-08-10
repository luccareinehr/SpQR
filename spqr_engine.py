from __future__ import annotations
import math
from typing import Optional, Union, NamedTuple

import torch
from tqdm.auto import tqdm
from weight_permutation import get_permutation_order
from quant_groups import Quantizer, quantize, only_quantize, only_dequantize

from datautils import IntN, tensor_to_IntN_list, IntN_list_to_bytestream

import pickle, struct
import numpy as np
from tqdm import trange

class SPQRUtil:
    """Learns GPTQ for a single linear layer"""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp): # Same code as in GPTQ (accumulated Hessian calculation)
        assert self.H is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        # Hessian is scaled down using a correction factor to adjust for the number of samples, 
        # to update more smoothly as new data batches are added:
        self.H *= self.nsamples / (self.nsamples + tmp) 
        # The nsambles variable is updated by adding the number of samples in the current batch ('tmp'):
        self.nsamples += tmp
        # The input data batch is normalized:
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # H = XX':
        self.H += inp.matmul(inp.t())

    def quantize(
        self,
        *,
        bits: int = 2,
        blocksize: int = 128,
        percdamp: float = 1e-2,
        groupsize: Optional[int] = None,
        keep_last_columns: int = 0,
        outlier_relative_threshold: float = float("inf"),
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
        simplified_outliers: bool = False,
        verbose=True,
        perchannel: bool = True,
        sym: bool = False,
        **kwargs,
    ) -> QuantizationResult:
        """
        :param bits: number of bits used at the lowest level (the full model size will be different!)
        :param blocksize: take blocks of this many input features at a time for GPTQ
        :note: blocksize affects runtime and memory, but does not affect the resulting matrix (up to machine precision)
        :param groupsize: fit quantization scaling / statistics to each group of this many input features
        :param percdamp: relative regularizer added to hessian diagonal before inversion
        :note: if groupsize_in_dim* is None, use the same quantization statistics across all input features
        :param keep_last_columns: if not None, keep the last (this many) input features un_quantized and return them
        :note: the un-quantized columns will be a part of the first returned result
        :param outlier_relative_threshold: threshold used for *UNSTRUCTURED* outliers, relative to
        :note: if keep_last_columns > 0, quantized_dequantized_weights[-keep_last_columns:] will be non-quantized
        :param permutation_order: re-order input features using a certain policy
        :param keep_H: if False, delete the accumulated hessian during quantize; if False, keep the accumulated hessian
        :param simplified_outliers: if True,do not perform leave-one-out evaluation when detecting outliers;
            works faster, but generally worse in perplexity
        :param verbose: if True, display a tqdm progressbar over input columns
        :param sym: if True, base weight quantization is symmetric
        :param perchannel: if True, base weight quantization will learn statistics for each output dimension separately
        :return: a QuantizationResult tuple that contains(
            weight, perm, _unused, _unused, _unused, _unused, quantization_errors, outlier_unstructured_mask
        ), see class QuantizationResult below for details
        """
        weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)
        perm = get_permutation_order(self.H, weight, permutation_order)
        weight = weight[:, perm]  # note: weight is modified
                
        H = self.H # Get accumulated hessian calculated during non-quantized inference
        if keep_H:
            H = H.clone()  # protect from in-place changes
        else:
            self.H = None

        H = H[perm][:, perm]
        self.dead = torch.diag(H) == 0  # indices of input features that do not affect outputs
        
        # Regularize matrix diagonal for inversion with percdamp value
        if percdamp > 0:
            ix = torch.arange(len(H), device=weight.device)
            H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
            del ix
        
        H[self.dead, self.dead] = 1
        weight[:, self.dead] = 0
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H)) # compute H^(-1)
        H_inv_cho = torch.linalg.cholesky(H_inv, upper=True) # Compute L, where (L)(L^T) = H^(-1)
        H_inv_cho_diag = torch.diag(H_inv_cho) # Extract the diagonal of L as a vector
        del H

        quantizer = Quantizer()
        quantizer.configure(bits, perchannel=perchannel, sym=sym, **kwargs)
        assert H_inv_cho.shape[0] == H_inv_cho.shape[1] == weight.shape[1], "weight must be [out_features, in_features]"
        out_dim, in_dim = weight.shape  # [out_features, in_features]

        if groupsize is None:
            groupsize = in_dim

        # prepare outlier detection
        outlier_column_indices = torch.empty(0, dtype=torch.int64, device=weight.device)
        del H_inv

        # scale = average value of (variance/diag^2) for all lines [i.e., compute the variance per-line and divide by the corresponding diagonal value]
        outlier_scale = (weight.var(dim=0) / torch.diag(H_inv_cho).square()).mean().item()
        unstructured_outlier_threshold = outlier_relative_threshold * outlier_scale
        
        in_group_index = -1  # index of current group (of input features, for group quantizer purposes)

        quantization_errors = torch.zeros_like(weight)
        unstructured_outlier_mask = torch.zeros_like(weight, dtype=torch.bool)

        Q = torch.zeros_like(weight) # weight matrix with quantized values
        quantizers = [] # list with quantizer stats for each group

        block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
        block_start_iter = tqdm(block_start_iter, leave=False) if verbose else block_start_iter
        for block_start in block_start_iter:
            block_end = min(block_start + blocksize, in_dim)
            for column_index in range(block_start, block_end):
                if column_index % groupsize == 0: # in the first column of the group
                    # fit weight quantizer on the upcoming group of weight columns (inputs), across all rows (outputs)
                    in_group_index += 1
                    group_weight = weight[:, column_index : column_index + groupsize]

                    if simplified_outliers or (unstructured_outlier_threshold == float("inf")):
                        quantizer.find_params(group_weight, weight=True)

                    else:
                        # objective: detect which weights will be designated as outliers, fit quantizer *without* these weights
                        # step 1: fit quantizer on a leave-one-out version of weights, i.e. in each group, drop one weight at a time
                        assert perchannel, "refitting quantizer is only implemented for perchannel=True"
                        group_diag_hessian_inv_cho = H_inv_cho_diag[column_index : column_index + groupsize]
                        loo_quantization_error_sq = get_leave_one_out_error( # calculates the error improvement by not quantizing each weight (output: matrix with same shape as 'group_weight')
                            group_weight, group_diag_hessian_inv_cho, bits=bits, sym=sym
                        )
                        # ^-- dequantized(quantized(group_weight)) using a quantizer trained on all weights except the reconstructed one

                        likely_unstructured_outlier_mask = (loo_quantization_error_sq > unstructured_outlier_threshold).float()

                        non_outlier_mask = 1 - likely_unstructured_outlier_mask
                        mean_over_non_outliers = torch.sum(group_weight * non_outlier_mask, dim=1, keepdim=True) / torch.sum(
                            non_outlier_mask, dim=1, keepdim=True
                        ).clamp_min(1)
                        group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (
                            1 - non_outlier_mask
                        ) # mean imputation to fill in where outliers are (in the group)
                        quantizer.find_params(group_weight_without_outliers, weight=True) # find scale and zero-point for the group
                        # quantizer.scale and quantizer.zero are calculated >per row< of the weight matrix (each row in the group of columns has a scaling factor)
                        del group_diag_hessian_inv_cho, loo_quantization_error_sq
                        del mean_over_non_outliers, group_weight_without_outliers, non_outlier_mask

                    del group_weight

                    # save quantization group statistics
                    
                    # first, re-quantize scale and zero since the default routine returns their de-quantized form
                    scale_quantized = only_quantize(
                        quantizer.scale,
                        quantizer.qq_scale.scale.repeat_interleave(quantizer.qq_groupsize, dim=0),
                        quantizer.qq_scale.zero.repeat_interleave(quantizer.qq_groupsize, dim=0),
                        quantizer.qq_scale.maxq,
                    )
                    zero_quantized = only_quantize(
                        quantizer.zero,
                        quantizer.qq_zero.scale.repeat_interleave(quantizer.qq_groupsize, dim=0),
                        quantizer.qq_zero.zero.repeat_interleave(quantizer.qq_groupsize, dim=0),
                        quantizer.qq_zero.maxq,
                    )

                    # to access, use: quantizers[group_index]['q']['scale'] etc.
                    # 'q' is for 1st-order statistics, 'qq' for 2nd-order statistics
                    quantizers.append(
                        {
                            'q': {'scale': scale_quantized, 'zero': zero_quantized},
                            'qq_scale': {'scale': quantizer.qq_scale.scale, 'zero': quantizer.qq_scale.zero},
                            'qq_zero': {'scale': quantizer.qq_zero.scale, 'zero': quantizer.qq_zero.zero},
                        }
                    )
                    # do the stat tensors change when a new iteration happens? or do they stay the same, as if they were cloned?
                    # it looks like they don't change! at least not when I tested it

                Q[:, column_index] = only_quantize(
                    weight[:, column_index].unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                ).reshape_as(weight[:, column_index]) # quantize column using group statistics (outliers get quantized too)

                weight_i_quantized = only_dequantize(
                    Q[:, column_index].unsqueeze(1), quantizer.scale, quantizer.zero
                ).reshape_as(weight[:, column_index])

                # compute sensitivity for each weight in the column
                delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                quantization_errors[:, column_index] = delta_weight_i / H_inv_cho[column_index, column_index]  # [out_dim]

                if unstructured_outlier_threshold != float("inf"):
                    unstructured_outlier_mask[:, column_index] = (
                        quantization_errors[:, column_index].square() > unstructured_outlier_threshold
                    )
                    # re-quantize without outliers
                    is_outlier = unstructured_outlier_mask[:, column_index].float()
                    weight_i_quantized_wo_outliers = quantize(
                        (weight[:, column_index] * (1 - is_outlier)).unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                    ).reshape_as(weight[:, column_index])
                    # include raw outliers
                    weight_i_quantized = (
                        weight_i_quantized_wo_outliers * (1 - is_outlier) + weight[:, column_index] * is_outlier
                    )  # [out_dim]
                    del weight_i_quantized_wo_outliers

                    delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                    quantization_errors[:, column_index] = delta_weight_i / H_inv_cho[column_index, column_index]  # [out_dim]

                weight[:, column_index] = weight_i_quantized # update column with quantized weight
                weight[:, column_index + 1 : block_end].addr_(
                    quantization_errors[:, column_index],
                    H_inv_cho[column_index, column_index + 1 : block_end],
                    alpha=-1,
                ) # on all subsequent columns, subtract [error]*H_inv_cho (error correction)

            weight[:, block_end:].addmm_(
                quantization_errors[:, block_start:block_end],
                H_inv_cho[block_start:block_end, block_end:],
                alpha=-1,
            ) # on all subsequent blocks, subtract [error]*H_inv_cho (error correction)

        if permutation_order != "identity":
            invperm = torch.argsort(perm)
            weight = weight[:, invperm]

            print(f"CAUTION: permutation was on during quantization. Quantized results are all permuted, and must be inversely permuted after dequantization. This affects all members of {self}, except {self}.weight.")

        # save outliers in CSR format
        outliers_sparse = weight * unstructured_outlier_mask # sparse matrix with only outliers
        outliers_csr = outliers_sparse.to_sparse_csr()

        return QuantizationResult(
            weight=weight,
            perm=perm,
            quantization_errors=quantization_errors,
            unstructured_outlier_threshold=unstructured_outlier_threshold,
            unstructured_outlier_mask=unstructured_outlier_mask,
            weight_quantized=Q,
            quantizers=quantizers,
            outliers_csr=outliers_csr,
        )
        # NOTE: the quantized data (weights, stats and outliers) are all returned permuted.
        # Only the dequantized 'weight' tensor is inversely permuted.
        # This means that, in inference, after the data has been dequantized, inverse permutation must be applied
        # to get back the original matrix correctly.
        #
        # We can't do otherwise because quantizers are saved per group/per block, and permutation is done
        # per column, so with more granularity.

    def dequantize(self, quantized_result: QuantizationResult, dtype: torch.dtype=torch.float32) -> torch.Tensor:
        raise NotImplementedError

class QuantizationResult():
    """A collection of codebooks, indices and assorted statistics produced by SPQRUtil; not memory-optimized!"""
    def __init__(self,
        weight: torch.FloatTensor,  # dequantized(quantized(weight)), same shape as the original
        perm: Optional[torch.LongTensor],  # optional input permutation indices that were used during quantization
        # NOTE: if permutation_order != identity, all subsequent tensors (incl. outlier indices) are permuted in that order!

        quantization_errors: torch.Tensor,  # per-element quantization errors, defined as (weight - quantized_weight) / diag(inverse_hessian_cholesky)
        unstructured_outlier_threshold: float,  # threshold on squared error increase used for determining *UNSTRUCTURED* outliers
        unstructured_outlier_mask: torch.Tensor,  # bool mask where True means that this is an individual outlier

        weight_quantized: torch.Tensor, # quantized(weight)
        quantizers: list, # list of quantization statistics dictionaries per group
        outliers_csr: torch.Tensor, # CSR-formatted tensor object of weight outliers
    ):
        self.weight = weight
        self.perm = perm
        self.quantization_errors = quantization_errors
        self.unstructured_outlier_threshold = unstructured_outlier_threshold
        self.unstructured_outlier_mask = unstructured_outlier_mask

        self.weight_quantized = weight_quantized
        self.quantizers = quantizers
        self.outliers_csr = outliers_csr
            
    def save(self, filename, args, format='pytorch'):
        """saves single spqr-quantized matrix"""
        if format.lower() == 'pytorch':
            # saves without converting data types
            with open(filename, 'wb') as wf:
                pickle.dump(self, wf)

        elif format.lower() == 'ggml':
            # saves in correct data types (specified in args)
            if args.permutation_order != "identity":
                raise TypeError('ggml format does not currently support permutation')
            
            GGML_TYPE_SPQR = 99 # NOTE: not final

            wbits = args.wbits
            qq_scale_bits = args.qq_scale_bits
            qq_zero_bits = args.qq_zero_bits

            groupsize = args.groupsize
            qq_groupsize = args.qq_groupsize

            dims = self.weight_quantized.shape
            ftype = GGML_TYPE_SPQR
            matrix_name = args.sublayer.encode('utf-8')

            with open(filename, "wb") as wf:
                # header
                wf.write(struct.pack("iii", len(dims), len(matrix_name), ftype)) # n_dims, len(matrix_name), ftype
                for dim in dims: wf.write(struct.pack("i", dim)) # dim1, ..., dimN 
                wf.write(matrix_name) # matrix_name

                # data
                for group in trange(
                    dims[1]//groupsize, desc="saving quantized matrix and stats", leave=False
                    ):
                    for block in range(dims[0]//qq_groupsize):
                        blocklines = []
                        for line in range(block*qq_groupsize, block*qq_groupsize + qq_groupsize): # line indices in the block
                            blocklines.append(
                                SPQRBlockLine(
                                    scale=IntN(self.quantizers[group]['q']['scale'][line].item(), qq_scale_bits),
                                    zero=IntN(self.quantizers[group]['q']['zero'][line].item(), qq_zero_bits),
                                    weights=tensor_to_IntN_list(self.weight_quantized[line, group*groupsize:(group*groupsize + groupsize)], wbits)
                                )
                            )
                        this_block =  SPQRBlock(
                                qq_scale4scales=self.quantizers[group]['qq_scale']['scale'][block].numpy(), # must be in numpy to save in fp16
                                qq_zero4scales=self.quantizers[group]['qq_scale']['zero'][block].numpy(),
                                qq_scale4zeros=self.quantizers[group]['qq_zero']['scale'][block].numpy(),
                                qq_zero4zeros=self.quantizers[group]['qq_zero']['zero'][block].numpy(),
                                blocklines=blocklines
                            )
                        # save data to file
                        this_block.qq_scale4scales.tofile(wf)
                        this_block.qq_zero4scales.tofile(wf)
                        this_block.qq_scale4zeros.tofile(wf)
                        this_block.qq_zero4zeros.tofile(wf)
                        concatenated_blocklines = []
                        for blockline in this_block.blocklines:
                            concatenated_blocklines += [blockline.scale, blockline.zero] + blockline.weights
                        wf.write(IntN_list_to_bytestream(concatenated_blocklines)) # NOTE: this gets padded with zeros

                # outliers
                crow_indices = self.outliers_csr.crow_indices().numpy().astype(np.uint32) # uint32
                col_indices = self.outliers_csr.col_indices().numpy().astype(np.uint16) # uint16
                outlier_values = self.outliers_csr.values().numpy().astype(np.float16) # fp16

                print(f"saving sparse outliers, starting at pos {hex(wf.tell())} in file")
                for row in trange(
                    dims[0], desc=f"saving sparse outliers", leave=False
                    ):
                    total_no_previous_outliers = crow_indices[row:row+1]
                    total_no_previous_outliers.tofile(wf)

                    outliers_in_row = outlier_values[crow_indices[row]:crow_indices[row+1]]
                    col_indices_in_row = col_indices[crow_indices[row]:crow_indices[row+1]]
                    for i in range(len(outliers_in_row)):
                        outlier = outliers_in_row[i]
                        col_idx = col_indices_in_row[i]
                        outlier.tofile(wf)
                        col_idx.tofile(wf)
                crow_indices[-1].tofile(wf) # last crow value
        
        else:
            raise ValueError('Supported formats: pytorch, ggml')

class SPQRBlock:
    def __init__(self, qq_scale4scales, qq_zero4scales, qq_scale4zeros, qq_zero4zeros, blocklines: list):
        self.qq_scale4scales = qq_scale4scales # numpy fp16
        self.qq_zero4scales = qq_zero4scales # numpy fp16
        self.qq_scale4zeros = qq_scale4zeros # numpy fp16
        self.qq_zero4zeros = qq_zero4zeros # numpy fp16
        self.blocklines = blocklines # default: list of SPQRBlockLine objects

class SPQRBlockLine(NamedTuple):
    scale: IntN
    zero: IntN
    weights: list


def get_leave_one_out_error(group_weight: torch.Tensor, group_diag_hessian_inv_cho: torch.Tensor, *, bits, sym):
    """EXPERIMENTAL! BEWARE - for each weight, fit quantizer without this_one_weight and return this one weight's reconstruction"""

    assert group_weight.ndim == 2
    loo_indices = torch.arange(group_weight.shape[1], device=group_weight.device)
    loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype)
    groupwise_loo_data = group_weight[:, loo_indices]  # [num_groups, num_loo = groupsize, groupsize - 1]
    fast_quantizer = Quantizer(shape=groupwise_loo_data.flatten(0, 1).shape)
    fast_quantizer.configure(bits, perchannel=True, sym=sym)
    fast_quantizer.find_params(groupwise_loo_data.flatten(0, 1), weight=True)

    # compute error improvement from not quantizing each one weight
    # to do so, we shall first train quantizer on leave-one-out data (which can be done faster since not all data affects quantization)
    loo_groupwise_reconstructed_weights = fast_quantizer.quantize(groupwise_loo_data.flatten(0, 1)).reshape_as(groupwise_loo_data)
    loo_group_diag_hessian_inv_cho = group_diag_hessian_inv_cho[loo_indices]  # [num_loo = groupsize, groupsize - 1]
    assert group_diag_hessian_inv_cho.ndim == 1

    # total quantization error consists of hessian-weighted mse on all remaining weights except for the one that's left out
    # -- this is because the left-out weights will not be quantized, and therefore, has zero quantization error
    loo_errors_sq = ((loo_groupwise_reconstructed_weights - groupwise_loo_data) / loo_group_diag_hessian_inv_cho).square().sum(-1)
    assert loo_errors_sq.shape == group_weight.shape  # [num_groups, num_loo = groupsize]

    # as a baseline error, quantize data normally without outliers
    base_quantizer = Quantizer(shape=group_weight.shape)
    base_quantizer.configure(bits, perchannel=True, sym=sym)
    base_quantizer.find_params(group_weight, weight=True)
    baseline_reconstructed_weights = base_quantizer.quantize(group_weight)
    baseline_errors_sq = (
        ((baseline_reconstructed_weights - group_weight) / group_diag_hessian_inv_cho).square().sum(dim=1, keepdim=True)
    )

    # outlier's usefulness = how much does mse decrease from treating this weight as an outlier
    reduction_in_squared_error = baseline_errors_sq - loo_errors_sq
    return reduction_in_squared_error
