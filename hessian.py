'''Calculates the Hessian for a sublayer'''
import torch
from tqdm import trange

from spqr_engine import SPQRUtil
from main import get_inps, find_sublayers, get_layers

@torch.no_grad() # note: without this, memory will leak
def hessian(model, sublayer_name, dataloader, args, device=torch.device("cpu")):
    """Returns the Hessian of a given sublayer, computed from a loaded dataset.

    Args:
        model: model object obtained with get_model()
        sublayer_name (str): sublayer name obtained with find_sublayers()
        dataloader: dataloader var obtained with get_loaders()
        args: same args from argparse in main.py
        device (torch.device, optional): pytorch device to run model on. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: n-by-n matrix with the sublayer's Hessian at the input dataset
    """
    print("\nCalculating Hessian from inputs...")

    sublayer = find_sublayers(model)[sublayer_name]
    layer_number = int(sublayer_name.split(".")[2])
    layer = get_layers(model)[layer_number]

    spqr_handler_sublayer = SPQRUtil(sublayer)

    # fetch inputs from dataset
    inps, forward_args = get_inps(model, dataloader, args, device)
    inps = inps.to(device)
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # add forward hook to layer
    def add_batch(name):
        # computes the sublayer's Hessian from input data
        def tmp(_, inp, out):
            spqr_handler_sublayer.add_batch(inp[0].data)
        return tmp
    
    handle = sublayer.register_forward_hook(add_batch(args.sublayer))

    # evaluate the non-quantized model
    for j in trange(
        args.nsamples, desc="calc outs (and Hessian) before quantization", leave=False
    ):
        outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]

    handle.remove()

    return spqr_handler_sublayer.H
    