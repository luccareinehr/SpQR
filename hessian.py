'''Calculates the Hessian for a sublayer'''
import torch
from tqdm import tqdm, trange

from spqr_engine import SPQRUtil
from main import get_inps, find_sublayers, get_layers

class StopModelInference(Exception):
    '''Custom class to stop model inference on a target layer.'''
    pass

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
        int: total number of samples used to compute the Hessian
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

    # add forward hooks to layer | TODO: change these to pre-hook?
    def add_batch(name):
        # computes the sublayer's Hessian from input data
        def tmp(_, inp):
            spqr_handler_sublayer.add_batch(inp[0].data)
        return tmp
    
    def stop_inference(_, inp):
        # raise flag to stop model inference at a certain layer
        # to prevent computations useless for calculating the Hessian
        raise StopModelInference()
    
    handles = (
        sublayer.register_forward_pre_hook(add_batch(args.sublayer)),
        sublayer.register_forward_pre_hook(stop_inference),
    )

    # evaluate the non-quantized model
    for j in trange(
        args.nsamples, desc="calc outs (and Hessian) before quantization", leave=False
    ):
        try:
            #outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]
            _ = layer(inps[j].unsqueeze(0), **forward_args)[0]
        except StopModelInference:
            tqdm.write(f"Stopping inference calculation at layer {sublayer_name} in iteration {j+1}/{args.nsamples}")

    for h in handles:
        h.remove()

    return spqr_handler_sublayer.H, spqr_handler_sublayer.nsamples
    