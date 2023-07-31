'''Module used to quantize a specific sublayer of a model using SpQR
To run, use: python sublayer.py (...arguments...)'''

import torch
from tqdm import trange

from modelutils import get_model
from datautils import get_loaders
from spqr_engine import SPQRUtil
from main import get_inps, find_sublayers, get_layers

def quantize_spqr_sublayer(model, dataloader, args, device):
    print("\nStarting SPQR quantization ...")

    inps, forward_args = get_inps(model, dataloader, args, device)
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    quantizers = {}

    normal_outlier_count_global, w_count_global = 0, 0

    sublayer_name = args.sublayer
    layer_number = int(sublayer_name.split(".")[2])
    sublayer = find_sublayers(model)[sublayer_name]
    layer = get_layers(model)[layer_number]

    print(f"\n---------------- Layer {layer_number}, sublayer {sublayer_name} ----------------")
    normal_outlier_count, w_count = 0, 0
    #start_time = time.time()

    # sublayer_dev_original = next(sublayer.parameters()).device  # quantized layer will return there
    # print(f"{sublayer_dev_original=}")
    # if sublayer_dev_original.type != "cuda":
    #     sublayer = sublayer.to(device)
    # else:
    #     sublayer = layers[i]
    sublayer_dev = next(sublayer.parameters()).device
    # all_sublayers = find_sublayers(layer)

    inps = inps.to(sublayer_dev)

    spqr_handler = SPQRUtil(sublayer)

    def add_batch(name):
        # computes the sublayer's Hessian from input data
        def tmp(_, inp, out):
            spqr_handler.add_batch(inp[0].data)
        return tmp

    # add forward hook to auto-calculate a sublayer's Hessian (used in quantization) during the non-quantized model evaluation
    handle = sublayer.register_forward_hook(add_batch(sublayer_name)) # when add_batch is called as a hook, the arguments inp and out are passed as a tuple (inp, out) by PyTorch
    
    # evaluate the non-quantized model
    for j in trange(
        args.nsamples, desc="calc outs (and Hessian) before quantization", leave=False
    ):
        outs[j] = layer(inps[j].unsqueeze(0), **forward_args)[0]
    
    # remove forward hook, to not recalculate Hessians during the quantized model evaluation
    handle.remove()

    if args.offload_activations:
        inps = inps.cpu()
        outs = outs.cpu()
        torch.cuda.empty_cache()

    print(f"Quantizing module {sublayer_name}")
    quantized = spqr_handler.quantize(
        percdamp=args.percdamp,
        bits=args.wbits,
        groupsize=args.groupsize,
        sym=args.sym,
        perchannel=args.perchannel,
        qq_groupsize=args.qq_groupsize,
        round_zero=args.round_zero,
        qq_scale_bits=args.qq_scale_bits,
        qq_zero_bits=args.qq_zero_bits,
        qq_zero_sym=args.qq_zero_sym,
        outlier_relative_threshold=args.outlier_threshold,
        permutation_order=args.permutation_order,
        simplified_outliers=args.simplified_outliers,
    )

    # spqr_handler.layer.weight.data = quantized.weight.to(
    #     spqr_handler.layer.weight.data.dtype
    # )

    return quantized

    # quantizers["model.layers.%d.%s" % (i, sublayer_name)] = ()  # to be updated

                # OUTLIER STATS per module:
        #         normal_outliers_count = quantized.unstructured_outlier_mask.to(torch.int32).sum()
        #         stats_payload[f"n_{sublayer_name}_ol_share"] = \
        #             (normal_outliers_count / quantized.weight.numel()).item()
        #         normal_outlier_count += normal_outliers_count.item()
        #         w_count += quantized.weight.numel()

        # # upload inputs back to the device
        # inps = inps.to(layer_dev)
        # outs = outs.to(layer_dev)

    #     out_losses = []
    #     for j in trange(args.nsamples, desc="calc outs after quantization", leave=False):
    #         outs_batch = layer(inps[j].unsqueeze(0), **forward_args)[0]
    #         if not args.skip_out_loss:
    #             outs_batch_loss = (outs_batch - outs[j]).float().square().view(outs_batch.shape[0], -1)\
    #                 .mean(dim=1).sqrt()
    #             outs_batch_loss /= outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
    #             out_losses.append(outs_batch_loss.item())
    #         outs[j] = outs_batch
    #     del outs_batch

    #     layers[i] = layer.to(layer_dev_original)
    #     del layer
    #     del spqr_handlers
    #     torch.cuda.empty_cache()

    #     inps, outs = outs, inps

    #     # Logging
    #     #stats_payload["layer_time"] = time.time() - start_time
    #     stats_payload["ol_share"] = normal_outlier_count / max(w_count, 1)
    #     stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
    #     stats_payload["Step"] = i

    #     normal_outlier_count_global += normal_outlier_count
    #     w_count_global += w_count

    #     print(stats_payload)

    # print("=====================\nFinal stats:")
    # print(f"global_ol_share:  {normal_outlier_count_global / w_count_global:.3%}")

    # wbits_avg = get_average_number_of_bits(
    #     wbits=args.wbits,
    #     qq_scale_bits=args.qq_scale_bits,
    #     qq_zero_bits=args.qq_zero_bits,
    #     qqq_scale_bits=16,
    #     qqq_zero_bits=16,
    #     groupsize=args.groupsize,
    #     qq_groupsize=args.qq_groupsize,
    #     round_zero=args.round_zero,
    #     global_ol_n_share=normal_outlier_count_global / w_count_global,
    # )

    # if args.wandb:
    #     wandb.log({"outlier_share": normal_outlier_count_global / w_count_global})
    #     wandb.log({"wbits_avg": wbits_avg})

    # model.config.use_cache = use_cache
    # return quantizers, wbits_avg

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["custom", "wikitext2", "ptb", "c4"],
        default="none",
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default=None,
        help="Path to load if specified.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--nearest", action="store_true", help="Whether to run the RTN baseline."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--permutation_order",
        type=str,
        default="identity",
        help="Weights permutation order; options: identity(default), spearman, act_order",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument("--sym", action="store_true", help="Symmetric quantization")
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="fit a unique quantizer to each output dim",
    )
    parser.add_argument(
        "--qq_scale_bits",
        type=int,
        default=None,
        help="Quantize quantization scale with this many bits (default=do not quantize)",
    )
    parser.add_argument(
        "--round_zero",
        type=int,
        default=None,
        help='whether to allow non-integer "zero" when quantizing weights non-symmetrically',
    )
    parser.add_argument(
        "--qq_zero_bits",
        type=int,
        default=None,
        help='Quantize quantization "zero" with this many bits (default=do not quantize)',
    )
    parser.add_argument(
        "--qq_zero_sym",
        action="store_true",
        help="enable sym=True in meta-quantization for groupwise zero, specifically",
    )
    parser.add_argument(
        "--qq_groupsize",
        type=int,
        default=16,
        help="Quantize quantization scale in groups of this many scales",
    )

    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=float("inf"),
        help="relative threshold for     outliers; higher threshold = more outliers.",
    )
    parser.add_argument(
        "--simplified_outliers",
        action="store_true",
        help="do not perform leave-one-out evaluation when detecting outliers; works faster, but generally worse in perplexity",
    )

    parser.add_argument(
        "--save_pt",
        type=str,
        default="",
        help="Save quantized checkpoint under this name.",
    )
    parser.add_argument(
        "--save_safetensors",
        type=str,
        default="",
        help="Save quantized `.safetensors` checkpoint under this name.",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Whether to use wandb or store locally."
    )
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="dtype to load the model.",
    )

    parser.add_argument(
        "--sublayer",
        type=str,
        required=True,
        help="Sublayer to quantize. You can find all sublayers by running modelutils.find_sublayer(model). Example (OpenLLaMa): model.layers.0.self_attn.q_proj"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("============  Loading model... ============")
    model = get_model(args.model_path, args.dtype).train(False)

    print("\n============ Quantizing model... ============")
    print("Loading data ...")
    dataloader, _ = get_loaders(
        args.dataset,
        custom_data_path=args.custom_data_path,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=model.seqlen,
    )
    # quantization_result = quantize_spqr_sublayer(model, dataloader, args, device)
    from hessian import hessian
    print(
        hessian(model, args.sublayer, dataloader, args)
    )

