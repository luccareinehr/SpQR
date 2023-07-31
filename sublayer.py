'''Module used to quantize a specific sublayer of a model using SpQR
To run, use: python sublayer.py (...arguments...)'''

import torch
from tqdm import trange

from modelutils import get_model
from datautils import get_loaders
from spqr_engine import SPQRUtil
from main import get_inps, find_sublayers, get_layers

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
    print("Loading data...")
    dataloader, _ = get_loaders(
        args.dataset,
        custom_data_path=args.custom_data_path,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=model.seqlen,
    )
    print("Data loaded!")
    
    sublayer = find_sublayers(model)[args.sublayer]
    spqr_handler = SPQRUtil(sublayer)

    # calculate hessian
    from hessian import hessian
    spqr_handler.H, spqr_handler.nsamples = hessian(model, args.sublayer, dataloader, args)

    # quantize given sublayer
    not_quantized = sublayer.weight.data.clone()

    print(f"Quantizing module {args.sublayer}")
    quantization_result = spqr_handler.quantize(
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

    quantized = quantization_result.weight.to(
        spqr_handler.layer.weight.data.dtype
    ).data # FIXME: quantization_result is giving the de-quantized version of the quantized layer (thus, fp32)

    # print results
    torch.set_printoptions(precision=10)
    print(f"Pre-quantization: {not_quantized}")
    print(f"Post-quantization: {quantized}")

    print("Save results? [y/N]")
    if input().lower() in ("yes", "y"):
        torch.save(not_quantized, f"{args.sublayer}_not_quantized.pt")
        torch.save(quantized, f"{args.sublayer}_not_quantized.pt")
        # TODO: this is not saving the outliers in CSR format,
        # neither the 1st- and 2nd-order group statistics
        # TODO: this is not saving in memory using the correct structure
        # (i.e., what information comes first in the memory file?)