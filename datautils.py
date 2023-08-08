import numpy as np
import torch
import random
import struct
import warnings
from transformers import AutoTokenizer, LlamaTokenizer
from datasets import load_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seqlen, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seqlen, tokenizer):
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")

    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seqlen, tokenizer):
    traindata = load_dataset(
        "allenai/c4", "allenai--c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )
    valdata = load_dataset(
        "allenai/c4", "allenai--c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation"
    )

    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_ptb_new(nsamples, seqlen, tokenizer):
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seqlen, tokenizer):
    traindata = load_dataset(
        "allenai/c4", "allenai--c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )
    valdata = load_dataset(
        "allenai/c4", "allenai--c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation"
    )

    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, custom_data_path=None, nsamples=128, seed=0, seqlen=2048, model_path=""):
    """
    Loads and prepares data for a Transformers model.
    Args:
        name (str): The name of the dataset to load. This can be one of 'wikitext2', 'c4', 'ptb' or 'custom'
        custom_data_path (str, optional): The path to a custom dataset file. Assumes name=='custom'
        nsamples (int, optional): The number of samples to load from the dataset. Defaults to 128.
        seed (int, optional): The random seed value for data shuffling and splitting. Defaults to 0.
        seqlen (int, optional): The maximum sequence length for input tokenization. Defaults to 2048.
        model_path (str, optional): The path to the pretrained model weights or full model name.
            used to detect llama to call proper tokenizer.
            see https://github.com/huggingface/transformers/issues/22222#issuecomment-1488578722 for reasons.
    Returns:
        train_loader (torch.utils.data.DataLoader or iterable): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader or iterable): DataLoader for the test dataset.
    """

    set_seed(seed)

    if custom_data_path:
        dataloader = torch.load(custom_data_path)[: nsamples]
        return dataloader, None

    assert name != "custom"

    if "llama" in model_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)

        # fix for transformer 4.28.0.dev0 compatibility
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
                print(f"bos/eos tokens updated: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
            except AttributeError:
                pass
                print(f"bos/eos tokens unchanged: {tokenizer.bos_token_id=},  {tokenizer.eos_token_id=}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if "wikitext2" in name:
        return get_wikitext2(nsamples, seqlen, tokenizer)
    elif "ptb" in name:
        if "new" in name:
            return get_ptb_new(nsamples, seqlen, tokenizer)
        return get_ptb(nsamples, seqlen, tokenizer)
    elif "c4" in name:
        if "new" in name:
            return get_c4_new(nsamples, seqlen, tokenizer)
        return get_c4(nsamples, seqlen, tokenizer)
    else:
        raise ValueError(
            f"Unable to load {name} - only wikitext2, ptb, c4 are supported."
        )

def sizeof_tensor(T: torch.Tensor) -> int:
    return T.element_size() * T.nelement()

one_byte = struct.calcsize("B") * 8

class ValueBitlength:
    """value with a specified length of max 8 bits"""
    def __init__(self, val, len):
        self.length = len
        if self.length > one_byte:
            raise ValueError(f"Max length supported for {self} is 1 byte.")

        self.value = val & (0xff >> (8-self.length))
        if self.value != val:
            warnings.warn(f"Value in {self} does not fit in the specified length. Overflowing bits were removed.",
                          stacklevel=2)

def values_to_bytestream(values: list) -> bytes:
    """converts a list of FixedLengthValue values to a bytestream"""
    # sum lengths of all values
    sum = 0
    for v in values:
        sum += v.length

    if (sum % one_byte) != 0:
        padding_bits = one_byte - (sum % one_byte)
        warnings.warn(f"""Input list is not multiple of a byte. Manually including {padding_bits} bits of zero-padding at the end to fix.""",
                      stacklevel=2)
        values.append(ValueBitlength(0, padding_bits))

    # create bytestream with all values in their correct bitwidth
    value_to_pack = 0x00
    offset = 0
    bytestream = b''
    for v in values:
        value_to_pack |= (v.value << offset) & 0xff
        offset += v.length

        if offset >= one_byte:
            bytestream += struct.pack("B", value_to_pack)

            included_bits = one_byte - (offset - v.length)
            value_to_pack = v.value >> included_bits
            offset = v.length - included_bits
    
    return bytestream