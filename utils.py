import dataclasses
import io
from typing import Optional, Sequence, Union

import openai
import tqdm
from openai import openai_object
import copy

import os
import sys
import shutil
import subprocess
import logging
import colorlog
import argparse
import copy
import pathlib
import shlex
import deepdish
from tqdm import tqdm
import time
import platform
import pickle
import yaml
import glob
import random
import msgpack
import importlib
import traceback
from PIL import Image
import functools
from functools import partial
import urllib.request
from warnings import simplefilter
from datetime import timedelta
from timeit import default_timer
from configobj import ConfigObj
import requests
import psutil
import hashlib
import imageio
import math
import h5py
import csv
import collections
import json
import json_lines
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from einops import rearrange, repeat
import torch.distributed as dist
from torchvision import datasets, transforms, utils
import torchvision
import cv2

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                choices = completion_batch.choices

                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def split_filename(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname

import pickle
import msgpack
import glob
import deepdish
import pandas as pd
import numpy as np
import csv
import h5py
import torch
import json_lines
from configobj import ConfigObj
import yaml
import importlib

def file2data(filename, type=None, printable=True, **kwargs):
    dirname, rootname, extname = split_filename(filename)
    print_load_flag = True
    if type:
        extname = type
    if extname == 'pkl':
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    elif extname == 'msg':
        with open(filename, 'rb') as f:
            data = msgpack.load(f, encoding="utf-8")
    elif extname == 'h5':
        split_num = kwargs.get('split_num')
        if split_num:
            print_load_flag = False
            if isinstance(split_num, int):
                filenames = ["%s_%i" % (filename, i) for i in range(split_num)]
                if split_num != len(glob.glob("%s*" % filename)):
                    print('Maybe you are giving a wrong split_num(%d) != seached num (%d)' % (
                        split_num, len(glob.glob("%s*" % filename))))

            elif split_num == 'auto':
                filenames = glob.glob("%s*" % filename)
                print('Auto located %d splits linked to %s' % (len(filenames), filename))
            else:
                raise ValueError("params['split_num'] got unexpected value: %s, which is not supported." % split_num)
            data = []
            for e in filenames:
                data.extend(deepdish.io.load(e))
            print('Loaded data from %s_(%s)' % (
                os.path.abspath(filename), ','.join(sorted([e.split('_')[-1] for e in filenames]))))
        else:
            data = deepdish.io.load(filename)
    elif extname == 'csv':
        data = pd.read_csv(filename)
    elif extname == 'tsv':  # Returns generator since tsv file is large.
        if not kwargs.get('delimiter'):  # Set default delimiter
            kwargs['delimiter'] = '\t'
        if not kwargs.get('fieldnames'):  # Check field names
            raise ValueError('You must specify fieldnames when load tsv data.')
        # Required args.
        key_str = kwargs.pop('key_str')
        decode_fn = kwargs.pop('decode_fn')
        # Optimal args.
        topk = kwargs.pop('topk', None)
        redis = kwargs.pop('redis', None)
        if not redis:
            data = dict()
        else:
            data = redis
        if not redis or not redis.check():
            with open(filename) as f:
                reader = csv.DictReader(f, **kwargs)
                for i, item in enumerate(tqdm(reader)):
                    if not redis:  # if memory way
                        decode_fn(item)
                    data[item[key_str]] = item
                    if topk is not None and i + 1 == topk:
                        break
        else:
            print('check_str %s in redis, skip loading.' % data.check_str)
    elif extname == 'hy':
        data = h5py.File(filename, 'r')
    elif extname in ['npy', 'npz']:
        try:
            data = np.load(filename, allow_pickle=True)
        except UnicodeError:
            print('%s is python2 format, auto use latin1 encoding.' % os.path.abspath(filename))
            data = np.load(filename, encoding='latin1', allow_pickle=True)
    elif extname == 'json':
        with open(filename) as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError as e:
                raise ValueError('[error] utils.file2data: failed to load json file %s' % filename)
    elif extname == 'jsonl':
        with open(filename, 'rb') as f:
            data = [e for e in json_lines.reader(f)]
    elif extname == 'ini':
        data = ConfigObj(filename, encoding='utf-8')
    elif extname in ['pth', 'ckpt']:
        data = torch.load(filename, map_location=kwargs.get('map_location'))
    elif extname == 'txt':
        top = kwargs.get('top', None)
        with open(filename, encoding='utf-8') as f:
            if top:
                data = [f.readline() for _ in range(top)]
            else:
                data = [e for e in f.read().split('\n') if e]
    elif extname == 'yaml':
        with open(filename, 'r') as f:
            data = yaml.load(f)
    else:
        raise ValueError('type can only support h5, npy, json, txt')
    if printable:
        if print_load_flag:
            print('Loaded data from %s' % os.path.abspath(filename))
    return data

def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

class BasicArgs:
    # default settings
    resume = True
    use_tqdm = True
    debug = False

    root_dir = '/f_data/G/'

    @staticmethod
    def parse_config_name(config_filename):
        """
        Example:
            Args:
                config_filename: 'config/t2i/t2i4ccF8S256.py'
            Return:
                task_name: 't2i'
                model_name: 't2i4ccF8S256'
        """
        name = os.path.normpath(config_filename).split(os.path.sep)[-3:]
        if len(name)==3:
            task_name = os.path.join(name[0], name[1])
        else:
            task_name = name[0]
        model_name = name[-1].split('.')[0]
        return task_name, model_name

def init_model_from_config(cfg_path='config/vae_kl/VAE_KL_F8D4.py'):
    cfg = import_filename(cfg_path)
    Model_net, Args_net = cfg.Net, cfg.args
    model = Model_net(Args_net)

    pretrained_model = getattr(Args_net, 'pretrained_path', None)
    if pretrained_model:
        model.load_state_dict(file2data(pretrained_model, map_location='cpu'), strict=False)
        # model.load_state_dict(file2data(pretrained_model, map_location='cpu')['models'], strict=False)

    return model

from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple, Any
import os
from shutil import copyfile

class LlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = {"vocab_file": "tokenizer.model"}
    pretrained_vocab_files_map = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        decode_with_prefix_space=False,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.decode_with_prefix_space = decode_with_prefix_space
        self.sp_model = SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        self._no_prefix_space_tokens = None

        """ Initialisation"""

    @property
    def no_prefix_space_tokens(self):
        if self._no_prefix_space_tokens is None:
            vocab = self.convert_ids_to_tokens(list(range(self.vocab_size)))
            self._no_prefix_space_tokens = {i for i, tok in enumerate(vocab) if not tok.startswith("▁")}
        return self._no_prefix_space_tokens

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    @property
    def pad_token_id(self) -> Optional[int]:
        return 0

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sp_model.eos_id()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Returns a tokenized string."""
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def _maybe_add_prefix_space(self, tokens, decoded):
        if tokens and tokens[0] not in self.no_prefix_space_tokens:
            return " " + decoded
        else:
            return decoded

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        out_string = self._maybe_add_prefix_space(tokens=tokens, decoded=out_string)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            print(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "tokenizer.model"
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is not None:
            output = output + token_ids_1

        if self.add_eos_token:
            output = output + [self.eos_token_id]

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]


import clip
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, reduce
from itertools import islice, cycle
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

class ParameterList(object):
    """a mock parameter list object until below issue is resolved
        https://github.com/pytorch/pytorch/issues/36035
    """

    def __init__(self, kls, prefix, length):
        self.ind = 0
        self.kls = kls
        self.prefix = prefix
        self.length = length

    def _keyname(self, prefix, ind):
        return f'{prefix}_{ind}'

    def append(self, x):
        setattr(self.kls, self._keyname(self.prefix, self.ind), x)
        self.ind += 1

    def to_list(self):
        return [getattr(self.kls, self._keyname(self.prefix, i)) for i in range(self.length)]

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, axial_shape, axial_dims=None):
        super().__init__()

        self.dim = dim
        axial_shape = tuple([shape for shape in axial_shape if shape != 1])  # remove the axis whose shape is 1
        self.shape = axial_shape
        self.max_seq_len = reduce(lambda x, y: x * y, axial_shape, 1)

        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(
            axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or not self.summed and sum(
            axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        self.weights = ParameterList(self, 'weights', len(axial_shape))

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x, idx=None, row_range=None, col_range=None):
        b, t, e = x.shape
        assert (t <= self.max_seq_len), f'Sequence length ({t}) must ' \
                                        f'be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights.to_list():
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, axial_dim)
            embs.append(emb)
        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)

        if idx is None and row_range is None:
            return pos_emb[:, :t].to(x)
        elif idx is not None:
            return pos_emb[:, idx:idx + 1].to(x)
        else:
            rows = torch.arange(row_range[0], row_range[1])
            cols = torch.arange(col_range[0], col_range[1])
            coords = torch.stack(torch.meshgrid([rows, cols]), dim=-1).reshape(-1, 2)
            coords[:, 0] = coords[:, 0] * self.shape[0]
            pos_idxs = coords.sum(dim=-1)
            return pos_emb[:, pos_idxs].to(x)


def adaptively_load_state_dict(target, state_dict):
    target_dict = target.state_dict()

    try:
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict and v.size() == target_dict[k].size()}
    except Exception as e:
        print('load error %s', e)
        common_dict = {k: v for k, v in state_dict.items() if k in target_dict}

    if 'param_groups' in common_dict and common_dict['param_groups'][0]['params'] != \
            target.state_dict()['param_groups'][0]['params']:
        print('Detected mismatch params, auto adapte state_dict to current')
        common_dict['param_groups'][0]['params'] = target.state_dict()['param_groups'][0]['params']
    target_dict.update(common_dict)
    target.load_state_dict(target_dict)

    missing_keys = [k for k in target_dict.keys() if k not in common_dict]
    unexpected_keys = [k for k in state_dict.keys() if k not in common_dict]

    if len(unexpected_keys) != 0:
        print(
            f"Some weights of state_dict were not used in target: {unexpected_keys}"
        )
    if len(missing_keys) != 0:
        print(
            f"Some weights of state_dict are missing used in target {missing_keys}"
        )
    if len(unexpected_keys) == 0 and len(missing_keys) == 0:
        print("Strictly Loaded state_dict.")

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True