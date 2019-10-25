import set_root

import time
import sys
import argparse

import torch

import src.train.atomic_train as train
import src.models.models as models
import src.data.data as data
import utils.utils as utils
import src.train.utils as train_utils
import src.data.config as cfg

from src.data.utils import TextEncoder
from src.train.opt import OpenAIAdam

import src.models.utils as model_utils
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import random

import ac.utils.config as ac_conf
from ac.data.loader import GenerationDataLoader
import ac.data.encode as encode
from ac.utils.io_utils import abs_path

parser = argparse.ArgumentParser()
parser.add_argument("--generation_set_size", type=str, default='full', choices=["full", "human"])
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--split", type=str, default="dev")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--experiment_num", type=str, default="0")
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--gen_len", type=int, default=100)

args = parser.parse_args()

if args.model_name == None:
    print("Please enter model name.")
    exit()

split = args.split

# configure evaluation run
config = ac_conf.load_default()
config.train.dynamic.bs = 32
#config.gpu_index = int(args.gpu_num)
meta = ac_conf.get_meta(config)

eval_opt = cfg.get_eval_parameters(config)

checkpoint = data.load_checkpoint(abs_path(args.model_name), gpu=False)
opt = checkpoint["opt"]
opt.eval.update(eval_opt)

# Set the random seeds
torch.manual_seed(opt.train.static.seed)
random.seed(opt.train.static.seed)
#if config.gpu_mode:
#    torch.cuda.manual_seed_all(opt.train.static.seed)

opt.train.dynamic.epoch = 0

print("Loading Data")

# load data
relations = encode.get_relations()
data_loader = GenerationDataLoader(relations)
data_loader.load_data()
data_loader.opt = config
data_loader.batch_size = config.train.dynamic.bs
data_loader.reset_offsets("train")
data.set_max_sizes(data_loader)
text_encoder = encode.build_text_encoder()
data_loader.vocab_encoder = text_encoder.encoder
data_loader.vocab_decoder = text_encoder.decoder
special_tokens = encode.get_special_tokens()


context_size_event = data_loader.max_event
context_size_effect = data_loader.max_effect

n_special = len(special_tokens)
n_ctx = context_size_event + context_size_effect
n_vocab = len(text_encoder.encoder) + n_ctx

print(data_loader.__dict__.keys())
opt.net.vSize = n_vocab

print("Building Model")

print(opt.exp)

model = models.make_model(
    opt, n_vocab, n_ctx, 0, load=False, return_acts=False, return_probs=True)

model.load_state_dict(checkpoint["state_dict"])

# if config.gpu_mode:
#     print("Pushing to GPU: {}".format(config.gpu_index))
#     cfg.device = config.gpu_index
#     cfg.do_gpu = True
#     torch.cuda.set_device(cfg.device)
#     model.cuda(cfg.device)
#     print("Done.")

model.eval()

device = cfg.device
#model.to(device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
#torch.cuda.manual_seed_all(args.seed)

lm_model = model

def make_batch(X):
    X = np.array(X)
    assert X.ndim in [1, 2]
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    pos_enc = np.arange(n_vocab + n_special, n_vocab + n_special + X.shape[-1])
    pos_enc = np.expand_dims(pos_enc, axis=0)
    batch = np.stack([X, pos_enc], axis=-1)
    batch = torch.tensor(batch, dtype=torch.long) #.to(device)
    return batch


def append_batch(X, next_idx, mask):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
    next_mask = torch.cat([mask, torch.ones(X.size(0), 1, device=mask.device)], 1)
    return torch.cat((X, next_x), 1), next_mask

data_loader.reset_offsets(splits=split, shuffle=False)

if args.generation_set_size == "full":
    b = [tuple(j) for j in data_loader.sequences[split]['total'][:, :data_loader.max_event + 1].tolist()]
    total = []
    set_total = set()
    for i, sequence in enumerate(b):
        if sequence not in set_total:
            total.append(i)
            set_total.add(sequence)
elif args.generation_set_size == "human":
    human_events = open("data/atomic/{}-human-eval-events.txt".format(split), "r").read().split("\n")
    found = []
    total = []
    for i, j in enumerate(data_loader.data[split]["total"]):
        if j[0] in human_events and (j[0], j[1]) not in found:
            found.append((j[0], j[1]))
            total.append(i)
else:
    total = list(range(int(args.generation_set_size)))

args.decoding_strategy = "greedy"

final_sequences = []

end_token = text_encoder.encoder[data.end_token]

eval_file_name = args.model_name.replace("gs_1000", "gs_{}".format(
    args.generation_set_size))
eval_file_name = eval_file_name[:-7] + "/{}.gens".format(split)
eval_file_name = eval_file_name.replace("models/", "results/gens/")

print("Saving generations to: {}".format(eval_file_name))

with torch.no_grad():
    for idx in tqdm(total[:1000]):
        sequence_all = {}

        batch, reset = data_loader.sample_batch(split=split, bs=1, idxs=[idx])

        XMB = batch["sequences"][:, :context_size_event + 1]
        Ref = batch["sequences"][:, context_size_event + 1:]
        MMB = batch["attention_mask"][:, :context_size_event + 1]

        init = "".join([text_encoder.decoder[i].replace('</w>', ' ').replace(
                "<blank>", "___ ") for i in XMB[:, :].squeeze().tolist() if i])

        XMB = model_utils.prepare_position_embeddings(
            opt, text_encoder.encoder, XMB.unsqueeze(-1))

        words = init.split()
        sequence_all["event"] = " ".join(words[:-1]) if len(words) > 1 else ""
        sequence_all["relation"] = words[-1]

        lm_probs = lm_model(XMB.unsqueeze(1), sequence_mask=MMB)
        dist = lm_probs[:, -1, :].squeeze()

        values, indices = lm_probs[:, -1, :].max(dim=-1)
        seqs = indices.clone().unsqueeze(0)

        next_pos = XMB[:, -1:, 1] + 1
        next_x = torch.cat((indices.view(1, -1), next_pos), -1).unsqueeze(1)
        XMB = torch.cat((XMB, next_x), 1)
        MMB = torch.cat([MMB, torch.ones(XMB.size(0), 1, device=MMB.device)], 1)

        # Sample from top k

        for _ in range(args.gen_len):
            lm_probs = lm_model(XMB.unsqueeze(1), sequence_mask=MMB)
            dist = lm_probs[:, -1, :].squeeze()

            # Sample from top k
            values, next_idx = lm_probs[:, -1, :].max(dim=-1)

            next_idx = next_idx.unsqueeze(1)

            seqs = torch.cat([seqs, next_idx], 1)

            if (next_idx.item() == end_token) or _ == context_size_effect - 1:
                break

            XMB, MMB = append_batch(XMB, next_idx, MMB)

        beams = []

        for beam in seqs:
            beams.append(" ".join("".join(
                [text_encoder.decoder[tok.item()].replace(
                    '</w>', ' ').replace('\n', '')
                 for tok in beam if tok != end_token]).split()))

        # print(beams[0])

        sequence_all['beams'] = beams
        final_sequences.append(sequence_all)

import pickle

utils.mkpath("/".join(eval_file_name.split("/")[:-1]))

with open(eval_file_name, "wb") as f:
    pickle.dump(final_sequences, f)

