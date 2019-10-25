import set_root

import src.data.config as cfg

cfg.device = 0

import torch
import argparse

import src.data.data as data
import src.data.config as cfg
import src.models.models as models
import src.evaluate.atomic_evaluate as evaluate
import utils.utils as utils
from src.data.utils import TextEncoder

import ac.utils.config as ac_conf
from ac.data.loader import GenerationDataLoader
import ac.data.encode as encode

# torch.cuda.set_device(cfg.device)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_num", type=str, default="0")
parser.add_argument("--split", type=str, default="dev")
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--gpu_num", type=str, default="0")

args = parser.parse_args()
if args.model_name == None:
    print("Please enter model name.")
    exit()

split = args.split

# configure evaluation run
config = ac_conf.load_default()
config.train.dynamic.bs = 32
config.gpu_index = int(args.gpu_num)
meta = ac_conf.get_meta(config)

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

# Get component segmentation of sequences
# context_size_event = maximum size of an event description
# context_size_effect = maximum size of an event effect/intent/etc.
context_size_event = data_loader.max_event
context_size_effect = data_loader.max_effect

n_special = len(special_tokens)
n_ctx = context_size_event + context_size_effect
n_vocab = len(text_encoder.encoder) + n_ctx

config.net.vSize = n_vocab

print("Building Model")

model = models.make_model(config, n_vocab, n_ctx, n_special, load=False)

print("Loading Weights")
model_file = torch.load(args.model_name, map_location=torch.device("cpu"))
model.load_state_dict(model_file['state_dict'])
print("Done Loading Weights")

model.eval()

# Initialize variable for # of examples to cycle through
data.set_max_sizes(data_loader, force_split=split)

evaluator = evaluate.make_evaluator(config, model, data_loader)
evaluator.batch_variables["split"] = split
# model.cuda(cfg.device)

loss = evaluator.epoch(config, model, data_loader, split)

data.save_eval_file(config, loss, "losses", split=split)

loss_str = []
loss_str.append("Per Token   Loss:       {}".format(loss["total_micro"]))
loss_str.append("Per Token   Perplexity: {}".format(loss["ppl_micro"]))
loss_str.append("Per Example Loss:       {}".format(loss["total_macro"]))
loss_str.append("Per Example Perplexity: {}".format(loss["ppl_macro"]))
loss_str = "\n".join(loss_str)
print(loss_str)

data.save_eval_file(config, loss_str, "losses", split=split, ext="txt")
