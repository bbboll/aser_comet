import utils.utils as utils
import src.data.utils as data_utils
import src.data.config as cfg

import json
import random
import math
import torch

from ac.util.io_util import abs_path


class DataLoader(object):
    def __init__(self, opt):
        self.data = {}
        self.data["train"] = {}
        self.data["dev"] = {}
        self.data["test"] = {}

        self.sequences = {}
        self.sequences["train"] = {}
        self.sequences["dev"] = {}
        self.sequences["test"] = {}

        self.masks = {}
        self.masks["train"] = {}
        self.masks["dev"] = {}
        self.masks["test"] = {}

        self.offsets = {}
        self.offsets["train"] = {}
        self.offsets["dev"] = {}
        self.offsets["test"] = {}

    def offset_summary(self, split):
        return self.offsets[split]["total"]


class GenerationDataLoader(DataLoader):
    def __init__(self, opt, categories):
        super(GenerationDataLoader, self).__init__(opt)

        self.categories = categories
        self.opt = opt

        for split in self.data:
            self.data[split] = {"total": []}
            self.offsets[split] = {"total": 0}

        self.vocab_encoder = None
        self.vocab_decoder = None
        self.special_chars = ["<START>", "<END>", "<BLANK>"]
        self.max_event = 18
        self.max_effect = 20

    def load_data(self, path):
        # TODO: load data

        # load tensors
        self.sequences["train"]["total"] = torch.load(abs_path("data/tensor_train.pickle"))
        self.sequences["dev"]["total"] = torch.load(abs_path("data/tensor_dev.pickle"))
        self.sequences["test"]["total"] = torch.load(abs_path("data/tensor_test.pickle"))

        # load masks
        self.masks["train"]["total"] = torch.load(abs_path("data/tensor_train_mask.pickle"))
        self.masks["dev"]["total"] = torch.load(abs_path("data/tensor_dev_mask.pickle"))
        self.masks["test"]["total"] = torch.load(abs_path("data/tensor_test_mask.pickle"))

        return True

    def sample_batch(self, split, bs, idxs=None):
        offset = self.offsets[split]["total"]

        batch = {}

        # Decided not to reduce computation on here because it's all parallel
        # anyway and we don't want to run out of memory in cases where we
        # don't see the longest version quickly enough

        if idxs:
            seqs = self.sequences[split]["total"].index_select(
                0, torch.LongTensor(idxs).to(
                    self.sequences[split]["total"].device))
        else:
            seqs = self.sequences[split]["total"][offset:offset + bs]
        batch["sequences"] = seqs.to(cfg.device)
        batch["attention_mask"] = make_attention_mask(seqs)
        batch["loss_mask"] = make_loss_mask(
            seqs, self.max_event, 1)
        batch["key"] = ("total", offset, offset + bs)

        offset += seqs.size(0)

        self.offsets[split]["total"] = offset

        if split == "train" and offset + bs > len(self.sequences[split]["total"]):
            return batch, True
        elif offset >= len(self.sequences[split]["total"]):
            return batch, True
        else:
            return batch, False

    def reset_offsets(self, splits=["train", "test", "dev"],
                      shuffle=True, keys=None):
        if isinstance(splits, str):
            splits = [splits]

        for split in splits:
            if keys is None:
                keys = ["total"]

            for key in keys:
                self.offsets[split][key] = 0

            if shuffle:
                self.shuffle_sequences(split, keys)

    def shuffle_sequences(self, split="train", keys=None):
        if keys is None:
            # print(type(self.data))
            # print(type(self.data.keys()))
            keys = self.data[split].keys()

        for key in keys:
            idxs = list(range(len(self.masks[split][key])))

            random.shuffle(idxs)

            self.sequences[split][key] = \
                self.sequences[split][key].index_select(
                    0, torch.LongTensor(idxs))

            #temp = [self.data[split][key][i] for i in idxs]
            #self.data[split][key] = temp
            temp = [self.masks[split][key][i] for i in idxs]
            self.masks[split][key] = temp


def make_attention_mask(sequences):
    return (sequences != 0).float().to(cfg.device)


def make_loss_mask(sequences, max_event, num_delim_tokens):
    # print(num_delim_tokens)
    # print(sequences.size())
    mask = (sequences != 0).float()
    mask[:, :max_event + num_delim_tokens] = 0
    return mask[:, 1:].to(cfg.device)
