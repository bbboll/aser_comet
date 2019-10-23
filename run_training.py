import set_root

import random
import torch
import argparse
from distutils.dir_util import mkpath

import utils.utils as utils
import src.train.utils as train_utils
import src.train.atomic_train as train
import ac.train.models as models
import src.data.data as data
import src.data.config as cfg

from src.data.utils import TextEncoder
from src.train.opt import OpenAIAdam

from ac.data.loader import GenerationDataLoader
import ac.data.encode as encode
import ac.utils.config as ac_conf
from ac.utils.io_utils import abs_path

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_num", type=str, default="0")
parser.add_argument("--gpu_num", type=str, default="0")

if __name__ == '__main__':
	args = parser.parse_args()

	# configure training run
	config = ac_conf.load_default()
	config.train.dynamic.bs = 32
	config.gpu_index = args.gpu_num
	meta = ac_conf.get_meta(config)

	# save training run config
	savedir = abs_path("results/run_{}".format(args.experiment_num))
	mkpath(savedir)
	ac_conf.save_config(config, savedir)

	print("Loading data")

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

	print("Done.")

	# compute training metadata
	n_special = len(special_tokens)
	n_ctx = config.data.maxe1 + config.data.maxe2
	n_vocab = len(text_encoder.encoder) + n_ctx
	config.net.vSize = n_vocab

	print("Building Model")

	model = models.make_model(
		config, n_vocab, n_ctx, n_special,
		load=True)

	print("Done.")

	if config.gpu_mode:
		print("Pushing to GPU: {}".format(config.gpu_index))
		cfg.device = int(config.gpu_index)
		cfg.do_gpu = True
		torch.cuda.device(cfg.device)
		model.cuda(cfg.device)
		print("Done.")

	print("Training")

	optimizer = OpenAIAdam(model.parameters(),
						   lr=config.train.dynamic.lr,
						   schedule=config.train.static.lrsched,
						   warmup=config.train.static.lrwarm,
						   t_total=meta.iterations,
						   b1=config.train.static.b1,
						   b2=config.train.static.b2,
						   e=config.train.static.e,
						   l2=config.train.static.l2,
						   vector_l2=config.train.static.vl2,
						   max_grad_norm=config.train.static.clip)

	scorers = ["bleu", "rouge", "cider"]
	trainer = train.make_trainer(
		config, meta, data_loader, model, optimizer)
	trainer.set_evaluator(config, model, data_loader)
	#import pdb; pdb.set_trace()
	
	trainer.run()