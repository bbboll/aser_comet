import json
import os

from ac.utils.io_utils import abs_path
from utils.utils import DD

def load_default():
	with open(abs_path("config_default.json"), 'r') as f:
		config = json.load(f)
	config =  DD(config)
	config.net = DD(config.net)
	config.train = DD(config.train)
	config.train.static = DD(config.train.static)
	config.train.dynamic = DD(config.train.dynamic)
	config.data = DD(config.data)
	config.eval = DD(config.eval)
	config.train.dynamic.epoch = 0
	return DD(config)

def get_meta(config):
	meta = DD()
	meta.iterations = int(config.iters)
	meta.cycle = config.cycle
	return meta

def save_config(config, dirpath):
	with open(os.path.join(dirpath, "config.json"), "w") as f:
		json.dump(config, f, indent=4)