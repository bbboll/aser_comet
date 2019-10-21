import pickle
from relations import relations

def load_dataset(path):
	"""
	Load pickle at given path.
	"""
	with open(path, 'rb') as f:
		return pickle.load(f)

def encode_dataset(dataset, text_encoder):
	"""
	For each tuple (ev1, rel, ev2) in the dataset, replace the
	... string ev1 and integer rel with a single encoded int sequence
	... string ev2 with an encoded int sequence
	This first looks up the relation name by its index and encodes it as a natural word.
	"""
	for (ev1, rel, ev2) in dataset:
		# work on input
		input_seq = text_encoder.encode(ev1, verbose=False)
		input_seq += text_encoder.encode(relations[rel], verbose=False)
		




def build_masks(encoded_dataset):
	masks = [(len(ev1), len(ev2)) for (ev1, rel, ev2) in encoded_dataset]
	return masks