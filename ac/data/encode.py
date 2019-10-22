from src.data.utils import TextEncoder
from ac.utils.io_utils import abs_path
from ac.utils.list_utils import flatten_list
import time

# text encoder filepaths
path_encoder = abs_path("COMET/model/encoder_bpe_40000.json")
path_bpe = abs_path("COMET/model/vocab_40000.bpe")

def get_relations():
	relations = [
		"Precedence",
		"Succession",
		"Synchronous",
		"Reason",
		"Result",
		"Condition",
		"Contrast",
		"Concession",
		"Conjunction",
		"Instantiation",
		"Restatement",
		"ChosenAlternative",
		"Alternative",
		"Exception",
		"CoOccurrence"	
	]
	return relations

def build_text_encoder():
	"""
	Load vocabulary, build text encoder, add special tokens to it.
	"""
	text_encoder = TextEncoder(path_encoder, path_bpe)
	for special_token in ["<START>", "<END>", "<BLANK>"]:
		vocab_size = len(text_encoder.encoder)
		text_encoder.decoder[vocab_size] = special_token
		text_encoder.encoder[special_token] = vocab_size
	# note, that special tokens are not canonically used by calling .encode()
	# instead, index the encoder.encoder dict directly
	return text_encoder

def encode_dataset(dataset, text_encoder):
	"""
	For each tuple (ev1, rel, ev2) in the dataset, replace the
	... string ev1 and integer rel with a single encoded int sequence
	... string ev2 with an encoded int sequence
	This first looks up the relation name by its index and encodes it as a natural word.
	"""
	dataset_size = len(dataset)
	print("Encoding", dataset_size, "tuples...")
	encoded_dataset = []
	masks = []
	relations = get_relations()
	start_it = time.time()
	for i, (ev1, rel, ev2) in enumerate(dataset):
		# encode text
		input_seq = text_encoder.encode(ev1.split(), verbose=False)
		input_seq += text_encoder.encode([relations[rel]], verbose=False)
		output_seq = text_encoder.encode(ev2.split(), verbose=False)
		output_seq.append([text_encoder.encoder["<END>"]])
				
		# flatten encodings
		input_seq = flatten_list(input_seq)
		output_seq = flatten_list(output_seq)
		
		encoded_dataset.append((input_seq, output_seq))
		(inlen, outlen) = (len(input_seq), len(output_seq))
		masks.append((inlen, outlen))

		if i % int(1e5) == 0 and i > 0:
			end_it = time.time()
			it_per_sec = i / (end_it-start_it)
			time_left = (dataset_size-i) / it_per_sec / 60
			print("{}/{} tuples processed. {} minutes remaining.".format(i, dataset_size, time_left))
	print("done.")
	return encoded_dataset, masks