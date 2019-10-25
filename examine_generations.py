import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gen_file", type=str, default=None)
args = parser.parse_args()
if args.gen_file == None:
    print("Please supply --gen_file argument.")
    exit()

with open(args.gen_file, "rb") as f:
	seqs = pickle.load(f)

for seq in seqs[:100]:
	print(seq["event"])
	print(seq["relation"], seq["beams"][0])
	print()