import sys
import os
if not os.getcwd() in sys.path:
	sys.path.append(os.getcwd())
if not os.path.join(os.getcwd(), "COMET") in sys.path:
	sys.path.append(os.path.join(os.getcwd(), "COMET"))

from src.data.utils import TextEncoder

from ac.utils.utils import abs_path, find_pickles, any_missing_file

# raw data filepaths
path_aserdb = abs_path("data/aser_v0.1.0.db")
path_relations_npz = abs_path("data/relations.npz")
path_ids = abs_path("data/ids.npy")
path_metadat = abs_path("data/meta.npy")
path_relations_ind = abs_path("data/relations_ind.npy")

# txt pickle filepaths
path_txt_datset_train = abs_path("data/train.pickle")
path_txt_datset_dev = abs_path("data/dev.pickle")
path_txt_datset_test = abs_path("data/test.pickle")

# tensor pickle filepaths
path_tensor_datset_train = abs_path("data/tensor_train.pickle")
path_tensor_datset_dev = abs_path("data/tensor_dev.pickle")
path_tensor_datset_test = abs_path("data/tensor_test.pickle")

# text encoder filepaths
path_encoder = abs_path("COMET/model/encoder_bpe_40000.json")
path_bpe = abs_path("COMET/model/vocab_40000.bpe")

def prepare_dataset():
    import ac.data.extract as ex

    # build compressed relations representation
    if not os.path.isfile(path_ids) or not os.path.isfile(path_relations_npz):
        ex.resolve_relations(
            path_aserdb,
            path_relations_npz,
            path_metadat,
            path_ids
        )

    # build index array from relations
    if not os.path.isfile(path_relations_ind):
        ex.extract_relation_ind(path_relations_npz, path_relations_ind)

def build_dataset_tensors():
    """
    """
    import ac.data.tensor as tensor
    if any_missing_file([path_tensor_datset_dev, path_tensor_datset_test, path_tensor_datset_train]):        
        encoder = TextEncoder(path_encoder, path_bpe)
        print(encoder.encoder["<END>"])
        exit()
        test_data = tensor.encode_dataset(path_txt_datset_test, encoder)
        

if __name__ == '__main__':
    #prepare_dataset()

    # TODO: split dataset into pickles. This is done in a notebook.

    build_dataset_tensors()

