import sys
import os
if not os.getcwd() in sys.path:
	sys.path.append(os.getcwd())
if not os.path.join(os.getcwd(), "COMET") in sys.path:
	sys.path.append(os.path.join(os.getcwd(), "COMET"))

from ac.utils.io_utils import *
import ac.data.encode as ac_encode
import torch

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

# encoded pickle filepaths
path_encoded_datset_train = abs_path("data/encoded_train.pickle")
path_encoded_datset_dev = abs_path("data/encoded_dev.pickle")
path_encoded_datset_test = abs_path("data/encoded_test.pickle")

# tensor pickle filepaths
path_tensor_datset_train = abs_path("data/tensor_train.pickle")
path_tensor_datset_dev = abs_path("data/tensor_dev.pickle")
path_tensor_datset_test = abs_path("data/tensor_test.pickle")

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

def build_dataset_encoded():
    """
    """
    encoded_paths = [
        path_encoded_datset_dev,
        path_encoded_datset_test,
        path_encoded_datset_train
    ]
    data_paths = [
        path_txt_datset_dev,
        path_txt_datset_test,
        path_txt_datset_train
    ]

    encoder = ac_encode.build_text_encoder()
    
    for encoded_path, data_path in zip(encoded_paths, data_paths):
        if missing_file(encoded_path):
            print("Working to create", encoded_path)
            # load + encode txt data
            data = load_txt_dataset(data_path)
            data, masks = ac_encode.encode_dataset(data, encoder)

            # save data tensor
            torch.save(data, encoded_path)
            mask_savepath = encoded_path[:-7]+"_mask.pickle"
            torch.save(masks, mask_savepath)
            print("Data saved to", encoded_path)
            print("and masks saved to", mask_savepath)
            del masks
            del data

def build_dataset_tensors():
    """
    """
    data_paths = [
        path_tensor_datset_dev,
        path_tensor_datset_test,
        path_tensor_datset_train
    ]
    
    if any_missing_file(data_paths):        
        encoder = ac_encode.build_text_encoder()
    
    # TODO: load from encoded because this enables to compute max(seq_len for all seq)

    # for data_path in data_paths:
    #     if missing_file(data_path):
    #         print("Working to create", data_path)
    #         # load + encode txt data
    #         data = load_txt_dataset(path_txt_datset_dev)
    #         data, masks, max_len = ac_encode.encode_dataset(data, encoder)

    #         # build data tensor
    #         num_elements = len(data)
    #         tensor = torch.IntTensor(num_elements, max_len[0]+max_len[1]).fill_(0)
    #         print("building data tensor...")
    #         for i, (in_seq, out_seq) in enumerate(data):
    #             tensor[i,:len(in_seq)] = torch.IntTensor(in_seq)
    #             tensor[i,max_len[0]:max_len[0]+len(out_seq)] = torch.IntTensor(out_seq)

    #         # save data tensor
    #         torch.save(tensor, data_path)
    #         print("Data saved to", data_path)
    #         del tensor
    #         del data

if __name__ == '__main__':
    #prepare_dataset()

    # TODO: split dataset into pickles. This is done in a notebook.

    build_dataset_encoded()
    #build_dataset_tensors()

