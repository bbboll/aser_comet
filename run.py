import sys
import os
if not os.getcwd() in sys.path:
	sys.path.append(os.getcwd())
if not os.path.join(os.getcwd(), "COMET") in sys.path:
	sys.path.append(os.path.join(os.getcwd(), "COMET"))
"""
You may regard this technique of managing python modules as hacky
and I agree. This is, however, the way the COMET authors chose
to do it and I will concede that it is quite a glorious hack.
"""

from ac.utils.utils import abs_path, find_pickles
import ac.data.extract as ex

# setup data filepaths
path_aserdb = abs_path("data/aser_v0.1.0.db")
path_relations_npz = abs_path("data/relations.npz")
path_ids = abs_path("data/ids.npy")
path_metadat = abs_path("data/meta.npy")
dirpath_datachunks = abs_path("data/chunks/")

# build compressed relations representation
if not os.path.isfile(path_ids) or not os.path.isfile(path_relations_npz):
	ex.resolve_relations(
		path_aserdb,
		path_relations_npz,
		path_metadat,
		path_ids
	)


print(find_pickles(dirpath_datachunks))