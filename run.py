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

from ac.utils.utils import abs_path
import ac.data.extract as ex

ex.resolve_relations(
	abs_path("data/aser_v0.1.0.db"),
	abs_path("data/relations.npz"),
	abs_path("data/meta.npy"),
	abs_path("data/ids.npy")
)