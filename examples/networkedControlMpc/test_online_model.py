"""Regression tests for the compact linear-update payload and CPU-only P."""
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from online_model import OnlineModel


class ShortWriter:
    def __init__(self):
        self.data = bytearray()

    def write(self, values):
        count = min(17,len(values))
        self.data.extend(values[:count])
        return count

    def flush(self):
        pass


class OnlineModelTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data = Path(directory.name)
        self.shapes = {
            "BTr": (1,3,5), "AB": (3,4,5), "K": (1,3,3),
            "ABKTr": (3,3,5), "lowCholesky": (1,1,3),
        }
        for name,shape in self.shapes.items():
            np.asarray(shape,dtype="<u8").tofile(self.data/f"dynamics_{name}_d.bt")
        (self.data/"data.json").write_text(json.dumps({"preconditioned":False}))

        def tree(gamma,phi,branching,trick_,plot):
            return None,None,None,np.array([0.,phi/2,phi,.2*gamma,.4*gamma])

        self.parameters = {"state_size":3,"num_states":2,"num_inputs":1,
            "Ac":np.diag([-.2,-.35]),"Bc":np.array([[.4],[.25]]),"max_delay":.1,
            "branching":[2,1],"trick":1.,"make_beta_tree":tree}

    def model(self):
        model = OnlineModel(self.parameters,self.data)
        model.update(.4,.8)
        return model

    def test_payload_contains_six_active_fields_only(self):
        model = self.model()
        expected = ("BTr","AB","K","KTr","ABKTr","lowCholesky")
        self.assertEqual(tuple(model.fields),expected)
        self.assertEqual(model.protocol,"linear-v2")
        self.assertEqual(model.payload_bytes,sum(array.nbytes for array in model.fields.values()))
        output = io.BytesIO()
        model.write_payload(output)
        self.assertEqual(len(output.getvalue()),model.payload_bytes)
        self.assertEqual(output.getvalue(),b"".join(array.tobytes() for array in model.fields.values()))

    def test_cpu_riccati_workspace_remains_finite_and_is_recomputed(self):
        model = self.model()
        expected = model.P.copy()
        self.assertNotIn("P",model.fields)
        self.assertNotIn("APB",model.fields)
        np.testing.assert_allclose(model.P[model.nonleaf:],np.broadcast_to(np.eye(3),(2,3,3)))
        self.assertGreater(np.linalg.norm(model.P[0]-np.eye(3)),0.)
        model.P.fill(np.nan)
        model.update(.4,.8)
        np.testing.assert_array_equal(model.P,expected)

    def test_buffers_are_reused_across_updates(self):
        model = self.model()
        pointers = {name:array.ctypes.data for name,array in model.fields.items()}
        p_pointer = model.P.ctypes.data
        for phi in [.2,.6,.9]:
            model.update(phi,.8)
            self.assertEqual(pointers,{name:array.ctypes.data for name,array in model.fields.items()})
            self.assertEqual(p_pointer,model.P.ctypes.data)

    def test_payload_writer_handles_partial_writes(self):
        model = self.model()
        output = ShortWriter()
        model.write_payload(output)
        self.assertEqual(bytes(output.data),b"".join(array.tobytes() for array in model.fields.values()))

    def test_reference_audit_includes_private_cpu_p(self):
        model = self.model()
        arrays = {**model.fields,"P":model.P.swapaxes(1,2)}
        for name,array in arrays.items():
            with (self.data/f"dynamics_{name}_d.bt").open("wb") as stream:
                np.asarray(array.shape[::-1],dtype="<u8").tofile(stream)
                array.tofile(stream)
        self.assertEqual(set(model.compare_reference()),set(arrays))
        model.P[0,0,0] += 1.
        with self.assertRaises(AssertionError):
            model.compare_reference()


if __name__ == "__main__":
    unittest.main()
