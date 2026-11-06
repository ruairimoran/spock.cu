"""Fixed-layout numerical updates for the stored diagonal Case Study 2 plant.

Static topology, costs, risk projections and scaling come from the original
factory once. Subsequent updates recompute only dynamics/projection factors.
Payload arrays use (node, column, row) order, matching GPUtils binary storage.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, cho_solve


class OnlineModel:
    names = ("BTr", "AB", "K", "KTr", "ABKTr", "lowCholesky")
    protocol = "linear-v2"

    def __init__(self, parameters, data):
        self.p = parameters
        self.data = Path(data)
        self.nx, self.nu = parameters["state_size"], parameters["num_inputs"]
        self.physical = parameters["num_states"]
        self.a = np.diag(parameters["Ac"])
        if not np.array_equal(parameters["Ac"], np.diag(self.a)):
            raise ValueError("This Case Study 2 online updater requires diagonal Ac")
        if self.nx != self.physical + self.nu:
            raise ValueError("Unexpected augmented state layout")
        self.fields = {}
        for name in self.names:
            if name == "KTr":
                k = self.fields["K"]
                self.fields[name] = np.empty((k.shape[0], k.shape[2], k.shape[1]), dtype="<f8")
            else:
                with (self.data / f"dynamics_{name}_d.bt").open("rb") as stream:
                    shape = np.fromfile(stream, dtype="<u8", count=3)
                self.fields[name] = np.zeros(tuple(int(x) for x in shape[::-1]), dtype="<f8")
        self.nodes, self.nonleaf = len(self.fields["AB"]), len(self.fields["K"])
        self.AB = self.fields["AB"].swapaxes(1, 2)
        self.A, self.B = self.AB[:, :, :self.nx], self.AB[:, :, self.nx:]
        # Retain the old per-matrix column-major layout for the CPU recursion.
        # P is private workspace; neither it nor APB belongs in the wire payload.
        self.P = np.zeros((self.nodes, self.nx, self.nx), dtype="<f8").swapaxes(1, 2)
        self.K = self.fields["K"].swapaxes(1, 2)
        self.identity_x, self.identity_u = np.eye(self.nx), np.eye(self.nu)
        config = json.loads((self.data / "data.json").read_text())
        scale = np.fromfile(self.data / "scaling_d.bt", dtype="<f8", offset=24) if config["preconditioned"] else np.ones(self.nx + self.nu)
        self.sx, self.su = scale[:self.nx], scale[self.nx:]
        self.h = parameters["max_delay"]
        self.exp_h = np.exp(self.a * self.h)
        self.full_integral = self.integral(self.h)
        self.branching = parameters["branching"]
        counts = [1]
        for factor in self.branching:
            counts.append(counts[-1] * factor)
        self.stage_from = np.cumsum([0] + counts[:-1])
        self.counts = counts
        if sum(counts) != self.nodes:
            raise ValueError("Configured topology differs from the static problem")

    def integral(self, duration):
        result = np.full_like(self.a, duration, dtype=float)
        np.divide(np.expm1(self.a * duration), self.a, out=result, where=self.a != 0)
        return result

    def update(self, phi, gamma):
        begin = time.perf_counter()
        _, _, _, values = self.p["make_beta_tree"](
            gamma, phi, self.branching, trick_=self.p["trick"], plot=False)
        if len(values) != self.nodes:
            raise ValueError("Online tree dimensions changed")
        for node in range(1, self.nodes):
            remaining = self.integral(self.h * (1 - values[node]))
            A, B = self.A[node], self.B[node]
            A.fill(0)
            B.fill(0)
            A[np.arange(self.physical), np.arange(self.physical)] = self.exp_h
            A[:self.physical, self.physical:] = ((self.full_integral - remaining)[:, None]
                                               * self.p["Bc"] * self.sx[:self.physical, None]
                                               / self.sx[None, self.physical:])
            B[:self.physical] = (remaining[:, None] * self.p["Bc"]
                                * self.sx[:self.physical, None] / self.su[None, :])
            B[self.physical:] = np.diag(self.sx[self.physical:] / self.su)
            self.fields["BTr"][node] = B
        self.P[self.nonleaf:] = self.identity_x
        for stage in reversed(range(len(self.branching))):
            start, count = self.stage_from[stage], self.counts[stage]
            child_start, factor = self.stage_from[stage + 1], self.branching[stage]
            for offset in range(count):
                parent = start + offset
                children = range(child_start + offset * factor, child_start + (offset + 1) * factor)
                sum_r = np.zeros((self.nu, self.nu))
                sum_k = np.zeros((self.nu, self.nx))
                for child in children:
                    btp = self.B[child].T @ self.P[child]
                    sum_r += btp @ self.B[child]
                    sum_k += btp @ self.A[child]
                lower = cholesky(self.identity_u + sum_r, lower=True, check_finite=False)
                self.fields["lowCholesky"][parent] = lower.T
                self.K[parent] = cho_solve((lower, True), -sum_k, check_finite=False)
                self.fields["KTr"][parent] = self.K[parent]
                sum_p = np.zeros((self.nx, self.nx))
                for child in children:
                    combined = self.A[child] + self.B[child] @ self.K[parent]
                    self.fields["ABKTr"][child] = combined
                    atp = combined.T @ self.P[child]
                    sum_p += atp @ combined
                self.P[parent] = self.identity_x + self.K[parent].T @ self.K[parent] + sum_p
        if not np.isfinite(self.P).all() or not all(np.isfinite(array).all() for array in self.fields.values()):
            raise ValueError("Nonfinite online dynamics/projection data")
        return time.perf_counter() - begin

    def compare_reference(self, data=None):
        data = Path(data) if data is not None else self.data
        errors = {}
        # Audit the CPU-only Riccati factors as well as the transmitted fields.
        arrays = {**self.fields, "P": self.P.swapaxes(1, 2)}
        for name, array in arrays.items():
            if name == "KTr":
                k = self.fields["K"]
                reference = np.fromfile(data / "dynamics_K_d.bt", dtype="<f8", offset=24).reshape(k.shape).swapaxes(1, 2)
            else:
                reference = np.fromfile(data / f"dynamics_{name}_d.bt", dtype="<f8", offset=24).reshape(array.shape)
            error = float(np.max(np.abs(array - reference)))
            errors[name] = error
            np.testing.assert_allclose(array, reference, rtol=2e-10, atol=1e-9,
                                       err_msg=f"Online {name} differs from the original factory")
        return errors

    @property
    def payload_bytes(self):
        return sum(array.nbytes for array in self.fields.values())

    def write_payload(self, stream):
        # memoryview avoids creating a multi-gigabyte bytes object.
        for array in self.fields.values():
            view = memoryview(array).cast("B")
            while view:
                count = stream.write(view)
                if count is None or count == 0:
                    raise BrokenPipeError("MPC worker stopped accepting dynamics data")
                view = view[count:]
        stream.flush()
