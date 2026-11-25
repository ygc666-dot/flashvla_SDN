"""TCP client for `scripts/p3/flashvla_server.py` -- the G3 `--method flashvla` adapter.

`policy/pizero/evaluate.py` only touches two members of a policy object
(`reset(instruction, seed=...)` and `step(image, instruction, proprio=...)`), so
this class implements exactly those and returns the same
`(raw_chunk, [action dict, ...])` shape as `PiZeroInference`. Everything else
(chunking, cold start, normalization) lives in the server, which runs in the
py3.12 `flashvla` env because SIMPLER's py3.10 env cannot import lerobot.

Keep this module dependency-free apart from numpy/socket/pickle: it is imported
by the `sdn` env (numpy 1.24), which cannot unpickle numpy 2.x objects, so only
plain Python crosses the wire.

One policy server serves one GPU. The port is derived from the worker's
`CUDA_VISIBLE_DEVICES` (`evaluate.py` sets it per worker in `_set_gpu`), so
`--num-gpus 2` with GPUs 0 and 4 talks to servers on 12400 and 12404.
"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import time

import numpy as np


def _recv_msg(conn) -> dict:
    raw = b""
    while len(raw) < 4:
        chunk = conn.recv(4 - len(raw))
        if not chunk:
            raise ConnectionError("policy server closed the connection")
        raw += chunk
    (n,) = struct.unpack(">I", raw)
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("policy server closed mid-message")
        buf += chunk
    return pickle.loads(buf)


def _send_msg(conn, obj) -> None:
    payload = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _worker_gpu_id() -> str:
    """The physical GPU id this worker was pinned to, or '' if unpinned."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [g.strip() for g in visible.split(",") if g.strip()]
    return ids[0] if ids else ""


class FlashVLAInference:
    """Drop-in stand-in for `PiZeroInference` backed by a remote FlashVLA server."""

    def __init__(
        self,
        checkpoint_path: str = "",
        port: int | None = None,
        port_base: int = 12400,
        host: str = "127.0.0.1",
        exec_steps: int = 4,
        timeout: float = 900.0,
        connect_retries: int = 20,
        connect_wait_s: float = 15.0,
        **unknown,
    ):
        if unknown:
            # evaluate.py mirrors the built config into 000.log; a typo in
            # --opts would otherwise be swallowed silently.
            raise TypeError(f"unexpected FlashVLA options: {sorted(unknown)}")

        gpu_id = _worker_gpu_id()
        if port is None:
            if not gpu_id:
                raise ValueError(
                    "cannot infer the policy-server port: CUDA_VISIBLE_DEVICES is empty; "
                    "pass `--opts flashvla_port <port>`"
                )
            port = port_base + int(gpu_id)
        self.port = int(port)
        self.host = host
        self.exec_steps = int(exec_steps)
        self.checkpoint_path = checkpoint_path
        self._conn = None
        self.last_gripper_raw = None
        self.last_was_cold = False
        self.last_latency_ms = 0.0
        self._connect(timeout, connect_retries, connect_wait_s)
        self.info = self._rpc({"cmd": "ping"})
        # Guard against labelling results with the wrong checkpoint: one server
        # per port, but the port is derived from the GPU id, so a stale server
        # from a previous run is a real possibility.
        served = str(self.info.get("ckpt", ""))
        if checkpoint_path and served and os.path.abspath(served) != os.path.abspath(checkpoint_path):
            raise RuntimeError(
                f"port {self.port} serves {served!r} but --checkpoint is "
                f"{checkpoint_path!r}; restart the server for this checkpoint"
            )

    # -- connection --------------------------------------------------------- #
    def _connect(self, timeout: float, retries: int, wait_s: float) -> None:
        last = None
        for attempt in range(1, retries + 1):
            try:
                self._conn = socket.create_connection((self.host, self.port), timeout=timeout)
                return
            except OSError as exc:  # server not up yet (model load takes ~40 s)
                last = exc
                if attempt < retries:
                    time.sleep(wait_s)
        raise ConnectionError(
            f"no FlashVLA policy server on {self.host}:{self.port} after {retries} attempts "
            f"({last}). Start one per GPU with scripts/p3/flashvla_server.py."
        )

    def _rpc(self, req: dict) -> dict:
        _send_msg(self._conn, req)
        resp = _recv_msg(self._conn)
        if not resp.get("ok", False):
            raise RuntimeError(f"policy server error: {resp.get('error')!r}")
        return resp

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    # -- PiZeroInference interface ----------------------------------------- #
    def reset(self, instruction, seed=None):
        resp = self._rpc({"cmd": "reset", "seed": int(seed or 0),
                          "instruction": str(instruction)})
        self.last_reset = resp
        return resp

    def step(self, image, instruction, proprio, *args, **kwargs):
        img = np.ascontiguousarray(np.asarray(image))
        if img.dtype != np.uint8:
            # SIMPLER hands over uint8 RGB; anything else would silently shift
            # the [-1, 1] rescale inside the policy.
            img = np.clip(img, 0, 255).astype(np.uint8)
        state = np.asarray(proprio, dtype=np.float32).reshape(-1)[:8]

        resp = self._rpc({
            "cmd": "act",
            "image": img.tobytes(),
            "shape": tuple(int(v) for v in img.shape),
            "state": state.tolist(),
            "instruction": str(instruction),
        })
        raw = np.asarray(resp["actions"], dtype=np.float32).reshape(-1, 7)
        self.last_latency_ms = float(resp.get("latency_ms", 0.0))
        self.last_was_cold = bool(resp.get("cold", False))
        self.last_gripper_raw = resp.get("gripper_raw")

        zero = np.zeros(1, dtype=np.float32)
        actions = [
            {
                "world_vector": row[:3],
                "rot_axangle": row[3:6],
                "gripper": row[6:7],
                "terminate_episode": zero,
            }
            for row in raw
        ]
        return raw, actions
