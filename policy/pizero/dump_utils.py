"""Per-episode dump of pi0* rollouts, for the FlashVLA bootstrap dataset (P1).

One `<out_dir>/episode_<seed:05d>.npz` + one `.mp4` per episode.

Row convention (frozen here, the P2 LeRobot converter depends on it):

    row t (0-based)  proprio   : obs['agent']['eef_pos'] (8-dim, raw, before the step)
                     image     : the RGB frame that action row t was applied to
                     action    : the exact 7-dim vector handed to `env.step`, i.e.
                                 [world_vector(3), rot_axangle(3), gripper(1)]
                     success   : info['success'] right after the step (same value that
                                 `eval_utils.stat_first` turns into `first_success`)

Rows are one per `env.step`, i.e. at the native control rate of the robot
(3 Hz for google_robot, 5 Hz for widowx). Because pi0* executes a 4-step chunk
open-loop (`horizon_steps=4`, `act_steps=4`), rows 1-3 of each chunk carry the
action that was *executed*, not the one a fresh observation would have produced;
`query_index` / `query_step` / `raw_chunk` keep enough information to rebuild
either view in P2:

    query_index[t]  : which policy query produced action row t
    query_step[q]   : env-step index at which query q was made (query_step[0] == 0)
    raw_chunk[q]    : that query's raw [horizon, 7] output, i.e. *before*
                      denormalization (the value `policy.step` returns first)

Images: `mp4` (default, h264 via imageio-ffmpeg, ~10x smaller and P2 re-encodes to
mp4 anyway), `npz` (lossless, inside the npz), or `none` (numeric-only smoke runs).
"""

import json
import os
import os.path as osp
import time

import numpy as np

DUMPER_VERSION = 1
# control_freq of the SIMPLER scenes: 3 Hz for google_robot, 5 Hz for widowx
FPS_BY_ROBOT = {"google_robot": 3, "widowx": 5}
CAMERA_BY_ROBOT = {"google_robot": "overhead_camera", "widowx": "3rd_view_camera"}
MAX_INSTRUCTION_CHARS = 192


def robot_type_from_task(task):
    if task.startswith("google_robot"):
        return "google_robot"
    if task.startswith("widowx"):
        return "widowx"
    raise NotImplementedError(f"unknown task prefix {task!r}")


def fps_for_robot(robot_type):
    if robot_type not in FPS_BY_ROBOT:
        raise NotImplementedError(f"unknown robot_type {robot_type!r}; expected one of {sorted(FPS_BY_ROBOT)}")
    return FPS_BY_ROBOT[robot_type]


def _to_numpy_1d(value, dtype=np.float64):
    """Tensor / ndarray / float -> 1-dim ndarray (works in the `sdn` env, no torch import).

    `dtype=float64` is the default on purpose: the executed command is the float64
    `np.concatenate` of the three raw components, so recording them at a lower
    precision would break the bit-exact `action` check.
    """
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=dtype).reshape(-1)


class EpisodeDumper:
    def __init__(self,
                 out_dir,
                 episode,
                 task,
                 robot_type=None,
                 camera_name=None,
                 image_mode="mp4",
                 fps=None,
                 method=None,
                 result_dir=None,
                 verify=True,
                 video_quality=9,
                 action_dim=7):
        assert image_mode in ("mp4", "npz", "none"), image_mode
        self.out_dir = out_dir
        self.episode = int(episode)
        self.task = task
        self.robot_type = robot_type if robot_type is not None else robot_type_from_task(task)
        self.camera_name = camera_name if camera_name is not None else CAMERA_BY_ROBOT[self.robot_type]
        self.image_mode = image_mode
        self.fps = fps_for_robot(self.robot_type) if fps is None else int(fps)
        self.method = method
        self.result_dir = result_dir
        self.verify = verify
        self.video_quality = video_quality
        self.action_dim = int(action_dim)

        self.instruction_vocab = []
        self._instruction_index = {}
        self.query_step = []
        self.query_instruction_id = []
        self.raw_chunks = []
        self.proprios = []
        self.commands = []
        self.world_vectors = []
        self.rot_axangles = []
        self.grippers = []
        self.rewards = []
        self.successes = []
        self.truncateds = []
        self.query_indices = []
        self.instruction_ids = []
        self.images = []
        self.frame_shape = None

    # ---------------------------------------------------------------- recording
    def _instruction_id(self, text):
        text = str(text)
        if len(text) > MAX_INSTRUCTION_CHARS:
            raise ValueError(f"instruction longer than {MAX_INSTRUCTION_CHARS} chars, would be truncated: {text!r}")
        if text not in self._instruction_index:
            self._instruction_index[text] = len(self.instruction_vocab)
            self.instruction_vocab.append(text)
        return self._instruction_index[text]

    def begin_query(self, raw_chunk, instruction, step):
        """Called once per `policy.step`, before the chunk is executed."""
        chunk = _to_numpy_1d(raw_chunk, dtype=np.float32)
        if chunk.size % self.action_dim != 0:
            raise ValueError(f"raw chunk of size {chunk.size} is not a multiple of action_dim={self.action_dim}")
        self.raw_chunks.append(chunk.reshape(-1, self.action_dim))
        self.query_step.append(int(step))
        self.query_instruction_id.append(self._instruction_id(instruction))

    def add_step(self,
                 image,
                 proprio,
                 command,
                 world_vector,
                 rot_axangle,
                 gripper,
                 reward,
                 success,
                 truncated,
                 instruction):
        """Called once per `env.step`, with the observation that step was applied to."""
        command = np.asarray(command, dtype=np.float64).reshape(-1)
        if command.shape != (self.action_dim,):
            raise AssertionError(f"command must be {self.action_dim}-dim, got {command.shape}")
        if not np.all(np.isfinite(command)):
            raise AssertionError(f"non-finite command {command}")

        if self.verify:
            # The dump is only useful if `action[t]` is bit-identical to what `env.step`
            # received; rebuild it from the recorded components and compare exactly.
            rebuilt = np.concatenate([
                _to_numpy_1d(world_vector),
                _to_numpy_1d(rot_axangle),
                _to_numpy_1d(gripper),
            ]).astype(np.float64)
            if not np.array_equal(rebuilt, command):
                raise AssertionError(f"recorded components do not rebuild the executed command: "
                                     f"{rebuilt.tolist()} != {command.tolist()}")

        if image is not None:
            image = np.asarray(image)
            if image.dtype != np.uint8:
                raise AssertionError(f"image dtype {image.dtype} != uint8")
            if image.ndim != 3 or image.shape[2] != 3:
                raise AssertionError(f"image shape {image.shape} is not HWC with 3 channels")
            if self.frame_shape is None:
                self.frame_shape = tuple(image.shape)
                if self.image_mode == "mp4" and (image.shape[0] % 2 or image.shape[1] % 2):
                    # libx264 + yuv420p needs even dimensions
                    raise AssertionError(f"odd image size {image.shape[:2]} cannot be encoded as h264")
            elif tuple(image.shape) != self.frame_shape:
                raise AssertionError(f"image shape changed within the episode: "
                                     f"{tuple(image.shape)} != {self.frame_shape}")
            if self.image_mode != "none":
                # copy: the renderer may hand out a reused buffer
                self.images.append(np.array(image, copy=True))

        self.proprios.append(np.array(proprio, dtype=np.float32, copy=True).reshape(-1))
        self.commands.append(command)
        self.world_vectors.append(_to_numpy_1d(world_vector))
        self.rot_axangles.append(_to_numpy_1d(rot_axangle))
        self.grippers.append(_to_numpy_1d(gripper))
        self.rewards.append(float(reward))
        self.successes.append(bool(success))
        self.truncateds.append(bool(truncated))
        self.query_indices.append(len(self.query_step) - 1)
        self.instruction_ids.append(self._instruction_id(instruction))

    # ------------------------------------------------------------------ writing
    def _write_video(self, path):
        import imageio
        with imageio.get_writer(path, fps=self.fps, codec="libx264", quality=self.video_quality,
                                macro_block_size=1) as writer:
            for frame in self.images:
                writer.append_data(frame)

    def finalize(self, first_success=None):
        """Writes the npz (+ mp4) and returns the per-episode summary written to the log."""
        n_steps = len(self.commands)
        if n_steps == 0:
            raise AssertionError(f"episode {self.episode} recorded no step")
        if len(self.raw_chunks) == 0:
            raise AssertionError(f"episode {self.episode} recorded no query")
        if self.verify and np.any(np.diff(self.query_indices) < 0):
            raise AssertionError("query_index went backwards")

        os.makedirs(self.out_dir, exist_ok=True)
        stem = osp.join(self.out_dir, f"episode_{self.episode:05d}")

        image_file = None
        if self.image_mode == "mp4":
            image_file = osp.basename(stem) + ".mp4"
            self._write_video(stem + ".mp4")
        elif self.image_mode == "npz":
            image_file = osp.basename(stem) + ".npz"

        # `first_success` follows `eval_utils.stat_first`: 0-based index into the
        # per-step list, -1 when the episode never succeeded.
        own_first_success = next((i for i, ok in enumerate(self.successes) if ok), -1)
        if first_success is not None and int(first_success) != own_first_success:
            raise AssertionError(f"first_success mismatch: evaluate.py={first_success} dumper={own_first_success}")

        arrays = {
            "proprio": np.stack(self.proprios).astype(np.float32),
            "action": np.stack(self.commands).astype(np.float64),
            "world_vector": np.stack(self.world_vectors),
            "rot_axangle": np.stack(self.rot_axangles),
            "gripper": np.stack(self.grippers),
            "reward": np.asarray(self.rewards, dtype=np.float32),
            "success": np.asarray(self.successes, dtype=bool),
            "truncated": np.asarray(self.truncateds, dtype=bool),
            "query_index": np.asarray(self.query_indices, dtype=np.int32),
            "query_step": np.asarray(self.query_step, dtype=np.int32),
            "query_instruction_id": np.asarray(self.query_instruction_id, dtype=np.int32),
            "raw_chunk": np.stack(self.raw_chunks).astype(np.float32),
            "instruction_id": np.asarray(self.instruction_ids, dtype=np.int32),
            "instruction_vocab": np.asarray(self.instruction_vocab, dtype=f"<U{MAX_INSTRUCTION_CHARS}"),
        }
        if self.image_mode == "npz":
            arrays["images"] = np.stack(self.images)

        meta = {
            "dumper_version": DUMPER_VERSION,
            "task": self.task,
            "episode": self.episode,
            "robot_type": self.robot_type,
            "camera": self.camera_name,
            "method": self.method,
            "result_dir": self.result_dir,
            "fps": self.fps,
            "steps": n_steps,
            "n_queries": len(self.raw_chunks),
            "horizon_steps": int(self.raw_chunks[0].shape[0]),
            "success": bool(own_first_success >= 0),
            "first_success": own_first_success,
            "image_mode": self.image_mode,
            "image_file": image_file,
            "image_shape": list(self.frame_shape) if self.frame_shape is not None else None,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        arrays["meta"] = np.asarray(json.dumps(meta))
        np.savez(stem + ".npz", **arrays)

        return {
            "episode": self.episode,
            "success": meta["success"],
            "first_success": own_first_success,
            "steps": n_steps,
            "n_queries": meta["n_queries"],
            "dump_npz": osp.basename(stem) + ".npz",
            "dump_image": image_file,
        }
