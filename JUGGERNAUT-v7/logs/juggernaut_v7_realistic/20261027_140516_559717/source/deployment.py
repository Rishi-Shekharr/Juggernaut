"""V7 deployment observation history contract.

Call reset(frame) with the first complete, already-scaled 46-value actor frame
of each episode, then append(frame) once for every control observation. The
returned 230 values are five frames in oldest-to-newest order. The caller
supplies previous leg actions and command-only arm estimates in each frame.
"""
import torch

V7_PIPELINE = "juggernaut_v7_external_arms_history_v4"
FRAME_OBSERVATIONS = 46
HISTORY_FRAMES = 5
ACTOR_OBSERVATIONS = FRAME_OBSERVATIONS * HISTORY_FRAMES


class ObservationHistory:
    def __init__(self, frames=HISTORY_FRAMES):
        if int(frames) != HISTORY_FRAMES:
            raise ValueError("V7 deployment requires exactly five observation frames")
        self.frames = int(frames)
        self._history = None

    @staticmethod
    def _check(frame):
        if not isinstance(frame, torch.Tensor) or frame.ndim not in (1, 2):
            raise ValueError("Actor frame must be a 46-value tensor or a batch of 46-value tensors")
        if frame.shape[-1] != FRAME_OBSERVATIONS:
            raise ValueError(f"Actor frame must contain exactly {FRAME_OBSERVATIONS} values")
        return frame.unsqueeze(0) if frame.ndim == 1 else frame

    def reset(self, first_frame):
        frame = self._check(first_frame)
        self._history = frame.unsqueeze(1).expand(-1, self.frames, -1).clone()
        return self._history.reshape(frame.shape[0], ACTOR_OBSERVATIONS).clone()

    def append(self, frame):
        frame = self._check(frame)
        if self._history is None:
            return self.reset(frame)
        if frame.shape[0] != self._history.shape[0]:
            raise ValueError("Actor frame batch size changed; reset history for the new batch")
        self._history = torch.cat((self._history[:, 1:], frame.unsqueeze(1)), dim=1)
        return self._history.reshape(frame.shape[0], ACTOR_OBSERVATIONS).clone()

    def reset_rows(self, rows, first_frames):
        """Repeat each reset row's first frame without leaking terminal history."""
        frame = self._check(first_frames)
        if self._history is None or frame.shape[0] != len(rows):
            raise ValueError("reset_rows requires initialized history and one frame per selected row")
        row_index = torch.as_tensor(rows, device=self._history.device, dtype=torch.long).reshape(-1)
        self._history[row_index] = frame.unsqueeze(1).expand(-1, self.frames, -1)
        return self._history.reshape(self._history.shape[0], ACTOR_OBSERVATIONS).clone()
