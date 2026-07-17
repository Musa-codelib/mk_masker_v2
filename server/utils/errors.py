"""Typed error categories for Mk Masker.

Raising an MkError (instead of a bare Exception) lets the frontend show a precise,
copy-pasteable `code` + `message` + `detail`. This is the runtime diagnostics channel
you can paste straight back into an AI editor to report exactly what failed.
"""

from typing import Optional


class MkError(Exception):
    """Base typed error. Always carries a stable `code` string.

    Every subclass overrides `code` with a unique, machine-readable identifier so the
    frontend can display it and the user can relay it verbatim to a coding assistant.
    """

    code: str = "UNKNOWN"

    def __init__(self, message: str, detail: Optional[str] = None):
        # `message` is what the user sees; `detail` is the raw trace/stderr for debugging.
        super().__init__(message)
        self.user_message = message
        self.detail = detail

    def to_payload(self) -> dict:
        """Serialize into the SocketIO `error_alert` payload sent to the renderer."""
        return {
            "code": self.code,
            "message": self.user_message,
            "detail": self.detail,
        }


# --- Specific error types (one per failure category) ---

# Model weights file (.pt / .pth) could not be found on disk.
class WeightsMissingError(MkError):
    code = "WEIGHTS_MISSING"


# No ffmpeg binary available (system or bundled in bin/).
class FFmpegMissingError(MkError):
    code = "FFMPEG_MISSING"


# ffmpeg ran but exited with a non-zero status (capture stderr into detail).
class FFmpegFailedError(MkError):
    code = "FFMPEG_FAILED"


# The source video reported zero readable frames (corrupt / unsupported container).
class EmptyVideoError(MkError):
    code = "EMPTY_VIDEO"


# cv2.VideoCapture could not open the file at all (wrong path / unsupported codec).
class UnsupportedVideoError(MkError):
    code = "UNSUPPORTED_VIDEO"


# SAM2 mode was started without any user click prompts.
class NoSelectionError(MkError):
    code = "NO_SELECTION"


# Expected mask/matte/frame PNG was missing during the export merge step.
class FrameMissingError(MkError):
    code = "FRAME_MISSING"


# A torch model (SAM2 / RVM) failed to load or build.
class EngineLoadError(MkError):
    code = "ENGINE_LOAD_FAILED"


def as_mk_error(exc: Exception) -> MkError:
    """Coerce ANY exception into an MkError for uniform UI handling.

    If it is already an MkError, return it unchanged; otherwise wrap the plain
    exception text in a generic MkError with code "UNKNOWN".
    """
    if isinstance(exc, MkError):
        return exc
    return MkError(message=str(exc), detail=None)
