"""Typed error categories for Mk Masker.

Raising an MkError (instead of a bare Exception) lets the frontend show a precise,
copy-pasteable code + message. This is the runtime diagnostics channel you can paste
straight back into an AI editor.
"""

from typing import Optional


class MkError(Exception):
    """Base typed error. Always carries a stable `code` string."""

    code: str = "UNKNOWN"

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(message)
        self.user_message = message
        self.detail = detail

    def to_payload(self) -> dict:
        return {
            "code": self.code,
            "message": self.user_message,
            "detail": self.detail,
        }


class WeightsMissingError(MkError):
    code = "WEIGHTS_MISSING"


class FFmpegMissingError(MkError):
    code = "FFMPEG_MISSING"


class FFmpegFailedError(MkError):
    code = "FFMPEG_FAILED"


class EmptyVideoError(MkError):
    code = "EMPTY_VIDEO"


class UnsupportedVideoError(MkError):
    code = "UNSUPPORTED_VIDEO"


class NoSelectionError(MkError):
    code = "NO_SELECTION"


class FrameMissingError(MkError):
    code = "FRAME_MISSING"


class EngineLoadError(MkError):
    code = "ENGINE_LOAD_FAILED"


def as_mk_error(exc: Exception) -> MkError:
    """Coerce any exception into an MkError payload for the UI."""
    if isinstance(exc, MkError):
        return exc
    return MkError(message=str(exc), detail=None)
