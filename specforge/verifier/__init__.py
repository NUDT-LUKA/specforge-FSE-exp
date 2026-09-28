"""验证器适配层。"""

from .base import Failure, VerifyResult, Verifier
from .dafny import DafnyVerifier

__all__ = [
    "Failure",
    "VerifyResult",
    "Verifier",
    "DafnyVerifier",
]
