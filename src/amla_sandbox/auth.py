"""Authentication and PCA creation for Amla sandbox.

This module provides utilities for creating test authorities and PCAs
for development and testing. The same code paths are used for tests
and production - the only difference is where keys come from.

For testing::

    from amla_sandbox.auth import EphemeralAuthority

    authority = EphemeralAuthority()  # Generates ephemeral keypair
    pca = authority.create_pca(capabilities=["tool_call:**"])

For production, load keys from secure storage and use the Authority class.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Any, cast


# =============================================================================
# Canonical CBOR Encoding (matching Rust's ciborium)
# =============================================================================
# Canonical CBOR requires:
# 1. Deterministic integer encoding (shortest form)
# 2. Map keys sorted by encoded length, then lexicographically
# =============================================================================


def _cbor_encode_uint(major_type: int, n: int) -> bytes:
    """Encode unsigned integer with major type in CBOR (canonical form)."""
    if n < 24:
        return bytes([major_type | n])
    elif n < 256:
        return bytes([major_type | 24, n])
    elif n < 65536:
        return bytes([major_type | 25]) + struct.pack(">H", n)
    elif n < 4294967296:
        return bytes([major_type | 26]) + struct.pack(">I", n)
    else:
        return bytes([major_type | 27]) + struct.pack(">Q", n)


def _cbor_encode_text(s: str) -> bytes:
    """Encode text string in CBOR."""
    data = s.encode("utf-8")
    return _cbor_encode_uint(0x60, len(data)) + data


def _cbor_encode_bytes(data: bytes) -> bytes:
    """Encode byte string in CBOR."""
    return _cbor_encode_uint(0x40, len(data)) + data


def _cbor_encode_array(items: list[bytes]) -> bytes:
    """Encode array in CBOR (items are already encoded)."""
    result = _cbor_encode_uint(0x80, len(items))
    for item in items:
        result += item
    return result


def _cbor_encode_map(pairs: list[tuple[bytes, bytes]]) -> bytes:
    """Encode map in CBOR preserving insertion order.

    Note: Rust's ciborium uses struct field order for serialization,
    NOT canonical CBOR order. We must match this exactly for signature
    verification to work.
    """
    # DO NOT sort - preserve insertion order to match Rust's serde behavior
    result = _cbor_encode_uint(0xA0, len(pairs))
    for key, value in pairs:
        result += key + value
    return result


def _cbor_encode_value(value: Any) -> bytes:
    """Encode a Python value to CBOR (canonical form)."""
    if isinstance(value, bool):
        return bytes([0xF5 if value else 0xF4])
    elif isinstance(value, int):
        if value >= 0:
            return _cbor_encode_uint(0x00, value)
        else:
            return _cbor_encode_uint(0x20, -1 - value)
    elif isinstance(value, str):
        return _cbor_encode_text(value)
    elif isinstance(value, bytes):
        return _cbor_encode_bytes(value)
    elif isinstance(value, list):
        items = [_cbor_encode_value(item) for item in cast(list[Any], value)]
        return _cbor_encode_array(items)
    elif isinstance(value, tuple):
        items = [_cbor_encode_value(item) for item in cast(tuple[Any, ...], value)]
        return _cbor_encode_array(items)
    elif isinstance(value, dict):
        pairs = [
            (_cbor_encode_text(k), _cbor_encode_value(v))
            for k, v in cast(dict[str, Any], value).items()
        ]
        return _cbor_encode_map(pairs)
    elif value is None:
        return bytes([0xF6])
    else:
        raise ValueError(f"Cannot encode {type(value)} to CBOR")


# =============================================================================
# PCA Data Classes
# =============================================================================


@dataclass
class PCA:
    """Policy/Capability Authority token.

    A PCA is a signed token that grants capabilities to an agent.
    """

    cbor_bytes: bytes = field(repr=False)
    """Serialized CBOR representation."""

    def to_cbor(self) -> bytes:
        """Get the CBOR-encoded PCA."""
        return self.cbor_bytes


@dataclass
class EphemeralAuthority:
    """Ephemeral authority for development and testing.

    Creates an Ed25519 keypair that can sign PCAs. The keypair is
    generated fresh each time, making tests isolated.

    Example::

        authority = EphemeralAuthority()
        pca = authority.create_pca(capabilities=["tool_call:**"])

        # Pass to runtime
        runtime = Runtime(RuntimeConfig(
            pca_bytes=pca.to_cbor(),
            trusted_authorities=[authority.public_key_hex()],
        ))
    """

    _private_key: bytes = field(repr=False, default=b"")
    _public_key: bytes = field(repr=False, default=b"")
    _signing_key: Any = field(repr=False, default=None)

    def __post_init__(self) -> None:
        """Generate keypair if not provided."""
        if not self._private_key:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PrivateKey,
                )

                private_key = Ed25519PrivateKey.generate()
                self._signing_key = private_key
                self._private_key = private_key.private_bytes_raw()
                self._public_key = private_key.public_key().public_bytes_raw()
            except ImportError:
                try:
                    from nacl.signing import SigningKey  # type: ignore[import-not-found]

                    # nacl has no type stubs - silence all type warnings for this block
                    signing_key: Any = SigningKey.generate()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                    self._signing_key = signing_key
                    self._private_key = bytes(signing_key)  # pyright: ignore[reportUnknownArgumentType]
                    self._public_key = bytes(signing_key.verify_key)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                except ImportError:
                    raise ImportError(
                        "Either 'cryptography' or 'pynacl' is required for EphemeralAuthority.\n"
                        "Install with: pip install cryptography"
                    )

    @classmethod
    def from_seed(cls, seed: bytes) -> "EphemeralAuthority":
        """Create a test authority from a 32-byte seed.

        This is useful for deterministic testing.

        Args:
            seed: 32-byte seed for key derivation.

        Returns:
            EphemeralAuthority with deterministic keypair.
        """
        if len(seed) != 32:
            raise ValueError("Seed must be exactly 32 bytes")

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            private_key = Ed25519PrivateKey.from_private_bytes(seed)
            auth = cls.__new__(cls)
            auth._signing_key = private_key
            auth._private_key = seed
            auth._public_key = private_key.public_key().public_bytes_raw()
            return auth
        except ImportError:
            try:
                from nacl.signing import SigningKey  # type: ignore[import-not-found]

                # nacl has no type stubs - silence all type warnings for this block
                signing_key: Any = SigningKey(seed)  # pyright: ignore[reportUnknownVariableType]
                auth = cls.__new__(cls)
                auth._signing_key = signing_key
                auth._private_key = seed
                auth._public_key = bytes(signing_key.verify_key)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                return auth
            except ImportError:
                raise ImportError(
                    "Either 'cryptography' or 'pynacl' is required.\n"
                    "Install with: pip install cryptography"
                )

    def public_key_bytes(self) -> bytes:
        """Get the raw public key bytes (32 bytes for Ed25519)."""
        return self._public_key

    def public_key_hex(self) -> str:
        """Get the public key in hex format with algorithm prefix.

        Returns:
            String like "ed25519:abc123..."
        """
        return f"ed25519:{self._public_key.hex()}"

    def _sign(self, message: bytes) -> bytes:
        """Sign a message with the private key.

        Returns:
            64-byte Ed25519 signature.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )

            if isinstance(self._signing_key, Ed25519PrivateKey):
                return self._signing_key.sign(message)
        except ImportError:
            pass

        try:
            from nacl.signing import SigningKey  # type: ignore[import-not-found]

            if isinstance(self._signing_key, SigningKey):
                signed: Any = self._signing_key.sign(message)
                return signed.signature
        except ImportError:
            pass

        raise RuntimeError("No signing implementation available")

    def create_pca(
        self,
        capabilities: list[str] | None = None,
        expires_in_secs: int = 3600,
        designated_executor: str | None = None,
    ) -> PCA:
        """Create a signed PCA with the specified capabilities.

        Args:
            capabilities: List of capability patterns like "tool_call:**".
                Defaults to ["tool_call:**"] (allow all tool calls).
            expires_in_secs: Time until expiry in seconds. Default 1 hour.
            designated_executor: Public key of designated executor (ed25519:hex).
                Defaults to issuer's public key (self-signed).

        Returns:
            Signed PCA ready for use with Runtime.

        Example::

            # Allow all tool calls
            pca = authority.create_pca()

            # Only allow specific tools
            pca = authority.create_pca(capabilities=[
                "tool_call:stripe/**",
                "tool_call:notion/search",
            ])
        """
        if capabilities is None:
            capabilities = ["tool_call:**"]

        # Calculate expiry timestamp (RFC3339 string matching Rust's chrono format)
        # Rust uses +00:00 instead of Z for UTC timezone
        expires_at = time.time() + expires_in_secs
        expires_str = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(expires_at))

        # Issuer is the signing key in "ed25519:hex" format
        issuer_str = self.public_key_hex()

        # Designated executor (default to issuer for self-signed test PCAs)
        if designated_executor is None:
            designated_executor = issuer_str

        # Create capability data matching Rust's CapabilityData
        cap_list: list[dict[str, Any]] = []
        for i, cap_pattern in enumerate(capabilities):
            if cap_pattern.startswith("tool_call:"):
                cap_type = "tool-call"
                pattern = cap_pattern.removeprefix("tool_call:")
            else:
                cap_type = "tool-call"
                pattern = cap_pattern

            # Payload is CBOR-encoded ToolCallCap {tool: pattern, constraints: []}
            # The constraints field is required by Rust's serde deserialization
            payload_cbor = _cbor_encode_value({"tool": pattern, "constraints": []})

            # CapabilityData field order matches Rust struct:
            # key, capability_type (renamed to "type"), data
            cap_list.append(
                {
                    "key": f"cap:{i}",
                    "type": cap_type,
                    "data": payload_cbor,  # CBOR bytes
                }
            )

        # Parse designated executor public key
        # Rust serializes PublicKey as [algorithm_u8, raw_bytes]
        if designated_executor.startswith("ed25519:"):
            executor_key_bytes = bytes.fromhex(
                designated_executor.removeprefix("ed25519:")
            )
            executor_wire = (0, executor_key_bytes)  # 0 = Ed25519
        else:
            raise ValueError(f"Unknown key algorithm in: {designated_executor}")

        # Build the signable content (matches Rust's PcaSignable)
        # Note: Rust uses indefinite-length CBOR maps, but we use definite-length.
        # The ciborium library should handle either format.
        signable = {
            "version": (0, 1),  # Order doesn't matter for maps in CBOR
            "capabilities": cap_list,
            "designated_executor": {
                "type": "pubkey",
                "value": executor_wire,  # [algo_u8, key_bytes], NOT string
            },
            "expires_at": expires_str,
            "issuer": issuer_str,  # Issuer IS a string (different from PublicKey)
        }

        # Encode signable to CBOR for signing
        signable_cbor = _cbor_encode_value(signable)

        # Sign the signable content
        signature = self._sign(signable_cbor)
        # Rust stores signature as hex string with algorithm prefix
        signature_str = f"ed25519:{signature.hex()}"

        # Build full PCA wire format (signable + signature)
        # The signature is flattened into the top-level map
        pca_wire = {
            "version": (0, 1),
            "capabilities": cap_list,
            "designated_executor": {
                "type": "pubkey",
                "value": executor_wire,
            },
            "expires_at": expires_str,
            "issuer": issuer_str,
            "signature": signature_str,
        }

        # Encode full PCA to CBOR
        full_cbor = _cbor_encode_value(pca_wire)

        return PCA(cbor_bytes=full_cbor)


__all__ = [
    "EphemeralAuthority",
    "PCA",
]

# Backwards compatibility alias
TestAuthority = EphemeralAuthority
