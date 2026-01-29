"""Tests for the auth module."""

from amla_sandbox.auth import EphemeralAuthority, PCA


class TestEphemeralAuthority:
    """Tests for EphemeralAuthority class."""

    def test_create_authority(self) -> None:
        """EphemeralAuthority can be created."""
        authority = EphemeralAuthority()
        assert authority.public_key_hex().startswith("ed25519:")
        assert len(authority.public_key_bytes()) == 32

    def test_from_seed_deterministic(self) -> None:
        """EphemeralAuthority.from_seed produces deterministic keys."""
        seed = b"x" * 32
        auth1 = EphemeralAuthority.from_seed(seed)
        auth2 = EphemeralAuthority.from_seed(seed)
        assert auth1.public_key_hex() == auth2.public_key_hex()

    def test_create_pca(self) -> None:
        """EphemeralAuthority can create a signed PCA."""
        authority = EphemeralAuthority()
        pca = authority.create_pca(capabilities=["tool_call:**"])
        assert isinstance(pca, PCA)
        assert len(pca.to_cbor()) > 0


class TestRuntimeWithAuth:
    """Tests for Runtime with real PCA authentication."""

    def test_runtime_for_testing(self) -> None:
        """Runtime.for_testing() creates a working runtime."""
        from amla_sandbox.runtime.wasm import Runtime

        runtime = Runtime.for_testing()
        assert runtime is not None

    def test_runtime_with_ephemeral_authority(self) -> None:
        """Runtime can be created with EphemeralAuthority PCA."""
        from amla_sandbox.runtime.wasm import Runtime, RuntimeConfig

        authority = EphemeralAuthority()
        pca = authority.create_pca(capabilities=["tool_call:**"])

        config = RuntimeConfig(
            pca_bytes=pca.to_cbor(),
            trusted_authorities=[authority.public_key_hex()],
        )
        runtime = Runtime(config)
        assert runtime is not None
