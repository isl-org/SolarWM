from __future__ import annotations

from pathlib import Path

import pytest

from solarwm.backends.wan22 import create_backend
from solarwm.config import load_config
from solarwm.errors import BackendContractError

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "configs" / "examples"
WAN_TRAINING_EXAMPLES = (
    EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_153f.yaml",
    EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_81f.yaml",
    EXAMPLES / "wan22_ti2v_5b" / "train_stage1_tf_anyflow_v1_5_81f.yaml",
    EXAMPLES / "wan22_ti2v_5b" / "train_stage1_tf_fm_81f.yaml",
    EXAMPLES / "wan22_ti2v_5b" / "train_stage2_sgf_81f.yaml",
    EXAMPLES / "wan22_i2v_a14b" / "train_stage0p5_fm_153f.yaml",
    EXAMPLES / "wan22_i2v_a14b" / "train_stage0p5_fm_81f.yaml",
)


@pytest.mark.parametrize(
    "path",
    sorted((EXAMPLES / "wan22_ti2v_5b").glob("*.yaml"))
    + sorted((EXAMPLES / "wan22_i2v_a14b").glob("*.yaml")),
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_example_config_resolves_and_satisfies_backend_contract(path: Path) -> None:
    resolved = load_config(path)
    family = str(resolved.values["model"]["family"])
    create_backend(family=family).validate_config(resolved.values)


@pytest.mark.parametrize(
    "path",
    WAN_TRAINING_EXAMPLES,
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_training_examples_enable_node_shard_lookahead(path: Path) -> None:
    assert load_config(path).values["data"]["gcs_prefetch_shards"] == 32


def test_negative_gcs_prefetch_shards_is_rejected() -> None:
    config = load_config(WAN_TRAINING_EXAMPLES[0]).mutable_copy()
    config["data"]["gcs_prefetch_shards"] = -1
    with pytest.raises(BackendContractError, match="gcs_prefetch_shards must be non-negative"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage1_training_requires_compiled_flex_attention() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage1_tf_anyflow_v1_5_81f.yaml"
    config = load_config(path).mutable_copy()
    assert config["runtime"]["compile_flex"] is True
    config["runtime"]["compile_flex"] = False
    with pytest.raises(BackendContractError, match=r"runtime\.compile_flex=true"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_camera_inference_defaults_to_circular_kv_cache() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    assert config["inference"]["kv_cache_mode"] == "circular"
    assert config["inference"]["max_rollout_latent_frames"] == 900
    del config["inference"]["kv_cache_mode"]
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_camera_inference_accepts_clone_kv_cache_fallback() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["inference"]["kv_cache_mode"] = "clone"
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_interactive_camera_inference_accepts_one_latent_block() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_interactive.yaml"
    config = load_config(path)
    create_backend(family="wan22_ti2v_5b").validate_config(config.values)


def test_stage2_camera_circular_kv_cache_rejects_sink() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["model"]["sink_size"] = 3
    with pytest.raises(BackendContractError, match="sink_size"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


@pytest.mark.parametrize("maximum", (0, 20))
def test_stage2_camera_inference_requires_chunk_aligned_buffer_capacity(maximum: int) -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["inference"]["max_rollout_latent_frames"] = maximum
    with pytest.raises(BackendContractError, match="max_rollout_latent_frames"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


@pytest.mark.parametrize(
    "kernel",
    (
        "none",
        "reference",
        "fused_rope_prope_sdpa_reference",
        "fused_rope_prope_sdpa",
        "sdpa",
        "fused_sdpa",
        "fused_rope_prope_sage",
        "sage",
        "fused_sage",
    ),
)
def test_stage2_fused_kernel_valid_configs(kernel: str) -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["runtime"]["stage2_fused_kernel"] = kernel
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_fused_kernel_rejects_invalid_kernel() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["runtime"]["stage2_fused_kernel"] = "flash_attention_unsupported"
    with pytest.raises(BackendContractError, match="stage2_fused_kernel"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_radial_attention_accepts_fused_camera_inference() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["runtime"].update(
        {
            "stage2_fused_kernel": "fused_rope_prope_sdpa",
            "stage2_radial_attention": True,
            "stage2_radial_decay": 0.8,
            "stage2_radial_sink_frames": 1,
            "stage2_radial_dense_blocks": 1,
            "stage2_radial_dense_steps": 1,
        }
    )
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_radial_attention_rejects_non_fused_kernel() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["runtime"]["stage2_radial_attention"] = True
    with pytest.raises(BackendContractError, match="requires a fused"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_fused_kernel_build_architecture() -> None:
    from solarwm.backends.wan22.runtime.loader import build_camera_transformer_architecture
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["runtime"]["stage2_fused_kernel"] = "reference"
    model = build_camera_transformer_architecture(config)
    assert model.fused_kernel == "reference"
    assert model.blocks[0].self_attn.fused_kernel == "reference"

    config["runtime"]["stage2_fused_kernel"] = "fused_rope_prope_sdpa"
    model = build_camera_transformer_architecture(config)
    assert model.fused_kernel == "fused_rope_prope_sdpa"
    assert model.blocks[0].self_attn.fused_kernel == "fused_rope_prope_sdpa"

    config["runtime"]["stage2_fused_kernel"] = "fused_rope_prope_sage"
    model = build_camera_transformer_architecture(config)
    assert model.fused_kernel == "fused_rope_prope_sage"
    assert model.blocks[0].self_attn.fused_kernel == "fused_rope_prope_sage"

    config["runtime"].update(
        {
            "stage2_fused_kernel": "fused_rope_prope_sdpa",
            "stage2_radial_attention": True,
            "stage2_radial_decay": 0.7,
            "stage2_radial_sink_frames": 2,
            "stage2_radial_dense_blocks": 1,
            "stage2_radial_dense_steps": 1,
        }
    )
    model = build_camera_transformer_architecture(config)
    assert model.radial_attention is True
    assert model.radial_decay == 0.7
    assert model.blocks[0].self_attn.radial_attention is True


def test_backend_factory_is_strict() -> None:
    with pytest.raises(BackendContractError, match="does not implement"):
        create_backend(family="wan22_future")


def test_executable_runtime_reports_missing_dependencies_before_import() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_81f.yaml"
    config = load_config(path).values
    with pytest.raises(BackendContractError, match="runtime is not ready"):
        create_backend(family="wan22_ti2v_5b").train(config)


def test_sp_peers_must_use_logical_dp_rng_and_sharding() -> None:
    path = EXAMPLES / "wan22_i2v_a14b" / "train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["distributed"]["rng_scope"] = "global_rank"
    with pytest.raises(BackendContractError, match="rng_scope"):
        create_backend(family="wan22_i2v_a14b").validate_config(config)


def test_full_resume_rejects_camera_transform_drift() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_153f.yaml"
    config = load_config(path).mutable_copy()
    config["model"]["camera_translation_transform"] = "logd4"
    with pytest.raises(BackendContractError, match="transform mismatch"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_81f_stage0p5_accepts_full_resume_contract() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["checkpoint"] = {
        "mode": "full_resume",
        "path": "/path/to/checkpoint_model_001500",
        "resume_step": 1500,
        "source_contract": {
            "stage": "stage0p5",
            "objective": "flow_matching",
            "camera_translation_transform": "linear",
        },
    }
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_bucket_transport_preserves_the_same_backend_contract() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["data"]["transport"] = {
        "kind": "gcs",
        "root": "gs://public-example-bucket",
        "cache_dir": "/path/to/cache",
        "cache_max_gib": 512,
    }
    config["data"]["index_root"] = "/path/to/staged-controls"
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_bucket_transport_accepts_a_separate_validation_cache() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["data"]["transport"] = {
        "kind": "gcs",
        "root": "gs://public-example-bucket",
        "cache_dir": "/path/to/train-cache",
        "cache_max_gib": 512,
    }
    config["runtime"]["data_cache_max_gib"] = 4096
    config["runtime"]["validation_cache_dir"] = "/path/to/validation-cache"
    config["runtime"]["validation_cache_max_gib"] = 64
    config["data"]["index_root"] = "/path/to/staged-controls"
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_bucket_transport_rejects_a_partial_validation_cache() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["data"]["transport"] = {
        "kind": "gcs",
        "root": "gs://public-example-bucket",
        "cache_dir": "/path/to/train-cache",
        "cache_max_gib": 512,
    }
    config["runtime"]["data_cache_max_gib"] = 4096
    config["runtime"]["validation_cache_dir"] = "/path/to/validation-cache"
    config["data"]["index_root"] = "/path/to/staged-controls"
    with pytest.raises(BackendContractError, match="configured together"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


@pytest.mark.parametrize(
    "overrides",
    (
        {"data_cache_dir": "/path/to/train-cache"},
        {"data_cache_max_gib": 64},
        {
            "validation_cache_dir": "/path/to/validation-cache",
            "validation_cache_max_gib": 64,
        },
    ),
)
def test_local_transport_rejects_runtime_cache_overrides(overrides: dict[str, object]) -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["runtime"].update(overrides)
    with pytest.raises(BackendContractError, match="only for gcs"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_inference_rejects_nonstandard_checkpoint_format() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["checkpoint"]["format"] = "unsupported_v1"
    with pytest.raises(BackendContractError, match="embedded_config_v1"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


@pytest.mark.parametrize(
    ("family", "filename", "release_name", "weight_role"),
    (
        (
            "wan22_ti2v_5b",
            "infer_stage0p5_fm_81f.yaml",
            "SolarWM-5B-bid-stage0p5-81f",
            "live",
        ),
        (
            "wan22_ti2v_5b",
            "infer_stage0p5_fm_153f.yaml",
            "SolarWM-5B-bid-stage0p5-153f",
            "ema",
        ),
        (
            "wan22_ti2v_5b",
            "infer_stage1_tf_anyflow_v1_5_81f.yaml",
            "SolarWM-5B-tf-stage1-81f",
            "ema",
        ),
        (
            "wan22_ti2v_5b",
            "infer_stage2_sgf_81f.yaml",
            "SolarWM-5B-sgf-stage2-81f",
            "ema",
        ),
        (
            "wan22_i2v_a14b",
            "infer_stage0p5_fm_81f.yaml",
            "SolarWM-14B-bid-stage0p5-81f",
            "live",
        ),
    ),
)
def test_inference_examples_name_the_public_checkpoint_transaction(
    family: str,
    filename: str,
    release_name: str,
    weight_role: str,
) -> None:
    config = load_config(EXAMPLES / family / filename).values
    assert config["checkpoint"] == {
        "path": f"/path/to/SolarWM-models/{release_name}",
        "weights": weight_role,
    }
    pass_roles = [item["weights"] for item in config["validation"]["passes"]]
    if filename == "infer_stage2_sgf_81f.yaml":
        assert pass_roles == ["live", "ema"]
    else:
        assert set(pass_roles) == {weight_role}


def test_stage2_camera_length_inference_selects_one_direct_model() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).values
    assert config["checkpoint"] == {"path": "/path/to/SolarWM-models/SolarWM-5B-sgf-stage2-81f"}
    assert config["inference"] == {
        "source": "validation",
        "length": "camera",
        "kv_cache_mode": "circular",
        "max_rollout_latent_frames": 900,
        "stage2_xpu_vae_pipeline": True,
        "stage2_xpu_compile_blocks": False,
        "output_layout": "dataset_triplet_v1",
        "run_id": "wan22-ti2v-5b-stage2-sgf-camera-length-inference",
    }
    assert config["data"]["random_start"] is False
    assert config["validation"]["max_rel_translation"] is None
    assert config["validation"]["max_camera_abs"] is None
    assert [item["weights"] for item in config["validation"]["passes"]] == ["model"]


def test_stage2_camera_length_inference_rejects_weight_role_selection() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["checkpoint"]["weights"] = "ema"
    with pytest.raises(BackendContractError, match="selects the checkpoint model automatically"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_camera_length_measurements_are_opt_in_and_validated() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["runtime"]["stage2_inference_measurements"] = True
    create_backend(family="wan22_ti2v_5b").validate_config(config)
    config["runtime"]["stage2_inference_global_peak"] = True
    create_backend(family="wan22_ti2v_5b").validate_config(config)
    config["runtime"]["stage2_xpu_profiler"] = True
    create_backend(family="wan22_ti2v_5b").validate_config(config)

    config["runtime"]["stage2_inference_measurements"] = "true"
    with pytest.raises(BackendContractError, match="must be boolean"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_camera_xpu_vae_pipeline_is_opt_out_and_camera_only() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["inference"]["stage2_xpu_vae_pipeline"] = False
    create_backend(family="wan22_ti2v_5b").validate_config(config)

    config["inference"]["stage2_xpu_vae_pipeline"] = "false"
    with pytest.raises(BackendContractError, match="must be boolean"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)

    fixed = load_config(EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_81f.yaml").mutable_copy()
    fixed["inference"]["stage2_xpu_vae_pipeline"] = True
    with pytest.raises(BackendContractError, match="only for standalone Stage2 camera-length"):
        create_backend(family="wan22_ti2v_5b").validate_config(fixed)


def test_stage2_camera_vae_optimizations_are_opt_out_and_camera_only() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    config["runtime"]["stage2_vae_channels_last"] = False
    config["runtime"]["stage2_vae_fused_kernels"] = False
    config["runtime"]["stage2_vae_int8_quarot"] = True
    create_backend(family="wan22_ti2v_5b").validate_config(config)

    config["runtime"]["stage2_vae_int8_quarot"] = "true"
    with pytest.raises(BackendContractError, match="stage2_vae_int8_quarot must be boolean"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)
    config["runtime"]["stage2_vae_int8_quarot"] = False

    config["runtime"]["stage2_vae_channels_last"] = "false"
    with pytest.raises(BackendContractError, match="stage2_vae_channels_last must be boolean"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)

    config["runtime"]["stage2_vae_channels_last"] = True
    config["runtime"]["stage2_vae_fused_kernels"] = "false"
    with pytest.raises(BackendContractError, match="stage2_vae_fused_kernels must be boolean"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)

    fixed = load_config(EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_81f.yaml").mutable_copy()
    fixed["runtime"]["stage2_vae_channels_last"] = False
    with pytest.raises(BackendContractError, match="only for standalone Stage2 camera-length"):
        create_backend(family="wan22_ti2v_5b").validate_config(fixed)

    fixed["runtime"]["stage2_vae_channels_last"] = True
    fixed["runtime"]["stage2_vae_fused_kernels"] = False
    with pytest.raises(BackendContractError, match="only for standalone Stage2 camera-length"):
        create_backend(family="wan22_ti2v_5b").validate_config(fixed)

    fixed["runtime"]["stage2_vae_fused_kernels"] = True
    fixed["runtime"]["stage2_vae_int8_quarot"] = True
    with pytest.raises(BackendContractError, match="only for standalone Stage2 camera-length"):
        create_backend(family="wan22_ti2v_5b").validate_config(fixed)


def test_stage2_camera_xpu_block_compile_is_opt_in_and_camera_only() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_camera_length.yaml"
    config = load_config(path).mutable_copy()
    assert config["inference"]["stage2_xpu_compile_blocks"] is False
    config["inference"]["stage2_xpu_compile_blocks"] = True
    create_backend(family="wan22_ti2v_5b").validate_config(config)

    config["inference"]["stage2_xpu_compile_blocks"] = "true"
    with pytest.raises(BackendContractError, match="must be boolean"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)

    fixed = load_config(EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_81f.yaml").mutable_copy()
    fixed["inference"]["stage2_xpu_compile_blocks"] = True
    with pytest.raises(BackendContractError, match="only for standalone Stage2 camera-length"):
        create_backend(family="wan22_ti2v_5b").validate_config(fixed)


def test_stage2_measurements_reject_non_camera_or_training_routes() -> None:
    config = load_config(
        EXAMPLES / "wan22_ti2v_5b" / "infer_stage2_sgf_81f.yaml"
    ).mutable_copy()
    config["runtime"]["stage2_inference_measurements"] = True
    with pytest.raises(BackendContractError, match="only for standalone Stage2 camera-length"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_ti2v_720p_preencoded_stage0p5_profile_is_supported() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_153f.yaml"
    config = load_config(path).mutable_copy()
    config["model"]["frame_sequence_length"] = 880
    config["data"].update(
        preencode_schema="solarwm.wan22_ti2v_5b.720p.153f.v1",
        latent_shape=[39, 48, 44, 80],
        height=704,
        width=1280,
    )
    config["checkpoint"].update(
        mode="weights_only",
        weights=["ema", "ema"],
        source_step=10000,
    )
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_ti2v_480p_preencoded_81f_profile_is_supported() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["data"].update(
        encoding="preencoded",
        preencode_schema="solarwm.wan22_ti2v_5b.480p.81f.v1",
        preencode_window_assignment="materialized_index_v1",
        train_index=("recipes/clean-81f/latent-wds/wan22-ti2v5b-81f-480p-v1/train-index.jsonl.gz"),
        random_start=False,
    )
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_ti2v_480p_preencoded_81f_dispatch_is_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import readiness, stage0p5

    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["data"].update(
        encoding="preencoded",
        preencode_schema="solarwm.wan22_ti2v_5b.480p.81f.v1",
        preencode_window_assignment="materialized_index_v1",
        random_start=False,
    )

    class Ready:
        def require_ready(self) -> None:
            return None

    monkeypatch.setattr(readiness, "probe_runtime", lambda *args, **kwargs: Ready())
    monkeypatch.setattr(stage0p5, "run_stage0p5_training", lambda _: 19)
    assert create_backend(family="wan22_ti2v_5b").train(config) == 19


def test_stage2_can_initialize_logd4_teacher_and_critic_from_ema() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage2_sgf_81f.yaml"
    config = load_config(path).mutable_copy()
    config["model"]["camera_translation_transform"] = "logd4"
    for role in config["checkpoint"]["roles"].values():
        role["camera_translation_transform"] = "logd4"
    for role in ("teacher", "critic"):
        config["checkpoint"]["roles"][role]["weights"] = "ema"
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_short_training_can_validate_live_before_ema_exists() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage2_sgf_81f.yaml"
    config = load_config(path).mutable_copy()
    config["validation"]["passes"] = [config["validation"]["passes"][0]]
    create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_stage2_role_config_rejects_unsupported_fields() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage2_sgf_81f.yaml"
    config = load_config(path).mutable_copy()
    roles = config["checkpoint"]["roles"]
    create_backend(family="wan22_ti2v_5b").validate_config(config)

    roles["student"]["unknown_field"] = "value"
    with pytest.raises(BackendContractError, match="unsupported fields"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_guided_anyflow_requires_base_relative_embedding() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage1_tf_anyflow_v1_5_81f.yaml"
    config = load_config(path).mutable_copy()
    assert config["train"]["anyflow_negative_embedding"] == ("conditioning/wan_negemb_cn.pth")
    create_backend(family="wan22_ti2v_5b").validate_config(config)

    config["train"]["anyflow_negative_embedding"] = "/repo/assets/deleted.pth"
    with pytest.raises(BackendContractError, match="base-model-relative"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_guided_anyflow_rejects_removed_embedding_digest_field() -> None:
    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage1_tf_anyflow_v1_5_81f.yaml"
    config = load_config(path).mutable_copy()
    config["train"]["anyflow_negative_embedding_digest"] = "ignored"
    with pytest.raises(BackendContractError, match="is not supported"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_153f_cached_recipe_can_select_online_codec_without_schema_drift() -> None:
    path = EXAMPLES / "wan22_i2v_a14b" / "train_stage0p5_fm_153f.yaml"
    config = load_config(path).mutable_copy()
    config["data"]["encoding"] = "online"
    create_backend(family="wan22_i2v_a14b").validate_config(config)


def test_ti2v_153f_resume_can_select_the_online_codec_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import readiness, stage0p5

    path = EXAMPLES / "wan22_ti2v_5b" / "train_stage0p5_fm_153f.yaml"
    config = load_config(path).mutable_copy()
    config["data"]["encoding"] = "online"
    create_backend(family="wan22_ti2v_5b").validate_config(config)

    class Ready:
        def require_ready(self) -> None:
            return None

    monkeypatch.setattr(readiness, "probe_runtime", lambda *args, **kwargs: Ready())
    monkeypatch.setattr(stage0p5, "run_stage0p5_training", lambda _: 17)
    assert create_backend(family="wan22_ti2v_5b").train(config) == 17
