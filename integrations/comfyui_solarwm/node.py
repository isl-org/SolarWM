"""Declarative ComfyUI node for opening the SolarWM interactive viewer."""

from __future__ import annotations

import json
import os


class SolarWMInteractive:
    """Create a viewer configuration without owning model/session state."""

    _ATTENTION_BACKENDS = ["sdpa", "sageattnv1"]
    _LINEAR_QUANTS = [
        "full_precision",
        "svdquant_k1",
        "svdquant_k2",
        "svdquant_k4",
    ]
    _VAE_QUANTS = ["full_precision", "quarot_w8a8"]
    _CAMERA_MOTIONS = [
        "straight_about_turn_10s",
        "circle_twice_10s",
        "manual_only",
    ]
    _DISPLAY_SCALES = ["1x", "2x", "fit"]
    _DISPLAY_FPS = [4, 8, 12, 16, 24, 30]

    @staticmethod
    def _image_options() -> tuple[list[str], dict[str, object]]:
        """Expose ComfyUI's uploadable image combo with a safe local fallback."""
        default = "assets/eastshade.webp"
        try:
            import folder_paths

            files = folder_paths.filter_files_content_types(
                os.listdir(folder_paths.get_input_directory()), ["image"]
            )
            options = sorted(files)
        except (ImportError, OSError, AttributeError):
            options = []
        if default not in options:
            options.insert(0, default)
        return options, {"default": default, "image_upload": True}

    @classmethod
    def INPUT_TYPES(cls):
        image_options, image_options_config = cls._image_options()
        return {
            "required": {
                "config_path": (
                    "STRING",
                    {
                        "default": (
                            "configs/examples/wan22_ti2v_5b/"
                            "infer_stage2_sgf_interactive.yaml"
                        )
                    },
                ),
                "initial_image": (
                    image_options,
                    {
                        **image_options_config,
                        "tooltip": "Upload or choose the local anchor image",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": (
                            "Walk slowly straight ahead for 4 seconds, take an about turn, "
                            "and walk back."
                        ),
                        "multiline": True,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 2**63 - 1,
                        "control_after_generate": True,
                    },
                ),
                "max_latent_frames": ("INT", {"default": 120, "min": 2, "max": 900}),
                "attention_backend": (
                    "COMBO",
                    {"options": cls._ATTENTION_BACKENDS, "default": "sdpa"},
                ),
                "linear_quant": (
                    "COMBO",
                    {"options": cls._LINEAR_QUANTS, "default": "full_precision"},
                ),
                "svdquant_weights_root": (
                    "STRING",
                    {
                        "default": (
                            "models/SolarWM-5B-sgf-stage2-81f/"
                            "svdquant_int4"
                        ),
                        "tooltip": "Root containing <attention_backend>_<k> SVDQuant weights",
                    },
                ),
                "stage2_vae_quant": (
                    "COMBO",
                    {"options": cls._VAE_QUANTS, "default": "full_precision"},
                ),
                "camera_motion": (
                    "COMBO",
                    {"options": cls._CAMERA_MOTIONS, "default": "straight_about_turn_10s"},
                ),
                "translation_speed": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 20.0}),
                "rotation_speed": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 6.28}),
                "display_scale": (
                    "COMBO",
                    {"options": cls._DISPLAY_SCALES, "default": "2x"},
                ),
                "display_fps": (
                    "COMBO",
                    {"options": cls._DISPLAY_FPS, "default": 8},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SOLARWM_SESSION_CONFIG",)
    FUNCTION = "build_config"
    CATEGORY = "SolarWM/Interactive"
    OUTPUT_NODE = True

    def build_config(
        self,
        config_path: str,
        initial_image: str,
        prompt: str,
        seed: int,
        max_latent_frames: int,
        attention_backend: str,
        linear_quant: str,
        svdquant_weights_root: str,
        stage2_vae_quant: str,
        camera_motion: str = "straight_about_turn_10s",
        translation_speed: float = 0.1,
        rotation_speed: float = 0.1,
        display_scale: str = "2x",
        display_fps: int = 8,
        control_after_generate: str = "fixed",
        **kwargs,
    ):
        payload = {
            "schema": "solarwm.comfyui.interactive-config.v1",
            "config_path": str(config_path),
            "initial_image": str(initial_image),
            "prompt": str(prompt),
            "seed": int(seed),
            "control_after_generate": str(control_after_generate),
            "max_latent_frames": int(max_latent_frames),
            "attention_backend": str(attention_backend),
            "linear_quant": str(linear_quant),
            "svdquant_weights_root": str(svdquant_weights_root),
            "stage2_vae_quant": str(stage2_vae_quant),
            "camera_motion": str(camera_motion),
            "translation_speed": float(translation_speed),
            "rotation_speed": float(rotation_speed),
            "display_scale": str(display_scale),
            "display_fps": int(display_fps),
            "controls": "keyboard-free-fly-v1",
        }
        return (json.dumps(payload, sort_keys=True),)


NODE_CLASS_MAPPINGS = {"SolarWMInteractive": SolarWMInteractive}
NODE_DISPLAY_NAME_MAPPINGS = {"SolarWMInteractive": "SolarWM Interactive Camera"}
