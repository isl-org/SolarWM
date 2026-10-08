"""Declarative ComfyUI node for opening the SolarWM interactive viewer."""

from __future__ import annotations

import json


class SolarWMInteractive:
    """Create a viewer configuration without owning model/session state."""

    @classmethod
    def INPUT_TYPES(cls):
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
                "initial_image": ("STRING", {"default": "", "tooltip": "Local anchor image path"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**63 - 1}),
                "max_latent_frames": ("INT", {"default": 120, "min": 2, "max": 900}),
                "translation_speed": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0}),
                "rotation_speed": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 6.28}),
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
        translation_speed: float,
        rotation_speed: float,
    ):
        payload = {
            "schema": "solarwm.comfyui.interactive-config.v1",
            "config_path": str(config_path),
            "initial_image": str(initial_image),
            "prompt": str(prompt),
            "seed": int(seed),
            "max_latent_frames": int(max_latent_frames),
            "translation_speed": float(translation_speed),
            "rotation_speed": float(rotation_speed),
            "controls": "keyboard-free-fly-v1",
        }
        return (json.dumps(payload, sort_keys=True),)


NODE_CLASS_MAPPINGS = {"SolarWMInteractive": SolarWMInteractive}
NODE_DISPLAY_NAME_MAPPINGS = {"SolarWMInteractive": "SolarWM Interactive Camera"}
