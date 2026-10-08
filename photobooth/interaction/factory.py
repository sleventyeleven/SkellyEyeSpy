from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from ..config import InteractionConfig, PhotoBoothConfig
from .animation import Performer
from .audio import AplayPlayer, SimPlayer
from .director import Director
from .triggers import TriggerWatcher
from .voicebank import VoiceBank

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Interaction:
    director: Director
    bank: VoiceBank
    skeleton: object
    watcher: TriggerWatcher | None
    trigger_input: object | None
    simulated_audio: bool
    simulated_gpio: bool

    def close(self) -> None:
        self.director.stop()
        if self.watcher is not None:
            self.watcher.stop()
        if self.trigger_input is not None:
            self.trigger_input.close()


def resolve_voice_dir(path: str) -> str:
    """Relative voice directories are looked up from the working directory first, then the project root."""
    if os.path.isabs(path) or os.path.isdir(path):
        return path
    return os.path.join(_PROJECT_ROOT, path)


def build_interaction(service, config: PhotoBoothConfig | InteractionConfig, skeleton=None, player=None, trigger_input=None) -> Interaction | None:
    """Wires the voice bank, audio, skeleton and trigger around a booth service.

    Returns None when the interaction is switched off. Hardware is opt-in:
    with use_gpio off the skeleton is simulated (audio only), and without
    `aplay` (or with play_audio off) the audio is simulated too. Pass the
    process's existing `skeleton` to share it: its GPIO lines can only be
    requested once per process.
    """
    cfg = config.interaction if isinstance(config, PhotoBoothConfig) else config
    if not cfg.enabled:
        return None

    from skelly import Skeleton
    from skelly.hardware import SimGpioBackend

    if cfg.use_gpio:
        skeleton = skeleton or Skeleton()
    else:
        skeleton = Skeleton(backend=SimGpioBackend())
    simulated_gpio = isinstance(skeleton.backend, SimGpioBackend)

    simulated_audio = player is None and not (cfg.play_audio and shutil.which("aplay"))
    if player is None:
        player = SimPlayer() if simulated_audio else AplayPlayer(cfg.audio_device)

    bank = VoiceBank.load(resolve_voice_dir(cfg.voice_dir))
    performer = Performer(player, skeleton, cfg)
    director = Director(service, bank, performer, cfg)

    watcher = None
    if cfg.trigger_line is not None:
        if trigger_input is None:
            if cfg.use_gpio:
                from skelly.inputs import PeripheryInput

                trigger_input = PeripheryInput(cfg.trigger_chip, cfg.trigger_line, pull_up=cfg.trigger_active_low)
            else:
                from skelly.inputs import SimInput

                trigger_input = SimInput(level=cfg.trigger_active_low)  # idle level
        kind = cfg.trigger_kind
        watcher = TriggerWatcher(
            trigger_input.read,
            lambda: director.start(skip_invites=(kind == "button"), reason=kind),
            active_low=cfg.trigger_active_low,
            debounce_s=cfg.trigger_debounce_s,
        )
        watcher.start()
    return Interaction(director, bank, skeleton, watcher, trigger_input, simulated_audio, simulated_gpio)
