import json
import random
import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from flask import Flask

from photobooth.config import InteractionConfig, PhotoBoothConfig
from photobooth.interaction import voicefx
from photobooth.interaction.animation import Performer, plan_eyes, plan_mouth
from photobooth.interaction.audio import AplayPlayer, SimPlayer
from photobooth.interaction.director import Director
from photobooth.interaction.factory import Interaction, build_interaction
from photobooth.interaction.triggers import TriggerWatcher
from photobooth.interaction.voicebank import ENVELOPE_HOP_S, Line, VoiceBank, loudness_envelope
from photobooth.web import create_blueprint
from skelly.hardware import SimGpioBackend, Skeleton


# ---------- fakes ----------

class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class RecordingBackend(SimGpioBackend):
    def __init__(self):
        super().__init__()
        self.writes = []

    def write(self, line, value):
        self.writes.append((line, value))
        super().write(line, value)


def make_line(line_id, category, seconds, loud=True):
    count = max(1, int(seconds / ENVELOPE_HOP_S))
    envelope = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0], np.float32), count // 5 + 1)[:count] if loud else np.zeros(count, np.float32)
    return Line(line_id, category, line_id, f"{line_id}.wav", seconds, envelope)


def make_bank():
    return VoiceBank([
        make_line("invite_1", "invite", 3.0), make_line("welcome_1", "welcome", 3.0), make_line("filler_1", "filler", 2.0),
        make_line("three", "three", 0.5), make_line("two", "two", 0.5), make_line("one", "one", 0.5), make_line("got_it", "got_it", 0.5),
        make_line("printing_1", "printing", 1.0), make_line("done_1", "done", 1.0),
        make_line("nobody_1", "nobody", 1.0), make_line("come_back_1", "come_back", 1.0),
    ])


class FakeSession:
    def __init__(self, clock, people_at, capture_people=1):
        self.clock, self.people_at, self.capture_people = clock, people_at, capture_people
        self.closed = False
        self.captured_at = None
        self.looks = 0

    def people(self):
        self.looks += 1
        return self.people_at(self.clock())

    def capture(self):
        self.captured_at = self.clock()
        return {"people": self.capture_people, "ignored": 0, "person_percent": 20.0}

    def close(self):
        self.closed = True


class FakeService:
    draft_stamp = 0

    def __init__(self, session):
        self.session = session
        self.rendered = 0
        self.printed = 0
        self.print_error = None

    def open_visitor_session(self):
        return self.session

    def draft_render(self, **_):
        self.rendered += 1
        self.draft_stamp += 1
        return {}

    def draft_print(self):
        self.printed += 1
        return SimpleNamespace(printed=self.print_error is None, error=self.print_error)


def fast_cfg(**overrides):
    cfg = InteractionConfig(enabled=True, cooldown_s=15.0)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_director(people_at, cfg=None, capture_people=1, stop_event=None):
    clock = FakeClock()
    cfg = cfg or fast_cfg()
    backend = RecordingBackend()
    skeleton = Skeleton(backend=backend)
    player = SimPlayer(clock=clock, sleep=clock.sleep)
    stop = stop_event or threading.Event()
    performer = Performer(player, skeleton, cfg, clock=clock, sleep=clock.sleep, stop_event=stop)
    session = FakeSession(clock, people_at, capture_people)
    service = FakeService(session)
    director = Director(service, make_bank(), performer, cfg, stop_event=stop, clock=clock, rng=random.Random(1))
    return SimpleNamespace(director=director, clock=clock, player=player, backend=backend, skeleton=skeleton,
                           session=session, service=service, cfg=cfg, stop=stop)


def played(rig):
    return [name.removesuffix(".wav") for name, _t, _v in rig.player.plays]


# ---------- voice effects ----------

def test_tighten_pauses_caps_long_silences_and_keeps_speech():
    sr = 16000
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(sr // 2) * 0.3).astype(np.float32)
    x = np.concatenate([speech, np.zeros(sr), speech])  # a 1 s gap in the middle
    out = voicefx.tighten_pauses(x, sr, max_pause_s=0.28)
    assert len(x) - len(out) == pytest.approx(0.72 * sr, abs=0.03 * sr)
    assert np.allclose(out[: len(speech) // 2], speech[: len(speech) // 2])  # speech untouched
    assert np.abs(out[-len(speech):]).max() > 0.1


def test_tighten_pauses_leaves_short_gaps_and_edges_alone():
    sr = 16000
    speech = np.full(sr // 4, 0.3, np.float32)
    x = np.concatenate([np.zeros(sr), speech, np.zeros(sr // 10), speech, np.zeros(sr)])
    out = voicefx.tighten_pauses(x, sr)
    assert len(out) == len(x)


@pytest.mark.parametrize("name", sorted(voicefx.PRESETS))
def test_every_preset_is_finite_and_normalized(name):
    sr = 16000
    t = np.arange(sr) / sr
    clip = (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    out = voicefx.apply_preset(clip, sr, name)
    assert np.isfinite(out).all() and 0.5 < np.abs(out).max() <= 0.75


def test_tomb_is_milder_than_crypt_and_stronger_than_ghoul():
    ghoul, tomb, crypt = (voicefx.PRESETS[n] for n in ("ghoul", "tomb", "crypt"))
    assert ghoul["semitones"] > tomb["semitones"] > crypt["semitones"]
    assert ghoul["lowpass"] > tomb["lowpass"] > crypt["lowpass"]


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError):
        voicefx.apply_preset(np.zeros(100, np.float32), 16000, "nope")


# ---------- voice bank ----------

def write_wav(path, seconds=0.5, rate=16000, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    pcm = (amp * np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def test_voicebank_loads_manifest_and_picks_without_repeating(tmp_path):
    for name in ("a", "b", "c"):
        write_wav(tmp_path / f"{name}.wav")
    entries = [{"id": "invite_1", "category": "invite", "text": "x", "file": "a.wav"},
               {"id": "invite_2", "category": "invite", "text": "y", "file": "b.wav"},
               {"id": "three", "category": "three", "text": "Three.", "file": "c.wav"}]
    (tmp_path / "manifest.json").write_text(json.dumps({"lines": entries}))
    bank = VoiceBank.load(str(tmp_path))
    assert len(bank) == 3 and bank.has("invite") and not bank.has("nobody")
    assert bank.get("three").seconds == pytest.approx(0.5, abs=0.01)
    rng = random.Random(0)
    last = None
    for _ in range(20):
        line = bank.pick("invite", rng, avoid=last)
        assert line.id != last
        last = line.id
    assert bank.pick("three", rng, avoid="three").id == "three"  # nothing else to choose
    with pytest.raises(KeyError):
        bank.pick("nobody")


def test_voicebank_without_a_manifest_is_empty(tmp_path):
    assert len(VoiceBank.load(str(tmp_path))) == 0


def test_loudness_envelope_is_normalized_and_tracks_silence():
    rate = 16000
    loud = 0.5 * np.sin(2 * np.pi * 200 * np.arange(rate) / rate)
    env = loudness_envelope(np.concatenate([loud, np.zeros(rate)]).astype(np.float32), rate)
    assert env.max() == pytest.approx(1.0) and env[:20].mean() > 0.8 and env[-20:].max() < 0.05


# ---------- animation planning ----------

def test_mouth_never_chatters_or_sticks_open():
    cfg = InteractionConfig()
    rapid = np.tile(np.array([1.0, 0.0], np.float32), 200)  # flips every 20 ms
    events = plan_mouth(rapid, ENVELOPE_HOP_S, cfg)
    opens = [e.t for e in events if e.kind == "mouth_open"]
    closes = [e.t for e in events if e.kind == "mouth_close"]
    assert all(b - a >= cfg.mouth_min_closed_s + cfg.mouth_min_open_s - 1e-9 for a, b in zip(opens, opens[1:]))
    assert len(opens) == len(closes) and events[-1].kind == "mouth_close"

    held = plan_mouth(np.ones(500, np.float32), ENVELOPE_HOP_S, cfg)  # 10 s of one loud tone
    spans = [c.t - o.t for o, c in zip((e for e in held if e.kind == "mouth_open"), (e for e in held if e.kind == "mouth_close"))]
    assert max(spans) <= cfg.mouth_max_open_s + ENVELOPE_HOP_S + 1e-6


def test_silence_never_opens_the_mouth_and_trailing_open_is_closed():
    cfg = InteractionConfig()
    assert plan_mouth(np.zeros(100, np.float32), ENVELOPE_HOP_S, cfg) == []
    events = plan_mouth(np.ones(3, np.float32), ENVELOPE_HOP_S, cfg)
    assert [e.kind for e in events] == ["mouth_open", "mouth_close"]


def test_eyes_are_lit_and_blink_slowly_and_irregularly():
    cfg = InteractionConfig()
    events = plan_eyes("welcome", 12.0, cfg, random.Random(3))
    assert events[0].kind == "eyes_on" and events[0].t == 0 and events[-1] == type(events[-1])(12.0, "eyes_off")
    dark = [(a.t, b.t) for a, b in zip(events, events[1:]) if a.kind == "eyes_off" and b.kind == "eyes_on"]
    assert len(dark) >= 5
    assert all(b - a == pytest.approx(cfg.eye_blink_s) for a, b in dark)  # each blink is the same slow length
    lit = [dark[i + 1][0] - dark[i][1] for i in range(len(dark) - 1)]
    assert all(cfg.eye_blink_min_gap_s - 1e-9 <= g <= cfg.eye_blink_max_gap_s + 1e-9 for g in lit)
    assert len({round(g, 3) for g in lit}) > 1  # not metronomic
    shortest = min(b.t - a.t for a, b in zip(events, events[1:]))
    assert shortest >= cfg.eye_blink_s - 1e-9


def test_short_lines_and_count_words():
    cfg = InteractionConfig()
    assert [e.kind for e in plan_eyes("welcome", 0.9, cfg, random.Random(1))] == ["eyes_on", "eyes_off"]  # no room to blink
    word = plan_eyes("two", 1.0, cfg)
    assert [e.kind for e in word] == ["eyes_on", "eyes_off"] and word[-1].t == pytest.approx(cfg.eye_blink_s)


def test_got_it_flashes_slowly_enough_for_the_hardware():
    cfg = InteractionConfig()
    events = plan_eyes("got_it", 1.07, cfg)
    assert sum(e.kind == "eyes_on" for e in events) >= 2
    assert min(b.t - a.t for a, b in zip(events, events[1:])) >= 0.2
    assert events[-1].kind == "eyes_off"  # ends dark


# ---------- performer ----------

def test_performer_moves_in_time_and_rests():
    clock = FakeClock()
    cfg = InteractionConfig()
    backend = RecordingBackend()
    skeleton = Skeleton(backend=backend)
    player = SimPlayer(clock=clock, sleep=clock.sleep)
    performer = Performer(player, skeleton, cfg, clock=clock, sleep=clock.sleep)
    line = make_line("welcome_1", "welcome", 2.0)
    assert performer.speak(line) is True
    assert clock.now == pytest.approx(2.0, abs=0.06)
    mouth = skeleton.pins.mouth
    mouth_writes = [v for ln, v in backend.writes if ln == mouth]
    assert mouth_writes and mouth_writes[0] is True and mouth_writes[-1] is False  # opened, ends closed
    assert backend.writes[-1][0] in (skeleton.pins.mouth, skeleton.pins.eyes)
    assert backend.levels[skeleton.pins.eyes] is False
    assert player.plays[0][2] == cfg.volume


def test_performer_interrupt_stops_audio_and_rests():
    clock = FakeClock()
    cfg = InteractionConfig()
    backend = RecordingBackend()
    skeleton = Skeleton(backend=backend)
    player = SimPlayer(clock=clock, sleep=clock.sleep)
    performer = Performer(player, skeleton, cfg, clock=clock, sleep=clock.sleep)
    assert performer.speak(make_line("x", "welcome", 5.0), interrupt=lambda: clock.now >= 1.0) is False
    assert player.handles[0].stopped
    assert backend.levels[skeleton.pins.mouth] is False and backend.levels[skeleton.pins.eyes] is False
    assert clock.now < 1.2


def test_performer_only_touches_what_it_may_move():
    clock = FakeClock()
    backend = RecordingBackend()
    performer = Performer(SimPlayer(clock=clock, sleep=clock.sleep), Skeleton(backend=backend), InteractionConfig(),
                          clock=clock, sleep=clock.sleep)
    performer.speak(make_line("x", "welcome", 1.0), move_mouth=False, flash_eyes=False)
    assert backend.writes == []  # audio only: no pin is driven, not even a "rest" write
    performer.speak(make_line("x", "welcome", 1.0), move_mouth=False, flash_eyes=True)
    assert {ln for ln, _ in backend.writes} == {Skeleton(backend=backend).pins.eyes}


def test_performer_stop_event_cuts_the_line_short():
    clock = FakeClock()
    stop = threading.Event()
    performer = Performer(SimPlayer(clock=clock, sleep=clock.sleep), Skeleton(backend=SimGpioBackend()), InteractionConfig(),
                          clock=clock, sleep=clock.sleep, stop_event=stop)
    stop.set()
    assert performer.speak(make_line("x", "welcome", 3.0)) is False
    assert clock.now == 0.0


# ---------- audio ----------

def test_aplay_player_scales_volume_to_stereo_and_cleans_up(tmp_path, monkeypatch):
    src = tmp_path / "line.wav"
    write_wav(src, seconds=0.2, amp=0.8)
    calls = {}

    class FakeProcess:
        def __init__(self, cmd, **kw):
            calls["cmd"] = cmd
            with wave.open(cmd[-1], "rb") as wav:
                calls["channels"], calls["rate"] = wav.getnchannels(), wav.getframerate()
                calls["pcm"] = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
            self.terminated = False

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.terminated = True

    handle = AplayPlayer("plughw:TEST", popen=FakeProcess).play(str(src), 0.2, volume=0.5)
    assert calls["cmd"][:4] == ["aplay", "-q", "-D", "plughw:TEST"]
    assert calls["channels"] == 2 and calls["rate"] == 16000
    assert calls["pcm"][0::2].tolist() == calls["pcm"][1::2].tolist()  # same signal both channels
    assert np.abs(calls["pcm"]).max() == pytest.approx(0.8 * 0.5 * 32767, rel=0.02)
    import os
    assert os.path.exists(calls["cmd"][-1])
    assert handle.wait(1.0) is True
    assert not os.path.exists(calls["cmd"][-1])  # temp file removed once playback is done


# ---------- director ----------

def test_full_sequence_with_a_visitor_who_walks_up():
    rig = make_director(lambda t: 1 if t >= 5 else 0)
    outcome = rig.director.run_sequence()
    assert outcome == "captured"
    names = played(rig)
    assert names[0] == "invite_1" and names[1] == "welcome_1"
    assert names[2:] == ["filler_1", "three", "two", "one", "got_it", "done_1"]
    assert rig.service.rendered == 1 and rig.service.printed == 0  # studio-first: no paper by default
    assert rig.session.closed
    got_it_start = [t for n, t, _ in rig.player.plays if n == "got_it.wav"][0]
    assert 0.0 <= rig.session.captured_at - got_it_start <= 0.5  # the photo lands on the word
    assert rig.backend.levels[rig.skeleton.pins.eyes] is False and rig.backend.levels[rig.skeleton.pins.mouth] is False
    status = rig.director.status()
    assert status["outcome"] == "captured" and status["state"] == "cooldown" and status["cooldown_s"] > 0


def test_invite_is_cut_short_when_a_visitor_arrives():
    rig = make_director(lambda t: 1 if t >= 1 else 0)
    rig.director.run_sequence()
    invite_start = [t for n, t, _ in rig.player.plays if n == "invite_1.wav"][0]
    welcome_start = [t for n, t, _ in rig.player.plays if n == "welcome_1.wav"][0]
    assert welcome_start - invite_start < 3.0  # did not wait for the whole 3 s invitation


def test_a_button_press_skips_the_invitations():
    rig = make_director(lambda t: 1)
    rig.director.run_sequence(skip_invites=True)
    assert "invite_1" not in played(rig) and played(rig)[0] == "welcome_1"


def test_nobody_comes():
    rig = make_director(lambda t: 0)
    assert rig.director.run_sequence() == "nobody"
    names = played(rig)
    assert names.count("invite_1") == rig.cfg.invite_max and names[-1] == "nobody_1"
    assert rig.service.rendered == 0 and rig.session.captured_at is None and rig.session.closed


def test_visitor_walks_off_mid_sequence():
    rig = make_director(lambda t: 1 if t < 4 else 0)
    assert rig.director.run_sequence(skip_invites=True) == "left"
    assert played(rig)[-1] == "come_back_1" and rig.session.captured_at is None
    assert "three" not in played(rig) and rig.service.rendered == 0


def test_nobody_in_the_photo_is_not_delivered():
    rig = make_director(lambda t: 1, capture_people=0)
    assert rig.director.run_sequence(skip_invites=True) == "left"
    assert rig.service.rendered == 0 and played(rig)[-1] == "come_back_1"


def test_stop_ends_the_sequence_and_rests_the_skeleton():
    stop = threading.Event()
    rig = make_director(lambda t: (stop.set() or 1) if t >= 5 else 1, stop_event=stop)
    assert rig.director.run_sequence(skip_invites=True) == "stopped"
    assert rig.session.captured_at is None and rig.session.closed
    assert rig.backend.levels[rig.skeleton.pins.mouth] is False and rig.backend.levels[rig.skeleton.pins.eyes] is False
    assert rig.director.status()["cooldown_s"] == 0  # a stopped run does not lock the booth


def test_camera_errors_are_reported_and_leave_the_booth_safe():
    def boom(_t):
        raise RuntimeError("camera unplugged")

    rig = make_director(boom)
    assert rig.director.run_sequence() == "error"
    status = rig.director.status()
    assert "camera unplugged" in status["error"] and rig.session.closed
    assert rig.backend.levels.get(rig.skeleton.pins.eyes, False) is False


def test_auto_print_sends_the_reel_and_announces_it():
    rig = make_director(lambda t: 1, cfg=fast_cfg(auto_print=True))
    assert rig.director.run_sequence(skip_invites=True) == "captured"
    names = played(rig)
    assert rig.service.printed == 1 and names[-2:] == ["printing_1", "done_1"]


def test_print_failure_is_an_error_outcome():
    rig = make_director(lambda t: 1, cfg=fast_cfg(auto_print=True))
    rig.service.print_error = "printer off"
    assert rig.director.run_sequence(skip_invites=True) == "error"
    assert "printer off" in rig.director.status()["error"]


def test_short_mode_skips_the_long_welcome():
    rig = make_director(lambda t: 1, cfg=fast_cfg(welcome=False))
    rig.director.run_sequence(skip_invites=True)
    assert "welcome_1" not in played(rig) and played(rig)[0] == "filler_1"


def test_start_runs_in_the_background_and_honors_the_cooldown():
    cfg = fast_cfg(invite_gap_s=0.05, presence_poll_s=0.01, hold_seconds=0.02, count_gap_s=0.01, cooldown_s=0.3)
    real = SimpleNamespace(now=0)
    session = FakeSession(lambda: 0.0, lambda t: 1)
    service = FakeService(session)
    bank = VoiceBank([make_line(i, c, 0.05) for i, c in
                      [("welcome_1", "welcome"), ("filler_1", "filler"), ("three", "three"), ("two", "two"),
                       ("one", "one"), ("got_it", "got_it"), ("done_1", "done")]])
    stop = threading.Event()
    performer = Performer(SimPlayer(), Skeleton(backend=SimGpioBackend()), cfg, stop_event=stop)
    director = Director(service, bank, performer, cfg, stop_event=stop)
    assert director.start(skip_invites=True) is True
    assert director.start(skip_invites=True) is False  # already running
    director.join(5)
    status = director.status()
    assert status["outcome"] == "captured" and not status["running"]
    assert director.start(skip_invites=True) is False  # cooling down
    assert director.start(skip_invites=True, ignore_cooldown=True) is True  # the studio button overrides it
    director.join(5)
    assert real.now == 0 and service.rendered == 2


def test_start_without_voice_lines_reports_why():
    cfg = fast_cfg()
    performer = Performer(SimPlayer(), Skeleton(backend=SimGpioBackend()), cfg)
    director = Director(FakeService(None), VoiceBank(), performer, cfg)
    assert director.start() is False
    assert "voice lines" in director.status()["error"]


def test_say_plays_one_line_and_rejects_while_busy():
    cfg = fast_cfg()
    player = SimPlayer()
    performer = Performer(player, Skeleton(backend=SimGpioBackend()), cfg)
    director = Director(FakeService(None), VoiceBank([make_line("three", "three", 0.1)]), performer, cfg)
    assert director.say("three") is True
    assert director.say("three") is False
    director.join(5)
    assert [p[0] for p in player.plays] == ["three.wav"]
    with pytest.raises(KeyError):
        director.say("nope")


# ---------- triggers ----------

def test_trigger_debounces_and_fires_once_per_press():
    clock = FakeClock()
    level = {"v": True}  # pulled up, idle
    fired = []
    watcher = TriggerWatcher(lambda: level["v"], lambda: fired.append(clock()), active_low=True, debounce_s=0.05, clock=clock, sleep=clock.sleep)

    level["v"] = False
    assert watcher.poll_once() is False  # just went active: wait out the debounce
    clock.sleep(0.02)
    level["v"] = True
    watcher.poll_once()  # a 20 ms glitch
    assert fired == []

    level["v"] = False
    watcher.poll_once()
    clock.sleep(0.06)
    assert watcher.poll_once() is True and len(fired) == 1
    for _ in range(10):  # button held down
        clock.sleep(0.5)
        assert watcher.poll_once() is False
    level["v"] = True
    watcher.poll_once()
    level["v"] = False
    watcher.poll_once()
    clock.sleep(0.06)
    watcher.poll_once()
    assert watcher.fired == 2


def test_trigger_active_high_for_a_motion_sensor():
    clock = FakeClock()
    level = {"v": False}
    watcher = TriggerWatcher(lambda: level["v"], lambda: None, active_low=False, debounce_s=0.0, clock=clock, sleep=clock.sleep)
    assert watcher.poll_once() is False
    level["v"] = True
    assert watcher.poll_once() is True


def test_trigger_thread_survives_a_failing_read():
    calls = {"n": 0}
    fired = threading.Event()

    def read():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("flaky line")
        return False

    watcher = TriggerWatcher(read, fired.set, active_low=True, debounce_s=0.0, poll_s=0.005)
    watcher.start()
    assert fired.wait(2.0)
    watcher.stop()


# ---------- factory ----------

def test_factory_returns_none_when_disabled():
    assert build_interaction(object(), PhotoBoothConfig()) is None


def test_factory_builds_a_simulated_booth_and_wires_the_button(tmp_path):
    write_wav(tmp_path / "w.wav")
    (tmp_path / "manifest.json").write_text(json.dumps({"lines": [{"id": "welcome_1", "category": "welcome", "text": "hi", "file": "w.wav"}]}))
    cfg = PhotoBoothConfig()
    cfg.interaction.enabled = True
    cfg.interaction.play_audio = False
    cfg.interaction.voice_dir = str(tmp_path)
    cfg.interaction.trigger_line = 5
    cfg.interaction.trigger_debounce_s = 0.0
    cfg.interaction.cooldown_s = 0.0
    started = []
    inter = build_interaction(FakeService(None), cfg)
    try:
        assert inter.simulated_audio and inter.simulated_gpio and len(inter.bank) == 1
        assert isinstance(inter.skeleton.backend, SimGpioBackend)
        inter.director.start = lambda **kw: started.append(kw) or True
        inter.trigger_input.level = False  # press the (active low) button
        for _ in range(100):
            if started:
                break
            threading.Event().wait(0.02)
        assert started and started[0]["skip_invites"] is True
    finally:
        inter.close()


def test_factory_does_not_touch_the_shared_skeleton_unless_gpio_is_on(tmp_path):
    shared = Skeleton(backend=RecordingBackend())
    cfg = PhotoBoothConfig()
    cfg.interaction.enabled = True
    cfg.interaction.play_audio = False
    cfg.interaction.voice_dir = str(tmp_path)
    inter = build_interaction(FakeService(None), cfg, skeleton=shared)
    assert inter.skeleton is not shared
    cfg.interaction.use_gpio = True
    inter2 = build_interaction(FakeService(None), cfg, skeleton=shared)
    assert inter2.skeleton is shared


# ---------- web ----------

@pytest.fixture
def client_with_voice():
    cfg = fast_cfg(invite_gap_s=0.05, presence_poll_s=0.01, hold_seconds=0.02, count_gap_s=0.01, cooldown_s=0.0)
    stop = threading.Event()
    player = SimPlayer()
    performer = Performer(player, Skeleton(backend=SimGpioBackend()), cfg, stop_event=stop)
    bank = VoiceBank([make_line(i, c, 0.05) for i, c in
                      [("welcome_1", "welcome"), ("filler_1", "filler"), ("three", "three"), ("two", "two"),
                       ("one", "one"), ("got_it", "got_it"), ("done_1", "done")]])
    service = FakeService(FakeSession(lambda: 0.0, lambda t: 1))
    director = Director(service, bank, performer, cfg, stop_event=stop)
    inter = Interaction(director, bank, Skeleton(backend=SimGpioBackend()), None, None, True, True)
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(SimpleNamespace(), inter))
    return app.test_client(), director, player


def test_voice_endpoints(client_with_voice):
    client, director, player = client_with_voice
    status = client.get("/api/photobooth/interact/status").get_json()
    assert status["enabled"] and status["simulated_audio"] and status["settings"]["move_mouth"] is True
    lines = client.get("/api/photobooth/interact/lines").get_json()["lines"]
    assert {l["id"] for l in lines} >= {"welcome_1", "got_it"}

    assert client.post("/api/photobooth/interact/say?line=nope").status_code == 404
    assert client.post("/api/photobooth/interact/say?line=three&volume=0.3").get_json()["status"] == "success"
    director.join(5)
    assert player.plays[0][2] == 0.3

    settings = client.post("/api/photobooth/interact/settings?auto_print=1&welcome=0&volume=2&move_mouth=0").get_json()["settings"]
    assert settings == {"auto_print": True, "welcome": False, "move_mouth": False, "flash_eyes": True, "volume": 1.0}

    assert client.post("/api/photobooth/interact/start?skip_invites=1").get_json()["status"] == "success"
    director.join(5)
    done = client.get("/api/photobooth/interact/status").get_json()
    assert done["outcome"] == "captured" and done["running"] is False and done["draft_stamp"] >= 1
    assert client.post("/api/photobooth/interact/stop").get_json()["status"] == "success"


def test_voice_endpoints_when_disabled():
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(SimpleNamespace()))
    client = app.test_client()
    assert client.get("/api/photobooth/interact/status").get_json() == {"enabled": False}
    assert client.post("/api/photobooth/interact/start").status_code == 404
    page = client.get("/photobooth/view").get_data(as_text=True)
    assert 'id="voicePanel"' in page and "/api/photobooth/interact/start" in page


# ---------- hardware speed check ----------

def test_wiggle_alternates_at_the_requested_rhythm_and_rests():
    clock = FakeClock()
    backend = RecordingBackend()
    skeleton = Skeleton(backend=backend)
    performer = Performer(SimPlayer(clock=clock, sleep=clock.sleep), skeleton, InteractionConfig(), clock=clock, sleep=clock.sleep)
    assert performer.wiggle("mouth", 0.6, 4) == 4
    assert clock.now == pytest.approx(2.4, abs=0.1)
    mouth = [v for ln, v in backend.writes if ln == skeleton.pins.mouth]
    assert mouth[:4] == [True, False, True, False] and mouth[-1] is False
    assert all(ln == skeleton.pins.mouth for ln, _ in backend.writes)  # the eyes were never touched


def test_wiggle_is_bounded_and_stoppable():
    clock = FakeClock()
    stop = threading.Event()
    backend = RecordingBackend()
    performer = Performer(SimPlayer(clock=clock, sleep=clock.sleep), Skeleton(backend=backend), InteractionConfig(),
                          clock=clock, sleep=clock.sleep, stop_event=stop)
    assert performer.wiggle("eyes", 0.01, 500) == 20  # period floor 0.1 s, at most 20 cycles
    assert clock.now == pytest.approx(2.0, abs=0.1)
    stop.set()
    assert performer.wiggle("eyes", 1.0, 5) == 0
    with pytest.raises(ValueError):
        performer.wiggle("body", 1.0, 1)  # never the body motor


# ---------- jaw charge budget ----------

def test_governor_spends_while_open_and_refills_while_closed():
    from photobooth.interaction.animation import MouthGovernor

    clock = FakeClock()
    cfg = InteractionConfig(mouth_budget_s=2.0, mouth_recharge_per_s=0.5, mouth_max_open_s=0.5)
    gov = MouthGovernor(cfg, clock)
    assert gov.try_open()
    clock.sleep(1.5)
    gov.closed()
    assert gov.level == pytest.approx(0.5)
    assert gov.try_open() is True  # exactly enough for one more longest open
    clock.sleep(0.4)
    gov.closed()
    clock.sleep(0.1)
    assert gov.try_open() is False and gov.skipped == 1  # flat: refused until it has recharged
    clock.sleep(2.0)
    assert gov.try_open() is True  # 2 s closed x 0.5 = 1.0 s regained


def test_continuous_speech_cannot_exceed_the_sustainable_duty_cycle():
    clock = FakeClock()
    cfg = InteractionConfig()
    backend = RecordingBackend()
    skeleton = Skeleton(backend=backend)
    performer = Performer(SimPlayer(clock=clock, sleep=clock.sleep), skeleton, cfg, clock=clock, sleep=clock.sleep)
    line = make_line("talk", "welcome", 30.0)  # loud on/off syllables for 30 s straight
    performer.speak(line, flash_eyes=False)

    assert performer.governor.skipped > 0  # it had to hold back
    # steady-state share open can't beat recharge/(1+recharge) by more than the starting budget allows
    allowed = cfg.mouth_budget_s + 30.0 * cfg.mouth_recharge_per_s / (1 + cfg.mouth_recharge_per_s) * 1.0
    opens = sum(1 for ln, v in backend.writes if ln == skeleton.pins.mouth and v is True)
    assert opens * cfg.mouth_min_open_s <= allowed + 1.0
    assert backend.levels[skeleton.pins.mouth] is False  # and it is left closed


def test_a_refused_open_also_skips_its_close():
    clock = FakeClock()
    cfg = InteractionConfig(mouth_budget_s=0.4, mouth_max_open_s=0.45)  # can never afford an open
    backend = RecordingBackend()
    performer = Performer(SimPlayer(clock=clock, sleep=clock.sleep), Skeleton(backend=backend), cfg, clock=clock, sleep=clock.sleep)
    performer.speak(make_line("x", "welcome", 3.0), flash_eyes=False)
    mouth_writes = [v for ln, v in backend.writes if ln == Skeleton(backend=backend).pins.mouth]
    assert mouth_writes == [False]  # only the final rest write; the jaw was never driven open
