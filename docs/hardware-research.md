# Pirate Photo Booth: hardware research

Findings from initial research (2026-09-14), before the printer is ordered
or the skelly is back out of storage. Written down now so decisions made
here don't have to be re-derived later -- see `docs/photobooth-architecture.md`
for how these findings shape the actual pipeline design.

## Camera: MIPI CSI on the Radxa Zero 3W

The Zero 3W has its own native camera connector -- a 4-lane, 22-pin,
0.5mm-pitch MIPI CSI interface on the board itself. It does **not** need
an IO board or a USB hub to add a camera; CSI is a separate connector
from the single USB-C port, so moving to CSI actually frees up the USB-C
port rather than needing one to add a hub.

**Officially supported cameras** (per Radxa's own docs, each with an
existing device-tree overlay):
- Raspberry Pi Camera V1.3 (OV5647 sensor)
- Raspberry Pi Camera V2 (IMX219 sensor)
- Radxa Camera 8M (IMX219 sensor)

**Radxa Camera 4K (IMX415 sensor) is *not* on that list.** It's documented
for the ROCK 3A/5A/5B family, not the Zero 3W. A Radxa forum thread
("IMX415 camera for Zero 3W") has a user attempting to bring it up: the
physical pinout matches, Radxa staff pointed to the ROCK 5 series' device
tree as a reference to adapt from, and one attempt hit a `dphy` link
error ("No link between dphy and sensor") on kernel 5.10. No one in that
thread reported a working result. **Treat the 4K/IMX415 as unproven,
non-trivial device-tree work, not a plug-and-play option.**

ArduCam modules aren't mentioned in Radxa's own compatibility list at
all -- same caution as the 4K camera, possibly worse since there's no
forum thread even attempting it on this specific board.

**Recommendation:** start with the Radxa Camera 8M (IMX219) or a
Raspberry Pi Camera V2 -- both have confirmed overlays and no driver
archaeology required. Revisit the 4K/IMX415 later only if 8MP/1080p
proves genuinely insufficient for the segmentation model's input needs
(unlikely -- MediaPipe Selfie Segmentation works fine well below 4K).

**Known hardware gotcha:** at least one batch of Zero 3W units shipped
with an inverted MIPI CSI connector clip (per the Radxa community
forum). Check cable orientation carefully when the board and camera are
first connected -- flipping the connector's black retention clip
resolves it if the ribbon won't seat correctly.

## Printer: Phomemo M02 protocol

The M02 (not M02 Pro/M02S/M02X -- those are different, incompatible
protocol families per community notes) has been fully reverse-engineered
by the `vivier/phomemo-tools` project by sniffing the Android app's
Bluetooth traffic. Key facts, confirmed by reading that project's actual
source (`cups/backend/phomemo.py`, `cups/filter/rastertopm02_t02.py`),
not just its README:

- **Transport is classic Bluetooth RFCOMM, not BLE GATT.** Plain
  `socket.socket(AF_BLUETOOTH, SOCK_STREAM, BTPROTO_RFCOMM)`, connect to
  the printer's Bluetooth MAC on channel 1. This is significantly
  simpler than a BLE GATT client -- Python's standard `socket` module on
  Linux handles it directly via BlueZ, no extra library needed. (A USB
  fallback also exists: vendor ID `0x0493`, product ID `0xb002`.)
- **Protocol is plain ESC/POS**, three commands cover everything needed:
  - `ESC @` -- initialize printer
  - `ESC a <justification>` -- set alignment (1 = center)
  - `GS v 0 <mode:1B> <bytesPerLine:2B LE> <numLines:2B LE> <raw bits>`
    -- print a raster bit-image block. Chunked at a max of 255 lines per
    block for a taller image (send repeated `GS v 0` blocks back to
    back).
  - `ESC d <n>` -- print and feed n lines, at the end.
- **Image format:** 1-bit monochrome, packed 8 pixels/byte, fixed at
  **384 px wide (48 bytes/line)** -- this matches the M02's 203 DPI head
  and ~48mm print width and is not configurable. Any composited image
  needs to be resized/cropped to exactly 384px wide before encoding.
- Converting a grayscale image to 1-bit with PIL's own `.convert('1')`
  (Floyd-Steinberg dithering by default) is what the reference filter
  uses, and it's genuinely a good fit here -- the dithering texture
  reads as film grain rather than as a compression artifact, which lines
  up with the vintage/Polaroid treatment already planned rather than
  fighting it.

**Radxa Zero 3W has onboard Bluetooth 5.4** (Radxa Wireless Module D8),
so the printer connects directly from the board's own radio -- no BLE
dongle or hub needed for this either. Between this and the CSI camera
finding above, the single USB-C port turns out not to be the bottleneck
originally assumed; it's free for whatever else needs it (most likely
just power, or a hub only if something else specifically needs USB).

## Skelly as found (2026-10-07)

Read-only assessment over SSH of the actual animatronic (no GPIO toggled,
no audio played, nothing reconfigured; a throwaway benchmark venv was
created in `/tmp` and removed afterward).

**Board and OS:** Radxa Zero 3W (reports "ZERO 3"), 4 GB RAM, Debian 12,
kernel 6.1.84-12-rk2410-nocsf, 115 GB card at 11% used. Clock is
NTP-synced (no RTC). Idle SoC temp ~58 C at 1.8 GHz `ondemand` with no
fan -- fine for bursts, consider a heatsink for sustained work.

**Network:** the ASIX USB ethernet adapter (192.168.4.41) and `wlan0`
(192.168.4.132, same /22) are both up; ethernet currently wins the
default route. SSH over Wi-Fi was verified (identical host key), so the
adapter can simply be unplugged. Key auth works with `~/.ssh/skelly`
(alias `skelly` added to the dev machine's ssh config; point it at
`.132` after the cutover).

**USB / camera:** the only USB device is the ethernet adapter; no
`/dev/video*` exist. A USB webcam worked on this board in 2025
(`fswebcam`, device `/dev/video10`, captures still on disk). Enabled
overlays: `external-antenna`, `rk3568-npu-enable`, `rk3568-pwm14-m0`,
`rock-max98357a`. Present but disabled: `radxa-zero3-radxa-camera-8m`,
`rpi-camera-v1.3`, `rpi-camera-v2`. There is no IMX415 overlay.

**NPU:** `rknpu` driver loaded (devfreq 600 MHz), `python3-rknnlite2`
2.3.0, `rknpu2-rk356x` 2.3.0, `rknn_model_zoo` v2.3.0 cloned, one
prebuilt model (`yolov8.rknn`). Segmentation examples (`yolov8_seg`,
`ppseg`, `deeplabv3`) exist but would need converting on an x86 host.

**Segmentation benchmark (CPU only, MediaPipe 1.1.0, selfie landscape
model, python 3.11):** model load 227 ms; median 55 ms @ 640x480,
93 ms @ 720p, 179 ms @ 1080p. The NPU is not required for this project.

**Bluetooth:** `hci0` up (AIC8800 over `ttyS1` via `hciattach`), BlueZ
5.66, no paired devices. HCI reports BR/EDR packet types (DM/DH/HV) and
the "BR/EDR not supported" feature bit is clear, so classic RFCOMM -- what
the M02 needs -- is available; Python exposes `AF_BLUETOOTH` and
`BTPROTO_RFCOMM`. Not yet tried against a real M02.

**GPIO (matches `app.py`):** `gpiochip3` line 20 = PIN_7 (mouth), 1 =
PIN_11 (eyes), 8 = PIN_15 (body motor), 7 = PIN_36 (spare); all
currently unused/input. PWM14_M0 overlay is enabled for pin 7.

**Audio:** MAX98357A I2S amp works via a custom overlay (took many
reboots to get right, per shell history). Card order is currently HDMI =
0, amp = 1, but `/etc/asound.conf` pins `card 0`; use the named device
`hw:CARD=sndmax98357adac,DEV=0`. Not played during this assessment, so
the default-device claim is inferred from config, not heard.

**Last year's test code (all relevant):**
- `~/skelly/app.py` (Oct 2025, 67 lines) is the original per-pin test
  harness: a Flask page toggling lines 20/1/8 (whitelist `[1, 7, 8, 20]`)
  via libgpiod v1, set-then-release, with the page reading each line back
  as an input and treating 1 as "ON". All three outputs use HIGH = ON.
- `~/SkellyEyeSpy` is byte-identical to this repo's original `app.py`
  (the dashboard version, with fswebcam capture; audio and streaming were
  stubs). `~/skelly-control-api` is an earlier copy of it.
- `~/skelly/test_pwm.py` / `duty_pwm.py` (Sep 2025) fade an LED with
  python-periphery PWM on `pwmchip0` channel 0 at 1 kHz. That is PWM14_M0,
  which the `rk3568-pwm14-m0` overlay (enabled in `extlinux.conf` on
  2025-09-21, the same day) routes to PIN_7, the mouth line. So a
  proportional (PWM) mouth was explored but never used: the later Radxa
  control code keeps `pwm_enabled = False` and drives the mouth as a plain
  on/off GPIO.
- Not part of the booth: `~/halloweave` (modified fork of an upstream
  project; only the untracked `skeleton_control_radxa.py` is relevant,
  see above) and `~/radxa-skelly` (Flask + Piper TTS voices).

**GPIO findings (input-only reads, no pin was driven):**
- Repeated libgpiod v1 line requests succeed, so the Radxa code's status
  helper does not leak lines; python-periphery opens and closes
  `/dev/gpiochip3` lines cleanly.
- Lines 20 (mouth), 8 (body) and 7 (spare) can all be requested as plain
  GPIO even though the PWM14 overlay is active, so the PWM overlay does
  not lock the mouth line out of GPIO use. It is a leftover of the PWM
  test and is not needed unless a proportional mouth is wanted.
- At idle, mouth (line 20) and eyes (line 1) read HIGH and body (line 8)
  reads LOW. The board had rebooted about an hour earlier, so this is a
  boot default (likely pull-ups), not a latched state from last year.
  Expect the mouth/eyes lines to sit HIGH until software sets them, and
  look at the skeleton during the first test to see what that does.
- The `halloweave` venv has `gpiod` 2.3.0, which lacks the v1
  `Chip.get_line` that the Radxa code calls.

## Open items

- Confirm the specific M02 unit's Bluetooth MAC address and pairing
  flow once it's ordered and in hand -- `phomemo-tools`' bluetooth scan
  logic (matching device names starting with `Mr.in` or exactly `T02`)
  may or may not match the M02's own advertised name; verify directly
  rather than assuming.
- The 255-line block chunking and overall throughput/timing haven't been
  measured against a real M02 yet -- `docs/backlog.md`-style tracking
  should pick this up once hardware exists.
