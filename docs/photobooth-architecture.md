# Pirate Photo Booth: pipeline design

Branch: `Pirate-Photo-Booth`. Goal: the skelly greets a visitor, then
prints a small monochrome souvenir photo of them composited onto a
pirate/shipwreck/saloon background with a vintage/Polaroid treatment.
See `docs/hardware-research.md` for the camera and printer research this
design is built on.

## Why snapshot-triggered, not continuous video

Unlike the Pose-Controller project's continuous gesture pipeline, this
is a one-shot capture: on a trigger (button, or later a person-detected
event), grab one frame, run it through the pipeline, print it. That
tolerates a second or two of end-to-end latency, which matters because
the Zero 3W's RK3566 NPU is roughly an order of magnitude weaker than
the Dragon boards used elsewhere in this project family -- fine for an
occasional snapshot, not for sustained per-frame processing.

## Pipeline

```
Camera capture -> Person segmentation -> Composite onto themed background
      -> Vintage treatment (sepia, vignette, grain) -> 1-bit dither/encode
      -> Phomemo M02 (Bluetooth RFCOMM)
```

- **`photobooth/capture/camera.py`** -- OpenCV `VideoCapture` wrapper,
  shared between the dev-laptop webcam and the Zero 3W's CSI camera
  (which appears as a plain V4L2 device once its overlay loads, so no
  separate code path is needed). Forces `cv2.CAP_V4L2` on Linux for
  integer device sources with a one-retry safety net -- this is the
  exact same intermittent-open-failure fix already found and validated
  on the Dragon boards in the sibling Pose-Controller project, applied
  here proactively rather than waiting to rediscover it.
- **`photobooth/segmentation/selfie.py`** -- MediaPipe's Selfie
  Segmentation model (Tasks API; the legacy `mp.solutions` API doesn't
  exist in current `mediapipe` releases, worth knowing before copying
  older example code). Chosen over an instance-segmentation model like
  YOLOv8-seg because it's purpose-built for exactly this
  person-cutout/background-replacement task and is much lighter --
  important given the Zero 3W's modest NPU/CPU budget. Uses the
  "landscape" model variant for a wider field of view suited to
  fuller-body framing. Model file isn't committed to git (binary, small,
  easy to re-fetch) -- run `scripts/download_models.py` once after
  cloning.
- **`photobooth/compose/compositor.py`** -- cuts the person out using a
  *feathered* (Gaussian-blurred) version of the segmentation mask rather
  than a hard cutout, so imperfect mask edges blend instead of showing a
  jagged boundary -- the vintage treatment downstream hides this even
  further.
- **`photobooth/compose/vintage.py`** -- sepia tone, radial vignette,
  then film grain last (grain on top of the vignette reads as print/scan
  texture). This step is deliberately forgiving of small compositing
  errors -- see `docs/hardware-research.md`'s note on how the printer's
  own 1-bit dithering reinforces rather than fights this look.
- **`photobooth/printer/raster.py` + `phomemo.py`** -- encodes a PIL
  image as the M02's plain ESC/POS raster protocol (see
  `docs/hardware-research.md` for exactly where each byte comes from)
  and sends it over a classic Bluetooth RFCOMM socket. A `dry_run_path`
  mode writes the raw encoded bytes to a file instead of opening a
  socket, so the encoding can be built and unit-tested (`tests/
  test_raster.py`, `tests/test_phomemo.py`) before the actual printer
  exists.
- **`photobooth/pipeline.py`** -- wires the above into one
  `render_photo()` call. `scripts/capture_test.py` is the current
  dev-loop entry point: capture one frame, run the pipeline, save a PNG,
  and dry-run encode it for the printer.

## Findings from the on-device assessment (2026-10-07)

Current status and next steps are in "Status and next steps" below.

On-device assessment of the real skelly is done (see "Skelly as found" in
`docs/hardware-research.md`). It changed the plan in four ways:

- **The NPU isn't needed.** MediaPipe selfie segmentation measured on the
  skelly's own RK3566 CPU: ~55 ms at 640x480, ~93 ms at 720p, ~179 ms at
  1080p. The printed image is only 384 px wide, so the whole pipeline can
  run at 640x480 (or smaller) end to end. RKNN stays an optional later
  experiment, not a dependency.
- **Wi-Fi worked at first** (`wlan0`, same subnet), but it later saw no
  networks at all (external antenna selected, none attached), so the booth
  has been running over USB ethernet through a hub.
- **Classic Bluetooth is available** (BR/EDR packet types present,
  Python has `AF_BLUETOOTH`/`BTPROTO_RFCOMM`), which removes the biggest
  printer-side risk before the M02 has even been paired.
- **`halloweave` is not part of this.** It is a modified fork of an
  upstream project; the only relevant piece is the uncommitted,
  never-wired-in `skeleton_control_radxa.py` (a Radxa port of the
  skeleton hardware control: pin map, eyes, body motor, audio-driven
  mouth). Its git history shows the Radxa work was never committed
  (see "Hygiene items" below).

## Photo strip mode (added 2026-10-07)

The 48 mm print width makes single photos small, which suits an
old-school photo-booth reel instead: N frames (default 4, up to 6) on one
tall strip, each on a different background, printed as a single job.

- **Flow:** the browser page (`/photobooth/view`) drives the session --
  `strip/start`, then one `strip/shot` per countdown (spoken pose prompt,
  beeps, "CHEESE"), then `strip/finish`, which lays out and prints the
  strip. Timing lives in the browser because only the person at the
  booth can see/hear the cues.
- **One-shot reel (the page's default):** a single capture is placed on
  each of the next N backgrounds (`strip/oneshot`), so a visitor poses
  once instead of N times. Each frame is composited separately (own
  grain), only the capture and mask are shared. The multi-shot flow
  (`strip/start` -> `strip/shot` x N -> `strip/finish`) remains as an
  option, and so does a single photo.
- **Every reel is unique (`compose/backgrounds.py`):** each reel draws a
  random **seed** (returned as `seed` in every result, shown on the page,
  and accepted as `?seed=N` or the page's seed box to reproduce a reel).
  From the seed come: (1) a random order of distinct backgrounds, never
  the same file twice in a row, reshuffled when the pool runs out;
  (2) per frame, a random zoom/pan window (up to `zoom_max` 1.4x), a
  coin-flip mirror, and small gamma/contrast/brightness and hue jitter;
  (3) with probability `morph_probability` (0.3), the frame is a **morph**
  of two themes through a soft random mask (a tilted ramp wobbled by
  low-frequency noise), e.g. kraken tentacles over the treasure island.
  Files whose name contains a `no_flip_markers` entry (default `saloon`,
  `noflip`) are never mirrored, because that would reverse lettering.
  Frame i's randomness is independent of the reel length, so the aim
  view's composite preview shows the real first background of the reel
  that follows (the preview seed is consumed by the next reel, then a new
  one is drawn). Same seed reproduces the backgrounds exactly; the final
  strip matches to within a rounding step, and film grain (if on) is not
  seeded. `compose.backgrounds` entries may be **glob patterns**
  (`assets/backgrounds/*_43_placeholder.png`), so adding real artwork is
  just dropping files in the folder. Switch it all off with
  `randomize_backgrounds`, `vary_backgrounds: false`,
  `morph_probability: 0`.
- **High resolution background pool (2026-10-08):** 21 generated 1200x896
  scenes in `assets/backgrounds/highres/` (about 25 MB, deliberately not
  committed without a decision on binary assets) replace the procedural
  placeholders via the glob `assets/backgrounds/highres/*.jpg`. They are
  downscaled once to `background_max_width` (720 px; final frames are
  360 px, so this keeps 2x headroom for zoom) to keep morphing and
  variation cheap on the RK3566. Three of them contain readable lettering
  (the two saloon signs and the RUM barrels) and are listed in
  `no_flip_markers` by filename, so they are never mirrored.
- **Studio: preview, tweak, then print (the page's default mode).** One
  capture (`draft/capture`, which also reports how many people were seen)
  can be re-rendered as often as wanted (`draft/render`, about 1.1 to 1.6 s
  on the board) before anything touches the printer; `draft/print` sends
  exactly the strip last rendered and never re-captures. The page shows the
  color reel beside the exact 1-bit print preview, per-frame thumbnails
  with a **reroll this frame** button, **shuffle all**, a seed box, and
  sliders for person size, outline, soften, morph, and a placement picker
  (center, left, right, alternate, random). Slider values persist between
  renders; `seed` pins a reel, `reroll=k` replaces only frame k, and
  building the backgrounds is cached per capture so framing sliders do not
  rebuild them.
- **Keeping the person readable without a drawn border (2026-10-08):**
  the scenes are detailed and the print is 1 bit, so a person can melt
  into them. A solid white outline fixed that but printed as an "angel
  glow" (reported as far too strong even at 1 px with grain and sepia on),
  so edge treatment is now a choice, `edge_mode`, over a band of
  `outline_px` (default 5) at `edge_strength` (0.3):
  - `separate`: no line at all, a soft local dodge and burn of
    the background near the person. The direction is decided locally
    against the nearest part of the person (a pale face beside pale sand
    gets the sand darkened even if the shirt is dark), and when tones are
    alike it burns, like a contact shadow. A first version that chose one
    direction from the person's average brightness made exactly that
    pale-face-on-sand case worse, which a zoomed 1-bit comparison exposed.
  - `stipple` (default; 5 px at 30% was picked by eye from real prints
    over `separate`): a fading scatter of ordered-dither dots, dense at the
    person and sparse outward, in the separating tone (white dots on a
    light scene, black on a similar or darker one): the "alternating
    pixels" idea.
  - `none`, and `ring` (the old solid halo, kept only for comparison).
  `person_pop` (0.15) adds a little saturation and tone contrast to the
  person themselves, and `background_soften` blurs, flattens and lifts
  dark scenes toward a target brightness before dithering. All of it is
  live in the studio (edge style, width, strength, pop). Evaluated so far
  only against a synthetic textured figure; real faces and clothes will
  compete with the background more, so these defaults are a starting point
  for tuning on real photos.
- **Size, placement and groups:** `person_height_ratio` sets how much of
  each frame the person's visible height fills (about 0.5 reveals roughly
  twice the scenery of 0.72). `anchor_mode` can stand people off center
  (`left`/`right` at `anchor_offset` from center, `alternate`, or
  `random`, all derived from the seed), which keeps a scene's focal point
  visible; the group is clamped so it never leaves the frame. Several
  people are treated as one group scaled by the tallest, so a child beside
  an adult keeps true proportions, and people smaller than
  `group_min_area_ratio` (12%) of the biggest are ignored so a distant
  bystander does not shrink everyone. Limits: touching people merge into
  one blob (the people count undercounts a huddle), and two people far
  apart in the frame become a wide group that is scaled down to fit.
- **Known dev machine issue:** on the Windows PC used for development, the
  same pure-numpy float expression returned different results from run to
  run (with numpy 2.5.3 in float32, and with 2.4.6 in float64), a few
  levels at most; the Radxa board's numpy 2.4.6 was perfectly stable in
  the same stress test. Seeds therefore reproduce the *look* of a reel and
  the tests compare with a tolerance (`tests/_tolerance.py`). If that PC
  does other numerical work, a memory/CPU stability check is worth doing.
  numpy is pinned to `>=2.2,<2.5` in `requirements.txt`.
- **Per shot:** capture -> segment -> clean the mask (drop specks below
  0.4% of the frame, close small gaps) -> **normalize the person's
  size** -> composite onto the next background in `compose.backgrounds`
  (cycles) -> vintage, all at the final 360x270 so grain acts on real
  print pixels.
- **Size normalization** (`compose/framing.py`): a person who fills the
  camera frame would hide the background entirely, and nobody's height
  or distance is controllable. So the person's *visible* height is scaled
  to `compose.person_height_ratio` (default 0.35, see "Tuned defaults")
  of each output frame, within `person_scale_min`/`max` (0.1-1.8x; a high
  floor silently overrides small ratios, and the cap keeps a very distant
  person from being blown up into mush), and a wide blob (a group) is also capped at
  92% of the frame width. They are centered horizontally and anchored to
  the bottom: flush with the bottom edge when the camera cut them off
  there, slightly above it when their feet are in view. Everything above
  and around them is background. Visible height is the only height the
  camera gives us, so a person framed only from the chest up is
  normalized as a bust shot, which is the intent. Disable with
  `compose.normalize_person: false`.
- **Tuned defaults (2026-10-08, from testing on the board with a real
  adult):** person size 35% (30 to 40% is the sweet spot against these
  scenes), the solid outline off (even 1 px reads as a heavy ring in the
  print and makes the person feel pasted on instead of part of the image;
  see the edge modes above for what replaced it), background soften 0.33, morph 0.33, and `alternate`
  placement. Seed and reroll were reported to work well for repeatability
  and interaction. Testing also exposed that `person_scale_min` was 0.35,
  which quietly turned a requested 35% into up to about 47% for a person
  filling the camera frame; the floor is now 0.1 and a regression test pins
  that 35% means 35%.
- **Aim view** shows the live camera (detected person tinted green) next
  to a preview of the *next* shot's composite, so the person can see
  what the booth will do before the countdown starts.
- **Layout** (`photobooth/compose/strip.py`): built at exactly 384 px so
  the 1-bit dither runs with no resampling; header text, black-bordered
  frames, date footer. 4 frames is ~153 mm of paper, 6 is ~225 mm.
- **Printing (paced, 2026-10-07):** the first real strip stopped at the
  start of its third frame (~30 KB in). The M02 has a small receive
  buffer and no flow control, so a ~60 KB strip sent in well under a
  second overflowed it and the tail was dropped. The job is now sent block
  by block (<=255 rows each), holding back whenever more than
  `printer.max_backlog_lines` (400) rows would be sitting unprinted,
  judged against an estimated print position of `printer.lines_per_second`
  (100; the M02 is rated 10-15 mm/s = ~80-120 lines/s). The connection
  is then held open until the estimated end of the print plus
  `tail_pad_s`, since closing early can cut a print off. A 4-frame strip
  therefore takes ~15 s end to end. If prints still stop, lower
  `lines_per_second` / `max_backlog_lines` (60 / 255 is one block at a
  time). Anything the printer sends back is recorded in
  `PhomemoM02Printer.responses` in case it reports its own status.
- **Old-timey overlay is optional:** `compose.vintage` (default on) or
  `vintage=0/1` per request / the page checkbox. Off skips the sepia,
  vignette and film grain; the 1-bit dither alone already reads as grain
  and keeps more detail (see the on/off print previews).
- **Backgrounds:** six procedural placeholders at 4:3
  (`scripts/make_themed_backgrounds.py`: pirate ship, shipwreck, saloon,
  treasure island, Jolly Roger, kraken). They are crude stand-ins, not
  final art.
- **Known tuning issue:** dark scenes (saloon, shipwreck) go muddy under
  the 1-bit dither plus film grain; brighter, higher-contrast
  backgrounds and/or a tone lift before dithering are the likely fix, to
  be judged with real photos of people in frame.

## Status and next steps (2026-10-08)

**Working and tuned by eye on the board:** capture from the USB webcam,
segmentation, size normalization (35% for an adult), alternating stand,
stipple edge (5 px at 30%), pop, soften, morph, the 21-scene background pool
with seeds and per-frame reroll, and the studio (preview, tweak, then print).
The seed and reroll controls were reported as working well, for both
repeatability and interaction.

**Built but not yet validated end to end:** paced printing of a full 4-frame
reel (the first strip stopped at its third frame before pacing was added),
the `skelly/hardware.py` GPIO layer on the real pins (the body motor in
particular, because the USB hub now sits near the arm's sweep), and
behavior with more than one person.

### Next steps, in order

1. **Live test with 1 to 3 people in frame, adults and kids.** Check that
   grouping and scaling behave as expected and that everyone is visible on
   the print. Things to look at:
   - A child beside an adult (the group is sized by the tallest, so a small
     child can end up quite small at 35%; decide whether the group needs its
     own size ratio or a bigger default).
   - Two or three people standing apart (a wide group is scaled down to fit)
     versus close together or touching (they merge into one blob and the
     people count undercounts).
   - Someone partly in frame, or walking in late: does the 12% bystander
     filter (`group_min_area_ratio`) drop the right people?
   - Whether `alternate`/`random` placement still clamps a wide group
     sensibly, and whether the aim view's people count matches reality.
   - Include one complete 4-frame print in this session, so the paced
     printing is confirmed (or tuned via `printer.lines_per_second` and
     `max_backlog_lines`) at the same time.
2. **Props, hats and accessories.** Check how basic props are cut out and
   whether they cause collisions: tall hats (a hat adds height and shrinks
   the person), held props such as swords, signs or drinks (the selfie model
   tends to drop or fragment thin or held objects), capes and loose
   costume parts, and props that overlap another person. Deal with them as
   well as is practical, for example by measuring height from a robust
   percentile instead of the bounding box so a hat or raised arm does not
   shrink everyone, closing and filling the mask, or capping how much a thin
   protrusion can influence the scale.
3. **Update, edit, review and publish the blog post** with the results of
   the live test (it currently says the group, props and long-print behavior
   are unverified).
4. **Turn the skelly into a proper prop around a vintage style antique
   camera**, which should have plenty of room to hide the hardware. Things
   to design: where the Radxa, the USB webcam (or a slimmer CSI camera, see
   `docs/hardware-research.md`) and the M02 printer live, with the lens
   behind the camera's lens and the sticker exiting through a slot; the
   shutter or trigger (a physical button, and later motion or person
   detection); visual and audio cues from the prop (a flash LED for the
   countdown, the skelly speaking it); cooling and ventilation (the board has
   no fan and idles near 58 C); power and cable management, including the USB
   hub; clearance for the arm motor; and Wi-Fi (the board currently sees no
   networks with the external antenna selected and none attached, so either
   attach one or switch to the onboard antenna).

### Other open items

- Decide whether the 25 MB of generated background images are committed
  (or kept out of the repository), and push the branch.
- First supervised GPIO test on the real pins (mouth and eyes first, body
  motor last), and check what the boot-time pin states do.
- Security before Halloween night: the booth page has no authentication, test
  photos persist on the board, the segmentation model download is not
  hash-checked, and the dashboard can move motors. A delete-after-print step
  and a retention limit are worth adding.
- Enforced delay between visitors, and a paper-out or disconnected-printer
  fallback so a dead printer cannot wedge the show.

## Hygiene items found on the board

The Radxa work in `~/halloweave` exists only on the skelly's card:
branch `sleventy/2025-dev` is 0 commits ahead of `main`, and
`skeleton_control_radxa.py` (345 lines) is untracked, with `main.py`
(porcupine import commented out) and `requirements.txt` (`+gpiod`)
modified. `main.py` still imports the original `skeleton_control`, not
the Radxa module. A local backup copy was taken 2026-10-07.


Not photo-booth work, but worth fixing before the board sits on a
Halloween-night network: `~/.env` (four API keys) is world-readable
(`chmod 600`); a Wi-Fi passphrase sits in plaintext in `~/.bash_history`;
the default `radxa` password is still active (key auth already works,
so password login can be disabled); and `/etc/asound.conf` hard-codes
`card 0`, which is now the HDMI output rather than the MAX98357A amp
(use the named device `hw:CARD=sndmax98357adac`).
