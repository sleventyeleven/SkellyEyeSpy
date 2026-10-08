from __future__ import annotations

from flask import Blueprint, Response, jsonify, request, send_file

from .service import PhotoBoothService


def create_blueprint(service: PhotoBoothService, interaction=None) -> Blueprint:
    bp = Blueprint("photobooth", __name__)

    def _vintage():
        """?vintage=0/1 overrides the configured overlay; absent means use the config."""
        value = request.args.get("vintage")
        return None if value is None else value != "0"

    @bp.post("/api/photobooth/snap")
    def snap():
        try:
            result = service.snap(print_it=request.args.get("print", "1") != "0", vintage=_vintage(), seed=request.args.get("seed", type=int))
        except Exception as exc:  # noqa: BLE001 - report to the caller, keep serving
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 500
        return jsonify(
            {
                "status": "success",
                "image": result.image_path,
                "preview": result.preview_path,
                "debug": result.debug_path,
                "printed": result.printed,
                "print_error": result.error,
                "seconds": round(result.seconds, 2),
                "seed": result.seed,
                "backgrounds": result.backgrounds,
            }
        )

    @bp.get("/api/photobooth/last")
    def last_image():
        if service.last is None:
            return jsonify({"status": "error", "message": "No photo taken yet"}), 404
        return send_file(service.last.image_path, mimetype="image/png")

    @bp.get("/api/photobooth/last_print")
    def last_print():
        if service.last is None:
            return jsonify({"status": "error", "message": "No photo taken yet"}), 404
        return send_file(service.last.preview_path, mimetype="image/png")

    def _error(exc: Exception, code: int = 500):
        return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), code

    @bp.post("/api/photobooth/strip/start")
    def strip_start():
        shots = request.args.get("shots", type=int)
        try:
            shots = service.strip_start(shots, seed=request.args.get("seed", type=int))
            return jsonify({"status": "success", "shots": shots, "seed": service.strip_seed})
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @bp.post("/api/photobooth/strip/shot")
    def strip_shot():
        try:
            return jsonify({"status": "success", **service.strip_shot(vintage=_vintage())})
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @bp.get("/api/photobooth/strip/shot/<int:index>")
    def strip_shot_image(index: int):
        path = service.strip_shot_path(index)
        if path is None:
            return jsonify({"status": "error", "message": "No such shot"}), 404
        return send_file(path, mimetype="image/png")

    @bp.post("/api/photobooth/strip/finish")
    def strip_finish():
        try:
            result = service.strip_finish(print_it=request.args.get("print", "1") != "0")
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
        return jsonify(
            {
                "status": "success",
                "image": result.image_path,
                "preview": result.preview_path,
                "printed": result.printed,
                "print_error": result.error,
                "seconds": round(result.seconds, 2),
                "seed": result.seed,
                "backgrounds": result.backgrounds,
            }
        )

    @bp.post("/api/photobooth/strip/oneshot")
    def strip_oneshot():
        try:
            result = service.strip_one_shot(
                shots=request.args.get("shots", type=int),
                print_it=request.args.get("print", "1") != "0",
                vintage=_vintage(),
                seed=request.args.get("seed", type=int),
            )
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
        return jsonify(
            {
                "status": "success",
                "image": result.image_path,
                "preview": result.preview_path,
                "printed": result.printed,
                "print_error": result.error,
                "seconds": round(result.seconds, 2),
                "seed": result.seed,
                "backgrounds": result.backgrounds,
            }
        )

    def _draft_settings():
        a = request.args
        return dict(
            size=a.get("size", type=float),
            outline=a.get("outline", type=int),
            edge=a.get("edge"),
            strength=a.get("strength", type=float),
            pop=a.get("pop", type=float),
            soften=a.get("soften", type=float),
            morph=a.get("morph", type=float),
            anchor=a.get("anchor"),
            offset=a.get("offset", type=float),
        )

    @bp.post("/api/photobooth/draft/capture")
    def draft_capture():
        try:
            return jsonify({"status": "success", **service.draft_capture()})
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @bp.post("/api/photobooth/draft/render")
    def draft_render():
        a = request.args
        try:
            result = service.draft_render(
                shots=a.get("shots", type=int),
                seed=a.get("seed", type=int),
                vintage=_vintage(),
                reroll=a.get("reroll", type=int),
                reroll_all=a.get("reroll_all") == "1",
                frame=a.get("frame", type=int),
                position=a.get("position"),
                restore=a.get("restore", type=int),
                **_draft_settings(),
            )
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
        return jsonify({"status": "success", **result})

    @bp.post("/api/photobooth/draft/retake")
    def draft_retake():
        index = request.args.get("index", type=int)
        if index is None:
            return jsonify({"status": "error", "message": "index is required"}), 400
        try:
            return jsonify({"status": "success", **service.draft_retake(index)})
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    @bp.post("/api/photobooth/draft/print")
    def draft_print():
        try:
            result = service.draft_print()
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
        return jsonify(
            {
                "status": "success",
                "printed": result.printed,
                "print_error": result.error,
                "seconds": round(result.seconds, 2),
                "seed": result.seed,
            }
        )

    @bp.get("/api/photobooth/draft/image/<name>")
    def draft_image(name: str):
        path = service.draft_path(name)
        if path is None:
            return jsonify({"status": "error", "message": "No such image"}), 404
        return send_file(path)

    @bp.get("/api/photobooth/last_debug")
    def last_debug():
        if service.last is None:
            return jsonify({"status": "error", "message": "No photo taken yet"}), 404
        return send_file(service.last.debug_path, mimetype="image/jpeg")

    @bp.get("/api/photobooth/aim")
    def aim():
        try:
            return Response(service.aim_jpeg(composite=request.args.get("composite") == "1"), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 500

    # ---- booth voice: the spoken, animated visitor sequence ----

    director = interaction.director if interaction is not None else None

    def _settings() -> dict:
        cfg = director._cfg  # noqa: SLF001 - the live config the sequence reads
        return {"auto_print": cfg.auto_print, "welcome": cfg.welcome, "move_mouth": cfg.move_mouth,
                "flash_eyes": cfg.flash_eyes, "volume": cfg.volume}

    def _need_director():
        if director is None:
            return jsonify({"status": "error", "message": "The booth voice is not enabled (interaction.enabled in the config)"}), 404
        return None

    @bp.get("/api/photobooth/interact/status")
    def interact_status():
        if director is None:
            return jsonify({"enabled": False})
        return jsonify({"enabled": True, **director.status(), "settings": _settings(),
                        "simulated_audio": interaction.simulated_audio, "simulated_gpio": interaction.simulated_gpio})

    @bp.get("/api/photobooth/interact/lines")
    def interact_lines():
        missing = _need_director()
        return missing or jsonify({"lines": director._bank.catalog()})  # noqa: SLF001

    @bp.post("/api/photobooth/interact/start")
    def interact_start():
        missing = _need_director()
        if missing:
            return missing
        started = director.start(skip_invites=request.args.get("skip_invites") == "1", reason="studio",
                                 ignore_cooldown=True)
        if not started:
            return jsonify({"status": "error", "message": "A sequence is already running (or there are no voice lines)"}), 409
        return jsonify({"status": "success"})

    @bp.post("/api/photobooth/interact/stop")
    def interact_stop():
        missing = _need_director()
        if missing:
            return missing
        director.stop()
        return jsonify({"status": "success"})

    @bp.post("/api/photobooth/interact/say")
    def interact_say():
        missing = _need_director()
        if missing:
            return missing
        a = request.args
        try:
            started = director.say(a.get("line", ""), volume=a.get("volume", type=float),
                                   move_mouth=None if a.get("mouth") is None else a.get("mouth") == "1",
                                   flash_eyes=None if a.get("eyes") is None else a.get("eyes") == "1")
        except KeyError:
            return jsonify({"status": "error", "message": "No such voice line"}), 404
        if not started:
            return jsonify({"status": "error", "message": "The booth is busy"}), 409
        return jsonify({"status": "success"})

    @bp.post("/api/photobooth/interact/wiggle")
    def interact_wiggle():
        missing = _need_director()
        if missing:
            return missing
        a = request.args
        try:
            started = director.wiggle(a.get("part", ""), a.get("period", 1.0, type=float), a.get("cycles", 5, type=int))
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        if not started:
            return jsonify({"status": "error", "message": "The booth is busy"}), 409
        return jsonify({"status": "success"})

    @bp.post("/api/photobooth/interact/settings")
    def interact_settings():
        missing = _need_director()
        if missing:
            return missing
        cfg, a = director._cfg, request.args  # noqa: SLF001
        for key in ("auto_print", "welcome", "move_mouth", "flash_eyes"):
            if a.get(key) is not None:
                setattr(cfg, key, a.get(key) == "1")
        if a.get("volume", type=float) is not None:
            cfg.volume = min(1.0, max(0.0, a.get("volume", type=float)))
        return jsonify({"status": "success", "settings": _settings()})

    @bp.get("/photobooth/view")
    def view():
        return _VIEW_HTML

    return bp


_VIEW_HTML = r"""<!doctype html><title>Photo booth</title>
<style>
  body { margin:0; background:#111; color:#ddd; font-family:sans-serif; text-align:center; }
  label { margin:0 8px; white-space:nowrap; }
  .row { margin:10px 6px; line-height:2; }
  button { font-size:18px; margin:4px; padding:4px 12px; }
  input[type=range] { vertical-align:middle; width:140px; }
  .val { display:inline-block; min-width:2.6em; color:#fc6; }
  #prompt { font-size:20px; margin:6px 0 0; color:#fc6; }
  #prompt:empty { display:none; }
  .thumb { display:inline-block; margin:4px; vertical-align:top; font-size:12px; color:#999; max-width:190px; }
  .thumb img { width:180px; display:block; }
  .err { color:#f66; }
  .card { max-width:1120px; margin:14px auto; padding:8px 16px 14px; border:1px solid #444; border-radius:10px; background:#171717; }
  .card > h3 { margin:6px 0 2px; text-align:left; font-size:17px; color:#fc6; letter-spacing:.03em; }
  .hint { color:#999; font-size:13px; margin:2px 0 6px; }
  .tools { margin:8px 0; padding:6px; border:1px solid #333; border-radius:8px; background:#1d1d1d; }
  button:disabled { opacity:.4; }
  .slot { display:inline-block; vertical-align:top; margin:4px; border:2px dashed #3a3a3a; border-radius:6px; color:#555;
          font-size:13px; display:inline-flex; align-items:center; justify-content:center; box-sizing:border-box; }
  .previews { min-height:520px; padding:8px 0; }
  .previews figure { display:inline-block; vertical-align:top; margin:4px 10px; }
  .previews figcaption { color:#999; font-size:12px; margin-top:4px; }
  .previews img { background:#fff; display:block; max-width:100%; }
  #thumbs { min-height:250px; padding-top:4px; }
  .thumb { border:1px solid #333; border-radius:6px; padding:6px; background:#1d1d1d; }
  .thumb .ftools { margin:4px 0; line-height:2.1; }
  .thumb button { font-size:13px; padding:2px 8px; margin:2px; }
  .thumb select { font-size:13px; }
</style>
<body>
<div id="voicePanel" class="card" style="display:none">
  <h3>Booth voice</h3>
  <span id="voiceMode" style="color:#999;font-size:13px"></span><br>
  <button id="voiceStart" style="background:#7a2;color:#fff">Run the visitor sequence</button>
  <button id="voiceStop">Stop</button>
  <label><input id="vSkip" type="checkbox"> skip invitations (someone is already here)</label>
  <br>
  <label>volume <input id="vVolume" type="range" min="0" max="1" step="0.05" value="1"><span class="val" id="vVolumeV"></span></label>
  <label><input id="vMouth" type="checkbox"> move mouth</label>
  <label><input id="vEyes" type="checkbox"> flash eyes</label>
  <label><input id="vWelcome" type="checkbox"> long welcome</label>
  <label><input id="vPrint" type="checkbox"> print automatically</label>
  <br>
  <label>test a line <select id="vLine"></select></label><button id="vSay">Say it</button>
  <br>
  <span id="wigglePart"><b>Hardware speed check</b> (moves the real parts when "LIVE"):</span>
  <label>part <select id="wPart"><option>mouth</option><option>eyes</option></select></label>
  <label>one cycle takes <select id="wPeriod"><option>1.6</option><option selected>1.0</option><option>0.7</option><option>0.5</option><option>0.35</option><option>0.25</option></select> s</label>
  <label>cycles <input id="wCycles" type="number" value="5" min="1" max="20" style="width:3em"></label>
  <button id="wGo">Wiggle</button>
  <div id="voiceStatus" style="color:#fc6;min-height:1.6em"></div>
</div>

<div class="card" id="photoCard">
  <h3>Photo</h3>
  <p class="hint">Live aim view: green = detected person, right = the next shot. Walk into frame and face the camera.</p>
  <img id="f" style="max-width:100%">

  <div class="tools">
    <div class="row">
      <label><input type="radio" name="mode" value="studio" checked> studio (preview, tweak, then print)</label>
      <label><input type="radio" name="mode" value="oneshot"> reel from one shot, print now</label>
      <label><input type="radio" name="mode" value="strip"> reel, a shot per frame</label>
      <label><input type="radio" name="mode" value="single"> single photo</label>
    </div>
    <div class="row">
      <label>frames <input id="shots" type="number" value="4" min="1" max="6" style="width:3em"></label>
      <label>first countdown <input id="secs" type="number" value="10" min="1" max="60" style="width:3em"> s</label>
      <label>between shots <input id="between" type="number" value="5" min="1" max="30" style="width:3em"> s</label>
      <label><input id="vintage" type="checkbox" checked> old-timey grain/sepia</label>
      <label><input id="comp" type="checkbox" checked> preview composite</label>
      <label id="directOnly">seed <input id="seed" type="number" placeholder="random" style="width:7em">
        <input id="doprint" type="checkbox"> print</label>
    </div>
    <div class="row" id="sliders">
      <label>person size <input id="size" type="range" min="0.15" max="0.95" step="0.01" value="0.35"><span class="val" id="sizeV"></span></label>
      <label>edge <select id="edge"><option>stipple</option><option>separate</option><option>none</option><option value="ring">ring (old glow)</option></select></label>
      <label>edge width <input id="outline" type="range" min="0" max="30" step="1" value="5"><span class="val" id="outlineV"></span></label>
      <label>edge strength <input id="estrength" type="range" min="0" max="1" step="0.01" value="0.3"><span class="val" id="estrengthV"></span></label>
      <label>pop <input id="pop" type="range" min="0" max="1" step="0.01" value="0.15"><span class="val" id="popV"></span></label>
      <label>soften <input id="soften" type="range" min="0" max="1" step="0.01" value="0.33"><span class="val" id="softenV"></span></label>
      <label>morph <input id="morph" type="range" min="0" max="1" step="0.01" value="0.33"><span class="val" id="morphV"></span></label>
      <label>stand
        <select id="anchor"><option>alternate</option><option>center</option><option>left</option><option>right</option><option>random</option></select></label>
    </div>
  </div>
  <div class="row"><button id="go">Start countdown</button></div>
  <div id="prompt"></div>
</div>

<div class="card" id="studio">
  <h3>Render</h3>
  <div class="tools" id="renderTools">
    <button id="shuffle" disabled>Shuffle all backgrounds</button>
    <label>seed <input id="useSeed" type="number" placeholder="seed" style="width:8em" disabled></label>
    <button id="applySeed" disabled>Use seed</button>
    <button id="retake" disabled>Retake whole photo</button>
    <label>frame retake countdown <input id="retakeSecs" type="number" value="5" min="1" max="30" style="width:3em"> s</label>
    <label>preview size <select id="previewSize"><option value="actual">actual print size (1:1)</option><option value="fit">fit to window (smooth)</option></select></label>
    <button id="printBtn" style="background:#2a7;color:#fff" disabled>Print this reel</button>
  </div>
  <div id="studioStatus" class="row">seed &mdash; | people &mdash; | render time &mdash; | nothing rendered yet. Take a photo above.</div>
  <div class="previews" id="previews">
    <figure><div class="slot" style="width:384px;height:520px">colour reel</div><figcaption>colour reel</figcaption></figure>
    <figure><div class="slot" style="width:384px;height:520px">black and white print</div><figcaption>what the printer will print</figcaption></figure>
  </div>
  <div id="thumbs">
    <div class="thumb slot" style="width:190px;height:230px">frame 1</div>
    <div class="thumb slot" style="width:190px;height:230px">frame 2</div>
    <div class="thumb slot" style="width:190px;height:230px">frame 3</div>
    <div class="thumb slot" style="width:190px;height:230px">frame 4</div>
  </div>
</div>
<div id="out"></div>

<script>
const $ = id => document.getElementById(id);
const f = $('f'), out = $('out'), go = $('go'), prompt_ = $('prompt');
const POSES = ["Arrr! Your best pirate grin", "Look suspicious", "Fighting pose!", "Guard the treasure", "Scared of the kraken!", "Hoist the Jolly Roger!"];
let busy = false, aimOn = true, hasDraft = false, timer = null;

function next() {
  if (!aimOn) { setTimeout(next, 400); return; }
  f.onload = f.onerror = () => setTimeout(() => { busy ? setTimeout(next, 500) : next(); }, 300);
  f.src = '/api/photobooth/aim?composite=' + ($('comp').checked ? 1 : 0) + '&t=' + Date.now();
}
next();

const ctx = new (window.AudioContext || window.webkitAudioContext)();
function beep(freq, ms) { const o = ctx.createOscillator(); o.frequency.value = freq; o.connect(ctx.destination); o.start(); setTimeout(() => o.stop(), ms); }
const sleep = ms => new Promise(r => setTimeout(r, ms));
// Audio cue only: a beep each second (higher for the last three) and a longer, higher one for the shot. No on-screen timer.
async function countdown(n) {
  for (; n > 0; n--) { beep(n <= 3 ? 880 : 440, 150); await sleep(1000); }
  beep(1320, 400);
}
async function post(url) { return (await fetch(url, {method: 'POST'})).json(); }
const mode = () => document.querySelector('input[name=mode]:checked').value;
const vintageFlag = () => '&vintage=' + ($('vintage').checked ? 1 : 0);
const flags = () => ($('doprint').checked ? '' : '&print=0') + vintageFlag() + ($('seed').value ? '&seed=' + parseInt($('seed').value, 10) : '');
const withFlags = url => url + (url.includes('?') ? '' : '?x=1') + flags();
function summary(d) {
  return '<p>' + d.seconds + 's, ' + (d.printed ? 'printed' : 'not printed' + (d.print_error ? ' (' + d.print_error + ')' : ''))
    + (d.seed !== undefined && d.seed !== null ? ' | seed ' + d.seed : '') + '</p>'
    + (d.backgrounds && d.backgrounds.length ? '<p style="font-size:13px;color:#999">' + d.backgrounds.join('<br>') + '</p>' : '');
}

/* ---------- sliders ---------- */
const fmt = {size: v => Math.round(v * 100) + '%', outline: v => v + 'px', estrength: v => Math.round(v * 100) + '%', pop: v => Math.round(v * 100) + '%', soften: v => Math.round(v * 100) + '%', morph: v => Math.round(v * 100) + '%'};
function syncLabels() { for (const id of Object.keys(fmt)) $(id + 'V').textContent = fmt[id]($(id).value); }
syncLabels();
const sliderQuery = () => '&size=' + $('size').value + '&outline=' + $('outline').value + '&soften=' + $('soften').value
  + '&morph=' + $('morph').value + '&anchor=' + $('anchor').value
  + '&edge=' + $('edge').value + '&strength=' + $('estrength').value + '&pop=' + $('pop').value;
function later(fn, ms) { clearTimeout(timer); timer = setTimeout(fn, ms); }
for (const id of Object.keys(fmt)) $(id).addEventListener('input', () => { syncLabels(); if (hasDraft) later(() => renderDraft(''), 450); });
$('anchor').addEventListener('change', () => hasDraft && renderDraft(''));
$('edge').addEventListener('change', () => hasDraft && renderDraft(''));
$('vintage').addEventListener('change', () => hasDraft && renderDraft(''));
$('shots').addEventListener('change', () => hasDraft && renderDraft(''));

function applyMode() {
  const studio = mode() === 'studio';
  $('studio').style.display = studio ? '' : 'none';
  $('sliders').style.display = studio ? '' : 'none';
  $('directOnly').style.display = studio ? 'none' : '';
  $('between').parentElement.style.display = mode() === 'strip' ? '' : 'none';
}
document.querySelectorAll('input[name=mode]').forEach(r => r.addEventListener('change', applyMode));
applyMode();

/* ---------- studio ---------- */
async function renderDraft(extra) {
  if (!hasDraft) return;
  $('studioStatus').textContent = 'Rendering...';
  const d = await post('/api/photobooth/draft/render?shots=' + parseInt($('shots').value, 10) + sliderQuery() + vintageFlag() + (extra || ''));
  if (d.status !== 'success') { $('studioStatus').innerHTML = '<span class="err">' + d.message + '</span>'; return; }
  showDraft(d);
}
const PH_STRIP = '<figure><div class="slot" style="width:384px;height:520px">colour reel</div><figcaption>colour reel</figcaption></figure>'
  + '<figure><div class="slot" style="width:384px;height:520px">black and white print</div><figcaption>what the printer will print</figcaption></figure>';
function setTools(on) { for (const id of ['shuffle', 'useSeed', 'applySeed', 'retake', 'printBtn']) $(id).disabled = !on; }
function previewImg(name, caption, t) {
  return '<figure><img data-name="' + name + '" src="/api/photobooth/draft/image/' + name + '?t=' + t + '" onload="fitPreview(this)"><figcaption>' + caption + '</figcaption></figure>';
}
// The print is 384 dots wide. Shown 1:1 (or enlarged) it is crisp with hard pixels; shown smaller than its real size, hard pixels
// (nearest neighbour) drop dots and the picture falls apart, so a scaled-down view must be smoothed instead.
function fitPreview(img) {
  const fit = $('previewSize').value === 'fit';
  img.style.width = ''; img.style.height = fit ? '560px' : '';
  img.style.maxWidth = fit ? 'none' : '100%';
  const shrunk = img.naturalWidth > 0 && img.clientWidth < img.naturalWidth;
  img.style.imageRendering = (img.dataset.name === 'print' && !shrunk) ? 'pixelated' : 'auto';
}
function refitPreviews() { document.querySelectorAll('#previews img').forEach(fitPreview); }
$('previewSize').addEventListener('change', refitPreviews);
window.addEventListener('resize', refitPreviews);

function showDraft(d) {
  const t = Date.now();
  setTools(true);
  $('previews').innerHTML = previewImg('strip', 'colour reel', t) + previewImg('print', 'what the printer will print (203 dpi, 1-bit)', t);
  let html = '';
  for (let i = 0; i < d.shots; i++) {
    const pos = (d.frame_anchor && d.frame_anchor[i]) || 'auto';
    const taken = (d.retaken || []).includes(i);
    const opt = v => '<option value="' + v + '"' + (pos === v ? ' selected' : '') + '>' + (v === 'auto' ? 'auto (' + $('anchor').value + ')' : v) + '</option>';
    html += '<div class="thumb"><img style="width:180px;display:block" src="/api/photobooth/draft/image/shot_' + i + '?t=' + t + '">'
      + '<div class="ftools"><b>Frame ' + (i + 1) + '</b><br>'
      + '<button onclick="renderDraft(\'&reroll=' + i + '\')">reroll background</button><br>'
      + '<label>stand <select onchange="renderDraft(\'&frame=' + i + '&position=\' + this.value)">' + ['auto', 'left', 'center', 'right'].map(opt).join('') + '</select></label><br>'
      + '<button onclick="retakeFrame(' + i + ')">retake photo' + (taken ? ' (again)' : '') + '</button>'
      + (taken ? '<button onclick="renderDraft(\'&restore=' + i + '\')">use original</button>' : '')
      + '</div><div style="font-size:12px;color:#999;max-width:190px">' + d.backgrounds[i] + '</div></div>';
  }
  $('thumbs').innerHTML = html;
  const people = d.people + ' person' + (d.people === 1 ? '' : 's') + (d.ignored ? ' (+' + d.ignored + ' ignored)' : '');
  $('studioStatus').innerHTML = 'seed ' + d.seed + ' | ' + people + ' | rendered in ' + d.seconds + 's | nothing printed yet';
}
async function retakeFrame(i) {
  if (!hasDraft) return;
  await ctx.resume();
  setTools(false);
  $('studioStatus').textContent = 'Retaking frame ' + (i + 1) + ': get into position, the photo is taken on the long beep...';
  await countdown(parseInt($('retakeSecs').value, 10));
  const r = await post('/api/photobooth/draft/retake?index=' + i);
  setTools(true);
  if (r.status !== 'success') { $('studioStatus').innerHTML = '<span class="err">' + r.message + '</span>'; return; }
  await renderDraft('');
}
$('shuffle').onclick = () => renderDraft('&reroll_all=1');
$('applySeed').onclick = () => $('useSeed').value && renderDraft('&seed=' + parseInt($('useSeed').value, 10));
$('retake').onclick = () => { aimOn = true; f.style.display = ''; $('photoCard').scrollIntoView({behavior: 'smooth'}); };
$('printBtn').onclick = async () => {
  const btn = $('printBtn'); btn.disabled = true; $('studioStatus').textContent = 'Printing... (about 15 s for a 4 frame reel)';
  const d = await post('/api/photobooth/draft/print');
  $('studioStatus').innerHTML = d.status === 'success'
    ? (d.printed ? 'Printed in ' + d.seconds + 's | seed ' + d.seed : '<span class="err">Not printed: ' + (d.print_error || 'unknown') + '</span>')
    : '<span class="err">' + d.message + '</span>';
  btn.disabled = false;
};

async function runStudio() {
  prompt_.textContent = 'Strike your best pose! (listen for the beeps)';
  await countdown(parseInt($('secs').value, 10));
  busy = true;
  const c = await post('/api/photobooth/draft/capture');
  if (c.status !== 'success') { out.innerHTML = '<p class="err">' + c.message + '</p>'; return; }
  hasDraft = true; aimOn = false; f.style.display = 'none';
  prompt_.textContent = 'Rendering...';
  await renderDraft('&reroll_all=1');
}

/* ---------- direct modes (print immediately) ---------- */
async function runSingle() {
  await countdown(parseInt($('secs').value, 10));
  busy = true;
  const d = await post(withFlags('/api/photobooth/snap'));
  out.innerHTML = d.status === 'success' ? summary(d) + '<img style="max-width:100%" src="/api/photobooth/last_debug?t=' + Date.now() + '">' : '<p class="err">' + d.message + '</p>';
}
async function runOneShot() {
  prompt_.textContent = 'Strike your best pose!';
  await countdown(parseInt($('secs').value, 10));
  busy = true;
  prompt_.textContent = 'Printing your reel...';
  const d = await post(withFlags('/api/photobooth/strip/oneshot?shots=' + parseInt($('shots').value, 10)));
  out.innerHTML = d.status === 'success'
    ? summary(d) + '<img style="height:520px;margin:4px;background:#fff" src="/api/photobooth/last?t=' + Date.now() + '"><img style="height:520px;margin:4px;image-rendering:pixelated" src="/api/photobooth/last_print?t=' + Date.now() + '">'
    : '<p class="err">' + d.message + '</p>';
}
async function runStrip() {
  const total = parseInt($('shots').value, 10);
  const started = await post(withFlags('/api/photobooth/strip/start?shots=' + total));
  if (started.status !== 'success') { out.innerHTML = '<p class="err">' + started.message + '</p>'; return; }
  out.innerHTML = '<div id="stripThumbs"></div>';
  for (let k = 0; k < total; k++) {
    prompt_.textContent = 'Shot ' + (k + 1) + ' of ' + total + ': ' + POSES[k % POSES.length];
    await countdown(parseInt(k === 0 ? $('secs').value : $('between').value, 10));
    busy = true;
    const shot = await post(withFlags('/api/photobooth/strip/shot'));
    busy = false;
    if (shot.status !== 'success') { out.innerHTML += '<p class="err">' + shot.message + '</p>'; return; }
    $('stripThumbs').insertAdjacentHTML('beforeend', '<img style="height:120px;margin:4px" src="/api/photobooth/strip/shot/' + k + '?t=' + Date.now() + '">');
  }
  prompt_.textContent = 'Printing your reel...';
  busy = true;
  const d = await post(withFlags('/api/photobooth/strip/finish'));
  out.innerHTML += d.status === 'success'
    ? summary(d) + '<img style="height:520px;margin:4px;background:#fff" src="/api/photobooth/last?t=' + Date.now() + '"><img style="height:520px;margin:4px;image-rendering:pixelated" src="/api/photobooth/last_print?t=' + Date.now() + '">'
    : '<p class="err">' + d.message + '</p>';
}

/* ---------- booth voice ---------- */
let voiceWasRunning = false;
const vSetting = (k, v) => fetch('/api/photobooth/interact/settings?' + k + '=' + v, {method: 'POST'});
async function voiceInit() {
  const d = await (await fetch('/api/photobooth/interact/status')).json();
  if (!d.enabled) return;
  $('voicePanel').style.display = '';
  $('voiceMode').textContent = '(' + (d.simulated_audio ? 'audio simulated' : 'speaker on') + ', ' + (d.simulated_gpio ? 'eyes and mouth simulated' : 'eyes and mouth LIVE') + ')';
  const lines = await (await fetch('/api/photobooth/interact/lines')).json();
  $('vLine').innerHTML = lines.lines.map(l => '<option value="' + l.id + '">' + l.id + ' (' + l.seconds + 's): ' + l.text.slice(0, 50) + '</option>').join('');
  const s = d.settings;
  $('vVolume').value = s.volume; $('vVolumeV').textContent = Math.round(s.volume * 100) + '%';
  $('vMouth').checked = s.move_mouth; $('vEyes').checked = s.flash_eyes; $('vWelcome').checked = s.welcome; $('vPrint').checked = s.auto_print;
  $('vVolume').oninput = () => { $('vVolumeV').textContent = Math.round($('vVolume').value * 100) + '%'; vSetting('volume', $('vVolume').value); };
  $('vMouth').onchange = () => vSetting('move_mouth', $('vMouth').checked ? 1 : 0);
  $('vEyes').onchange = () => vSetting('flash_eyes', $('vEyes').checked ? 1 : 0);
  $('vWelcome').onchange = () => vSetting('welcome', $('vWelcome').checked ? 1 : 0);
  $('vPrint').onchange = () => vSetting('auto_print', $('vPrint').checked ? 1 : 0);
  $('voiceStart').onclick = async () => {
    await ctx.resume();
    aimOn = false; f.style.display = 'none';
    const r = await post('/api/photobooth/interact/start?skip_invites=' + ($('vSkip').checked ? 1 : 0));
    if (r.status !== 'success') { $('voiceStatus').innerHTML = '<span class="err">' + r.message + '</span>'; aimOn = !hasDraft; f.style.display = hasDraft ? 'none' : ''; }
  };
  $('voiceStop').onclick = () => post('/api/photobooth/interact/stop');
  $('wGo').onclick = async () => {
    const r = await post('/api/photobooth/interact/wiggle?part=' + $('wPart').value + '&period=' + $('wPeriod').value + '&cycles=' + $('wCycles').value);
    if (r.status !== 'success') $('voiceStatus').innerHTML = '<span class="err">' + r.message + '</span>';
  };
  $('vSay').onclick = async () => {
    const r = await post('/api/photobooth/interact/say?line=' + $('vLine').value);
    if (r.status !== 'success') $('voiceStatus').innerHTML = '<span class="err">' + r.message + '</span>';
  };
  setInterval(voicePoll, 700);
}
const STATE_TEXT = {idle: 'Idle', inviting: 'Inviting visitors over...', greeting: 'Greeting the visitor', holding: 'Hold still...', counting: 'Counting down',
  capturing: 'Got it!', rendering: 'Making the reel...', printing: 'Printing...', finishing: 'Signing off', cooldown: 'Cooling down'};
async function voicePoll() {
  let d;
  try { d = await (await fetch('/api/photobooth/interact/status')).json(); } catch (e) { return; }
  if (!d.enabled) return;
  let text = STATE_TEXT[d.state] || d.state;
  if (d.running && d.line) text += ' [' + d.line + ']';
  if (!d.running && d.outcome) text = 'Last: ' + d.outcome + (d.error ? ' (' + d.error + ')' : '') + (d.cooldown_s > 0 ? ', ready again in ' + d.cooldown_s + 's' : '');
  $('voiceStatus').textContent = text;
  if (d.running) { voiceWasRunning = true; aimOn = false; f.style.display = 'none'; }
  else if (voiceWasRunning) {
    voiceWasRunning = false;
    if (d.outcome === 'captured') { document.querySelector('input[name=mode][value=studio]').checked = true; applyMode(); hasDraft = true; aimOn = false; f.style.display = 'none'; await renderDraft(''); }
    else if (!hasDraft) { aimOn = true; f.style.display = ''; }
  }
}
voiceInit();

go.onclick = async () => {
  go.disabled = true; out.textContent = ''; prompt_.textContent = '';
  await ctx.resume();
  try { await ({studio: runStudio, oneshot: runOneShot, strip: runStrip, single: runSingle}[mode()])(); }
  catch (e) { out.textContent = 'Request failed'; }
  busy = false; prompt_.textContent = ''; go.disabled = false;
};
</script></body>
"""
