from __future__ import annotations

from flask import Blueprint, Response, jsonify, request, send_file

from .service import PhotoBoothService


def create_blueprint(service: PhotoBoothService) -> Blueprint:
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
                **_draft_settings(),
            )
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return _error(exc)
        return jsonify({"status": "success", **result})

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
  #count { font-size:96px; min-height:110px; }
  #prompt { font-size:28px; min-height:36px; color:#fc6; }
  .thumb { display:inline-block; margin:4px; vertical-align:top; font-size:12px; color:#999; max-width:190px; }
  .thumb img { width:180px; display:block; }
  .err { color:#f66; }
</style>
<body>
<p>Live aim view (green = detected person, right = the next shot). Walk into frame, face the camera.</p>
<img id="f" style="max-width:100%">

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
<div class="row"><button id="go">Start countdown</button></div>

<div id="prompt"></div>
<div id="count"></div>

<div id="studio" style="display:none">
  <div class="row">
    <button id="shuffle">Shuffle all backgrounds</button>
    <label>seed <input id="useSeed" type="number" placeholder="seed" style="width:8em"></label>
    <button id="applySeed">Use seed</button>
    <button id="retake">Retake photo</button>
    <button id="printBtn" style="background:#2a7;color:#fff">Print this reel</button>
  </div>
  <div id="studioStatus" class="row"></div>
  <div>
    <img id="strip" style="height:560px;margin:4px;background:#fff">
    <img id="stripPrint" style="height:560px;margin:4px;image-rendering:pixelated;background:#fff">
  </div>
  <div id="thumbs"></div>
</div>
<div id="out"></div>

<script>
const $ = id => document.getElementById(id);
const f = $('f'), count = $('count'), out = $('out'), go = $('go'), prompt_ = $('prompt');
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
async function countdown(n) {
  for (; n > 0; n--) { count.textContent = n; beep(n <= 3 ? 880 : 440, 150); await sleep(1000); }
  count.textContent = 'CHEESE'; beep(1320, 400);
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
function showDraft(d) {
  const t = Date.now();
  $('studio').style.display = '';
  $('strip').src = '/api/photobooth/draft/image/strip?t=' + t;
  $('stripPrint').src = '/api/photobooth/draft/image/print?t=' + t;
  let html = '';
  for (let i = 0; i < d.shots; i++) {
    html += '<div class="thumb"><img src="/api/photobooth/draft/image/shot_' + i + '?t=' + t + '">'
      + '<button onclick="renderDraft(\'&reroll=' + i + '\')">reroll frame ' + (i + 1) + '</button><br>' + d.backgrounds[i] + '</div>';
  }
  $('thumbs').innerHTML = html;
  const people = d.people + ' person' + (d.people === 1 ? '' : 's') + (d.ignored ? ' (+' + d.ignored + ' ignored)' : '');
  $('studioStatus').innerHTML = 'seed ' + d.seed + ' | ' + people + ' | rendered in ' + d.seconds + 's | nothing printed yet';
}
$('shuffle').onclick = () => renderDraft('&reroll_all=1');
$('applySeed').onclick = () => $('useSeed').value && renderDraft('&seed=' + parseInt($('useSeed').value, 10));
$('retake').onclick = () => { aimOn = true; f.style.display = ''; };
$('printBtn').onclick = async () => {
  const btn = $('printBtn'); btn.disabled = true; $('studioStatus').textContent = 'Printing... (about 15 s for a 4 frame reel)';
  const d = await post('/api/photobooth/draft/print');
  $('studioStatus').innerHTML = d.status === 'success'
    ? (d.printed ? 'Printed in ' + d.seconds + 's | seed ' + d.seed : '<span class="err">Not printed: ' + (d.print_error || 'unknown') + '</span>')
    : '<span class="err">' + d.message + '</span>';
  btn.disabled = false;
};

async function runStudio() {
  prompt_.textContent = 'Strike your best pose!';
  await countdown(parseInt($('secs').value, 10));
  busy = true;
  const c = await post('/api/photobooth/draft/capture');
  if (c.status !== 'success') { out.innerHTML = '<p class="err">' + c.message + '</p>'; return; }
  hasDraft = true; aimOn = false; f.style.display = 'none';
  prompt_.textContent = 'Rendering...'; count.textContent = '';
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
  prompt_.textContent = 'Printing your reel...'; count.textContent = '';
  const d = await post(withFlags('/api/photobooth/strip/oneshot?shots=' + parseInt($('shots').value, 10)));
  out.innerHTML = d.status === 'success'
    ? summary(d) + '<img style="height:520px;margin:4px;background:#fff" src="/api/photobooth/last?t=' + Date.now() + '"><img style="height:520px;margin:4px;image-rendering:pixelated" src="/api/photobooth/last_print?t=' + Date.now() + '">'
    : '<p class="err">' + d.message + '</p>';
}
async function runStrip() {
  const total = parseInt($('shots').value, 10);
  const started = await post(withFlags('/api/photobooth/strip/start?shots=' + total));
  if (started.status !== 'success') { out.innerHTML = '<p class="err">' + started.message + '</p>'; return; }
  const thumbs = $('thumbs'); $('studio').style.display = 'none'; out.innerHTML = '<div id="stripThumbs"></div>';
  for (let k = 0; k < total; k++) {
    prompt_.textContent = 'Shot ' + (k + 1) + ' of ' + total + ': ' + POSES[k % POSES.length];
    await countdown(parseInt(k === 0 ? $('secs').value : $('between').value, 10));
    busy = true;
    const shot = await post(withFlags('/api/photobooth/strip/shot'));
    busy = false;
    if (shot.status !== 'success') { out.innerHTML += '<p class="err">' + shot.message + '</p>'; return; }
    $('stripThumbs').insertAdjacentHTML('beforeend', '<img style="height:120px;margin:4px" src="/api/photobooth/strip/shot/' + k + '?t=' + Date.now() + '">');
  }
  prompt_.textContent = 'Printing your reel...'; count.textContent = '';
  busy = true;
  const d = await post(withFlags('/api/photobooth/strip/finish'));
  out.innerHTML += d.status === 'success'
    ? summary(d) + '<img style="height:520px;margin:4px;background:#fff" src="/api/photobooth/last?t=' + Date.now() + '"><img style="height:520px;margin:4px;image-rendering:pixelated" src="/api/photobooth/last_print?t=' + Date.now() + '">'
    : '<p class="err">' + d.message + '</p>';
}

go.onclick = async () => {
  go.disabled = true; out.textContent = ''; prompt_.textContent = '';
  await ctx.resume();
  try { await ({studio: runStudio, oneshot: runOneShot, strip: runStrip, single: runSingle}[mode()])(); }
  catch (e) { out.textContent = 'Request failed'; }
  busy = false; count.textContent = ''; prompt_.textContent = ''; go.disabled = false;
};
</script></body>
"""
