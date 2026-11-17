"""Replay animation from the JSON event logs only (no simulator, no JAX, no GPU).

  python viz.py --replay outputs/run/replays/Agent_LRU_tight_test0.json --mode mp4|html --fps 10
  python viz.py --replay A.json B.json [C.json D.json] --mode html        # 2 or 4 panels in lockstep
  python viz.py --scenarios --root outputs/run [--slots 200]               # the three showcase scenarios

Scene: CPU node = square, GPU node = hexagon; disk bar under each node with cached images as
coloured blocks (width = size / capacity), white outline = pinned by a running task, partial fill =
download progress. Tasks (circle = CPU service, triangle = GPU service) fly from their origin to the
chosen node; green flash = cache hit, orange ring = pull in progress, red = deadline missed.
Evicted blocks fade out in red with the victim image id.
"""
import argparse
import json
import math
from pathlib import Path

TRAVEL = 0.6          # visual flight time of a task (slots)
FADE = 1.5            # eviction ghost lifetime (slots)
OK, MISS, LATE, ACCENT = "#1baf7a", "#eb6834", "#e34948", "#2a78d6"
INK, MUTED, SURFACE, NODE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d3"


def image_colors(n):
    """Consistent per-image colours across every panel and every output (golden-angle hue walk)."""
    import colorsys
    out = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1
        l, s = (0.55, 0.65) if i % 3 == 0 else ((0.42, 0.7) if i % 3 == 1 else (0.68, 0.6))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        out.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
    return out


# ------------------------------------------------------------------------------------ preprocessing
def prepare(path, slots=None):
    """JSON log -> panel dict with tasks, cache residency intervals and per-slot metrics."""
    raw = json.loads(Path(path).read_text())
    S = min(slots or raw["slots"], raw["slots"])
    tasks = []
    for t in raw["tasks"]:
        tid, slot, gpu, img, node, outcome, evicted, dls, dlr, fin, met, x, y, ddl = t
        if slot >= S:
            continue
        tasks.append(dict(id=tid, t=float(slot), gpu=gpu, img=img, node=node, out=outcome, ev=evicted,
                          dls=dls, dlr=dlr, fin=fin, met=met, x=x, y=y, ddl=ddl))
    # Cache residency intervals per node, reconstructed from miss / eviction events.
    entries, open_ = [], {}
    for t in tasks:
        if t["node"] < 0:
            continue
        for v in t["ev"]:
            e = open_.pop((t["node"], v), None)
            if e is None:                     # resident before the log window (not possible: caches start empty)
                e = dict(node=t["node"], img=v, tin=0., dls=0., dlr=0.)
                entries.append(e)
            e["tout"] = t["t"]
        if t["out"] == 3:
            e = dict(node=t["node"], img=t["img"], tin=t["t"], dls=t["dls"], dlr=t["dlr"], tout=None)
            open_[(t["node"], t["img"])] = e
            entries.append(e)
    for e in entries:
        e["size"] = raw["images"][e["img"]]["size"]
    # Per-slot cumulative metrics (computed from events known by the end of each slot).
    metrics = []
    for s in range(S):
        now = s + 1.0
        arrived = [t for t in tasks if t["t"] < now]
        done = [t for t in arrived if t["node"] < 0 or (t["fin"] is not None and t["fin"] <= now)]
        lat = [(t["fin"] - t["t"]) if t["node"] >= 0 else (t["ddl"] - t["t"] + 5) for t in done]
        win = [t for t in arrived if t["t"] >= now - 10]
        metrics.append(dict(lat=sum(lat) / len(lat) if lat else 0., hit=sum(t["out"] == 1 for t in arrived) / max(1, len(arrived)),
                            ev=sum(len(t["ev"]) for t in arrived), ontime=sum(t["met"] for t in done) / max(1, len(done)),
                            roll=sum(t["out"] == 1 for t in win) / max(1, len(win)), n=len(arrived)))
    title = f"{raw['method']} · {raw['policy']} · {raw['disk']} disk ({100 * raw['disk_frac']:.0f}% of catalogue/node)"
    return dict(title=title, method=raw["method"], policy=raw["policy"], disk=raw["disk"], slots=S, nodes=spread(raw["nodes"]),
                capacity=raw["capacity"], images=raw["images"], tasks=tasks, entries=entries, metrics=metrics,
                trace=raw.get("trace", ""))


def spread(nodes, dx=13.0, dy=9.0, iters=300):
    """Display-only layout: push apart nodes whose glyph + disk bar would overlap (simulation untouched)."""
    pts = [[n["x"], n["y"]] for n in nodes]
    for _ in range(iters):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                ox, oy = dx - abs(pts[i][0] - pts[j][0]), dy - abs(pts[i][1] - pts[j][1])
                if ox > 0 and oy > 0:
                    moved = True
                    if ox < oy * dx / dy:
                        sgn = 1 if pts[i][0] >= pts[j][0] else -1
                        pts[i][0] += sgn * ox / 2; pts[j][0] -= sgn * ox / 2
                    else:
                        sgn = 1 if pts[i][1] >= pts[j][1] else -1
                        pts[i][1] += sgn * oy / 2; pts[j][1] -= sgn * oy / 2
        for q in pts:
            q[0], q[1] = min(94, max(6, q[0])), min(92, max(6, q[1]))
        if not moved:
            break
    return [dict(n, x=round(q[0], 2), y=round(q[1], 2)) for n, q in zip(nodes, pts)]


def state_at(P, t):
    """Everything to draw at continuous time t (slots)."""
    caches = [[] for _ in P["nodes"]]
    ghosts = []
    for e in P["entries"]:
        if e["tin"] <= t and (e["tout"] is None or t < e["tout"]):
            caches[e["node"]].append(e)
        elif e["tout"] is not None and e["tout"] <= t < e["tout"] + FADE:
            ghosts.append(e)
    pinned = set()
    live = []
    for k in P["tasks"]:
        if k["t"] > t:
            break
        end = k["fin"] if k["node"] >= 0 else k["t"] + TRAVEL + 1
        if t < end + 0.3:
            live.append(k)
        if k["node"] >= 0 and k["t"] <= t < (k["fin"] or 0):
            pinned.add((k["node"], k["img"]))
    return caches, ghosts, pinned, live


# ------------------------------------------------------------------------------------ matplotlib
def render_video(panels, out, fps=10, frames_per_slot=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Rectangle, RegularPolygon, Circle, Wedge
    colors = image_colors(len(panels[0]["images"]))
    n = len(panels)
    rows, cols = (1, n) if n <= 2 else (2, 2)
    fig = plt.figure(figsize=(7.0 * cols, 7.4 * rows), facecolor=SURFACE)
    gs = fig.add_gridspec(rows * 6, cols, hspace=0.45, wspace=0.06, left=.02, right=.98, top=.93 if rows == 1 else .95,
                          bottom=.07 if rows == 1 else .04)
    fig.text(.5, .008, "■ CPU node   ⬢ GPU node   ● CPU service   ▲ GPU service   green ring = cache hit   orange arc = image pull"
             "   red = evicted image / missed deadline   black outline = pinned   bar = disk (block width = image size)",
             ha="center", color=MUTED, fontsize=8.5)
    scenes, infos, sparks = [], [], []
    for i in range(n):
        r, c = divmod(i, cols)
        scenes.append(fig.add_subplot(gs[r * 6:r * 6 + 5, c]))
        sub = gs[r * 6 + 5, c].subgridspec(1, 2, width_ratios=(3, 1), wspace=0.05)
        infos.append(fig.add_subplot(sub[0]))
        sparks.append(fig.add_subplot(sub[1]))
    S = min(p["slots"] for p in panels)
    total = S * frames_per_slot
    sup = fig.suptitle("", color=INK, fontsize=13)

    def draw(f):
        t = f / frames_per_slot
        sup.set_text(f"Trace {panels[0]['trace']}   ·   slot {int(t):3d} / {S}")
        for P, ax, ai, sp in zip(panels, scenes, infos, sparks):
            ax.clear(); ai.clear(); sp.clear()
            ax.set_xlim(-2, 104); ax.set_ylim(106, -4); ax.set_aspect("equal"); ax.axis("off")
            ax.set_facecolor(SURFACE)
            ax.set_title(P["title"], color=INK, fontsize=10.5)
            caches, ghosts, pinned, live = state_at(P, t)
            cap = P["capacity"]
            for ni, nd in enumerate(P["nodes"]):
                x, y = nd["x"], nd["y"]
                if nd["gpu"]:
                    ax.add_patch(RegularPolygon((x, y), 6, radius=3.0, facecolor=NODE, edgecolor=MUTED, lw=1))
                else:
                    ax.add_patch(Rectangle((x - 2.4, y - 2.4), 4.8, 4.8, facecolor=NODE, edgecolor=MUTED, lw=1))
                bw, bx, by = 10., x - 5., y + 3.6
                ax.add_patch(Rectangle((bx, by), bw, 1.8, facecolor="white", edgecolor=MUTED, lw=.6))
                off = 0.
                for e in sorted(caches[ni], key=lambda e: e["tin"]):
                    w = bw * e["size"] / cap
                    prog = 1. if e["dlr"] is None or t >= e["dlr"] else max(0., (t - e["dls"]) / max(1e-6, e["dlr"] - e["dls"]))
                    ax.add_patch(Rectangle((bx + off, by), w, 1.8, facecolor=colors[e["img"]], alpha=.28, lw=0))
                    ax.add_patch(Rectangle((bx + off, by + 1.8 * (1 - prog)), w, 1.8 * prog, facecolor=colors[e["img"]], lw=0))
                    if (ni, e["img"]) in pinned:
                        ax.add_patch(Rectangle((bx + off, by), w, 1.8, fill=False, edgecolor=INK, lw=1.1))
                    off += w
            for e in ghosts:
                nd = P["nodes"][e["node"]]
                a = max(0., 1 - (t - e["tout"]) / FADE)
                ax.add_patch(Rectangle((nd["x"] - 5, nd["y"] + 3.6), 10 * e["size"] / cap, 1.8, facecolor=LATE, alpha=.8 * a, lw=0))
                ax.text(nd["x"] + 5.5, nd["y"] - 3, f"−{e['img']}", color=LATE, alpha=a, fontsize=8, weight="bold")
            for k in live:
                if k["node"] < 0:
                    continue
                nd = P["nodes"][k["node"]]
                u = min(1., (t - k["t"]) / TRAVEL)
                ang = (k["id"] * 2.39996) % (2 * math.pi)
                tx, ty = nd["x"] + 4.2 * math.cos(ang), nd["y"] + 4.2 * math.sin(ang)
                px, py = k["x"] + (tx - k["x"]) * u, k["y"] + (ty - k["y"]) * u
                late = t > k["ddl"] and (k["fin"] is None or t < k["fin"] or not k["met"])
                col = LATE if late else colors[k["img"]]
                if k["gpu"]:
                    ax.add_patch(RegularPolygon((px, py), 3, radius=1.3, facecolor=col, edgecolor="white", lw=.6))
                else:
                    ax.add_patch(Circle((px, py), 1.0, facecolor=col, edgecolor="white", lw=.6))
                if u >= 1:
                    if k["out"] == 1 and t - k["t"] < TRAVEL + .5:
                        ax.add_patch(Circle((nd["x"], nd["y"]), 3.8 + 3 * (t - k["t"] - TRAVEL), fill=False, edgecolor=OK, lw=2,
                                            alpha=max(0, 1 - 2 * (t - k["t"] - TRAVEL))))
                    elif k["out"] in (2, 3) and k["dlr"] and t < k["dlr"]:
                        prog = max(0., min(1., (t - k["dls"]) / max(1e-6, k["dlr"] - k["dls"]))) if k["dls"] is not None else 0
                        ax.add_patch(Circle((px, py), 1.9, fill=False, edgecolor=MISS, lw=.8, alpha=.5))
                        ax.add_patch(Wedge((px, py), 2.1, 90 - 360 * prog, 90, width=.5, facecolor=MISS, lw=0))
            m = P["metrics"][min(int(t), len(P["metrics"]) - 1)]
            ai.axis("off")
            ai.text(0, .55, f"avg latency {m['lat']:.2f}   hit rate {m['hit']:.1%}\nevictions {m['ev']:d}   on-time {m['ontime']:.1%}",
                    color=INK, fontsize=10, va="center", family="monospace")
            roll = [q["roll"] for q in P["metrics"][:int(t) + 1]]
            sp.plot(roll, color=ACCENT, lw=1.5)
            sp.set_xlim(0, S); sp.set_ylim(0, 1); sp.set_xticks([]); sp.set_yticks([0, 1])
            sp.tick_params(labelsize=7, colors=MUTED)
            for s in ("top", "right"):
                sp.spines[s].set_visible(False)
            sp.set_title("rolling hit rate", fontsize=7, color=MUTED)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=total, interval=1000 / fps, blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg unavailable")
        anim.save(out.with_suffix(".mp4"), writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=90)
        result = out.with_suffix(".mp4")
    except Exception as exc:                       # GIF fallback (Pillow)
        print(f"  mp4 failed ({exc}); writing GIF", flush=True)
        anim.save(out.with_suffix(".gif"), writer=animation.PillowWriter(fps=fps), dpi=60)
        result = out.with_suffix(".gif")
    plt.close(fig)
    print(f"  wrote {result}", flush=True)
    return result


# ------------------------------------------------------------------------------------ HTML
HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CHES Replay</title><style>
:root{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;--node:#d9d8d3;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#222220;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--node:#4a4945;--accent:#3987e5}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#222220;--ink:#f0efec;--muted:#c3c2b7;--line:#3a3936;--node:#4a4945;--accent:#3987e5}
body{margin:0;background:var(--surface);color:var(--ink);font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:12px 16px 4px}h1{font-size:18px;margin:0}p.sub{margin:2px 0 0;color:var(--muted);font-size:12.5px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:8px 16px;position:sticky;top:0;background:var(--surface);z-index:2;border-bottom:1px solid var(--line)}
button,select{font:inherit;padding:5px 12px;border-radius:6px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
input[type=range]{flex:1;min-width:140px;accent-color:var(--accent)}#slot{font-variant-numeric:tabular-nums;min-width:110px}
.grid{display:grid;gap:12px;padding:12px 16px;grid-template-columns:repeat(var(--cols),minmax(0,1fr))}
@media (max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.card h2{font-size:13.5px;margin:2px 0 4px;font-weight:600}canvas{width:100%;display:block}
.stats{display:grid;grid-template-columns:repeat(4,1fr) 1.3fr;gap:6px;align-items:end;margin-top:4px}
.stat b{display:block;font-size:16px;font-variant-numeric:tabular-nums}.stat span{font-size:11px;color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:14px;padding:0 16px 14px;color:var(--muted);font-size:12px}
.legend i{display:inline-block;width:11px;height:11px;margin-right:5px;vertical-align:-1px;border-radius:2px}
</style></head><body>
<header><h1>CHES scheduling replay</h1><p class="sub" id="sub"></p></header>
<div class="bar"><button id="play">▶ Play</button><input type="range" id="seek" min="0" step="0.01" value="0">
<span id="slot"></span><label>speed <select id="speed"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="8">8×</option></select></label></div>
<div class="grid" id="grid"></div>
<div class="legend"><span>■ CPU node</span><span>⬢ GPU node</span><span>● CPU service</span><span>▲ GPU service</span>
<span><i style="background:#1baf7a"></i>cache hit</span><span><i style="background:#eb6834"></i>image pull in progress</span>
<span><i style="background:#e34948"></i>evicted / deadline missed</span><span><i style="border:2px solid var(--ink);box-sizing:border-box"></i>pinned image</span>
<span>disk bar = node capacity; block width = image size; block colour = image id (same in every panel)</span></div>
<script>
const D=__DATA__;const TRAVEL=0.6,FADE=1.5,OK="#1baf7a",MISS="#eb6834",LATE="#e34948";
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const S=Math.min(...D.panels.map(p=>p.slots));const grid=document.getElementById("grid");
grid.style.setProperty("--cols",D.panels.length===1?1:2);
document.getElementById("sub").textContent=D.subtitle;
const seek=document.getElementById("seek");seek.max=S-0.01;
const views=D.panels.map((P,i)=>{const c=document.createElement("div");c.className="card";
c.innerHTML=`<h2>${P.title}</h2><canvas></canvas><div class="stats"><div class="stat"><b>–</b><span>avg latency (slots)</span></div><div class="stat"><b>–</b><span>hit rate</span></div><div class="stat"><b>–</b><span>evictions</span></div><div class="stat"><b>–</b><span>on-time</span></div><div class="stat"><canvas class="spark" height="34"></canvas><span>rolling hit rate</span></div></div>`;
grid.appendChild(c);return{P,cv:c.querySelector("canvas"),sp:c.querySelector(".spark"),st:[...c.querySelectorAll(".stat b")]};});
function fit(cv,h){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=r.width*d;cv.height=(h||r.width*0.95)*d;cv.style.height=(h||r.width*0.95)+"px";const g=cv.getContext("2d");g.setTransform(d,0,0,d,0,0);return[g,r.width];}
function hex(g,x,y,r){g.beginPath();for(let k=0;k<6;k++){const a=Math.PI/3*k+Math.PI/6;g.lineTo(x+r*Math.cos(a),y+r*Math.sin(a));}g.closePath();}
function tri(g,x,y,r){g.beginPath();for(let k=0;k<3;k++){const a=-Math.PI/2+2*Math.PI/3*k;g.lineTo(x+r*Math.cos(a),y+r*Math.sin(a));}g.closePath();}
function draw(t){document.getElementById("slot").textContent=`slot ${Math.floor(t)} / ${S}`;
for(const V of views){const P=V.P;const[g,W]=fit(V.cv);const s=W/108,X=x=>(x+2)*s,Y=y=>(y+4)*s;
g.fillStyle=css("--panel");g.fillRect(0,0,W,W);const cap=P.capacity;const pinned=new Set();const live=[];
for(const k of P.tasks){if(k.t>t)break;const end=k.node>=0?k.fin:k.t+TRAVEL+1;if(t<end+0.3)live.push(k);if(k.node>=0&&k.t<=t&&t<(k.fin||0))pinned.add(k.node*1000+k.img);}
P.nodes.forEach((nd,ni)=>{g.fillStyle=css("--node");g.strokeStyle=css("--muted");g.lineWidth=1;
if(nd.gpu){hex(g,X(nd.x),Y(nd.y),3*s);g.fill();g.stroke();}else{g.fillRect(X(nd.x-2.4),Y(nd.y-2.4),4.8*s,4.8*s);g.strokeRect(X(nd.x-2.4),Y(nd.y-2.4),4.8*s,4.8*s);}
const bx=nd.x-5,by=nd.y+3.6;g.fillStyle="#fff";g.fillRect(X(bx),Y(by),10*s,1.8*s);g.lineWidth=.6;g.strokeRect(X(bx),Y(by),10*s,1.8*s);
let off=0;for(const e of P.entries){if(e.node!==ni||e.tin>t||(e.tout!==null&&t>=e.tout))continue;const w=10*e.size/cap;
const prog=(e.dlr===null||t>=e.dlr)?1:Math.max(0,(t-e.dls)/Math.max(1e-6,e.dlr-e.dls));g.globalAlpha=.28;g.fillStyle=D.colors[e.img];g.fillRect(X(bx+off),Y(by),w*s,1.8*s);
g.globalAlpha=1;g.fillRect(X(bx+off),Y(by+1.8*(1-prog)),w*s,1.8*prog*s);if(pinned.has(ni*1000+e.img)){g.strokeStyle=css("--ink");g.lineWidth=1.2;g.strokeRect(X(bx+off),Y(by),w*s,1.8*s);}off+=w;}});
for(const e of P.entries){if(e.tout===null||t<e.tout||t>=e.tout+FADE)continue;const nd=P.nodes[e.node],a=1-(t-e.tout)/FADE;g.globalAlpha=.85*a;g.fillStyle=LATE;
g.fillRect(X(nd.x-5),Y(nd.y+3.6),10*e.size/cap*s,1.8*s);g.font=`bold ${Math.max(9,2.4*s)}px system-ui`;g.fillText("−"+e.img,X(nd.x+5.5),Y(nd.y-3));g.globalAlpha=1;}
for(const k of live){if(k.node<0)continue;const nd=P.nodes[k.node];const u=Math.min(1,(t-k.t)/TRAVEL);const ang=(k.id*2.39996)%(2*Math.PI);
const tx=nd.x+4.2*Math.cos(ang),ty=nd.y+4.2*Math.sin(ang);const px=k.x+(tx-k.x)*u,py=k.y+(ty-k.y)*u;
const late=t>k.ddl&&(k.fin===null||t<k.fin||!k.met);g.fillStyle=late?LATE:D.colors[k.img];g.strokeStyle="#fff";g.lineWidth=1;
if(k.gpu){tri(g,X(px),Y(py),1.4*s);}else{g.beginPath();g.arc(X(px),Y(py),1.05*s,0,7);}g.fill();g.stroke();
if(u>=1){const dt=t-k.t-TRAVEL;if(k.out===1&&dt<.5){g.globalAlpha=Math.max(0,1-2*dt);g.strokeStyle=OK;g.lineWidth=2.5;g.beginPath();g.arc(X(nd.x),Y(nd.y),(3.8+3*dt)*s,0,7);g.stroke();g.globalAlpha=1;}
else if((k.out===2||k.out===3)&&k.dlr&&t<k.dlr){const pr=Math.max(0,Math.min(1,(t-k.dls)/Math.max(1e-6,k.dlr-k.dls)));g.strokeStyle=MISS;g.lineWidth=.8;g.globalAlpha=.45;g.beginPath();g.arc(X(px),Y(py),2*s,0,7);g.stroke();g.globalAlpha=1;g.lineWidth=2.4;g.beginPath();g.arc(X(px),Y(py),2*s,-Math.PI/2,-Math.PI/2+2*Math.PI*pr);g.stroke();}}}
const m=P.metrics[Math.min(Math.floor(t),P.metrics.length-1)];V.st[0].textContent=m.lat.toFixed(2);V.st[1].textContent=(100*m.hit).toFixed(1)+"%";V.st[2].textContent=m.ev;V.st[3].textContent=(100*m.ontime).toFixed(1)+"%";
const[sg,sw]=fit(V.sp,34);sg.clearRect(0,0,sw,34);sg.strokeStyle=css("--line");sg.strokeRect(0,1,sw,32);sg.strokeStyle=css("--accent");sg.lineWidth=1.6;sg.beginPath();
P.metrics.slice(0,Math.floor(t)+1).forEach((q,i)=>{const x=i/S*sw,y=33-32*q.roll;i?sg.lineTo(x,y):sg.moveTo(x,y);});sg.stroke();}}
let t=0,playing=false,last=null;const btn=document.getElementById("play");
btn.onclick=()=>{playing=!playing;btn.textContent=playing?"❚❚ Pause":"▶ Play";last=null;if(playing&&t>=S-0.02)t=0;requestAnimationFrame(loop);};
seek.oninput=()=>{t=+seek.value;draw(t);};
function loop(ts){if(!playing)return;if(last!==null){t+=(ts-last)/1000*(+document.getElementById("speed").value)*1.33;}last=ts;
if(t>=S-0.01){t=S-0.01;playing=false;btn.textContent="▶ Play";}seek.value=t;draw(t);if(playing)requestAnimationFrame(loop);}
addEventListener("resize",()=>draw(t));draw(0);
</script></body></html>"""


def render_html(panels, out, subtitle=""):
    data = dict(panels=[{k: P[k] for k in ("title", "slots", "nodes", "capacity", "tasks", "entries", "metrics")} for P in panels],
                colors=image_colors(len(panels[0]["images"])), subtitle=subtitle)
    out = Path(out).with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(f"  wrote {out}", flush=True)
    return out


# ------------------------------------------------------------------------------------ scenarios / CLI
def scenarios(root, policy="LRU"):
    r = Path(root) / "replays"
    return {
        "agent_vs_greedy": ([r / f"Agent_{policy}_tight_test0.json", r / f"Greedy_latency_{policy}_tight_test0.json"],
                            f"CHES agent vs the best heuristic (Greedy_latency), {policy} eviction, tight disk"),
        "four_policies": ([r / f"Agent_{p}_tight_test0.json" for p in ("LRU", "LFU", "FIFO", "GDSF")],
                          "CHES agent (one trained network per policy) under LRU / LFU / FIFO / GDSF eviction, tight disk"),
        "disk_pressure": ([r / f"Agent_{policy}_large_test0.json", r / f"Agent_{policy}_tight_test0.json"],
                          f"Same workload, large vs tight disk ({policy}): eviction pressure appears"),
    }


def render(paths, mode, fps, slots, out, subtitle=""):
    panels = [prepare(p, slots) for p in paths]
    if len({P["trace"] for P in panels}) > 1:
        raise ValueError("All panels must replay the same trace")
    if mode in ("html", "both"):
        render_html(panels, out, subtitle)
    if mode in ("mp4", "both"):
        render_video(panels, out, fps)


def render_scenarios(root, fps=10, slots=200, mode="both"):
    figs = Path(root) / "figures"
    for name, (paths, sub) in scenarios(root).items():
        print(f"Scenario {name}", flush=True)
        render(paths, mode, fps, slots, figs / f"replay_{name}", sub)


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--replay", nargs="+", help="1, 2 or 4 replay JSON files (same trace)")
    pa.add_argument("--mode", choices=("mp4", "html", "both"), default="html")
    pa.add_argument("--fps", type=int, default=10)
    pa.add_argument("--slots", type=int, default=None, help="limit the number of slots rendered")
    pa.add_argument("--out", default=None)
    pa.add_argument("--scenarios", action="store_true")
    pa.add_argument("--smoke", action="store_true", help="scenarios with a few slots only")
    pa.add_argument("--root", default="outputs/run")
    a = pa.parse_args()
    if a.scenarios or a.smoke:
        render_scenarios(a.root, a.fps, 8 if a.smoke else (a.slots or 200), "both" if a.smoke else a.mode)
    elif a.replay:
        if len(a.replay) not in (1, 2, 4):
            pa.error("use 1, 2 or 4 replay files")
        out = a.out or str(Path(a.replay[0]).parent.parent / "figures" / ("replay_" + "_vs_".join(Path(p).stem for p in a.replay)))
        render(a.replay, a.mode, a.fps, a.slots, out)
    else:
        pa.error("use --replay or --scenarios")


if __name__ == "__main__":
    main()
