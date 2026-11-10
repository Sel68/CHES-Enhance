"""Replay JSON -> matplotlib video/GIF or dependency-free, self-contained HTML.

No simulator, JAX, torch, or GPU is imported. HTML mode uses only the stdlib.
"""
import argparse
import json
import math
import os
from pathlib import Path

PALETTE = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#F0E442", "#0072B2", "#D55E00", "#88CCAA"]


def load_panels(paths):
    panels = []
    for path in paths:
        value = json.loads(Path(path).read_text())
        panels.extend(value["panels"] if "panels" in value else [value])
    if len(panels) not in (1, 2, 4):
        raise ValueError("Provide 1, 2, or 4 panels")
    if len({(p["trace"], p["slot_seconds"]) for p in panels}) != 1:
        raise ValueError("Comparison panels must use the same trace and slot duration")
    for p in panels:
        if p["version"] != 1 or not p["frames"]:
            raise ValueError("Unsupported or empty replay")
    return panels


def visible_state(panel, t):
    slot = min(panel["slots"] - 1, int(t / panel["slot_seconds"]))
    frames = [f for f in panel["frames"] if f["slot"] <= slot]
    frame = frames[-1] if frames else panel["frames"][0]
    arrived = [e for e in panel["tasks"] if e["arrival"] <= t]
    completed = [e for e in arrived if not e["rejected"] and e["finish"] <= t]
    latency = sum(e["latency"] for e in completed) / max(1, len(completed))
    hit = sum(e["hit"] for e in arrived) / max(1, len(arrived))
    evictions = sum(len(e["evicted"]) for e in arrived)
    ratio = sum(e["deadline_met"] for e in completed) / max(1, len(arrived))
    return slot, frame, arrived, completed, (latency, hit, evictions, ratio)


def export_video(panels, output, fps=10, slots=None):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ches-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter, writers
    from matplotlib.patches import Rectangle, Circle, Arc
    count = len(panels)
    rows, cols = (2, 2) if count == 4 else (1, count)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6.2 * rows), squeeze=False, facecolor="#0B1220")
    fig.subplots_adjust(wspace=.08, hspace=.16, top=.93, bottom=.04, left=.025, right=.975)
    duration = min(p["slots"] for p in panels)
    if slots:
        duration = min(duration, slots)
    # 0.75 screen seconds per simulator slot; speed is explicit in HTML too.
    frame_count = max(2, int(duration * .75 * fps))
    def draw(frame_index):
        t = (frame_index / (frame_count - 1)) * max(0., duration * panels[0]["slot_seconds"] - .001)
        for ax, p in zip(axes.flat, panels):
            ax.clear(); ax.set_facecolor("#111C2F")
            ax.set_xlim(-5, 105); ax.set_ylim(-27, 112); ax.set_aspect("equal"); ax.axis("off")
            slot, state, arrived, completed, counters = visible_state(p, t)
            pinned_images = {(e["node"], e["image"]) for e in arrived if not e["rejected"] and e["finish"] > t}
            ax.text(0, 108, f"{p['method']}  /  {p['policy']}", color="white", fontsize=13, weight="bold")
            ax.text(0, 102, f"{p['disk']} disk ×{p['disk_scale']:.3f}    SLOT {slot:03d}    t={t:.1f}s", color="#94A3B8", fontsize=10)
            ax.plot([0, 100], [99, 99], color="#293951", lw=1)
            for n, cache in zip(p["nodes"], state["nodes"]):
                x, y = n["x"], n["y"]
                color = "#A9C4DC" if n["type"] == "CPU" else "#CC79A7"
                ax.scatter([x], [y], marker="s" if n["type"] == "CPU" else "h", s=230, c=color, edgecolor="#DBEAFE", linewidth=.8, zorder=5)
                ax.text(x, y + 4.2, f"{n['type'][0]}{n['id']}", ha="center", fontsize=7, color="#B9C8DA")
                left, yy = x - 7, y - 6
                ax.add_patch(Rectangle((left, yy), 14, 2.8, facecolor="#263349", edgecolor="#526379", lw=.7))
                for c in cache:
                    width = 14 * c["size"] / n["capacity"]
                    progress = min(1., max(0., (t - c["start"]) / max(1e-9, c["ready"] - c["start"])))
                    pinned = (n["id"], c["image"]) in pinned_images
                    color = PALETTE[c["image"] % len(PALETTE)]
                    ax.add_patch(Rectangle((left, yy), width, 2.8, facecolor=color, alpha=.2, edgecolor="none"))
                    ax.add_patch(Rectangle((left, yy), width * progress, 2.8, facecolor=color, edgecolor="none"))
                    ax.add_patch(Rectangle((left, yy), width, 2.8, fill=False, edgecolor="white" if pinned else "#111C2F", lw=1.2 if pinned else .4))
                    left += width
            for e in arrived:
                age = t - e["arrival"]
                if e["node"] < 0:
                    if age < p["slot_seconds"]:
                        ax.scatter(*e["origin"], c="#EE6677", marker="x", s=36)
                    continue
                n = p["nodes"][e["node"]]
                visual_travel = min(.7 * p["slot_seconds"], max(.45 * p["slot_seconds"], e["upload_finish"] - e["arrival"]))
                progress = min(1., age / visual_travel)
                x = e["origin"][0] + (n["x"] - e["origin"][0]) * progress
                y = e["origin"][1] + (n["y"] - e["origin"][1]) * progress
                late = t > e["deadline"] and not e["deadline_met"]
                if age < visual_travel or e["finish"] > t or t - e["finish"] < .8:
                    ax.plot([e["origin"][0], n["x"]], [e["origin"][1], n["y"]], color=PALETTE[e["image"] % len(PALETTE)], alpha=.12, lw=.7)
                    ax.scatter([x], [y], marker="^" if e["type"] == "GPU" else "o", c="#EE6677" if late else PALETTE[e["image"] % len(PALETTE)], s=25, edgecolor="white", linewidth=.3, zorder=8)
                if e["hit"] and visual_travel <= age < visual_travel + p["slot_seconds"] * .6:
                    ax.add_patch(Circle((n["x"], n["y"]), 4 + age, fill=False, edgecolor="#009E73", lw=2, alpha=max(0, 1 - (age - visual_travel) / (p["slot_seconds"] * .6))))
                elif not e["hit"] and age >= visual_travel and e["download_finish"] > t:
                    ax.add_patch(Circle((n["x"], n["y"]), 4.2, fill=False, edgecolor="#E69F00", lw=1.5, alpha=.7))
                    q = min(1., max(0., (t - e["download_start"]) / max(1e-9, e["download_finish"] - e["download_start"])))
                    ax.add_patch(Arc((n["x"], n["y"]), 8.4, 8.4, theta1=90, theta2=90 + 360 * q, edgecolor="#F0E442", lw=2))
                if age < p["slot_seconds"]:
                    for vi, victim in enumerate(e["evicted"]):
                        ax.text(n["x"], n["y"] - 11 - 3 * vi - age * 2, f"− image {victim['image']}", color="#EE6677", fontsize=8, ha="center", alpha=max(0, 1 - age / p["slot_seconds"]))
                        ax.add_patch(Rectangle((n["x"] - 7, n["y"] - 6), 14 * victim["size"] / n["capacity"], 2.8, facecolor="#EE6677", alpha=.5 * max(0, 1 - age / p["slot_seconds"])))
            latency, hit, evicted, ratio = counters
            ax.text(0, -8, f"{latency:5.1f}s latency     {hit:5.1%} hit     {evicted:3d} evictions", color="white", fontsize=10, family="monospace")
            ax.text(0, -15, f"On-time / arrived: {ratio:.1%}   •   completed: {len(completed)}/{len(arrived)}", color="#94A3B8", fontsize=9)
            history = [sum(z["hit"] for z in arrived[max(0, i - 19):i + 1]) / min(20, i + 1) for i in range(len(arrived))]
            if history:
                ax.plot([60 + 40 * i / max(1, len(history) - 1) for i in range(len(history))], [-25 + 7 * h for h in history], color="#56B4E9", lw=1)
            ax.text(0, -24, "CPU square | GPU hexagon | white outline = pinned", color="#64748B", fontsize=7)
        return []
    animation = FuncAnimation(fig, draw, frames=frame_count, interval=1000 / fps, blit=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if writers.is_available("ffmpeg"):
        output = output.with_suffix(".mp4")
        animation.save(output, writer=FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-threads", "2"]), dpi=110)
    else:
        output = output.with_suffix(".gif")
        animation.save(output, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"Video: {output}", flush=True)
    return output


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CHES · scheduling replay</title><style>
*{box-sizing:border-box}body{margin:0;background:#080f1c;color:#dbe7f3;font:14px system-ui,sans-serif}main{max-width:1550px;margin:auto;padding:28px}
header{display:flex;align-items:end;justify-content:space-between;gap:16px}h1{font-size:28px;margin:4px 0 8px;font-weight:650;letter-spacing:-1px}.eyebrow{font-size:11px;letter-spacing:3px;color:#56b4e9}p{color:#94a3b8;margin:5px 0 18px}.badge{border:1px solid #26384f;border-radius:20px;padding:7px 12px;color:#b2c9dd;font-size:11px}
.toolbar{display:flex;gap:16px;align-items:center;background:#111c2f;border:1px solid #25344a;padding:14px 18px;border-radius:12px;margin:10px 0 18px}button,select{background:#1c3049;color:white;border:1px solid #3b516c;border-radius:7px;padding:8px 14px;cursor:pointer}input{flex:1;accent-color:#56b4e9;min-width:50px}#stamp{font-variant-numeric:tabular-nums;min-width:130px}#panels{display:grid;gap:16px;grid-template-columns:repeat(var(--cols),minmax(0,1fr))}.panel{background:#111c2f;border:1px solid #25344a;border-radius:14px;overflow:hidden}.heading{padding:17px 20px 0;display:flex;justify-content:space-between;align-items:center}.heading strong{font-size:17px}.heading span{color:#94a3b8;font-size:12px}canvas{display:block;width:100%;aspect-ratio:1.25}.counters{display:grid;grid-template-columns:repeat(4,1fr);padding:12px 18px;background:#0d1829;border-top:1px solid #25344a;gap:6px}.metric b{display:block;font-size:20px;color:white;font-variant-numeric:tabular-nums}.metric small{font-size:10px;letter-spacing:.6px;color:#94a3b8}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0;color:#94a3b8;font-size:12px}footer{font-size:11px;color:#73859b;line-height:1.8}.key{padding:2px 6px;background:#1c3049;border-radius:4px}@media(max-width:800px){main{padding:14px}#panels{grid-template-columns:1fr}header{align-items:start}.badge{display:none}.toolbar{gap:8px;padding:10px}.metric b{font-size:16px}}
</style><main><header><div><div class="eyebrow">HETEROGENEOUS EDGE / LIVE REPLAY</div><h1>Where the next service lands.</h1><p>Shared trace. Visible cache pressure. Every placement, pull and eviction.</p></div><div class="badge">OFFLINE · SELF-CONTAINED</div></header>
<div class="toolbar"><button id="play">Pause</button><input id="seek" type="range" min="0" step="0.01" value="0" aria-label="Replay time"><span id="stamp"></span><select id="speed" aria-label="Playback speed"><option value="1">1×</option><option value="2">2×</option><option value="0.5">0.5×</option></select></div>
<div id="panels"></div><div class="legend"><span>□ CPU node</span><span>⬡ GPU node</span><span>● CPU task · ▲ GPU task</span><span style="color:#009e73">◯ Cache hit</span><span style="color:#e69f00">◯ Pull / waiting</span><span style="color:#ee6677">■ Eviction / missed deadline</span><span>White border = pinned</span></div>
<footer id="note"></footer><footer>Bars show reserved disk space; their fill shows download progress. Task travel is slowed for visibility. Live latency uses completed tasks; hit rate uses arrivals; on-time is completed before deadline / arrivals. The sparkline uses the last 20 arrivals. <span class="key">Space</span> play/pause · drag slider to seek.</footer></main>
<script id="replay-data" type="application/json">__DATA__</script><script>
'use strict';const panels=JSON.parse(document.getElementById('replay-data').textContent),palette=__PALETTE__;
const limit=Math.min(...panels.map(p=>p.slots)),dt=panels[0].slot_seconds,end=limit*dt-.001;
const container=document.getElementById('panels');container.style.setProperty('--cols',panels.length===1?1:2);
document.getElementById('note').textContent=`Trace ${panels[0].trace} · ${limit} recorded slots · image colors are fixed across panels · ${panels[0].trace.startsWith('test')?'held-out trace':'request trace'}.`;
const views=panels.map(p=>{let el=document.createElement('section');el.className='panel';el.innerHTML='<div class="heading"><strong></strong><span></span></div><canvas></canvas><div class="counters">'+['AVG LATENCY','CACHE HIT','EVICTIONS','ON-TIME / ARRIVED'].map(x=>`<div class="metric"><b>—</b><small>${x}</small></div>`).join('')+'</div>';el.querySelector('strong').textContent=p.method+' / '+p.policy;el.querySelector('span').textContent=p.disk+' disk ×'+p.disk_scale.toFixed(3);container.append(el);return {p,canvas:el.querySelector('canvas'),metrics:[...el.querySelectorAll('.metric b')]}});
function poly(c,x,y,r,count,rotation=0){c.beginPath();for(let i=0;i<count;i++){let a=2*Math.PI*i/count+rotation;c.lineTo(x+r*Math.cos(a),y+r*Math.sin(a))}c.closePath()}
function render(v,time){const {p,canvas,metrics}=v,dpr=window.devicePixelRatio||1,w=canvas.clientWidth;
const W=1000,H=800;const rw=Math.round(w*dpr),rh=Math.round(w*.8*dpr);if(canvas.width!==rw||canvas.height!==rh){canvas.width=rw;canvas.height=rh}const c=canvas.getContext('2d');c.setTransform(canvas.width/W,0,0,canvas.height/H,0,0);c.clearRect(0,0,W,H);
const X=x=>65+x*8.7,Y=y=>680-y*6.4,color=i=>palette[i%palette.length],slot=Math.min(p.slots-1,Math.floor(time/dt));
let f=p.frames[0];for(const q of p.frames){if(q.slot>slot)break;f=q}const arrived=p.tasks.filter(e=>e.arrival<=time),done=arrived.filter(e=>!e.rejected&&e.finish<=time),pins=new Set(arrived.filter(e=>!e.rejected&&e.finish>time).map(e=>e.node+':'+e.image));
c.strokeStyle='#1c2a3e';c.lineWidth=1;for(let x=0;x<=100;x+=10){c.beginPath();c.moveTo(X(x),Y(0));c.lineTo(X(x),Y(100));c.stroke()}for(let y=0;y<=100;y+=10){c.beginPath();c.moveTo(X(0),Y(y));c.lineTo(X(100),Y(y));c.stroke()}
c.font='12px system-ui';c.fillStyle='#61758e';c.fillText('NODE POSITIONS · x / y',X(0),24);c.textAlign='right';c.fillText('SLOT '+String(slot).padStart(3,'0'),X(100),24);c.textAlign='left';
for(let ni=0;ni<p.nodes.length;ni++){let n=p.nodes[ni],x=X(n.x),y=Y(n.y);c.fillStyle=n.type==='CPU'?'#a9c4dc':'#cc79a7';c.strokeStyle='#dbeafe';c.lineWidth=1.5;poly(c,x,y,16,n.type==='CPU'?4:6,n.type==='CPU'?Math.PI/4:0);c.fill();c.stroke();c.fillStyle='#b9c8da';c.textAlign='center';c.font='11px system-ui';c.fillText(n.type[0]+n.id,x,y-24);let left=x-38,yy=y+23;c.fillStyle='#263349';c.fillRect(left,yy,76,10);c.strokeStyle='#526379';c.strokeRect(left,yy,76,10);
for(const b of f.nodes[ni]){let width=76*b.size/n.capacity,progress=Math.max(0,Math.min(1,(time-b.start)/(b.ready-b.start||1))),pinned=pins.has(n.id+':'+b.image);c.globalAlpha=.2;c.fillStyle=color(b.image);c.fillRect(left,yy,width,10);c.globalAlpha=1;c.fillRect(left,yy,width*progress,10);c.lineWidth=pinned?1.5:.5;c.strokeStyle=pinned?'white':'#111c2f';c.strokeRect(left,yy,width,10);left+=width} }
for(const e of arrived){let age=time-e.arrival;if(e.node<0){if(age<dt){c.fillStyle='#ee6677';c.textAlign='center';c.fillText('×',X(e.origin[0]),Y(e.origin[1]))}continue}let n=p.nodes[e.node],nx=X(n.x),ny=Y(n.y),travel=Math.min(.7*dt,Math.max(.45*dt,e.upload_finish-e.arrival)),q=Math.min(1,age/travel),x=X(e.origin[0])+(nx-X(e.origin[0]))*q,y=Y(e.origin[1])+(ny-Y(e.origin[1]))*q,late=time>e.deadline&&!e.deadline_met;
if(age<travel||e.finish>time||time-e.finish<.8){c.globalAlpha=.18;c.strokeStyle=color(e.image);c.lineWidth=1;c.beginPath();c.moveTo(X(e.origin[0]),Y(e.origin[1]));c.lineTo(nx,ny);c.stroke();c.globalAlpha=1;c.fillStyle=late?'#ee6677':color(e.image);c.strokeStyle='white';c.lineWidth=.5;if(e.type==='GPU')poly(c,x,y,5,3,-Math.PI/2);else{c.beginPath();c.arc(x,y,4,0,Math.PI*2)}c.fill();c.stroke()}
if(e.hit&&age>=travel&&age<travel+dt*.6){c.globalAlpha=Math.max(0,1-(age-travel)/(dt*.6));c.strokeStyle='#009e73';c.lineWidth=3;c.beginPath();c.arc(nx,ny,20+age*5,0,Math.PI*2);c.stroke();c.globalAlpha=1}else if(!e.hit&&age>=travel&&e.download_finish>time){let q=Math.max(0,Math.min(1,(time-e.download_start)/(e.download_finish-e.download_start||1)));c.strokeStyle='#694b20';c.lineWidth=3;c.beginPath();c.arc(nx,ny,23,0,Math.PI*2);c.stroke();c.strokeStyle='#e69f00';c.beginPath();c.arc(nx,ny,23,-Math.PI/2,-Math.PI/2+q*2*Math.PI);c.stroke()}
if(age<dt)for(const [bi,b] of e.evicted.entries()){c.globalAlpha=Math.max(0,1-age/dt);c.fillStyle='#ee6677';c.font='12px system-ui';c.textAlign='center';c.fillText('− image '+b.image,nx,ny+52+bi*14+age*5);c.globalAlpha*=.6;c.fillRect(nx-38,ny+23,76*b.size/n.capacity,10);c.globalAlpha=1}}
let avg=done.reduce((a,e)=>a+e.latency,0)/(done.length||1),hit=arrived.filter(e=>e.hit).length/(arrived.length||1),ev=arrived.reduce((a,e)=>a+e.evicted.length,0),ratio=done.filter(e=>e.deadline_met).length/(arrived.length||1);[avg.toFixed(1)+'s',(100*hit).toFixed(1)+'%',ev,(100*ratio).toFixed(1)+'%'].forEach((x,i)=>metrics[i].textContent=x);
c.fillStyle='#73859b';c.textAlign='left';c.font='12px system-ui';c.fillText('ROLLING CACHE HIT · 20 REQUESTS',65,740);c.strokeStyle='#56b4e9';c.lineWidth=2;c.beginPath();arrived.forEach((e,i)=>{let start=Math.max(0,i-19),hit=arrived.slice(start,i+1).filter(t=>t.hit).length/(i-start+1);c.lineTo(430+500*i/Math.max(1,arrived.length-1),770-hit*45)});c.stroke();c.fillStyle='#73859b';c.fillText(done.length+' completed / '+arrived.length+' arrived',65,767)}
let time=0,playing=true,last=null;const seek=document.getElementById('seek'),play=document.getElementById('play');seek.max=end;
function draw(){for(const v of views)render(v,time);seek.value=time;document.getElementById('stamp').textContent='Slot '+Math.floor(time/dt)+' · '+time.toFixed(1)+'s'}
function toggle(){playing=!playing;play.textContent=playing?'Pause':'Play';if(playing&&time>=end)time=0}play.onclick=toggle;seek.oninput=()=>{time=Number(seek.value);draw()};window.addEventListener('keydown',e=>{if(e.code==='Space'&&e.target.tagName!=='INPUT'&&e.target.tagName!=='SELECT'){e.preventDefault();toggle()}});window.addEventListener('resize',draw);
function tick(now){const needsDraw=playing;if(last!==null&&playing){time=Math.min(end,time+(now-last)/1000*dt/.75*Number(document.getElementById('speed').value));if(time>=end){playing=false;play.textContent='Play'}}last=now;if(needsDraw)draw();requestAnimationFrame(tick)}requestAnimationFrame(tick);
</script></html>'''


def export_html(panels, output, slots=None):
    if slots:
        panels = [dict(p, slots=min(slots, p["slots"])) for p in panels]
    html = HTML.replace("__DATA__", json.dumps(panels, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c"))
    html = html.replace("__PALETTE__", json.dumps(PALETTE))
    output = Path(output).with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f"HTML: {output}", flush=True)
    return output


def render_scenarios(root, fps=10, slots=None):
    root = Path(root)
    replays = root / "replays"
    candidate = sorted(replays.glob("FrozenAgent_LRU_tight_*.json"))
    if not candidate:
        raise FileNotFoundError("Evaluate the LRU/tight agent before rendering scenarios")
    frozen = candidate[0]
    trace = json.loads(frozen.read_text())["trace"]
    scenarios = {
        "agent_vs_greedy": [frozen, replays / f"Greedy_latency_LRU_tight_{trace}.json"],
        "four_policies": [replays / f"FrozenAgent_{p}_tight_{trace}.json" for p in ("LRU", "LFU", "FIFO", "GDSF")],
        "disk_pressure": [frozen, replays / f"FrozenAgent_LRU_large_{trace}.json"],
    }
    for name, paths in scenarios.items():
        panels = load_panels(paths)
        (replays / f"scenario_{name}.json").write_text(json.dumps(dict(panels=panels), separators=(",", ":")))
        export_html(panels, root / "figures" / f"replay_{name}.html", slots)
        export_video(panels, root / "figures" / f"replay_{name}.mp4", fps, slots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", nargs="+")
    parser.add_argument("--mode", choices=("mp4", "html"), default="html")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output")
    parser.add_argument("--slots", type=int)
    parser.add_argument("--scenarios", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Render all three scenarios, first 16 slots")
    parser.add_argument("--root", default="outputs")
    args = parser.parse_args()
    if args.smoke:
        args.scenarios, args.slots = True, args.slots or 16
    if args.fps < 1 or (args.slots is not None and args.slots < 1):
        parser.error("FPS and slots must be positive")
    if args.scenarios:
        render_scenarios(args.root, args.fps, args.slots)
    elif args.replay:
        panels = load_panels(args.replay)
        out = args.output or str(Path(args.root) / "figures" / f"replay_{Path(args.replay[0]).stem}")
        if args.mode == "html":
            export_html(panels, out, args.slots)
        else:
            export_video(panels, out, args.fps, args.slots)
    else:
        parser.error("Provide --replay or --scenarios")


if __name__ == "__main__":
    main()
