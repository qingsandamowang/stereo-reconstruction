"""Standalone offline Canvas point-cloud viewer; no CDN, server, or external files."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np


def write_viewer(path: Path, points: np.ndarray, colors: np.ndarray, max_points: int = 20_000,
                 title: str = "Stereo reconstruction") -> int:
    """Embed a deterministic, uniformly sampled subset of XYZ-mm/RGB in one HTML file."""
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("Expected matching Nx3 XYZ and RGB arrays")
    if not isinstance(max_points, int) or max_points < 1:
        raise ValueError("Viewer maximum points must be a positive integer")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("Viewer points and colors must be finite")
    count = min(len(points), max_points)
    selected = np.linspace(0, len(points) - 1, count, dtype=np.int64) if count else np.empty(0, dtype=np.int64)
    data = {"xyz": points[selected].tolist(), "rgb": np.clip(colors[selected], 0, 255).astype(int).tolist(),
            "total": len(points), "units": "mm"}
    document = _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", json.dumps(data, separators=(",", ":"), allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return count


_TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;background:#101923;color:#dce6ef;font:15px system-ui,sans-serif}
header{padding:20px 24px;background:#182735;border-bottom:1px solid #334b5e}h1{font-size:22px;margin:0 0 8px}
p{margin:6px 0;color:#a9bdcc}.tools{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:12px 24px}
button{padding:8px 16px;border:1px solid #4a728c;background:#234357;color:white;border-radius:6px;cursor:pointer}
canvas{display:block;width:100%;height:calc(100vh - 235px);min-height:320px;touch-action:none;cursor:grab;outline-color:#78cef4}
label{display:flex;gap:8px;align-items:center}#empty{padding:32px;text-align:center;color:#ffc378}footer{padding:10px 24px;font-size:13px}
</style></head><body><header><h1>__TITLE__</h1><p id="stats"></p>
<p>Drag to rotate · Wheel or +/− to zoom · Arrow keys to rotate · R to reset. All coordinates: millimetres (mm).</p>
</header><div class="tools"><button id="reset">Reset view</button><label>Zoom <input id="zoom" type="range" min="0.1" max="8" step="0.05" value="1"></label><span id="angles"></span></div>
<p id="empty" hidden>No valid 3D points. Inspect disparity, calibration, and any depth clipping.</p>
<canvas id="cloud" tabindex="0" aria-label="Interactive colored 3D point cloud"></canvas>
<footer>Offline file · Deterministic point subset · X right, Y down, Z forward in the left camera · Orthographic display, equal scale on all axes.</footer>
<script id="cloud-data" type="application/json">__DATA__</script><script>
"use strict";
const data=JSON.parse(document.getElementById("cloud-data").textContent);
const canvas=document.getElementById("cloud"),ctx=canvas.getContext("2d"),zoom=document.getElementById("zoom");
const n=data.xyz.length,bounds=[[Infinity,Infinity,Infinity],[-Infinity,-Infinity,-Infinity]];
for(const p of data.xyz)for(let k=0;k<3;k++){bounds[0][k]=Math.min(bounds[0][k],p[k]);bounds[1][k]=Math.max(bounds[1][k],p[k]);}
const center=n?bounds[0].map((v,k)=>(v+bounds[1][k])/2):[0,0,0];
const radius=n?Math.max(...bounds[1].map((v,k)=>v-bounds[0][k]))/2:1;
const normalized=data.xyz.map(p=>p.map((v,k)=>(v-center[k])/(radius||1)));
document.getElementById("stats").textContent=`Showing ${n.toLocaleString()} / ${data.total.toLocaleString()} exported points · Units: mm`+(n?` · Displayed Z range: ${bounds[0][2].toFixed(2)}–${bounds[1][2].toFixed(2)} mm`:"");
document.getElementById("empty").hidden=!!n;
let yaw=0,pitch=0,drag=null,scheduled=false;
function requestDraw(){if(!scheduled){scheduled=true;requestAnimationFrame(draw);}}
function draw(){scheduled=false;const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle="#101923";ctx.fillRect(0,0,w,h);
const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),scale=Math.min(w,h)*0.4*Number(zoom.value);
const transformed=normalized.map((p,i)=>{const x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2];return [x,cp*p[1]-sp*z,sp*p[1]+cp*z,i];});
transformed.sort((a,b)=>b[2]-a[2]);for(const [x,y,z,i]of transformed){const c=data.rgb[i];ctx.fillStyle=`rgb(${c[0]},${c[1]},${c[2]})`;ctx.fillRect(w/2+x*scale,h/2+y*scale,2,2);}
document.getElementById("angles").textContent=`Yaw ${Math.round(yaw*180/Math.PI)}° · Pitch ${Math.round(pitch*180/Math.PI)}°`;
}
function reset(){yaw=0;pitch=0;zoom.value=1;requestDraw();}
canvas.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);canvas.focus();});
canvas.addEventListener("pointermove",e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.008;pitch+=(e.clientY-drag[1])*0.008;drag=[e.clientX,e.clientY];requestDraw();});
canvas.addEventListener("pointerup",()=>{drag=null;});canvas.addEventListener("pointercancel",()=>{drag=null;});
canvas.addEventListener("wheel",e=>{e.preventDefault();zoom.value=Math.max(0.1,Math.min(8,Number(zoom.value)*Math.exp(-e.deltaY*0.001)));requestDraw();},{passive:false});
canvas.addEventListener("keydown",e=>{let used=true;switch(e.key){case"ArrowLeft":yaw-=0.1;break;case"ArrowRight":yaw+=0.1;break;case"ArrowUp":pitch-=0.1;break;case"ArrowDown":pitch+=0.1;break;case"+":case"=":zoom.value=Math.min(8,Number(zoom.value)*1.1);break;case"-":zoom.value=Math.max(0.1,Number(zoom.value)/1.1);break;case"r":case"R":reset();break;default:used=false;}if(used){e.preventDefault();requestDraw();}});
zoom.addEventListener("input",requestDraw);document.getElementById("reset").addEventListener("click",reset);window.addEventListener("resize",requestDraw);requestDraw();
</script></body></html>'''
