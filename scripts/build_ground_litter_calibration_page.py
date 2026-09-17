"""Build a self-contained browser editor for ground-litter camera zones.

The generated page embeds the selected profile's reference images, so it can be
opened locally or served by ``python -m http.server`` without exposing RTSP
URLs. It only downloads a new JSON profile; it never edits the input file.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def image_data(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"reference image missing: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.profiles.read_text(encoding="utf-8"))
    cameras = payload.get("cameras") or []
    if payload.get("schema_version") != 2 or not cameras:
        parser.error("expected schema_version 2 with cameras")
    embedded = {}
    for camera in cameras:
        source = Path(camera["reference_image"])
        if not source.is_absolute():
            candidates = [args.profiles.parent / source, Path.cwd() / source]
            if args.profiles.parent.name == "config":
                candidates.append(args.profiles.parent.parent / source)
            source = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        embedded[camera["device_code"]] = image_data(source)
    args.output.mkdir(parents=True, exist_ok=True)
    page = PAGE.replace("__PROFILE__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    page = page.replace("__IMAGES__", json.dumps(embedded).replace("</", "<\\/"))
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"page": str(args.output / "index.html"), "cameras": len(cameras)}, ensure_ascii=False))
    return 0


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>零散垃圾识别区域校准</title>
<style>
:root{font:14px system-ui,-apple-system,sans-serif;color:#20333b;background:#eef3f5}body{margin:0}header{padding:14px 22px;background:#163b4a;color:white}main{display:grid;grid-template-columns:minmax(620px,1fr) 360px;gap:16px;padding:16px;max-width:1600px;margin:auto}.card{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 5px #0002}#stage{position:relative;width:100%;background:#111;overflow:hidden}canvas{display:block;width:100%;cursor:crosshair}label{display:block;margin:8px 0 3px;font-weight:600}select,input,button{font:inherit;padding:7px;border:1px solid #b8c7cc;border-radius:5px}input,select{width:100%;box-sizing:border-box}button{cursor:pointer;background:#f5fafb;margin:3px 2px}button.primary{background:#176b73;color:white;border-color:#176b73}.help{color:#536a72;font-size:12px}.warn{color:#9a4c00;background:#fff5e7;border-radius:5px;padding:8px}.zone{border:1px solid #d5e0e3;border-radius:5px;padding:7px;margin:5px 0;cursor:pointer}.zone.selected{border:2px solid #176b73;background:#edfafa}.zone small{color:#657980}.tools{display:flex;flex-wrap:wrap;gap:3px}.tools button.active{background:#176b73;color:#fff}.scroll{max-height:310px;overflow:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}.grid input{width:100%}@media(max-width:900px){main{grid-template-columns:1fr}.card{min-width:0}}
</style></head><body><header><b>零散垃圾识别区域校准</b>　<span id="state">草案</span></header><main>
<section class="card"><label>摄像头</label><select id="camera"></select><div id="stage"><canvas id="canvas"></canvas></div>
<p class="help">坐标按原始画面归一化保存。黄色为识别区域，红色为排除区域，蓝色为当前编辑形状。</p></section>
<aside class="card"><div class="warn">这里只修改下载的 profile 草案，不会启动识别、上传图片或发送通知。</div>
<label>编辑类型</label><div class="tools"><button data-mode="polygon">多边形</button><button data-mode="rect">矩形</button><button data-mode="grid">矩形网格</button><button data-mode="exclude">排除多边形</button></div>
<p class="help" id="modeHelp">多边形：逐点点击，双击或点“完成”闭合。</p><div class="grid"><div><label>网格行数</label><input id="rows" type="number" min="1" max="20" value="2"></div><div><label>网格列数</label><input id="cols" type="number" min="1" max="20" value="2"></div></div>
<button id="finish">完成当前形状</button><button id="undo">撤销最后一点</button><button id="clearDraft">清空未完成</button>
<hr><label>区域列表（点击选择）</label><div id="zones" class="scroll"></div><button id="delete">删除选中区域</button>
<label>区域 ID</label><input id="region" placeholder="例如 shop_01"><label>区域名称</label><input id="name" placeholder="门前人行道"><label>商户 ID（可留空）</label><input id="merchant" placeholder="merchant_001"><label>商户名称（可留空）</label><input id="merchantName" placeholder="某某商户"><button id="apply" class="primary">保存区域信息</button>
<hr><button id="download" class="primary">下载 profile JSON</button><button id="copy">复制 JSON</button><p class="help">下载文件需人工审核后替换部署 profile；页面始终保持 enabled=false、calibration_status=draft。</p></aside></main>
<script>
const profile=__PROFILE__, images=__IMAGES__; let cam=profile.cameras[0],mode='polygon',draft=[],drag=null,selected=-1;
const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d'),image=new Image();
const $=id=>document.getElementById(id), clamp=x=>Math.max(0,Math.min(1,x));
function init(){profile.cameras.forEach((c,i)=>{let o=document.createElement('option');o.value=c.device_code;o.textContent=c.device_code+' · '+(c.notes||'');$('camera').append(o)});$('camera').value=cam.device_code; loadCamera(); document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));}
function loadCamera(){cam=profile.cameras.find(c=>c.device_code===$('camera').value)||profile.cameras[0];selected=-1;draft=[];image.onload=()=>{canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;draw()};image.src=images[cam.device_code];renderZones();}
function xy(e){let r=canvas.getBoundingClientRect();return [clamp((e.clientX-r.left)/r.width),clamp((e.clientY-r.top)/r.height)]}
function px(p){return [p[0]*canvas.width,p[1]*canvas.height]}
function polygon(points,color,fill=false){if(points.length<2)return;ctx.beginPath();points.forEach((p,i)=>{let q=px(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});if(points.length>2)ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=Math.max(2,canvas.width/900);ctx.stroke();if(fill){ctx.fillStyle=color.replace(')',',0.12)').replace('rgb','rgba');ctx.fill()}}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);cam.zones.forEach((z,i)=>{let color=i===selected?'rgb(23,107,115)':'rgb(235,190,0)';polygon(z.polygon,color,true);(z.exclude_zones||[]).forEach(x=>polygon(x,'rgb(215,45,45)',true));});if(draft.length){polygon(draft,mode==='exclude'?'rgb(215,45,45)':'rgb(40,130,230)');draft.forEach(p=>{let q=px(p);ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(...q,Math.max(4,canvas.width/400),0,7);ctx.fill()});}if(drag){let p=[Math.min(drag[0][0],drag[1][0]),Math.min(drag[0][1],drag[1][1])],q=[Math.max(drag[0][0],drag[1][0]),Math.max(drag[0][1],drag[1][1])];polygon([p,[q[0],p[1]],q,[p[0],q[1]]],'rgb(40,130,230)');}}
function setMode(x){mode=x;draft=[];drag=null;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===x));$('modeHelp').textContent=x==='polygon'?'多边形：逐点点击，双击或点“完成”闭合。':x==='rect'?'矩形：拖拽一个识别矩形。':x==='grid'?'矩形网格：拖拽外框后按行列拆成多个区域。':'排除多边形：先选中一个区域，再逐点圈出固定设施。';draw();}
function newZone(points,index){let n='region_'+String(cam.zones.length+index+1).padStart(2,'0');return {region_id:n,name:n,merchant_id:null,merchant_name:null,physical_area_id:null,polygon:points,exclude_zones:[],minimum_short_side_px:12,minimum_box_area_px:160}}
function finish(){if(mode==='polygon'||mode==='exclude'){if(draft.length<3)return alert('至少需要3个点');if(mode==='exclude'){if(selected<0)return alert('请先选择要排除的区域');cam.zones[selected].exclude_zones.push(draft); }else cam.zones.push(newZone(draft,0));}else if(drag){let a=drag[0],b=drag[1],x0=Math.min(a[0],b[0]),x1=Math.max(a[0],b[0]),y0=Math.min(a[1],b[1]),y1=Math.max(a[1],b[1]);if(x1-x0<.002||y1-y0<.002)return;let rows=Math.max(1,Math.min(20,+$('rows').value||1)),cols=Math.max(1,Math.min(20,+$('cols').value||1));if(mode==='grid'){for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)cam.zones.push(newZone([[x0+(x1-x0)*c/cols,y0+(y1-y0)*r/rows],[x0+(x1-x0)*(c+1)/cols,y0+(y1-y0)*r/rows],[x0+(x1-x0)*(c+1)/cols,y0+(y1-y0)*(r+1)/rows],[x0+(x1-x0)*c/cols,y0+(y1-y0)*(r+1)/rows]],r*cols+c));}else cam.zones.push(newZone([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],0));}draft=[];drag=null;selected=cam.zones.length-1;renderZones();draw();}
function renderZones(){let box=$('zones');box.innerHTML='';cam.zones.forEach((z,i)=>{let d=document.createElement('div');d.className='zone'+(i===selected?' selected':'');d.innerHTML='<b>'+z.region_id+'</b> <small>'+((z.name||''))+'</small><br><small>商户：'+(z.merchant_name||z.merchant_id||'待绑定')+' · 排除区：'+(z.exclude_zones||[]).length+'</small>';d.onclick=()=>{selected=i;fillFields();renderZones();draw()};box.append(d)});fillFields();}
function fillFields(){let z=cam.zones[selected];$('region').value=z?.region_id||'';$('name').value=z?.name||'';$('merchant').value=z?.merchant_id||'';$('merchantName').value=z?.merchant_name||'';}
canvas.addEventListener('pointerdown',e=>{if(mode==='polygon'||mode==='exclude'){draft.push(xy(e));draw()}else drag=xy(e);});canvas.addEventListener('pointermove',e=>{if(drag){drag[1]=xy(e);draw()}});canvas.addEventListener('pointerup',e=>{if(drag){drag[1]=xy(e);draw()}});canvas.addEventListener('dblclick',e=>{if(mode==='polygon'||mode==='exclude'){draft.pop();finish()}});
$('finish').onclick=finish;$('undo').onclick=()=>{if(draft.length){draft.pop();draw()}};$('clearDraft').onclick=()=>{draft=[];drag=null;draw()};$('delete').onclick=()=>{if(selected>=0){cam.zones.splice(selected,1);selected=-1;renderZones();draw()}};$('apply').onclick=()=>{if(selected<0)return alert('请先选择区域');let z=cam.zones[selected];z.region_id=$('region').value.trim()||z.region_id;z.name=$('name').value.trim()||z.region_id;z.merchant_id=$('merchant').value.trim()||null;z.merchant_name=$('merchantName').value.trim()||null;renderZones();draw()};$('camera').onchange=loadCamera;
function exported(){let out=JSON.parse(JSON.stringify(profile));out.cameras.forEach(c=>{c.enabled=false;c.calibration_status='draft'});out.status='design_draft_not_stream_api_payload';return out}function download(){let b=new Blob([JSON.stringify(exported(),null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='ground_litter_profiles.edited.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}$('download').onclick=download;$('copy').onclick=async()=>{await navigator.clipboard.writeText(JSON.stringify(exported(),null,2));$('state').textContent='JSON已复制';};init();setMode('polygon');
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
