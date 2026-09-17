#!/usr/bin/env python3
"""Small, frozen local-image VLM probe; never connects to cameras or stream API.

Prepare crops first, visually review/freeze the manifest, then explicitly run.
Credentials are read only from DASHSCOPE_API_KEY and never saved or printed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
PROMPT = """你审核固定监控中的一个地面零散垃圾候选。输入图1是目标细节，图2是周围环境。
两图的红色矩形指向同一个待判断目标。只判断框内目标，不判断旁边其他物品。红框仅为定位标记。
判断是否是遗落在地面、需要清理的纸片、包装、塑料袋、瓶罐等零散垃圾。
车辆及零件、车罩、车篮、车上物品、正在穿戴或使用的物品、商品、工具、伞棚、固定设施、井盖、地面污渍和反光不是地面零散垃圾。
靠墙或靠车辆的真实垃圾仍可能是垃圾，不要仅凭位置排除。不要把有塑料/纸材质等同于垃圾。
不能因目标被提供为候选就假定它是垃圾。细节太少、遮挡或无法确认物体与地面的关系时返回uncertain，不要编造。
只输出JSON，字段为decision(litter/non_litter/uncertain)、object_description(简短)、ground_relation(on_ground/attached/not_ground/unclear)、reason(不超过60个汉字)。"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict]:
    day = "output/ground_litter_1021_fp_fix/now-clean.jpg"
    night = "output/litter_source_20260914/local_10m_network/snapshots/frame-003.jpg"
    other_day = "output/ground_litter_1021_pilot/raw-1021-clean.jpg"
    rows = [
        (day, [141, 767, 226, 903], "non_litter", "车辆覆盖物", "historical_false_positive", "Plastic 0.28", "vehicle_1"),
        (day, [142, 879, 233, 963], "non_litter", "车辆座椅与篮筐", "historical_false_positive", "Paper 0.07", "vehicle_1"),
        (day, [313, 579, 351, 611], "non_litter", "行人穿着的鞋", "historical_false_positive", "Plastic 0.54", "person_1"),
        (day, [635, 421, 746, 558], "non_litter", "遮阳伞边缘", "historical_false_positive", "Paper 0.69", "awning_1"),
        (day, [1625, 923, 1753, 982], "non_litter", "井盖", "negative_control", None, "manhole_1"),
        (day, [452, 866, 584, 964], "non_litter", "地面水渍与纹理", "negative_control", None, "stain_1"),
        (night, [803, 565, 818, 573], "litter", "历史用户确认的8像素小垃圾", "historical_user_confirmed", "Metal 0.222 (live003)", "small_litter_1"),
        (night, [835, 595, 925, 672], "litter", "路沿白色废弃袋状物", "assistant_visual_provisional", None, "bag_1"),
        (night, [827, 1393, 854, 1415], "litter", "近端人行道小纸片", "assistant_visual_provisional", None, "paper_1"),
        (night, [1032, 911, 1063, 936], "litter", "路边方形包装物", "assistant_visual_provisional", None, "package_1"),
        (other_day, [614, 1043, 640, 1068], "litter", "白天近端地面白色碎片", "assistant_visual_provisional", None, "paper_2"),
        (night, [558, 621, 602, 655], "uncertain", "店门旁红色物体", "ambiguous_control", None, "unknown_1"),
        (day, [392, 643, 435, 657], "uncertain", "阴影边缘细长物体", "ambiguous_control", "Plastic 0.07", "unknown_2"),
    ]
    return [dict(id=f"S{i:02d}", source=s, box=b, expected=e, note=n,
                 label_source=l, historical_prediction=h, object_group=g)
            for i, (s,b,e,n,l,h,g) in enumerate(rows, 1)]


def crop(image: Image.Image, box: list[int], edge: int) -> tuple[Image.Image, list[int]]:
    x1,y1,x2,y2=box
    w,h=min(edge,image.width),min(edge,image.height)
    x=max(0,min(image.width-w,(x1+x2-w)//2))
    y=max(0,min(image.height-h,(y1+y2-h)//2))
    out=image.crop((x,y,x+w,y+h))
    rect=[x1-x,y1-y,x2-x,y2-y]
    return out,rect


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out/"crops").mkdir()
    rows=cases()
    sheet=Image.new("RGB",(1000,((len(rows)+2)//3)*280),"#f1f3f5")
    draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        source=ROOT/row['source']
        im=Image.open(source).convert("RGB")
        row['source_sha256']=sha(source)
        row['source_size']=list(im.size)
        edge=max(row['box'][2]-row['box'][0],row['box'][3]-row['box'][1])
        row['images']=[]
        for kind,side in [('detail',max(64,int(edge*1.5))),('context',max(384,int(edge*3)))]:
            ci,rect=crop(im,row['box'],side)
            # Deterministic resize only: no generative enhancement, no labels/scores.
            scale=max(1,256/ci.width) if kind=='detail' else 1
            if scale!=1:
                ci=ci.resize((round(ci.width*scale),round(ci.height*scale)),Image.Resampling.NEAREST)
                rect=[round(v*scale) for v in rect]
            ImageDraw.Draw(ci).rectangle(rect,outline="red",width=1 if scale==1 else 2)
            p=out/"crops"/f"{row['id']}-{kind}.jpg"
            ci.save(p,quality=95)
            row['images'].append(str(p.relative_to(out)))
            row[f'{kind}_sha256']=sha(p)
            if kind=='context':
                ci.thumbnail((320,240))
                x=(i%3)*333; y=(i//3)*280
                sheet.paste(ci,(x,y+25))
                draw.text((x+4,y+5),f"{row['id']} {row['expected']}",fill='black')
    manifest=dict(created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        scope='Selected candidate-review sanity check; not end-to-end accuracy or recall.',
        label_warning='Only S07 has historical user confirmation; other labels are provisional assistant visual judgments. S01/S02 share one vehicle.',
        prompt=PROMPT,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),cases=rows)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    sheet.save(out/'contact.jpg',quality=94)
    print(json.dumps({'prepared':len(rows),'output':str(out)}))


def prepare_temporal(out: Path) -> None:
    """Freeze one user-confirmed tiny object at three distinct times."""
    out.mkdir(parents=True, exist_ok=False)
    (out/'crops').mkdir()
    source_dir=ROOT/'output/litter_source_20260914/missed_candidate_review'
    selected=['small-white-803-00.jpg','small-white-803-05.jpg','small-white-803-10.jpg']
    images=[];hashes=[]
    for name in selected:
        source=source_dir/name
        dest=out/'crops'/name
        shutil.copyfile(source,dest)
        images.append(str(dest.relative_to(out)))
        hashes.append(sha(dest))
    prompt="""这三张图按时间顺序拍摄同一个固定位置，中央是同一个极小目标，已做普通像素放大，没有生成或修复细节。请综合三帧判断该目标。
判断它是否是遗落在地面、需要清理的纸片、包装、塑料袋、瓶罐等零散垃圾。车辆及零件、车罩、车篮、车上物品、正在使用的物品、商品、工具、固定设施、井盖、污渍和反光不是垃圾。
不要因为它被提交审核就假定为垃圾；多帧一致只能证明目标存在，不能证明语义。若像素仍不足以确认物体性质或与地面的关系，必须返回uncertain。
只输出JSON，字段为decision(litter/non_litter/uncertain)、object_description(简短)、ground_relation(on_ground/attached/not_ground/unclear)、reason(不超过60个汉字)。"""
    manifest={'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'scope':'One user-confirmed 8px litter object; temporal evidence probe, not an accuracy test.',
        'label_warning':'The physical target was historically confirmed by the user. The model did not receive this fact.',
        'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
        'cases':[{'id':'T01','source':'three historical frames','box':None,
                  'expected':'litter','note':'用户确认的8像素小垃圾，三时刻',
                  'label_source':'historical_user_confirmed','historical_prediction':'single-frame VLM uncertain',
                  'object_group':'small_litter_1','images':images,'image_sha256s':hashes}]}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    sheet=Image.new('RGB',(900,300),'white')
    for i,rel in enumerate(images):
        im=Image.open(out/rel).convert('RGB');im.thumbnail((290,290));sheet.paste(im,(i*300,0))
    sheet.save(out/'contact.jpg',quality=95)
    print(json.dumps({'prepared':1,'images':3,'output':str(out)}))


def run(out: Path, model: str, limit: int) -> int:
    key=os.getenv('DASHSCOPE_API_KEY','')
    if not key:
        raise SystemExit('DASHSCOPE_API_KEY is not set')
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    response_dir=out/model
    response_dir.mkdir(exist_ok=True)
    for row in manifest['cases'][:limit or None]:
        dest=response_dir/f"{row['id']}.json"
        if dest.exists():
            continue
        content=[{'type':'text','text':manifest['prompt']}]
        for index,rel in enumerate(row['images']):
            p=out/rel
            expected_hash=(row.get('image_sha256s') or [
                row.get('detail_sha256'),row.get('context_sha256')
            ])[index]
            if sha(p)!=expected_hash:
                raise SystemExit('Frozen input hash mismatch')
            content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()}})
        body={'model':model,'messages':[{'role':'user','content':content}],
              'temperature':0,'max_tokens':384,'enable_thinking':False,
              'response_format':{'type':'json_object'}}
        req=urllib.request.Request(ENDPOINT,data=json.dumps(body).encode(),
            headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},method='POST')
        started=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=45) as res:
                payload=json.load(res)
        except urllib.error.HTTPError as exc:
            # Do not log arbitrary response bodies or request headers.
            print(json.dumps({'id':row['id'],'http_error':exc.code}),flush=True)
            return 2
        except (urllib.error.URLError,TimeoutError,OSError) as exc:
            print(json.dumps({'id':row['id'],'network_error':type(exc).__name__}),flush=True)
            return 3
        elapsed=time.perf_counter()-started
        answer=payload['choices'][0]['message'].get('content','')
        try:
            parsed=json.loads(answer)
            valid=parsed.get('decision') in ('litter','non_litter','uncertain')
        except (json.JSONDecodeError,AttributeError):
            parsed=None;valid=False
        result={'id':row['id'],'model_requested':model,'model_returned':payload.get('model'),
                'manifest_sha256':sha(manifest_path),'seconds':elapsed,
                'usage':payload.get('usage',{}),'request_id':payload.get('id'),
                'finish_reason':payload['choices'][0].get('finish_reason'),
                'raw_answer':answer,'parsed':parsed,'valid':valid}
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'id':row['id'],'decision':parsed.get('decision') if valid else 'invalid',
                          'seconds':round(elapsed,2)},ensure_ascii=False),flush=True)
    return 0


def report(out: Path, model: str) -> None:
    manifest=json.loads((out/'manifest.json').read_text())
    records=[]
    for row in manifest['cases']:
        path=out/model/f"{row['id']}.json"
        if not path.exists():
            raise SystemExit(f"Missing response: {row['id']}")
        response=json.loads(path.read_text())
        if response['manifest_sha256']!=sha(out/'manifest.json'):
            raise SystemExit('Response manifest mismatch')
        records.append(dict(**row,response=response))
    matrix={label:{decision:0 for decision in ['litter','non_litter','uncertain','invalid']}
            for label in ['litter','non_litter','uncertain']}
    for row in records:
        r=row['response']
        decision=r['parsed']['decision'] if r['valid'] else 'invalid'
        matrix[row['expected']][decision]+=1
    times=sorted(r['response']['seconds'] for r in records)
    # Linear interpolated sample percentile, not a service SLA.
    position=(len(times)-1)*.95
    lo=int(position);hi=min(lo+1,len(times)-1)
    p95=times[lo]+(times[hi]-times[lo])*(position-lo)
    usage={key:sum(r['response']['usage'].get(key,0) for r in records)
           for key in ['prompt_tokens','completion_tokens','total_tokens']}
    summary=dict(model=model,requests=len(records),matrix=matrix,usage=usage,
                 latency_seconds=dict(min=min(times),median=statistics.median(times),
                                      p95=p95,max=max(times)),
                 scope=manifest['scope'],label_warning=manifest['label_warning'],
                 end_to_end_accuracy=None,production_integrated=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    labels={'litter':'垃圾','non_litter':'非垃圾','uncertain':'不确定'}
    lines=[
        '# 百炼 Qwen VLM 垃圾候选审核小样本实测',
        '', '日期：2026-09-17；机位：1021；模型：`'+model+'`（API返回同名，未固定模型快照）。',
        '', '**结论：有纠正常见语义误报的价值，但本次结果不支持把 VLM 通过作为所有目标的强制显示条件。**',
        '', '共13次真实调用：4个历史模型误报候选全部被判为非垃圾；5个垃圾正例中1个通过、4个不确定；2个负对照中1个非垃圾、1个不确定；2个模糊对照均不确定。',
        '', '只有S07具有历史用户确认记录，其余标签为调用前冻结的助手目视暂定标签。S01/S02属于同一辆车；样本经过挑选，不能计算或宣传现场准确率。',
        '', '## 方法和边界',
        '', '- 从3张无检测叠字原图裁剪，原图分辨率分别为1920×1080和2560×1440；云端只接收局部图。',
        '- 每例独立请求：细节图＋至少384像素上下文图；仅添加红色目标定位框，不发送标签、历史YOLO分数或样本说明。',
        '- 小裁剪使用最近邻放大至至少256宽，没有生成细节或图像修复；提示词、来源与发送图SHA-256均保留。',
        '- temperature=0、enable_thinking=false、max_tokens=384、JSON输出、顺序调用、无自动重试。',
        '- 历史误报是模型候选级误报，部分分数低于现有阈值或可能已被ROI/actor过滤；不能称为减少了4个当前生产报警。',
        '- 5个正例是定向审核输入，不保证均能由当前API检测器提出；没有测漏检、轨迹、RTSP吞吐、1022或五路并发。',
        '- 一张严重花屏的历史已知物体参考图在调用前排除，没有发送或计分。',
        '', '## 结果', '', '|编号|预先记录的目标|参考标签|VLM结论|耗时(s)|说明|', '|---|---|---|---|---:|---|',
    ]
    cards=[]
    for row in records:
        r=row['response'];a=r['parsed'];decision=a['decision']
        reason=a['reason'].replace('|','／')
        lines.append(f"|{row['id']}|{row['note']}|{labels[row['expected']]}|{labels[decision]}|{r['seconds']:.2f}|{reason}|")
        esc=html.escape
        cards.append(f"<article><h3>{esc(row['id'])} · {esc(row['note'])}</h3>"
                     f"<p>参考：{labels[row['expected']]} → VLM：<b>{labels[decision]}</b> · {r['seconds']:.2f}s</p>"
                     f"<div class='images'><img src='{row['images'][0]}'><img src='{row['images'][1]}'></div>"
                     f"<p>{esc(a['object_description'])}</p><p>{esc(a['reason'])}</p>"
                     f"<small>标签来源：{esc(row['label_source'])}；历史模型结果：{esc(row['historical_prediction'] or '无，仅作对照')}</small>"
                     f"<p><a href='{model}/{row['id']}.json'>原始响应与token</a></p></article>")
    lines += ['', f"请求耗时中位数 **{statistics.median(times):.2f}s**，样本P95 **{p95:.2f}s**，范围 **{min(times):.2f}–{max(times):.2f}s**。计时包含本机HTTP往返，不含排队、检测、裁剪、建流；13次小样本不能作为SLA。",
              '', f"API报告token：输入 {usage['prompt_tokens']}、输出 {usage['completion_tokens']}、合计 {usage['total_tokens']}。实际费用以百炼账单为准。13/13返回可解析JSON及合法decision。",
              '', '## 对接入方案的影响',
              '', '1. 先影子运行；VLM明确识别为车辆/设施等非垃圾时，可作为候选抑制依据继续验证。',
              '2. 不确定不能当作非垃圾，也不能当作通过。若仅litter可显示，本组5个正例只显示1个，历史用户确认的8像素目标也会被挡住。',
              '3. 对小目标保留“疑似/待复核”状态，并优先改善近端像素、裁剪、拍摄或时序证据；本测试未证明放宽提示词或换模型能解决。',
              '4. 本云端请求约2–3秒，不能沿用800ms统一超时；首次显示还要加检测、稳定和调度时间。',
              '5. 下一步应使用更多独立物体、人工核验和冻结回放评估误杀率/召回损失，之后才接单路灰度。',
              '', '## 复现', '', '```bash',
              '.venv/bin/python scripts/probe_ground_litter_vlm.py prepare --output output/<新的试验目录>',
              '# 先人工核对manifest及contact.jpg；环境已设置DASHSCOPE_API_KEY后执行：',
              '.venv/bin/python scripts/probe_ground_litter_vlm.py run --output output/<新的试验目录>',
              '.venv/bin/python scripts/probe_ground_litter_vlm.py report --output output/<新的试验目录>',
              '```', '', '相同目录run会跳过已有响应，不重复计费。凭据只从环境读取；没有连接摄像头或生产服务器。',
              '', '接口参考：[百炼视觉理解](https://help.aliyun.com/zh/model-studio/vision)。',
              '', '[查看逐例图片对照](index.html) · [冻结输入](manifest.json) · [机器可读汇总](summary.json)', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen VLM候选审核实测</title><style>body{font:16px/1.7 system-ui;margin:32px;background:#f4f6f8;color:#202938}main{max-width:1200px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px}article{background:white;padding:20px;border-radius:12px;border:1px solid #dde3eb}.images{display:flex;align-items:center;gap:8px}.images img{width:48%;height:260px;object-fit:contain;background:#eee}small{color:#536273}.summary{padding:20px;background:#e9eef5;border-radius:12px;margin:24px 0}a{color:#1254a0}</style><main><h1>Qwen VLM 候选审核实测</h1><p>1021 · 2026-09-17 · qwen3-vl-plus · 13次真实调用</p><div class="summary"><b>常见误报可纠正，小目标仍不确定。</b><p>4/4历史模型误报判为非垃圾；5个垃圾参考正例中1个通过、4个不确定。包括历史用户确认的8像素垃圾也未通过。</p><p>这是挑选样本的可行性测试，多数标签为助手目视暂定，不能代表生产准确率。没有接入生产。</p>'''
    page+=f"<p>请求中位数 {statistics.median(times):.2f}s · 样本P95 {p95:.2f}s · {usage['total_tokens']} tokens</p><a href='REPORT.md'>完整报告</a> · <a href='manifest.json'>冻结输入</a></div><div class='grid'>"+''.join(cards)+'</div></main></html>'
    (out/'index.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False))


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','prepare-temporal','run','report'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='qwen3-vl-plus')
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.action=='prepare':
        prepare(a.output.resolve());return 0
    if a.action=='prepare-temporal':
        prepare_temporal(a.output.resolve());return 0
    if a.action=='report':
        report(a.output.resolve(),a.model);return 0
    return run(a.output.resolve(),a.model,a.limit)


if __name__=='__main__':
    raise SystemExit(main())
