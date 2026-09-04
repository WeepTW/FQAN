#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_XLSX = WORKSPACE_ROOT / "data" / "src" / "narratives" / "narrative2.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "FinFlier" / "econ_014_consistency_examples"
SOURCE_ID = "Econ_014"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
PALETTE = {
    "Actual": "#cb5353",
    "Goldman Sachs, 2003": "#1e7896",
    "Goldman Sachs, 2011": "#6d5a99",
    "Goldman Sachs, 2022": "#607446",
    "OECD, 2021": "#b05d25",
    "Capital Economics, 2023": "#1f6f8b",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_econ014(xlsx: Path) -> dict[str, Any]:
    workbook = openpyxl.load_workbook(xlsx, data_only=True, read_only=True)
    sheet = workbook["label"]
    rows = sheet.iter_rows(values_only=True)
    headers = [str(value) if value is not None else "" for value in next(rows)]
    for row in rows:
        record = dict(zip(headers, row))
        if record.get("Source") == SOURCE_ID:
            return {
                "source": SOURCE_ID,
                "chart_type": record.get("type"),
                "data": json.loads(record["data"]),
                "text": record.get("text") or "",
                "result": json.loads(record["result"]),
                "reason": record.get("reason") or "",
                "xlsx": str(xlsx),
            }
    raise ValueError(f"{SOURCE_ID} not found in {xlsx}")


def columns(rows: list[dict[str, Any]]) -> list[str]:
    return list(rows[0].keys()) if rows else []


def metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    return [name for name in columns(rows) if name != "Year"]


def normalize_binding(binding: dict[str, Any]) -> dict[str, Any]:
    object_names = binding.get("ObjectName") or []
    if not isinstance(object_names, list):
        object_names = [object_names]
    nums = binding.get("Num") or []
    if not isinstance(nums, list):
        nums = [nums]
    return {
        "object_names": [str(item) for item in object_names if item not in (None, "")],
        "data_name": binding.get("DataName"),
        "positions": [
            {"begin": item.get("Begin", []), "end": item.get("End", [])}
            for item in binding.get("Position", [])
            if isinstance(item, dict)
        ],
        "trend": binding.get("Trend") or "None",
        "nums": nums,
        "evidence_text": binding.get("Text") or "",
        "raw": binding,
    }


def merged_error_bindings(bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(bindings) < 3:
        raise ValueError("Econ_014 is expected to have at least three bindings")
    first = deepcopy(bindings[0])
    second = deepcopy(bindings[1])
    third = deepcopy(bindings[2])
    merged = {
        "ObjectName": (first.get("ObjectName") or []) + (second.get("ObjectName") or []),
        "DataName": f"{first.get('DataName')}; {second.get('DataName')}",
        "Position": [
            {
                "Begin": first.get("Position", [{}])[0].get("Begin", []),
                "End": second.get("Position", [{}])[0].get("End", []),
            }
        ],
        "Trend": "None",
        "Num": (first.get("Num") or []) + (second.get("Num") or []),
        "Text": f"{first.get('Text', '')}; {second.get('Text', '')}",
    }
    return [merged, third]


def trend_zh(value: Any) -> str:
    text = str(value or "None")
    return "無(None)" if text.lower() in {"none", "null", ""} else text


def fmt_num(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:,.1f}" if abs(value - round(value)) > 1e-9 else f"{int(value):,}"
    return str(value)


def html_document(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    title = html.escape(payload["title"])
    palette_json = json.dumps(PALETTE, ensure_ascii=False)
    return f'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
  :root{{--bg:#eef3f6;--panel:#ffffff;--border:#d9e2e7;--muted:#60717d;--text:#1d2c35;--blue:#1e7896;--blue2:#176d88;--red:#cb5353;--red-bg:#fff0f0;--shadow:0 8px 22px rgba(26,45,61,.10);}}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:"Noto Sans TC","Microsoft JhengHei","PingFang TC",Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;color:var(--text);background:linear-gradient(180deg,#f3f7fa 0%,var(--bg) 100%);}}
  .page{{width:1100px;min-height:760px;margin:0 auto;padding:6px 0 24px;}}
  .stats{{display:grid;grid-template-columns:1fr 1fr 1fr 2.15fr 1.1fr;gap:10px;margin:0 0 14px;}}
  .stat{{background:var(--panel);border:1px solid var(--border);border-radius:7px;min-height:62px;padding:12px 14px;box-shadow:var(--shadow);}}
  .label{{font-size:11px;letter-spacing:.10em;color:#758793;font-weight:800;margin-bottom:8px;}}
  .value{{font-size:15px;font-weight:800;color:#1e2f39;line-height:1.15;word-break:break-word;}}
  .value.small{{font-size:12px;line-height:1.25;}}
  .layout{{display:grid;grid-template-columns:1.58fr .92fr;gap:14px;}}
  .card{{background:var(--panel);border:1px solid var(--border);border-radius:8px;box-shadow:var(--shadow);}}
  .main-card{{padding:14px;}}
  .side-card{{padding:12px;min-height:280px;}}
  .section-title{{font-size:11px;letter-spacing:.10em;color:#718591;font-weight:800;margin-bottom:8px;}}
  .chart-head{{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;gap:12px;}}
  h1{{font-size:16px;margin:0;font-weight:800;}}
  .legend{{font-size:11px;color:#536774;display:flex;align-items:center;justify-content:flex-end;gap:7px;flex-wrap:wrap;}}
  .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;}}
  .chart-wrap{{height:342px;border:1px solid #dce6eb;border-radius:6px;padding:8px 10px 4px;overflow:hidden;background:#fff;}}
  svg text{{font-family:"Noto Sans TC","Microsoft JhengHei","PingFang TC",Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;fill:#566b76;}}
  .table-wrap{{margin-top:12px;border:1px solid #dce6eb;border-radius:6px;overflow:auto;background:#fff;max-height:220px;}}
  table{{border-collapse:collapse;width:100%;font-size:11px;min-width:760px;}}
  thead th{{background:#f4f8fa;color:#2c414d;text-align:left;padding:9px 8px;border-bottom:1px solid #dbe5ea;font-weight:800;white-space:nowrap;}}
  tbody td{{padding:8px;border-bottom:1px solid #e5edf1;color:#2e4552;white-space:nowrap;text-align:right;}}
  tbody td:first-child,thead th:first-child{{text-align:left;}}
  tbody tr:nth-child(even){{background:#fbfdfe;}}
  td.active-cell{{background:#fff4f4;color:#9d3333;font-weight:900;outline:1.5px solid #d16060;outline-offset:-2px;}}
  .narrative{{border:1px solid #dce6eb;border-radius:7px;background:#fff;padding:12px 13px;margin-bottom:12px;}}
  .narrative p{{font-size:13px;color:#43535e;margin:0;line-height:1.55;overflow-wrap:anywhere;}}
  .binding-title{{font-size:12px;letter-spacing:.08em;color:#748995;font-weight:900;margin:10px 0 5px;}}
  .matched{{font-size:16px;font-weight:900;margin:0 0 8px;color:#1f2e38;}}
  .binding-options{{display:grid;grid-template-columns:1fr;gap:6px;margin-bottom:10px;}}
  .binding-option{{border:1px solid #d9e3e8;border-radius:7px;background:#fff;color:#445965;padding:8px 10px;text-align:left;cursor:pointer;font-weight:800;font-size:12px;}}
  .binding-option.active{{border-color:var(--red);color:#9d3333;background:#fff4f4;}}
  .evidence{{border:1.5px solid var(--red);border-radius:7px;padding:12px;background:var(--red-bg);}}
  .evidence .obj{{font-size:14px;font-weight:900;margin-bottom:8px;color:#26343d;}}
  .tags{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;}}
  .tag{{font-size:11px;color:#4b6573;background:#fff;border:1px solid #d9e3e8;border-radius:999px;padding:4px 7px;font-weight:700;max-width:100%;}}
  .evidence-text{{font-size:12px;line-height:1.55;color:#7a5b5b;margin:0;overflow-wrap:anywhere;}}
  .reason{{margin-top:12px;border:1px solid #dce6eb;border-radius:7px;background:#fff;padding:12px 13px;min-height:45px;}}
  .reason p{{font-size:12px;color:#43535e;line-height:1.55;margin:0;overflow-wrap:anywhere;}}
</style>
</head>
<body>
<div class="page">
  <div class="stats">
    <div class="stat"><div class="label">列數</div><div class="value" id="rowCount"></div></div>
    <div class="stat"><div class="label">欄數</div><div class="value" id="colCount"></div></div>
    <div class="stat"><div class="label">資料繫結數</div><div class="value" id="bindingCount"></div></div>
    <div class="stat"><div class="label">目前資料</div><div class="value small" id="activeData"></div></div>
    <div class="stat"><div class="label">趨勢</div><div class="value" id="activeTrend"></div></div>
  </div>
  <div class="layout">
    <section class="card main-card"><div class="section-title">圖表</div><div class="chart-head"><h1>Econ_014 · 多線圖</h1><div class="legend" id="legend"></div></div><div class="chart-wrap"><svg id="chart" width="666" height="318" viewBox="0 0 666 318" role="img" aria-label="中國GDP相對美國GDP預測與實際值"></svg></div><div class="table-wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div></section>
    <aside class="card side-card"><div class="narrative"><div class="section-title">敘事文字</div><p id="narrative"></p></div><div class="binding-title">資料繫結(Data Binding)</div><div class="binding-options" id="bindingOptions"></div><div class="matched">配對證據(Matched Evidence)</div><div class="evidence"><div class="obj" id="objectName"></div><div class="tags" id="tags"></div><p class="evidence-text" id="evidenceText"></p></div><div class="reason"><div class="section-title">理由(Reason)</div><p id="reason"></p></div></aside>
  </div>
</div>
<script>
const payload = {payload_json};
const colors = {palette_json};
let selectedBinding = 0;
function fmt(value){{ if(value===null || value===undefined || value==='') return ''; if(typeof value==='number') return Number.isInteger(value) ? String(value) : value.toLocaleString(undefined, {{maximumFractionDigits:1}}); return String(value); }}
function esc(value){{ return String(value ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;'); }}
function trendZh(value){{ const text=String(value || 'None'); return ['None','none','null',''].includes(text) ? '無(None)' : text; }}
function currentBinding(){{ return payload.bindings[selectedBinding]; }}
function activeCells(binding){{ const cells=new Set(); (binding.positions || []).forEach(pos=>{{ const b=pos.begin || []; const e=pos.end || []; if(b.length>=2 && e.length>=2){{ for(let r=Math.min(b[0],e[0]); r<=Math.max(b[0],e[0]); r++) for(let c=Math.min(b[1],e[1]); c<=Math.max(b[1],e[1]); c++) cells.add(`${{r}}:${{c}}`); }} }}); return cells; }}
function renderSummary(){{ const b=currentBinding(); document.getElementById('rowCount').textContent=payload.rows.length; document.getElementById('colCount').textContent=payload.columns.length; document.getElementById('bindingCount').textContent=payload.bindings.length; document.getElementById('activeData').textContent=b?.data_name || 'None'; document.getElementById('activeTrend').textContent=trendZh(b?.trend); }}
function renderBindings(){{ const box=document.getElementById('bindingOptions'); box.innerHTML=payload.bindings.map((b,i)=>`<button class="binding-option${{i===selectedBinding?' active':''}}" type="button" data-index="${{i}}">${{esc(`繫結 ${{i+1}}：${{(b.object_names || []).join('、') || 'ObjectName=None'}}`)}}</button>`).join(''); box.querySelectorAll('button').forEach(btn=>btn.addEventListener('click',()=>{{ selectedBinding=Number(btn.dataset.index); render(); }})); }}
function renderEvidence(){{ const b=currentBinding(); document.getElementById('objectName').textContent=(b.object_names || []).join('、') || 'ObjectName=None'; const nums=Array.isArray(b.nums) ? b.nums.map(fmt).join('、') : fmt(b.nums); document.getElementById('tags').innerHTML=`<span class="tag">資料(Data): ${{esc(b.data_name)}}</span><span class="tag">趨勢(Trend): ${{esc(trendZh(b.trend))}}</span><span class="tag">數值(Num): ${{esc(nums)}}</span>`; document.getElementById('evidenceText').textContent=b.evidence_text || ''; document.getElementById('reason').textContent=payload.reason || ''; }}
function renderTable(){{ const active=activeCells(currentBinding()); document.getElementById('thead').innerHTML='<tr>'+payload.columns.map(c=>`<th>${{esc(c)}}</th>`).join('')+'</tr>'; document.getElementById('tbody').innerHTML=payload.rows.map((row,r)=>'<tr>'+payload.columns.map((c,ci)=>`<td class="${{active.has(`${{r}}:${{ci}}`)?'active-cell':''}}">${{esc(fmt(row[c]))}}</td>`).join('')+'</tr>').join(''); }}
function svgEl(name, attrs){{ const e=document.createElementNS('http://www.w3.org/2000/svg', name); Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v)); return e; }}
function renderChart(){{ const svg=document.getElementById('chart'); svg.innerHTML=''; const W=666,H=318,m={{l:44,r:16,t:18,b:44}}, plotW=W-m.l-m.r, plotH=H-m.t-m.b; const metrics=payload.metrics; const values=[]; payload.rows.forEach(row=>metrics.forEach(metric=>{{ const v=row[metric]; if(typeof v==='number') values.push(v); }})); const min=Math.min(...values), max=Math.max(...values), pad=(max-min)*0.08, yMin=min-pad, yMax=max+pad; const x=i=>m.l+(i/(payload.rows.length-1))*plotW; const y=v=>m.t+(1-(v-yMin)/(yMax-yMin))*plotH; for(let i=0;i<=4;i++){{ const gy=m.t+plotH/4*i; const val=yMax-(yMax-yMin)/4*i; svg.appendChild(svgEl('line',{{x1:m.l,y1:gy,x2:W-m.r,y2:gy,stroke:'#e6edf1','stroke-width':1}})); const t=svgEl('text',{{x:m.l-8,y:gy+4,'font-size':10,'text-anchor':'end'}}); t.textContent=fmt(val); svg.appendChild(t); }} svg.appendChild(svgEl('line',{{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,stroke:'#9badb6'}})); svg.appendChild(svgEl('line',{{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,stroke:'#9badb6'}})); const active=activeCells(currentBinding()); metrics.forEach(metric=>{{ let d=''; payload.rows.forEach((row,i)=>{{ const v=row[metric]; if(typeof v==='number') d += `${{d?'L':'M'}}${{x(i)}} ${{y(v)}} `; }}); svg.appendChild(svgEl('path',{{d,fill:'none',stroke:colors[metric] || '#1e7896','stroke-width':String(metric===currentBinding().data_name ? 3 : 1.8),opacity:String(metric===currentBinding().data_name ? 1 : .55)}})); payload.rows.forEach((row,i)=>{{ const v=row[metric]; if(typeof v!=='number') return; const ci=payload.columns.indexOf(metric), isActive=active.has(`${{i}}:${{ci}}`), r=isActive?5:2.8; svg.appendChild(svgEl('circle',{{cx:x(i),cy:y(v),r,fill:colors[metric] || '#1e7896',stroke:isActive?'#fff':'none','stroke-width':isActive?2:0}})); if(isActive){{ const label=svgEl('text',{{x:x(i)+7,y:y(v)-7,'font-size':11,fill:'#b93d3d','font-weight':700}}); label.textContent=fmt(v); svg.appendChild(label); }} }}); }}); payload.rows.forEach((row,i)=>{{ const label=svgEl('text',{{x:x(i),y:H-17,'font-size':9,'text-anchor':'middle'}}); label.textContent=row.Year; svg.appendChild(label); }}); document.getElementById('legend').innerHTML=metrics.map(metric=>`<span><span class="dot" style="background:${{colors[metric] || '#1e7896'}}"></span>${{esc(metric)}}</span>`).join(''); }}
function render(){{ document.getElementById('narrative').textContent=payload.text; renderSummary(); renderBindings(); renderEvidence(); renderTable(); renderChart(); }}
render();
</script>
</body>
</html>
'''


def build_payload(base: dict[str, Any], variant: str, bindings: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    rows = base["data"]
    return {
        "source": SOURCE_ID,
        "variant": variant,
        "title": "Econ_014 · 中國GDP預測資料繫結範例",
        "generated_at_utc": utc_now(),
        "source_xlsx": base["xlsx"],
        "chart_type": base["chart_type"],
        "columns": columns(rows),
        "metrics": metric_columns(rows),
        "rows": rows,
        "text": base["text"],
        "reason": reason,
        "bindings": [normalize_binding(item) for item in bindings],
        "raw_result": bindings,
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, fnt: ImageFont.FreeTypeFont, fill: str, line_gap: int = 4) -> int:
    x, y = xy
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for char in paragraph:
            candidate = current + char
            if draw.textlength(candidate, font=fnt) <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def make_preview_png(payload: dict[str, Any], path: Path) -> None:
    image = Image.new("RGB", (1100, 780), "#eef3f6")
    draw = ImageDraw.Draw(image)
    regular, small, bold, title_font, label_font = font(14), font(11), font(15, True), font(18, True), font(11, True)
    rows, metrics, binding = payload["rows"], payload["metrics"], payload["bindings"][0]
    labels = [("列數", str(len(rows))), ("欄數", str(len(payload["columns"]))), ("資料繫結數", str(len(payload["bindings"]))), ("目前資料", binding["data_name"]), ("趨勢", trend_zh(binding["trend"]))]
    x = 0
    for (label, value), w in zip(labels, [130, 130, 130, 470, 220]):
        draw.rounded_rectangle((x + 6, 8, x + w - 4, 70), radius=7, fill="#ffffff", outline="#d9e2e7")
        draw.text((x + 20, 20), label, font=label_font, fill="#758793")
        draw_wrapped(draw, (x + 20, 40), value, w - 36, bold, "#1d2c35", 2)
        x += w
    draw.rounded_rectangle((6, 86, 686, 754), radius=8, fill="#ffffff", outline="#d9e2e7")
    draw.rounded_rectangle((704, 86, 1094, 754), radius=8, fill="#ffffff", outline="#d9e2e7")
    draw.text((22, 104), "圖表", font=label_font, fill="#718591")
    draw.text((22, 124), "Econ_014 · 多線圖", font=title_font, fill="#1d2c35")
    cx0, cy0, cx1, cy1 = 22, 154, 670, 474
    draw.rounded_rectangle((cx0, cy0, cx1, cy1), radius=6, fill="#ffffff", outline="#dce6eb")
    values = [row[m] for row in rows for m in metrics if isinstance(row.get(m), (int, float))]
    mn, mx = min(values), max(values)
    pad = (mx - mn) * 0.08
    y_min, y_max = mn - pad, mx + pad
    ml, mr, mt, mb = 48, 12, 18, 42
    plot = (cx0 + ml, cy0 + mt, cx1 - mr, cy1 - mb)
    def sx(i: int) -> float:
        return plot[0] + i / (len(rows) - 1) * (plot[2] - plot[0])
    def sy(v: float) -> float:
        return plot[1] + (1 - (v - y_min) / (y_max - y_min)) * (plot[3] - plot[1])
    active_cells = set()
    for pos in binding["positions"]:
        b, e = pos["begin"], pos["end"]
        if len(b) >= 2 and len(e) >= 2:
            for rr in range(min(b[0], e[0]), max(b[0], e[0]) + 1):
                for cc in range(min(b[1], e[1]), max(b[1], e[1]) + 1):
                    active_cells.add((rr, cc))
    for i in range(5):
        yy = plot[1] + (plot[3] - plot[1]) / 4 * i
        draw.line((plot[0], yy, plot[2], yy), fill="#e6edf1")
        draw.text((plot[0] - 42, yy - 6), fmt_num(y_max - (y_max - y_min) / 4 * i), font=small, fill="#566b76")
    for metric in metrics:
        pts = [(sx(i), sy(row[metric])) for i, row in enumerate(rows) if isinstance(row.get(metric), (int, float))]
        color = PALETTE.get(metric, "#1e7896")
        for a, b in zip(pts, pts[1:]):
            draw.line((*a, *b), fill=color, width=3 if metric == binding["data_name"] else 2)
        for i, row in enumerate(rows):
            if not isinstance(row.get(metric), (int, float)):
                continue
            px, py = sx(i), sy(row[metric])
            col = payload["columns"].index(metric)
            active = (i, col) in active_cells
            r = 5 if active else 3
            draw.ellipse((px-r, py-r, px+r, py+r), fill=color, outline="#ffffff" if active else color, width=2)
            if active:
                draw.text((px + 8, py - 12), fmt_num(row[metric]), font=small, fill="#cb5353")
    for i, row in enumerate(rows):
        draw.text((sx(i) - 13, cy1 - 26), str(row["Year"]), font=small, fill="#566b76")
    table_y = 492
    draw.rounded_rectangle((22, table_y, 670, 738), radius=6, fill="#ffffff", outline="#dce6eb")
    preferred_cols = ["Year", "Actual", "OECD, 2021", "Capital Economics, 2023"]
    cols = [col for col in preferred_cols if col in payload["columns"]]
    col_w = [70, 100, 160, 190][: len(cols)]
    tx = 30
    for c, w in zip(cols, col_w):
        draw.text((tx, table_y + 12), c, font=small, fill="#1d2c35")
        tx += w
    for ri, row in enumerate(rows[:8]):
        yy, tx = table_y + 36 + ri * 24, 30
        for c, w in zip(cols, col_w):
            ci = payload["columns"].index(c)
            active = (ri, ci) in active_cells
            if active:
                draw.rectangle((tx - 3, yy - 3, tx + w - 6, yy + 18), fill="#fff0f0", outline="#cb5353")
            draw.text((tx, yy), fmt_num(row[c]), font=small, fill="#9d3333" if active else "#2e4552")
            tx += w
    draw.text((720, 104), "敘事文字", font=label_font, fill="#718591")
    y = draw_wrapped(draw, (720, 124), payload["text"], 350, regular, "#43535e", 5) + 14
    draw.text((720, y), "資料繫結(Data Binding)", font=label_font, fill="#748995")
    y += 22
    for i, b in enumerate(payload["bindings"]):
        fill, outline = ("#fff4f4", "#cb5353") if i == 0 else ("#ffffff", "#d9e3e8")
        draw.rounded_rectangle((720, y, 1078, y + 38), radius=7, fill=fill, outline=outline)
        draw_wrapped(draw, (732, y + 9), f"繫結 {i+1}：{'、'.join(b['object_names'])}", 330, small, "#9d3333" if i == 0 else "#445965", 2)
        y += 44
    y += 4
    draw.text((720, y), "配對證據(Matched Evidence)", font=title_font, fill="#1d2c35")
    y += 30
    draw.rounded_rectangle((720, y, 1078, y + 142), radius=7, fill="#fff0f0", outline="#cb5353", width=2)
    draw_wrapped(draw, (734, y + 12), "、".join(binding["object_names"]), 326, bold, "#1d2c35", 3)
    draw_wrapped(draw, (734, y + 42), f"資料(Data): {binding['data_name']}  趨勢(Trend): {trend_zh(binding['trend'])}  數值(Num): {'、'.join(fmt_num(v) for v in binding['nums'])}", 326, small, "#4b6573", 3)
    draw_wrapped(draw, (734, y + 86), binding["evidence_text"], 326, small, "#7a5b5b", 4)
    y += 158
    draw.text((720, y), "理由(Reason)", font=label_font, fill="#718591")
    draw_wrapped(draw, (720, y + 20), payload["reason"], 350, small, "#43535e", 4)
    image.save(path)


def write_readme(output_dir: Path, files: dict[str, str], xlsx: Path) -> None:
    readme = f'''# Econ_014 consistency examples

Source: `{xlsx}` (`label` sheet, `Source=Econ_014`).

## Files
- Correct interface: `{files['correct_html']}`
- Merged-error interface: `{files['error_html']}`
- Correct payload: `{files['correct_payload']}`
- Merged-error payload: `{files['error_payload']}`
- Correct preview PNG: `{files['correct_png']}`
- Merged-error preview PNG: `{files['error_png']}`

## Construction
- Correct version keeps the three original bindings from `narrative2.xlsx`.
- Merged-error version intentionally fuses binding 1 and binding 2 into one binding, leaving two bindings total.
- UI wording follows the Traditional Chinese labels used in `preview.html`: `列數`, `欄數`, `資料繫結數`, `目前資料`, `趨勢`, `圖表`, `敘事文字`, `資料繫結(Data Binding)`, `配對證據(Matched Evidence)`, and `理由(Reason)`.
- PNG files are deterministic PIL previews generated from the same payloads. This environment has no Chromium/Playwright/Selenium browser renderer installed, so they are not browser raster screenshots.
'''
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Econ_014 FinFlier consistency examples.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = load_econ014(args.xlsx)
    correct_bindings = base["result"]
    error_bindings = merged_error_bindings(correct_bindings)
    correct = build_payload(base, "correct", correct_bindings, base["reason"])
    error_reason = "The OECD 2021 forecast and Capital Economics 2023 forecast are treated as one combined binding, while US debt remains a separate reference binding."
    error = build_payload(base, "merged_error", error_bindings, error_reason)
    outputs = {
        "correct_html": "econ_014_correct_zh.html",
        "error_html": "econ_014_merged_error_zh.html",
        "correct_payload": "econ_014_correct_payload.json",
        "error_payload": "econ_014_merged_error_payload.json",
        "correct_png": "econ_014_correct_preview.png",
        "error_png": "econ_014_merged_error_preview.png",
    }
    (args.output_dir / outputs["correct_payload"]).write_text(json.dumps(correct, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / outputs["error_payload"]).write_text(json.dumps(error, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / outputs["correct_html"]).write_text(html_document(correct), encoding="utf-8")
    (args.output_dir / outputs["error_html"]).write_text(html_document(error), encoding="utf-8")
    make_preview_png(correct, args.output_dir / outputs["correct_png"])
    make_preview_png(error, args.output_dir / outputs["error_png"])
    write_readme(args.output_dir, outputs, args.xlsx)
    report = {
        "status": "completed",
        "output_dir": str(args.output_dir),
        "source_xlsx": str(args.xlsx),
        "correct_binding_count": len(correct["bindings"]),
        "merged_error_binding_count": len(error["bindings"]),
        "files": {key: str(args.output_dir / value) for key, value in outputs.items()},
        "readme": str(args.output_dir / "README.md"),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
