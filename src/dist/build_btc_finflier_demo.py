#!/usr/bin/env python3
"""Build static FinFlier-style BTC demo pages for five model routes.

The builder intentionally separates ground truth from model output. If a model
runtime is not executable, the page records a runtime_blocked status instead of
copying the gold binding as a prediction.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_SOURCE = WORKSPACE_ROOT / "data" / "src" / "BTC-USD_2024-12-24.xlsx"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "data" / "financial_narratives" / "demo"


MODEL_ROUTES = [
    ("A", "GPT-5.5", "gpt5_5", "non_adapter_generator"),
    ("B", "finqa_flan_m", "finqa_flan_m", "fine_tuned_retriever"),
    ("C", "finqa_mistral_m", "finqa_mistral_m", "fine_tuned_retriever"),
    ("D", "Qwen3.6", "qwen3_6", "non_adapter_generator"),
    ("E", "Mistral4", "mistral4", "non_adapter_generator"),
]


FINFLIER_STYLE_SYSTEM = (
    "You are a financial data binding assistant. Given chart data and narrative text, "
    "extract every data-text binding. Return only: result: "
    "[{\"ObjectName\":[],\"DataName\":\"\",\"Position\":[{\"Begin\":[],\"End\":[]}],"
    "\"Trend\":\"None\",\"Num\":[],\"Text\":\"\"}] reason: \"\""
)


@dataclass(frozen=True)
class RuntimeStatus:
    status: str
    reason: str
    route_status: str | None = None
    missing: list[str] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_case_id(value: Any, row_number: int) -> str:
    text = str(value or "").strip()
    if not text:
        return f"row_{row_number}"
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or f"row_{row_number}"


def parse_jsonish(value: Any, fallback: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def clean_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def normalize_num(value: Any) -> list[str]:
    values = value if isinstance(value, list) else ([] if value in (None, "", "None") else [value])
    normalized: list[str] = []
    for item in values:
        if item in (None, "", "None"):
            continue
        try:
            normalized.append(f"{float(str(item).replace(',', '').replace('%', '')):.12g}")
        except ValueError:
            normalized.append(str(item).strip())
    return normalized


def normalize_field(item: dict[str, Any], field: str) -> Any:
    value = item.get(field)
    if field == "ObjectName":
        if isinstance(value, list):
            return [str(v).strip() for v in value]
        if value in (None, "", "None"):
            return []
        return [str(value).strip()]
    if field == "Num":
        return normalize_num(value)
    if value in (None, ""):
        return "None"
    return str(value).strip()


def compare_binding_fields(gold: list[dict[str, Any]], prediction: list[dict[str, Any]] | None) -> dict[str, Any]:
    fields = ["ObjectName", "DataName", "Trend", "Num"]
    if prediction is None:
        return {
            "status": "not_compared",
            "reason": "model_runtime_blocked",
            "fields": fields,
            "exact_match": False,
        }
    rows: list[dict[str, Any]] = []
    exact = len(gold) == len(prediction)
    for index, gold_item in enumerate(gold):
        pred_item = prediction[index] if index < len(prediction) else {}
        field_rows = {}
        for field in fields:
            gold_value = normalize_field(gold_item, field)
            pred_value = normalize_field(pred_item, field)
            matched = gold_value == pred_value
            exact = exact and matched
            field_rows[field] = {
                "gold": gold_value,
                "prediction": pred_value,
                "match": matched,
            }
        rows.append({"binding_index": index, "fields": field_rows})
    return {
        "status": "completed",
        "fields": fields,
        "exact_match": exact,
        "rows": rows,
    }


def compare_cases(cases: list[dict[str, Any]], predictions: list[list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    fields = ["ObjectName", "DataName", "Trend", "Num"]
    if predictions is None:
        return {
            "status": "not_compared",
            "reason": "model_runtime_blocked",
            "fields": fields,
            "exact_match": False,
            "case_count": len(cases),
            "cases": [
                {
                    "case_id": case["case_id"],
                    "status": "not_compared",
                    "reason": "model_runtime_blocked",
                    "exact_match": False,
                }
                for case in cases
            ],
        }
    case_rows = []
    exact = len(cases) == len(predictions)
    for index, case in enumerate(cases):
        prediction = predictions[index] if index < len(predictions) else []
        comparison = compare_binding_fields(case["gold_binding"], prediction)
        exact = exact and comparison["exact_match"]
        case_rows.append({"case_id": case["case_id"], **comparison})
    return {
        "status": "completed",
        "fields": fields,
        "exact_match": exact,
        "case_count": len(cases),
        "cases": case_rows,
    }


def build_prompt(data_text: str, narrative_text: str) -> str:
    messages = [
        {"role": "system", "content": FINFLIER_STYLE_SYSTEM},
        {"role": "user", "content": f"data:{data_text}\ntext:[{narrative_text}]"},
    ]
    return json.dumps(messages, ensure_ascii=False, indent=2)


def infer_runtime_status(route_id: str, route_type: str) -> RuntimeStatus:
    if route_type == "fine_tuned_retriever":
        adapter = REPO_ROOT / "Experiment" / route_id / "retriever" / "model" / "adapter_config.json"
        if not adapter.is_file():
            return RuntimeStatus("runtime_blocked", f"missing adapter_config.json: {adapter}")
        return RuntimeStatus(
            "runtime_blocked",
            "fine-tuned retriever is available, but formal RetFact-to-Binding conversion requires an executable binding converter; GPT-5.5 is currently credential-blocked.",
            route_status="converter_blocked",
            missing=["OPENAI_API_KEY or CODEX_API_KEY or CODEX_CLI_ASSUME_AUTH=1"],
        )

    try:
        sys.path.insert(0, str(REPO_ROOT))
        import new_full_finqa_run as runtime  # noqa: WPS433

        config = runtime.resolve_engine(route_id, credential_purpose="execute")
        route_status = runtime.route_execution_status(config)
        if config.available:
            return RuntimeStatus("ready_not_run", "runtime appears executable, but this builder is report-only.", route_status=route_status)
        return RuntimeStatus(
            "runtime_blocked",
            "model runtime is not executable in the current shell.",
            route_status=route_status,
            missing=list(config.missing_credentials),
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return RuntimeStatus("runtime_blocked", f"runtime inspection failed: {exc.__class__.__name__}: {exc}")


def chart_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = list(rows[0].keys()) if rows else []
    numeric_columns: list[str] = []
    for column in columns:
        values = [row.get(column) for row in rows]
        numeric_count = sum(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
        if numeric_count:
            numeric_columns.append(column)
    x_column = next((candidate for candidate in ["time", "Time", "date", "Date"] if candidate in columns), columns[0] if columns else "")
    return {
        "columns": columns,
        "x_column": x_column,
        "metric_columns": [column for column in numeric_columns if column != x_column],
    }


def build_cases(source: Path) -> list[dict[str, Any]]:
    frame = pd.read_excel(source, sheet_name="Sheet1")
    cases: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        table = parse_jsonish(row.get("data"), [])
        table = [{str(k): clean_scalar(v) for k, v in item.items()} for item in table if isinstance(item, dict)]
        gold = parse_jsonish(row.get("result"), [])
        cases.append(
            {
                "case_id": stable_case_id(row.get("file"), index + 2) + f"_{index + 1}",
                "pattern": clean_scalar(row.get("pattern")),
                "chart_type": clean_scalar(row.get("type")),
                "file": clean_scalar(row.get("file")),
                "table": table,
                "schema": chart_schema(table),
                "text": clean_scalar(row.get("text")),
                "gold_binding": gold if isinstance(gold, list) else [],
                "gold_reason": clean_scalar(row.get("reason")),
                "prompt": build_prompt(str(row.get("data") or ""), str(row.get("text") or "")),
            }
        )
    return cases


def json_script_tag(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return "<script id=\"demo-data\" type=\"application/json\">" + html.escape(text) + "</script>"


def render_html(payload: dict[str, Any]) -> str:
    title = f"{payload['folder']} - {payload['model_label']}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f6f8; --panel: #fff; --ink: #17212b; --muted: #647285;
      --line: #d9e0e7; --accent: #1f6f8b; --warn: #b05d25; --bad: #b42318;
      --good: #2f6f4e; --shadow: 0 12px 28px rgba(31,45,61,.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Segoe UI", Arial, sans-serif; }}
    main {{ padding: 24px; max-width: 1420px; margin: 0 auto; }}
    header {{ display: flex; justify-content: space-between; align-items: end; gap: 18px; margin-bottom: 16px; }}
    .controls {{ display: flex; align-items: end; gap: 12px; flex-wrap: wrap; justify-content: flex-end; }}
    label {{ display: block; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 5px; }}
    select {{ min-width: 260px; height: 38px; border: 1px solid var(--line); border-radius: 7px; background: #fff; color: var(--ink); padding: 0 10px; font: inherit; }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 34px; }}
    h2 {{ font-size: 18px; }}
    .eyebrow {{ margin: 0 0 5px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }}
    .status {{ padding: 9px 12px; border-radius: 8px; background: #fff7ed; color: var(--warn); border: 1px solid #fed7aa; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(360px, .8fr); gap: 14px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); padding: 14px; }}
    .summary {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; box-shadow: var(--shadow); }}
    .metric b {{ display: block; overflow-wrap: anywhere; }}
    .chart {{ height: 360px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfd; margin-top: 12px; }}
    svg {{ width: 100%; height: 100%; display: block; }}
    .axis {{ stroke: #7d8997; }}
    .gridline {{ stroke: #e7edf2; }}
    .line {{ fill: none; stroke-width: 2.4; }}
    .point {{ stroke: #fff; stroke-width: 1.5; }}
    .bound {{ fill: var(--bad); stroke: #fff; stroke-width: 2; }}
    .band {{ fill: rgba(180, 35, 24, .08); }}
    .label {{ fill: var(--muted); font-size: 11px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 9px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef3f6; }}
    td.bound-cell {{ color: var(--bad); background: #fff0f1; font-weight: 700; }}
    .binding {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; margin-top: 8px; background: #fbfcfd; }}
    .binding.active {{ border-color: var(--bad); background: #fff7f7; }}
    .pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }}
    .pill {{ border-radius: 999px; background: #edf2f5; color: #394b5d; font-size: 12px; padding: 3px 8px; font-weight: 700; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #111827; color: #f9fafb; padding: 12px; border-radius: 8px; max-height: 320px; overflow: auto; }}
    .small {{ color: var(--muted); line-height: 1.5; }}
    .ok {{ color: var(--good); }} .bad {{ color: var(--bad); }}
    @media (max-width: 980px) {{ .grid, .summary {{ grid-template-columns: 1fr; }} header {{ display: block; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <p class="eyebrow">FinFlier2 static demo</p>
      <h1>{html.escape(title)}</h1>
    </div>
    <div class="controls">
      <div>
        <label for="case-select">Case</label>
        <select id="case-select"></select>
      </div>
      <div class="status">{html.escape(payload['runtime_status']['status'])}</div>
    </div>
  </header>
  <section class="summary" id="summary"></section>
  <section class="grid">
    <div class="card">
      <p class="eyebrow">Chart and table</p>
      <h2 id="case-title"></h2>
      <div class="chart"><svg id="chart" role="img" aria-label="demo chart"></svg></div>
      <div id="table"></div>
    </div>
    <aside>
      <section class="card">
        <p class="eyebrow">Narrative</p>
        <p id="narrative" class="small"></p>
      </section>
      <section class="card" style="margin-top:14px">
        <p class="eyebrow">Ground truth binding</p>
        <div id="bindings"></div>
      </section>
      <section class="card" style="margin-top:14px">
        <p class="eyebrow">Model status</p>
        <p class="small" id="runtime"></p>
        <p class="small" id="comparison"></p>
      </section>
    </aside>
  </section>
  <section class="card" style="margin-top:14px">
    <p class="eyebrow">FinFlier-style prompt from Excel cells</p>
    <pre id="prompt"></pre>
  </section>
</main>
{json_script_tag(payload)}
<script>
const payload = JSON.parse(document.getElementById('demo-data').textContent);
let caseIndex = 0;
let bindingIndex = 0;
const colors = ['#1f6f8b', '#b05d25', '#607446', '#6d5a99'];
function fmt(value) {{
  if (value === null || value === undefined || value === '') return 'null';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toLocaleString(undefined, {{maximumFractionDigits: 4}});
  return String(value);
}}
function num(value) {{
  if (typeof value === 'number') return value;
  const parsed = Number(String(value).replaceAll(',', '').replace('%', ''));
  return Number.isFinite(parsed) ? parsed : null;
}}
function record() {{ return payload.cases[caseIndex]; }}
function binding() {{ return record().gold_binding[bindingIndex] || null; }}
function initCaseSelect() {{
  const select = document.getElementById("case-select");
  select.innerHTML = payload.cases.map((c, i) => "<option value=\"" + i + "\">" + (i + 1) + " · " + c.file + " · " + c.chart_type + "</option>").join("");
  select.addEventListener("change", event => {{ caseIndex = Number(event.target.value); bindingIndex = 0; render(); }});
}}
function isBoundCell(r, c) {{
  const b = binding(); if (!b || !Array.isArray(b.Position)) return false;
  return b.Position.some(p => {{
    const br = p.Begin || []; const en = p.End || [];
    const r0 = Math.min(br[0], en[0]); const r1 = Math.max(br[0], en[0]);
    const c0 = Math.min(br[1], en[1]); const c1 = Math.max(br[1], en[1]);
    return r >= r0 && r <= r1 && c >= c0 && c <= c1;
  }});
}}
function render() {{
  const r = record();
  document.getElementById("case-select").value = String(caseIndex);
  document.getElementById('summary').innerHTML = [
    ['Folder', payload.folder], ['Model', payload.model_label], ['Route', payload.route_id],
    ['Cases', payload.cases.length], ['Generated', payload.generated_at_utc]
  ].map(([k,v]) => `<div class="metric"><p class="eyebrow">${{k}}</p><b>${{fmt(v)}}</b></div>`).join('');
  document.getElementById('case-title').textContent = `${{r.file}} · ${{r.chart_type}}`;
  document.getElementById('narrative').textContent = r.text || '';
  document.getElementById('prompt').textContent = r.prompt || '';
  document.getElementById('runtime').textContent = `${{payload.runtime_status.reason}} Missing: ${{(payload.runtime_status.missing || []).join('; ') || 'none'}}`;
  document.getElementById('comparison').innerHTML = payload.comparison.exact_match
    ? '<span class="ok">ObjectName/DataName/Trend/Num match ground truth.</span>'
    : '<span class="bad">No verified model prediction equals ground truth. See rerun_report.json.</span>';
  renderBindings(r); renderTable(r); renderChart(r);
}}
function renderBindings(r) {{
  document.getElementById('bindings').innerHTML = r.gold_binding.map((b, i) => `
    <div class="binding ${{i === bindingIndex ? 'active' : ''}}" onclick="bindingIndex=${{i}}; render();">
      <h3>${{(b.ObjectName || []).join(', ') || 'ObjectName=None'}}</h3>
      <div class="pills"><span class="pill">${{fmt(b.DataName)}}</span><span class="pill">Trend: ${{fmt(b.Trend)}}</span><span class="pill">Num: ${{fmt(b.Num)}}</span></div>
      <p class="small">${{fmt(b.Text)}}</p>
    </div>`).join('');
}}
function renderTable(r) {{
  const cols = r.schema.columns || [];
  const head = '<tr>' + cols.map(c => `<th>${{c}}</th>`).join('') + '</tr>';
  const body = r.table.map((row, ri) => '<tr>' + cols.map((c, ci) => `<td class="${{isBoundCell(ri, ci) ? 'bound-cell' : ''}}">${{fmt(row[c])}}</td>`).join('') + '</tr>').join('');
  document.getElementById('table').innerHTML = `<table><thead>${{head}}</thead><tbody>${{body}}</tbody></table>`;
}}
function renderChart(r) {{
  const svg = document.getElementById('chart'); const rows = r.table; const metrics = r.schema.metric_columns || []; const xCol = r.schema.x_column;
  const w = 900, h = 360, m = {{top:24,right:24,bottom:50,left:70}}, pw = w-m.left-m.right, ph = h-m.top-m.bottom;
  svg.setAttribute('viewBox', `0 0 ${{w}} ${{h}}`); svg.innerHTML = '';
  const values = []; metrics.forEach(k => rows.forEach(row => {{ const v = num(row[k]); if (v !== null) values.push(v); }}));
  if (!rows.length || !metrics.length || !values.length) {{ svg.innerHTML = '<text x="24" y="42" class="label">No chartable data.</text>'; return; }}
  const min = Math.min(...values), max = Math.max(...values), span = max-min || Math.max(1, Math.abs(max));
  const yMin = min - span*.08, yMax = max + span*.08;
  const x = i => m.left + (rows.length === 1 ? pw/2 : i/(rows.length-1)*pw);
  const y = v => m.top + (1 - (v-yMin)/(yMax-yMin))*ph;
  for (let i=0;i<=4;i++) {{ const yy=m.top+ph/4*i; svg.insertAdjacentHTML('beforeend', `<line class="gridline" x1="${{m.left}}" y1="${{yy}}" x2="${{w-m.right}}" y2="${{yy}}"></line>`); svg.insertAdjacentHTML('beforeend', `<text class="label" x="${{m.left-8}}" y="${{yy+4}}" text-anchor="end">${{fmt(yMax-(yMax-yMin)/4*i)}}</text>`); }}
  svg.insertAdjacentHTML('beforeend', `<line class="axis" x1="${{m.left}}" y1="${{h-m.bottom}}" x2="${{w-m.right}}" y2="${{h-m.bottom}}"></line><line class="axis" x1="${{m.left}}" y1="${{m.top}}" x2="${{m.left}}" y2="${{h-m.bottom}}"></line>`);
  const b = binding();
  if (b && Array.isArray(b.Position)) b.Position.forEach(p => {{ const s=Math.max(0, Math.min(p.Begin[0], p.End[0])); const e=Math.min(rows.length-1, Math.max(p.Begin[0], p.End[0])); svg.insertAdjacentHTML('beforeend', `<rect class="band" x="${{x(s)-10}}" y="${{m.top}}" width="${{Math.max(20, x(e)-x(s)+20)}}" height="${{ph}}"></rect>`); }});
  metrics.forEach((k, mi) => {{ const pts = rows.map((row,i) => {{ const v=num(row[k]); return v===null ? null : `${{x(i)}},${{y(v)}}`; }}).filter(Boolean); if (pts.length > 1) svg.insertAdjacentHTML('beforeend', `<polyline class="line" points="${{pts.join(' ')}}" stroke="${{colors[mi%colors.length]}}"></polyline>`); rows.forEach((row,i) => {{ const v=num(row[k]); if (v!==null) svg.insertAdjacentHTML('beforeend', `<circle class="point" cx="${{x(i)}}" cy="${{y(v)}}" r="3.5" fill="${{colors[mi%colors.length]}}"></circle>`); }}); }});
  if (b && Array.isArray(b.Position)) b.Position.forEach(p => {{ for (let ri=Math.min(p.Begin[0],p.End[0]); ri<=Math.max(p.Begin[0],p.End[0]); ri++) {{ for (let ci=Math.min(p.Begin[1],p.End[1]); ci<=Math.max(p.Begin[1],p.End[1]); ci++) {{ const col = r.schema.columns[ci]; const v = num(rows[ri] && rows[ri][col]); if (v!==null) svg.insertAdjacentHTML('beforeend', `<circle class="bound" cx="${{x(ri)}}" cy="${{y(v)}}" r="6"></circle>`); }} }} }});
  const step = Math.max(1, Math.ceil(rows.length/7)); rows.forEach((row,i) => {{ if (i%step===0 || i===rows.length-1) svg.insertAdjacentHTML('beforeend', `<text class="label" x="${{x(i)}}" y="${{h-m.bottom+22}}" text-anchor="middle">${{fmt(row[xCol])}}</text>`); }});
}}
initCaseSelect();
render();
</script>
</body>
</html>
"""


def write_demo(args: argparse.Namespace) -> dict[str, Any]:
    cases = build_cases(args.source)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": utc_now(),
        "source_xlsx": str(args.source),
        "output_root": str(args.output_root),
        "folders": [],
    }
    for folder, model_label, route_id, route_type in MODEL_ROUTES:
        runtime_status = infer_runtime_status(route_id, route_type)
        comparison = compare_cases(cases)
        payload = {
            "folder": folder,
            "model_label": model_label,
            "route_id": route_id,
            "route_type": route_type,
            "generated_at_utc": report["generated_at_utc"],
            "source_xlsx": str(args.source),
            "runtime_status": {
                "status": runtime_status.status,
                "reason": runtime_status.reason,
                "route_status": runtime_status.route_status,
                "missing": runtime_status.missing or [],
            },
            "comparison": comparison,
            "cases": cases,
        }
        out_dir = args.output_root / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(render_html(payload), encoding="utf-8")
        (out_dir / "payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["folders"].append(
            {
                "folder": folder,
                "model_label": model_label,
                "route_id": route_id,
                "route_type": route_type,
                "index_html": str(out_dir / "index.html"),
                "payload_json": str(out_dir / "payload.json"),
                "runtime_status": payload["runtime_status"],
                "comparison": comparison,
            }
        )
    (args.output_root / "rerun_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(write_demo(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
