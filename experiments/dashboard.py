#!/usr/bin/env python3
"""回测结果展示面板。

用法:
  python experiments/dashboard.py
  python experiments/dashboard.py --port 8080
"""
import argparse
import json
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import yaml

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "outputs" / "results"

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Quant Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f1117;color:#e1e4e8;min-height:100vh}
header{background:#161b22;border-bottom:1px solid #30363d;padding:16px 24px;position:sticky;top:0;z-index:10}
header h1{font-size:20px;color:#58a6ff}
.container{max-width:1400px;margin:0 auto;padding:24px}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:14px}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid #21262d}
th{background:#161b22;color:#8b949e;font-weight:600;position:sticky;top:57px;z-index:5}
tr:hover{background:#161b22}
.positive{color:#3fb950}.negative{color:#f85149}.neutral{color:#8b949e}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:16px 0}
.card h2{font-size:16px;color:#58a6ff;margin-bottom:12px}
.chart-wrap{position:relative;height:350px;margin:16px 0}
.chart-wrap canvas{width:100%!important}
.flex{display:flex;gap:16px;flex-wrap:wrap}
.flex>*{flex:1;min-width:300px}
.metric{text-align:center}
.metric .val{font-size:28px;font-weight:700}
.metric .lbl{font-size:12px;color:#8b949e;margin-top:4px}
button,.btn{background:#238636;color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:14px;text-decoration:none;display:inline-block}
button:hover{background:#2ea043}
#compare-btn{background:#1f6feb}#compare-btn:hover{background:#388bfd}
.compare-bar{position:fixed;bottom:0;left:0;right:0;background:#161b22;border-top:1px solid #30363d;padding:12px 24px;display:none;z-index:20}
.compare-bar.active{display:flex;align-items:center;gap:12px}
.compare-bar .chips{display:flex;gap:8px;flex-wrap:wrap;flex:1}
.chip{background:#1f6feb;color:#fff;padding:4px 10px;border-radius:12px;font-size:12px;display:flex;align-items:center;gap:6px}
.chip .x{cursor:pointer;opacity:.7}.chip .x:hover{opacity:1}
a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
.compact-table{font-size:12px}
.compact-table td,.compact-table th{padding:6px 10px}
details summary{cursor:pointer;color:#58a6ff;margin:8px 0}
pre{background:#0d1117;padding:12px;border-radius:6px;overflow-x:auto;font-size:12px}
</style>
</head>
<body>
<header><h1>Quant Dashboard</h1></header>
<div class="container">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
    <span id="status">加载中...</span>
    <button id="compare-btn" onclick="toggleCompare()">对比模式</button>
  </div>
  <div class="flex" id="metrics-bar"></div>
  <div class="card">
    <h2>实验列表</h2>
    <table id="exp-table"><thead><tr>
      <th></th><th>实验</th><th>时间</th><th>总收益</th><th>年化收益</th><th>Sharpe</th><th>最大回撤</th><th>胜率</th><th>交易数</th><th>Rank IC 5d</th>
    </tr></thead><tbody></tbody></table>
  </div>
  <div id="detail" style="display:none">
    <div class="flex" id="detail-metrics"></div>
    <div class="card"><h2>净值曲线</h2><div class="chart-wrap"><canvas id="equity-chart"></canvas></div></div>
    <div class="card"><h2>回撤</h2><div class="chart-wrap"><canvas id="dd-chart"></canvas></div></div>
    <div class="card"><h2>实验配置</h2><pre id="config-text"></pre></div>
  </div>
  <div id="compare-view" style="display:none">
    <div class="card"><h2>净值对比</h2><div class="chart-wrap"><canvas id="compare-chart"></canvas></div></div>
    <div class="card"><h2>指标对比</h2><table class="compact-table" id="compare-table"><thead></thead><tbody></tbody></table></div>
  </div>
</div>
<div class="compare-bar" id="compare-bar"><span>对比 (<span id="compare-count">0</span>)</span><div class="chips" id="compare-chips"></div><button onclick="showCompare()">对比</button><button onclick="clearCompare()" style="background:#30363d">清空</button></div>
<script>
let data={},selected=null,compareList=[],charts={};
async function load(){let r=await fetch('/api/experiments');data=await r.json();renderTable();renderOverview();document.getElementById('status').textContent=data.length+' 个实验'}
function fmt(n,dec=2){if(n==null||isNaN(n))return'-';let s=n.toFixed(dec);return n>=0?'+'+s:s}
function pct(n){if(n==null||isNaN(n))return'-';return (n*100).toFixed(2)+'%'}
function dateFmt(s){if(!s)return'';let d=s.replace('T',' ').split('.')[0];return d.length>16?d.substring(0,16):d}
function renderOverview(){
  let all=data.filter(d=>d.backtest&&d.backtest.metrics);
  if(!all.length)return;
  let avg=all.reduce((a,d)=>{let m=d.backtest.metrics;a.sharpe+=m.sharpe_ratio||0;a.dd+=m.max_drawdown||0;a.ret+=m.total_return||0;a.n++;return a},{sharpe:0,dd:0,ret:0,n:0});
  let html='';
  html+=`<div class="card metric"><div class="val">${all.length}</div><div class="lbl">实验总数</div></div>`;
  html+=`<div class="card metric"><div class="val ${avg.ret/avg.n>=0?'positive':'negative'}">${pct(avg.ret/avg.n)}</div><div class="lbl">平均总收益</div></div>`;
  html+=`<div class="card metric"><div class="val">${(avg.sharpe/avg.n).toFixed(2)}</div><div class="lbl">平均 Sharpe</div></div>`;
  html+=`<div class="card metric"><div class="val negative">${pct(avg.dd/avg.n)}</div><div class="lbl">平均最大回撤</div></div>`;
  document.getElementById('metrics-bar').innerHTML=html;
}
function renderTable(){
  let tbody=document.querySelector('#exp-table tbody');
  tbody.innerHTML=data.map((d,i)=>{
    let m=d.backtest?.metrics||{};
    let ev=d.eval_metrics||{};
    return `<tr>
      <td><input type="checkbox" onchange="toggleCmp('${d.name}','${d.ts}')" ${compareList.some(c=>c.name==d.name&&c.ts==d.ts)?'checked':''}></td>
      <td><a href="javascript:selectExp('${d.name}','${d.ts}')">${d.name}</a></td>
      <td>${dateFmt(d.ts)}</td>
      <td class="${m.total_return>=0?'positive':'negative'}">${fmt(m.total_return?m.total_return*100:null)}%</td>
      <td class="${m.annualized_return>=0?'positive':'negative'}">${fmt(m.annualized_return?m.annualized_return*100:null)}%</td>
      <td>${fmt(m.sharpe_ratio)}</td>
      <td class="negative">${fmt(m.max_drawdown?m.max_drawdown*100:null)}%</td>
      <td>${fmt(m.win_rate?m.win_rate*100:null)}%</td>
      <td>${m.n_trades||'-'}</td>
      <td>${fmt(ev.rank_ic_5d)}</td>
    </tr>`;
  }).join('');
}
async function selectExp(name,ts){
  let r=await fetch(`/api/experiments/${name}/${ts}`);
  let d=await r.json();selected=d;
  document.getElementById('detail').style.display='block';
  document.getElementById('compare-view').style.display='none';
  let m=d.backtest?.metrics||{},ev=d.eval_metrics||{};
  document.getElementById('detail-metrics').innerHTML=`
    <div class="card metric"><div class="val ${m.total_return>=0?'positive':'negative'}">${pct(m.total_return)}</div><div class="lbl">总收益</div></div>
    <div class="card metric"><div class="val">${m.sharpe_ratio?.toFixed(2)||'-'}</div><div class="lbl">Sharpe</div></div>
    <div class="card metric"><div class="val negative">${pct(m.max_drawdown)}</div><div class="lbl">最大回撤</div></div>
    <div class="card metric"><div class="val">${pct(m.win_rate)}</div><div class="lbl">胜率</div></div>
    <div class="card metric"><div class="val">${ev.rank_ic_5d?.toFixed(4)||'-'}</div><div class="lbl">Rank IC 5d</div></div>
    <div class="card metric"><div class="val">${ev.rank_ic_score?.toFixed(4)||'-'}</div><div class="lbl">Rank IC Score</div></div>`;
  document.getElementById('config-text').textContent=d.config_yaml||'';
  drawEquity(d);drawDD(d);
}
function drawEquity(d){
  let eq=d.backtest?.equity_curve||[];
  if(!eq.length)return;
  if(charts.equity)charts.equity.destroy();
  let ctx=document.getElementById('equity-chart').getContext('2d');
  charts.equity=new Chart(ctx,{type:'line',data:{labels:eq.map(e=>e.date),datasets:[{label:d.name,data:eq.map(e=>e.value),borderColor:'#58a6ff',borderWidth:1.5,pointRadius:0,fill:false}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:20}},y:{ticks:{color:'#8b949e',callback:v=>v.toLocaleString()}}}}});
}
function drawDD(d){
  let eq=d.backtest?.equity_curve||[];
  if(!eq.length)return;
  let peak=0,dd=eq.map(e=>{peak=Math.max(peak,e.value);return{date:e.date,dd:-(peak-e.value)/peak*100}});
  if(charts.dd)charts.dd.destroy();
  let ctx=document.getElementById('dd-chart').getContext('2d');
  charts.dd=new Chart(ctx,{type:'line',data:{labels:dd.map(d=>d.date),datasets:[{data:dd.map(d=>d.dd),borderColor:'#f85149',borderWidth:1,pointRadius:0,fill:true,backgroundColor:'rgba(248,81,73,0.1)'}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:20}},y:{ticks:{color:'#8b949e',callback:v=>v+'%'}}}}});
}
function toggleCmp(name,ts){
  let idx=compareList.findIndex(c=>c.name==name&&c.ts==ts);
  if(idx>=0)compareList.splice(idx,1);else compareList.push({name,ts});
  updateCompareBar();
}
function updateCompareBar(){
  let bar=document.getElementById('compare-bar');
  bar.classList.toggle('active',compareList.length>0);
  document.getElementById('compare-count').textContent=compareList.length;
  document.getElementById('compare-chips').innerHTML=compareList.map(c=>`<span class="chip">${c.name}<span class="x" onclick="toggleCmp('${c.name}','${c.ts}')">&times;</span></span>`).join('');
}
async function showCompare(){
  if(compareList.length<2)return;
  document.getElementById('detail').style.display='none';
  document.getElementById('compare-view').style.display='block';
  let all=[];
  for(let c of compareList){let r=await fetch(`/api/experiments/${c.name}/${c.ts}`);all.push(await r.json())}
  drawCompareChart(all);
  drawCompareTable(all);
}
function drawCompareChart(all){
  if(charts.compare)charts.compare.destroy();
  let colors=['#58a6ff','#3fb950','#f85149','#d29922','#bc8cff','#79c0ff'];
  let datasets=all.map((d,i)=>{let eq=d.backtest?.equity_curve||[];return{label:d.name,data:eq.map(e=>e.value),borderColor:colors[i%colors.length],borderWidth:1.5,pointRadius:0,fill:false}});
  let labels=all[0]?.backtest?.equity_curve?.map(e=>e.date)||[];
  let ctx=document.getElementById('compare-chart').getContext('2d');
  charts.compare=new Chart(ctx,{type:'line',data:{labels,datasets},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#e1e4e8',usePointStyle:true}}},scales:{x:{ticks:{color:'#8b949e',maxTicksLimit:20}},y:{ticks:{color:'#8b949e',callback:v=>v.toLocaleString()}}}}});
}
function drawCompareTable(all){
  let th=document.querySelector('#compare-table thead');
  let tb=document.querySelector('#compare-table tbody');
  th.innerHTML='<tr><th>指标</th>'+all.map(d=>`<th>${d.name}</th>`).join('')+'</tr>';
  let rows=[];
  ['total_return','annualized_return','sharpe_ratio','max_drawdown','win_rate','n_trades'].forEach(k=>{
    let vals=all.map(d=>d.backtest?.metrics?.[k]);
    rows.push(`<tr><td>${k}</td>${vals.map(v=>`<td>${typeof v==='number'?(k.includes('return')||k.includes('drawdown')||k.includes('win')?pct(v):fmt(v)):v||'-'}</td>`).join('')}</tr>`);
  });
  ['rank_ic_5d','rank_ic_score'].forEach(k=>{
    let vals=all.map(d=>d.eval_metrics?.[k]);
    rows.push(`<tr><td>${k}</td>${vals.map(v=>`<td>${fmt(v,4)}</td>`).join('')}</tr>`);
  });
  tb.innerHTML=rows.join('');
}
function toggleCompare(){let bar=document.getElementById('compare-bar');if(bar.classList.contains('active'))clearCompare();else bar.classList.add('active')}
function clearCompare(){compareList=[];updateCompareBar();document.getElementById('compare-view').style.display='none';renderTable()}
load();
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)

        if p.path == "/":
            self._send_html(HTML)
        elif p.path == "/api/experiments":
            self._api_list_experiments()
        elif p.path.startswith("/api/experiments/"):
            parts = p.path[len("/api/experiments/"):].strip("/").split("/")
            if len(parts) == 2:
                self._api_get_experiment(parts[0], parts[1])
            else:
                self._send_json({"error": "invalid path"}, 400)
        else:
            super().do_GET()

    def _send_html(self, content, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(content.encode())

    def _send_json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str, ensure_ascii=False).encode())

    def _api_list_experiments(self):
        if not RESULTS.exists():
            return self._send_json([])

        experiments = []
        for exp_dir in sorted(RESULTS.iterdir(), reverse=True):
            if not exp_dir.is_dir():
                continue
            for run_dir in sorted(exp_dir.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                exp = {"name": exp_dir.name, "ts": run_dir.name}

                # read backtest metrics
                bt_path = run_dir / "backtest_metrics.json"
                if bt_path.exists():
                    exp["backtest"] = json.loads(bt_path.read_text())
                    # keep equity_curve for detail, remove from list view
                    if "equity_curve" in exp.get("backtest", {}):
                        exp["backtest"] = {k: v for k, v in exp["backtest"].items()
                                           if k != "equity_curve"}
                    exp["backtest"]["metrics"] = {k: v for k, v in exp["backtest"].items()}

                # read eval metrics
                eval_path = run_dir / "metrics.json"
                if eval_path.exists():
                    exp["eval_metrics"] = json.loads(eval_path.read_text())

                experiments.append(exp)
        self._send_json(experiments)

    def _api_get_experiment(self, name, ts):
        run_dir = RESULTS / name / ts
        if not run_dir.exists():
            return self._send_json({"error": "not found"}, 404)

        exp = {"name": name, "ts": ts}

        for fname in ["backtest_metrics.json", "metrics.json"]:
            fpath = run_dir / fname
            if fpath.exists():
                key = "eval_metrics" if fname == "metrics.json" else "backtest"
                exp[key] = json.loads(fpath.read_text())

        # read config yaml
        cfg_path = run_dir / "config.yaml"
        if cfg_path.exists():
            exp["config_yaml"] = cfg_path.read_text()

        # read equity curve
        eq_path = run_dir / "equity_curve.csv"
        if eq_path.exists():
            import pandas as pd
            df = pd.read_csv(eq_path)
            exp.setdefault("backtest", {})["equity_curve"] = [
                {"date": str(r["date"]), "value": float(r["value"])}
                for _, r in df.iterrows()
            ]

        self._send_json(exp)

    def log_message(self, format, *args):
        pass  # suppress logs


def main():
    parser = argparse.ArgumentParser(description="Quant Dashboard")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"\n  Dashboard: http://localhost:{args.port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
        server.server_close()


if __name__ == "__main__":
    main()
