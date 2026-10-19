import React, { useEffect, useRef, useState, useCallback } from "react";
import axios from "axios";
import { Network, DataSet } from "vis-network/standalone";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

// Backend URL lives in ONE place. Override with REACT_APP_RISK_API in frontend/.env
const API = process.env.REACT_APP_RISK_API || "http://localhost:8001";
const POLL_MS = 2000;

const riskColor = (r) => {
  const hue = 120 - 120 * Math.min(Math.max(r, 0), 1); // green -> red
  return `hsl(${hue}, 70%, 45%)`;
};
const classColor = { LOW: "#177a3c", MEDIUM: "#b36b00", HIGH: "#c62828" };
const MSG = {
  LOW: ["All quiet", "No attack activity detected in this time window."],
  MEDIUM: ["Suspicious activity", "Unusual traffic seen. Keep an eye on the highlighted services."],
  HIGH: ["Likely active attack", "Check the highlighted services first, they are the most exposed."],
};

export default function RiskDashboard() {
  const [meta, setMeta] = useState(null);
  const [graph, setGraph] = useState(null);
  const [history, setHistory] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [replay, setReplay] = useState({ running: false, sent: 0, total: 0 });
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);

  const holder = useRef(null);
  const net = useRef(null);
  const fitted = useRef(false);
  const nodesDS = useRef(new DataSet());
  const edgesDS = useRef(new DataSet());

  // ---- polling ----
  const refresh = useCallback(async () => {
    try {
      const [h, a, r] = await Promise.all([
        axios.get(`${API}/history`, { params: { limit: 60 } }),
        axios.get(`${API}/alerts`, { params: { min_level: "MEDIUM", limit: 20 } }),
        axios.get(`${API}/replay/status`),
      ]);
      setHistory(h.data.items);
      setAlerts([...a.data.items].reverse());
      setReplay(r.data);
      setError(null);
      try {
        const g = await axios.get(`${API}/graph/latest`);
        setGraph(g.data);
      } catch (e) { /* 400 = no data yet */ }
    } catch (e) {
      setError(`Cannot reach backend at ${API}`);
    }
  }, []);

  useEffect(() => {
    axios.get(`${API}/meta`).then((r) => setMeta(r.data)).catch(() => {});
    refresh();
    const t = setInterval(refresh, POLL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  // ---- graph drawing ----
  useEffect(() => {
    if (!holder.current || net.current) return;
    net.current = new Network(
      holder.current,
      { nodes: nodesDS.current, edges: edgesDS.current },
      {
        physics: { stabilization: { iterations: 150 }, barnesHut: { springLength: 140 } },
        nodes: { shape: "dot", size: 16, font: { size: 11 } },
        edges: { arrows: "to", smooth: false },
        interaction: { hover: true },
      }
    );
    net.current.on("click", (p) => setSelected(p.nodes[0] || null));
    return () => { net.current && net.current.destroy(); net.current = null; };
  }, []);

  useEffect(() => {
    if (!graph) return;
    // When overall risk is LOW the node ranking isn't meaningful: draw all nodes the same.
    const low = graph.risk_class === "LOW";
    nodesDS.current.update(graph.nodes.map((n) => ({
      id: n.id, label: n.id,
      color: low ? "#9aa0a6" : riskColor(n.risk),
      size: low ? 14 : 12 + n.risk * 22,
      title: low ? `${n.id} (${n.type})` : `${n.id}\nrisk ${n.risk.toFixed(2)} (${n.type})`,
    })));
    const wanted = new Set();
    const edges = graph.edges.map((e) => {
      const id = `${e.source}>${e.target}>${e.type}`;
      wanted.add(id);
      return {
        id, from: e.source, to: e.target, width: 1 + Math.min(e.weight, 6) / 2,
        color: e.is_new ? "#c62828" : "#9aa8b8", dashes: e.is_new,
      };
    });
    edgesDS.current.update(edges);
    edgesDS.current.remove(edgesDS.current.getIds().filter((i) => !wanted.has(i)));
    if (!fitted.current) {  // once, after the layout settles, fit the whole graph in the box
      fitted.current = true;
      setTimeout(() => net.current && net.current.fit(), 1500);
    }
  }, [graph]);

  // ---- actions ----
  const post = (path, body) => axios.post(`${API}${path}`, body).then(refresh)
    .catch((e) => setError(e.response?.data?.error?.message || e.message));

  // ---- derived ----
  const latest = history[history.length - 1];
  const chartData = history.map((h) => ({
    t: (h.window_end || "").slice(11, 19), ...h.probabilities,
  }));
  const node = graph && selected ? graph.nodes.find((n) => n.id === selected) : null;
  const nodeEdges = graph && selected
    ? graph.edges.filter((e) => e.source === selected || e.target === selected) : [];
  const nodePaths = alerts.length
    ? (alerts[0].top_paths || []).filter((p) => p.nodes.includes(selected)) : [];
  const isLow = graph && graph.risk_class === "LOW";
  const isMock = meta && /mock/i.test(meta.model.name);

  const top = graph && !isLow ? [...graph.nodes].sort((a, b) => b.risk - a.risk)[0] : null;
  const lvl = latest ? latest.risk_class : "NONE";
  const pct = replay.total ? Math.round((100 * replay.sent) / replay.total) : 0;

  return (
    <div>
      {error && <div className="err"><b>Can't reach the backend.</b> {error}. Start it with <code>python -m uvicorn backend_api.main:app --port 8001</code>.</div>}

      <div className={"banner " + lvl}>
        <div>
          <div className="lvl">{latest ? latest.risk_class : "No data"}</div>
          {isMock && <div className="sub">Demo scores (mock model)</div>}
        </div>
        <div className="grow">
          <div className="msg">{latest ? MSG[lvl][0] : "Nothing to show yet"}</div>
          <div className="sub">{latest ? MSG[lvl][1] : "Press Start replay to play recorded traffic through the model."}</div>
          {top && <div className="sub">Most exposed service: <b>{top.id}</b> (risk {top.risk.toFixed(2)})</div>}
        </div>
        {latest && (
          <div className="probs">
            {["LOW", "MEDIUM", "HIGH"].map((k) => (
              <div className="prow" key={k}>
                <span>{k[0] + k.slice(1).toLowerCase()}</span>
                <div className="bar"><i style={{ width: `${(latest.probabilities[k] || 0) * 100}%`, background: classColor[k] }} /></div>
                <span>{Math.round((latest.probabilities[k] || 0) * 100)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="ctrl">
        <button className="btn go" onClick={() => post("/replay/start", { speed: 150 })} disabled={replay.running}>Start replay</button>
        <button className="btn" onClick={() => post("/replay/stop")} disabled={!replay.running}>Stop</button>
        <div className="prog">
          <div className="bar"><i style={{ width: pct + "%" }} /></div>
          <small>{replay.running ? "Playing" : "Stopped"}: {replay.sent} of {replay.total} time windows</small>
        </div>
        <span className="hint">Replay plays saved traffic as if it were live.</span>
      </div>

      <div className="grid">
        <div className="panel-box">
          <div className="h">
            <b>Service map</b><span className="hint">Click a service to see details</span>
            <button className="btn" onClick={() => net.current && net.current.fit()}>Fit view</button>
          </div>
          <div ref={holder} style={{ height: 440 }} />
          {!graph && <div className="empty">The map appears once the replay starts.</div>}
          <div className="legend">
            <span>Lower risk</span><span className="ramp" /><span>Higher risk</span>
            <span><span className="dash" /> new connection (possible attack path)</span>
            {isLow && <span>Colours stay grey while overall risk is low.</span>}
          </div>
        </div>

        <div className="panel-box node">
          <div className="h"><b>Service details</b></div>
          {!node ? <div className="empty">Click a service on the map, or a row in the alerts table.</div> : (
            <div>
              <h3>{node.id}</h3>
              <div className="chip">{node.type}</div>
              {isLow
                ? <div className="hint">Risk ranking is hidden while overall risk is low.</div>
                : <div>Risk score <b style={{ color: riskColor(node.risk) }}>{node.risk.toFixed(2)}</b> out of 1</div>}
              <h4>Connected to ({nodeEdges.length})</h4>
              <ul>{nodeEdges.map((e, i) => (
                <li key={i} className={e.is_new ? "new" : ""}>{e.source} → {e.target}{e.is_new ? " (new)" : ""}</li>))}</ul>
              <h4>Possible attack routes through it <span className="hint">(estimate, not validated)</span></h4>
              {nodePaths.length === 0 ? <div className="hint">None in the latest alert.</div> : (
                <ul>{nodePaths.map((p, i) => <li key={i}>{p.nodes.join(" → ")}</li>)}</ul>)}
            </div>
          )}
        </div>

        <div className="panel-box">
          <div className="h"><b>Risk over time</b><span className="hint">How likely each risk level is</span></div>
          {chartData.length === 0 ? <div className="empty">No history yet.</div> : (
            <div style={{ height: 230 }}>
              <ResponsiveContainer>
                <AreaChart data={chartData}>
                  <XAxis dataKey="t" tick={{ fontSize: 11 }} /><YAxis domain={[0, 1]} tick={{ fontSize: 11 }} /><Tooltip />
                  <Area type="monotone" dataKey="LOW" stackId="1" stroke={classColor.LOW} fill={classColor.LOW} fillOpacity={0.35} isAnimationActive={false} />
                  <Area type="monotone" dataKey="MEDIUM" stackId="1" stroke={classColor.MEDIUM} fill={classColor.MEDIUM} fillOpacity={0.45} isAnimationActive={false} />
                  <Area type="monotone" dataKey="HIGH" stackId="1" stroke={classColor.HIGH} fill={classColor.HIGH} fillOpacity={0.55} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>)}
        </div>

        <div className="panel-box">
          <div className="h"><b>Alerts</b><span className="hint">Medium and high, newest first</span></div>
          <div className="scroll">
            {alerts.length === 0 ? <div className="empty">No alerts so far.</div> : (
              <table>
                <thead><tr><th>Time</th><th>Level</th><th>Most exposed</th></tr></thead>
                <tbody>{alerts.map((a, i) => (
                  <tr className="row" key={i} onClick={() => { setSelected(a.top_node); net.current && net.current.selectNodes([a.top_node]); }}>
                    <td>{(a.window_end || "").slice(0, 19).replace("T", " ")}</td>
                    <td><span className={"pill " + a.risk_class}>{a.risk_class}</span></td>
                    <td>{a.top_node}</td>
                  </tr>))}</tbody>
              </table>)}
          </div>
        </div>
      </div>
    </div>
  );
}