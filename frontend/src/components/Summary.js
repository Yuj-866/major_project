import { useState } from "react";
import { get, itemsOf, latestOf, usePoll, LEVEL_STRONG, NO_DATA, DOWN } from "../risk/riskApi";

export default function Summary() {
  const [s, setS] = useState(null);
  const [err, setErr] = useState(null);

  usePoll(async () => {
    try {
      const [meta, hist, alerts] = await Promise.all([
        get("/meta"),
        get("/history", { limit: 500 }),
        get("/alerts", { min_level: "MEDIUM", limit: 50 }).catch(() => ({ items: [] })),
      ]);
      const h = itemsOf(hist);
      const count = (c) => h.filter((x) => x.risk_class === c).length;
      const last = latestOf(h);
      const la = latestOf(itemsOf(alerts));
      setS({
        nodes: (meta.node_order || []).length,
        scored: h.length,
        high: count("HIGH"),
        medium: count("MEDIUM"),
        low: count("LOW"),
        now: last ? last.risk_class : "—",
        paths: la && la.top_paths ? la.top_paths.length : 0,
        model: meta.model ? meta.model.name : "—",
      });
      setErr(h.length === 0 ? NO_DATA : null);
    } catch (e) {
      setS(null);
      setErr(e.response ? NO_DATA : DOWN);
    }
  }, 3000);

  if (!s) return <div>{err || "Loading summary..."}</div>;
  const cards = [
    ["Nodes", s.nodes, null],
    ["Windows scored", s.scored, null],
    ["High", s.high, LEVEL_STRONG.HIGH],
    ["Medium", s.medium, LEVEL_STRONG.MEDIUM],
    ["Low", s.low, LEVEL_STRONG.LOW],
    ["Now", s.now, LEVEL_STRONG[s.now] || null],
    ["Path candidates", s.paths, null],
  ];
  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        {cards.map(([k, v, c]) => (
          <div key={k} style={{ border: "1px solid #ccc", padding: 12, minWidth: 110 }}>
            <div>{k}</div>
            <b style={c ? { color: c } : undefined}>{v}</b>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 13, marginTop: 10 }}>
        {err && <div>{err}</div>}
        Model: {s.model}. High / Medium / Low count the windows replayed so far. Path candidates come from the latest
        Medium-or-higher window (heuristic, not validated).
      </div>
    </div>
  );
}