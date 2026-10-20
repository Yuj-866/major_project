import { useEffect, useRef, useState } from "react";
import { Network } from "vis-network";
import { DataSet } from "vis-data";
import { get, itemsOf, latestOf, usePoll, LEVEL_COLOR, NO_DATA, DOWN } from "../risk/riskApi";

export default function PathsPanel() {
  const box = useRef(null);
  const [state, setState] = useState(null); // {end, cls, paths, alertEnd}
  const [sel, setSel] = useState(0);
  const [step, setStep] = useState(0);
  const [msg, setMsg] = useState(null);

  usePoll(async () => {
    try {
      const [g, alerts] = await Promise.all([
        get("/graph/latest"),
        get("/alerts", { min_level: "MEDIUM", limit: 20 }).catch(() => ({ items: [] })),
      ]);
      const la = latestOf(itemsOf(alerts));
      const paths = g.top_paths || (la && la.top_paths) || [];
      setState({ end: g.window_end, cls: g.risk_class, paths, alertEnd: la ? la.window_end : null });
      setMsg(null);
    } catch (e) {
      setState(null);
      setMsg(e.response ? NO_DATA : DOWN);
    }
  }, 3000);

  const paths = state ? state.paths : [];
  const path = paths[Math.min(sel, paths.length - 1)];
  const chain = path ? path.nodes : [];
  const chainKey = chain.join(">");

  // draw the selected candidate path and step through it hop by hop
  useEffect(() => {
    const el = box.current;
    if (!el || !chain.length) return undefined;
    const nodes = new DataSet(chain.map((n) => ({ id: n, label: n, shape: "box", color: LEVEL_COLOR.LOW })));
    const edges = new DataSet(
      chain.slice(1).map((n, i) => ({ id: i, from: chain[i], to: n, arrows: "to", color: "#848484", width: 2 }))
    );
    const network = new Network(
      el,
      { nodes, edges },
      {
        layout: { hierarchical: { enabled: true, direction: "LR" } },
        interaction: { dragNodes: true, zoomView: true, hover: true },
        physics: { enabled: false },
      }
    );
    let k = 0;
    const anim = setInterval(() => {
      const hi = k % chain.length; // current hop
      chain.forEach((n, j) =>
        nodes.update({ id: n, color: j < hi ? LEVEL_COLOR.HIGH : j === hi ? LEVEL_COLOR.MEDIUM : LEVEL_COLOR.LOW })
      );
      setStep(hi);
      k += 1;
    }, 900);
    return () => {
      clearInterval(anim);
      network.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chainKey]);

  return (
    <div>
      <h3>Critical attack-path candidates</h3>
      <div style={{ marginBottom: 8, padding: 8, background: "#fff8e1", border: "1px solid #ccc", fontSize: 13 }}>
        Heuristic ranking: directed connections starting from the 3 riskiest nodes, scored by mean node risk along the
        path. Not validated against real attack chains. The animation steps through the candidate path; it does not mean
        those hops happened.
      </div>
      {msg && <div>{msg}</div>}
      {state && (
        <div style={{ marginBottom: 8 }}>
          Current window ending {state.end}: <b>{state.cls}</b>
          {state.alertEnd && state.alertEnd !== state.end && (
            <span> — paths below are from the latest Medium-or-higher window ({state.alertEnd})</span>
          )}
        </div>
      )}
      {state && paths.length === 0 && <div>No path candidates yet (no elevated-risk window so far).</div>}
      {paths.length > 0 && (
        <ol style={{ paddingLeft: 20 }}>
          {paths.map((p, i) => (
            <li key={i} onClick={() => setSel(i)} style={{ cursor: "pointer", fontWeight: i === sel ? 700 : 400 }}>
              {p.nodes.join(" → ")} <span style={{ fontWeight: 400 }}>(score {Number(p.score).toFixed(2)})</span>
            </li>
          ))}
        </ol>
      )}
      <div ref={box} style={{ height: 320, border: "1px solid #ccc", display: chain.length ? "block" : "none" }} />
      {chain.length > 0 && (
        <div style={{ fontSize: 13, marginTop: 6 }}>
          Hop {step + 1} of {chain.length}. Green = later nodes, amber = current node, red = already passed. Click a path
          above to switch.
        </div>
      )}
    </div>
  );
}