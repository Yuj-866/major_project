import { useEffect, useRef, useState } from "react";
import { Network } from "vis-network";
import { DataSet } from "vis-data";
import { get, usePoll, riskLevel, LEVEL_COLOR, NO_DATA, DOWN } from "../risk/riskApi";

const NEW_EDGE = "#d32f2f";

export default function AttackGraph() {
  const box = useRef(null);
  const nodes = useRef(new DataSet());
  const edges = useRef(new DataSet());
  const net = useRef(null);
  const newIds = useRef([]);
  const [info, setInfo] = useState(null);
  const [msg, setMsg] = useState(null);

  // create the network once; later refreshes only update the data sets (no flicker)
  useEffect(() => {
    net.current = new Network(
      box.current,
      { nodes: nodes.current, edges: edges.current },
      {
        layout: { hierarchical: { enabled: true, direction: "LR", sortMethod: "directed" } },
        interaction: { dragNodes: true, zoomView: true, hover: true },
        physics: { enabled: false },
      }
    );
    let big = false;
    const pulse = setInterval(() => {
      big = !big;
      if (newIds.current.length) edges.current.update(newIds.current.map((id) => ({ id, width: big ? 8 : 4 })));
    }, 600);
    return () => {
      clearInterval(pulse);
      if (net.current) net.current.destroy();
      net.current = null;
    };
  }, []);

  usePoll(async () => {
    try {
      const g = await get("/graph/latest");
      nodes.current.update(
        (g.nodes || []).map((n) => {
          const lvl = riskLevel(n.risk);
          return {
            id: n.id,
            label: n.id,
            shape: "box",
            color: LEVEL_COLOR[lvl],
            title: `${n.id}<br/>risk ${Number(n.risk).toFixed(2)} (${lvl})`,
          };
        })
      );
      const list = (g.edges || []).map((e) => {
        const id = `${e.source}->${e.target}`;
        const isNew = !!e.is_new;
        return {
          id,
          from: e.source,
          to: e.target,
          arrows: "to",
          color: isNew ? NEW_EDGE : "#848484",
          width: isNew ? 4 : Math.min(1 + (e.weight || 1) / 20, 6),
          title: `${id} w=${e.weight || 1}${isNew ? " NEW CONNECTION" : ""}`,
        };
      });
      const keep = new Set(list.map((x) => x.id));
      const stale = edges.current.getIds().filter((id) => !keep.has(id));
      if (stale.length) edges.current.remove(stale);
      edges.current.update(list);
      newIds.current = list.filter((x) => x.color === NEW_EDGE).map((x) => x.id);
      setInfo({ cls: g.risk_class, end: g.window_end, fresh: newIds.current.length });
      setMsg(null);
    } catch (e) {
      setInfo(null);
      setMsg(e.response ? NO_DATA : DOWN);
    }
  }, 3000);

  return (
    <div>
      <h3>
        Attack graph
        {info && (
          <span style={{ fontWeight: 400, fontSize: 14 }}>
            {" "}
            — window ending {info.end}, class <b>{info.cls}</b>, {info.fresh} new connection(s)
          </span>
        )}
      </h3>
      {msg && <div>{msg}</div>}
      <div ref={box} style={{ height: 520, border: "1px solid #ccc" }} />
      <div style={{ fontSize: 13, marginTop: 6 }}>
        Node colour = model risk (green low, amber medium, red high). Red pulsing edges = connections that are not part of
        the normal topology. Refreshes every 3 s.
      </div>
    </div>
  );
}