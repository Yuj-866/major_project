import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from "recharts";
import { get, usePoll, riskLevel, LEVEL_STRONG, NO_DATA, DOWN } from "../risk/riskApi";

export default function RiskChart() {
  const [data, setData] = useState([]);
  const [info, setInfo] = useState(null);
  const [msg, setMsg] = useState(null);

  usePoll(async () => {
    try {
      const g = await get("/graph/latest");
      const rows = (g.nodes || [])
        .map((n) => ({ node: n.id, risk: Number(n.risk) }))
        .sort((a, b) => b.risk - a.risk);
      setData(rows);
      setInfo({ cls: g.risk_class, end: g.window_end });
      setMsg(null);
    } catch (e) {
      setData([]);
      setInfo(null);
      setMsg(e.response ? NO_DATA : DOWN);
    }
  }, 3000);

  return (
    <div>
      <h3>
        Node risk — latest window
        {info && (
          <span style={{ fontWeight: 400, fontSize: 14 }}>
            {" "}
            (window ending {info.end}, class <b>{info.cls}</b>)
          </span>
        )}
      </h3>
      {msg && <div>{msg}</div>}
      <div style={{ height: 560 }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 130 }}>
            <XAxis type="number" domain={[0, 1]} />
            <YAxis type="category" dataKey="node" width={130} tick={{ fontSize: 11 }} interval={0} />
            <Tooltip formatter={(v) => Number(v).toFixed(3)} />
            <Bar dataKey="risk" isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.node} fill={LEVEL_STRONG[riskLevel(d.risk)]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ fontSize: 13, marginTop: 6 }}>
        Risk = model's node score scaled by how likely the window is not LOW, so quiet windows show low risk everywhere.
        Colour: green below 0.3, amber 0.3 to 0.6, red above 0.6. Refreshes every 3 s.
      </div>
    </div>
  );
}