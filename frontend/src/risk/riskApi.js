// frontend/src/risk/riskApi.js -- shared helpers for every tab that reads the new risk API (port 8001)
import { useEffect, useRef } from "react";
import axios from "axios";

export const API = process.env.REACT_APP_RISK_API || "http://localhost:8001";

export const get = (path, params) => axios.get(`${API}${path}`, { params }).then((r) => r.data);

// list endpoints may return {items: [...]} or a plain array
export const itemsOf = (d) => (Array.isArray(d) ? d : (d && d.items) || []);

// newest item by window_end (ISO strings sort correctly)
export const latestOf = (items) =>
  items.reduce((a, b) => (!a || String(b.window_end) >= String(a.window_end) ? b : a), null);

// display thresholds on node_risk (0..1); only used for colouring
export const riskLevel = (r) => (r >= 0.6 ? "HIGH" : r >= 0.3 ? "MEDIUM" : "LOW");
export const LEVEL_COLOR = { HIGH: "#ef9a9a", MEDIUM: "#ffe082", LOW: "#a5d6a7" };
export const LEVEL_STRONG = { HIGH: "#d32f2f", MEDIUM: "#f9a825", LOW: "#2e7d32" };

export const NO_DATA = "No data yet — open the Live risk tab and start the replay.";
export const DOWN = `Backend not reachable on ${API}`;

// call fn now and then every `ms` milliseconds
export function usePoll(fn, ms) {
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    let alive = true;
    const run = () => {
      if (alive) saved.current();
    };
    run();
    const t = setInterval(run, ms);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [ms]);
}