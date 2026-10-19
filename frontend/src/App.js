import { useState } from "react";
import Summary from "./components/Summary";
import RiskChart from "./components/RiskChart";
import AttackGraph from "./components/AttackGraph";
import PathsPanel from "./components/PathsPanel";
import RiskDashboard from "./risk/RiskDashboard";
import "./risk/risk.css";

const TABS = [["live", "Live risk"], ["summary", "Summary"], ["risk", "Risk chart"], ["graph", "Attack graph"], ["paths", "Critical paths"]];

function App() {
  const [tab, setTab] = useState("live");
  return (
    <div>
      <header className="app-top">
        <h1>Fintech risk dashboard</h1>
        <p>Spots attack activity across your cloud services and shows where it could spread.</p>
        <nav className="tabs">
          {TABS.map(([id, label]) => (
            <button key={id} className={"tab" + (tab === id ? " on" : "")} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
      </header>
      <main className="app-body">
        {tab === "live" && <RiskDashboard />}
        {tab !== "live" && (
          <div className="panel-box">
            {tab === "summary" && <Summary />}
            {tab === "risk" && <RiskChart />}
            {tab === "graph" && <AttackGraph />}
            {tab === "paths" && <PathsPanel />}
          </div>
        )}
      </main>
    </div>
  );
}
export default App;