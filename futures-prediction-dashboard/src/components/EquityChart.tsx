"use client";

import React, { useMemo, useState } from "react";
import { EquityPoint } from "@/types";
import { TrendingUp, Maximize2 } from "lucide-react";

interface EquityChartProps {
  data: EquityPoint[];
  startingBalance: number;
}

export const EquityChart: React.FC<EquityChartProps> = ({ data, startingBalance }) => {
  const [hoveredPoint, setHoveredPoint] = useState<EquityPoint | null>(null);

  const { points, minEq, maxEq, pathD, areaD, btcPathD } = useMemo(() => {
    if (!data || data.length < 2) {
      return { points: [], minEq: startingBalance * 0.98, maxEq: startingBalance * 1.02, pathD: "", areaD: "", btcPathD: "" };
    }

    const equities = data.map((d) => d.equity);
    const btcPrices = data.map((d) => d.btc_price);

    let min = Math.min(...equities, startingBalance);
    let max = Math.max(...equities, startingBalance);
    if (min === max) {
      min -= 100;
      max += 100;
    }
    const margin = (max - min) * 0.08;
    min -= margin;
    max += margin;

    const minBtc = Math.min(...btcPrices);
    const maxBtc = Math.max(...btcPrices);
    const btcRange = maxBtc - minBtc || 1;

    const width = 800;
    const height = 240;

    const pts = data.map((d, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((d.equity - min) / (max - min)) * height;
      const btcY = height - ((d.btc_price - minBtc) / btcRange) * height;
      return { x, y, btcY, point: d };
    });

    // SVG path string
    const pD = pts.reduce((acc, curr, i) => `${acc} ${i === 0 ? "M" : "L"} ${curr.x.toFixed(1)} ${curr.y.toFixed(1)}`, "");
    const aD = `${pD} L ${width} ${height} L 0 ${height} Z`;
    const bD = pts.reduce((acc, curr, i) => `${acc} ${i === 0 ? "M" : "L"} ${curr.x.toFixed(1)} ${curr.btcY.toFixed(1)}`, "");

    return { points: pts, minEq: min, maxEq: max, pathD: pD, areaD: aD, btcPathD: bD };
  }, [data, startingBalance]);

  const latestEquity = data.length > 0 ? data[data.length - 1].equity : startingBalance;
  const isProfit = latestEquity >= startingBalance;

  return (
    <div className="p-4 rounded-xl bg-zinc-900/80 border border-zinc-800 flex flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center space-x-2">
          <TrendingUp className="w-4 h-4 text-emerald-400" />
          <h3 className="text-xs uppercase font-mono text-zinc-300 font-bold">
            Simulated Portfolio Equity Curve
          </h3>
          <span className="text-[10px] font-mono text-zinc-500">
            ({data.length} Bars)
          </span>
        </div>

        <div className="flex items-center space-x-4 text-xs font-mono">
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-0.5 bg-emerald-400 rounded-full inline-block" />
            <span className="text-zinc-300">Portfolio Equity</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-0.5 bg-sky-400/60 rounded-full inline-block" />
            <span className="text-zinc-400">BTC Benchmark</span>
          </div>
        </div>
      </div>

      {/* SVG Chart */}
      <div className="w-full relative h-[240px] select-none">
        {data.length < 2 ? (
          <div className="flex items-center justify-center h-full text-zinc-500 font-mono text-xs">
            Accumulating live 10-second tick history...
          </div>
        ) : (
          <svg
            viewBox="0 0 800 240"
            className="w-full h-full overflow-visible"
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            {/* Grid Lines */}
            <line x1="0" y1="60" x2="800" y2="60" stroke="#27272a" strokeDasharray="3 3" />
            <line x1="0" y1="120" x2="800" y2="120" stroke="#27272a" strokeDasharray="3 3" />
            <line x1="0" y1="180" x2="800" y2="180" stroke="#27272a" strokeDasharray="3 3" />

            {/* Area Fill */}
            <path d={areaD} fill="url(#equityGrad)" />

            {/* BTC Benchmark Normalized Trajectory */}
            <path d={btcPathD} fill="none" stroke="#38bdf8" strokeWidth="1.2" strokeOpacity="0.4" strokeDasharray="4 3" />

            {/* Portfolio Equity Line */}
            <path d={pathD} fill="none" stroke={isProfit ? "#10b981" : "#f43f5e"} strokeWidth="2.2" strokeLinecap="round" />

            {/* Hover Guides */}
            {hoveredPoint && (
              <g>
                <line
                  x1={(data.indexOf(hoveredPoint) / (data.length - 1)) * 800}
                  y1="0"
                  x2={(data.indexOf(hoveredPoint) / (data.length - 1)) * 800}
                  y2="240"
                  stroke="#71717a"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                />
              </g>
            )}
          </svg>
        )}

        {/* Hover Tooltip Overlay */}
        {data.map((d, idx) => (
          <div
            key={idx}
            className="absolute top-0 bottom-0 cursor-crosshair"
            style={{
              left: `${(idx / (data.length - 1)) * 100}%`,
              width: `${100 / data.length}%`,
            }}
            onMouseEnter={() => setHoveredPoint(d)}
            onMouseLeave={() => setHoveredPoint(null)}
          />
        ))}

        {hoveredPoint && (
          <div className="absolute top-2 left-4 bg-zinc-950/90 border border-zinc-700/80 p-2 rounded-lg text-xs font-mono shadow-xl z-20 pointer-events-none">
            <div className="text-zinc-400 text-[10px]">
              {new Date(hoveredPoint.timestamp).toLocaleTimeString()}
            </div>
            <div className="text-emerald-400 font-bold">
              Equity: ${hoveredPoint.equity.toLocaleString()}
            </div>
            <div className="text-sky-300 text-[11px]">
              BTC: ${hoveredPoint.btc_price.toLocaleString()}
            </div>
            <div className="text-zinc-500 text-[10px]">
              DD: -{hoveredPoint.drawdown_pct}%
            </div>
          </div>
        )}
      </div>

      <div className="flex justify-between items-center text-[10px] font-mono text-zinc-500 pt-2 border-t border-zinc-800/60 mt-2">
        <span>Min: ${minEq.toFixed(0)}</span>
        <span>Starting Balance: ${startingBalance.toLocaleString()}</span>
        <span>Max: ${maxEq.toFixed(0)}</span>
      </div>
    </div>
  );
};
