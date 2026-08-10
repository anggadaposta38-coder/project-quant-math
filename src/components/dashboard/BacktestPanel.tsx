import { useEffect, useMemo, useRef, useState } from "react";
import type { Candle, Interval } from "@/lib/market.server";
import {
  DEFAULT_BACKTEST_CONFIG,
  runWalkForwardBacktest,
  type BacktestComparison,
  type WalkForwardFoldSummary,
} from "@/lib/quant/backtest";
import { fmtPct } from "@/lib/quant/analysis";
import { Panel, Stat } from "./Panel";
import { EquityChart } from "./EquityChart";

function MetricsGrid({ label, m }: { label: string; m: BacktestComparison["composite"]["metrics"] }) {
  return (
    <div>
      <div className="mono-label mb-2">{label}</div>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <Stat
          label="Total return"
          value={fmtPct(m.totalReturn, 1)}
          tone={m.totalReturn >= 0 ? "bull" : "bear"}
          hint={`vs buy&hold ${fmtPct(m.buyHoldReturn, 1)}`}
        />
        <Stat label="CAGR" value={fmtPct(m.cagr, 1)} tone={m.cagr >= 0 ? "bull" : "bear"} />
        <Stat label="Sharpe" value={m.sharpe.toFixed(2)} />
        <Stat label="Sortino" value={Number.isFinite(m.sortino) ? m.sortino.toFixed(2) : "∞"} />
        <Stat
          label="Max drawdown"
          value={fmtPct(m.maxDrawdown, 1)}
          tone="bear"
        />
        <Stat
          label="Win rate"
          value={`${(m.winRate * 100).toFixed(0)}%`}
          hint={`${m.numTrades} trade`}
        />
        <Stat label="Profit factor" value={Number.isFinite(m.profitFactor) ? m.profitFactor.toFixed(2) : "∞"} />
        <Stat label="Avg trade" value={fmtPct(m.avgTradeReturn, 2)} tone={m.avgTradeReturn >= 0 ? "bull" : "bear"} />
        <Stat label="Exposure" value={`${(m.exposurePct * 100).toFixed(0)}%`} hint="% bar dengan posisi terbuka" />
      </div>
    </div>
  );
}


function formatFoldDate(timestamp: number): string {
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    timeZone: "UTC",
  }).format(timestamp);
}

function FoldRobustness({ label, folds }: { label: string; folds: WalkForwardFoldSummary[] }) {
  const positiveFolds = folds.filter((fold) => fold.totalReturn > 0).length;
  const firstFold = folds[0];
  if (firstFold === undefined) return null;
  const worstFold = folds.reduce((worst, fold) =>
    fold.totalReturn < worst.totalReturn ? fold : worst, firstFold);

  return (
    <div>
      <div className="mono-label mb-2">{label}</div>
      <div className="mb-2 grid grid-cols-2 gap-2 md:grid-cols-3">
        <Stat label="Fold positif" value={`${positiveFolds}/${folds.length}`} />
        <Stat label="Fold terburuk" value={fmtPct(worstFold.totalReturn, 1)} tone={worstFold.totalReturn >= 0 ? "bull" : "bear"} />
        <Stat label="MDD terburuk" value={fmtPct(Math.min(...folds.map((fold) => fold.maxDrawdown)), 1)} tone="bear" />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-[11px]">
          <thead className="text-muted-foreground">
            <tr>
              <th className="pb-1 pr-3 font-normal">Fold</th>
              <th className="pb-1 pr-3 font-normal">Periode</th>
              <th className="pb-1 pr-3 text-right font-normal">Return</th>
              <th className="pb-1 pr-3 text-right font-normal">MDD</th>
              <th className="pb-1 pr-3 text-right font-normal">Trades</th>
              <th className="pb-1 text-right font-normal">Win rate</th>
            </tr>
          </thead>
          <tbody className="tabular">
            {folds.map((fold) => (
              <tr key={fold.fold} className="border-t border-border/50">
                <td className="py-1 pr-3">{fold.fold}</td>
                <td className="py-1 pr-3 text-muted-foreground">
                  {formatFoldDate(fold.startTime)} → {formatFoldDate(fold.endTime)}
                </td>
                <td className={`py-1 pr-3 text-right ${fold.totalReturn >= 0 ? "text-bull" : "text-bear"}`}>
                  {fmtPct(fold.totalReturn, 1)}
                </td>
                <td className="py-1 pr-3 text-right text-bear">{fmtPct(fold.maxDrawdown, 1)}</td>
                <td className="py-1 pr-3 text-right">{fold.numTrades}</td>
                <td className="py-1 text-right">{(fold.winRate * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function BacktestPanel({
  symbol,
  candles,
  interval,
}: {
  symbol: string;
  candles: Candle[];
  interval: Interval;
}) {
  const [result, setResult] = useState<BacktestComparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const runTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (runTimerRef.current !== null) {
        clearTimeout(runTimerRef.current);
        runTimerRef.current = null;
      }
    };
  }, []);

  const run = () => {
    setRunning(true);
    setError(null);
    // setTimeout(0) agar state "running" sempat ter-render sebelum komputasi
    // sinkron (walk-forward refit HMM berulang) memblokir main thread sejenak.
    runTimerRef.current = setTimeout(() => {
      try {
        const r = runWalkForwardBacktest(candles, interval, DEFAULT_BACKTEST_CONFIG);
        setResult(r);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Backtest gagal dijalankan.");
        setResult(null);
      } finally {
        setRunning(false);
        runTimerRef.current = null;
      }
    }, 30);
  };

  const cfg = DEFAULT_BACKTEST_CONFIG;
  const equitySeries = useMemo(
    () => [
      { name: "Skor komposit", colorClass: "stroke-primary", points: result?.composite.equityCurve ?? [] },
      { name: "OU zone", colorClass: "stroke-accent", points: result?.ouZone.equityCurve ?? [] },
    ],
    [result],
  );

  return (
    <Panel
      title={`Backtest Walk-Forward — ${symbol.replace("USDT", "")}/USDT`}
      subtitle={`Refit model tiap ${cfg.refitInterval} bar setelah warmup ${cfg.warmupBars} bar · biaya ${(cfg.feeBps + cfg.slippageBps) * 2} bps round-trip · tanpa lookahead (sinyal di checkpoint t hanya memakai bar ≤ t)`}
      right={
        <button
          type="button"
          onClick={run}
          disabled={running}
          className="rounded-md border border-primary/60 bg-surface-2 px-3 py-1.5 text-xs text-foreground transition-colors hover:bg-surface disabled:opacity-50"
        >
          {running ? "menjalankan…" : result ? "jalankan ulang" : "jalankan backtest"}
        </button>
      }
    >
      {error ? (
        <div className="rounded-md border border-bear/40 bg-bear/5 p-3 text-xs text-bear">{error}</div>
      ) : !result ? (
        <div className="grid h-24 place-items-center text-sm text-muted-foreground">
          {running
            ? "Mem-fit model di tiap checkpoint walk-forward…"
            : "Klik \"jalankan backtest\" untuk menguji dua strategi pada data historis simbol ini."}
        </div>
      ) : (
        <div className="space-y-5">
          <MetricsGrid label="Strategi 1 · Skor komposit (LONG/SHORT/WAIT)" m={result.composite.metrics} />
          <MetricsGrid label="Strategi 2 · Zona entry mean-reversion (OU)" m={result.ouZone.metrics} />

          <div>
            <div className="mono-label mb-2">Equity curve (mulai = 1.0)</div>
            <EquityChart series={equitySeries} />
          </div>

          <div className="space-y-4">
            <FoldRobustness label="OOS fold robustness · Skor komposit" folds={result.folds.composite} />
            <FoldRobustness label="OOS fold robustness · OU zone" folds={result.folds.ouZone} />
          </div>

          {result.ouZone.trades.length > 0 ? (
            <div>
              <div className="mono-label mb-2">
                Trade terakhir · OU zone ({result.ouZone.trades.length} total)
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[11px]">
                  <thead className="text-muted-foreground">
                    <tr>
                      <th className="pb-1 pr-3 font-normal">Arah</th>
                      <th className="pb-1 pr-3 font-normal">Entry</th>
                      <th className="pb-1 pr-3 font-normal">Exit</th>
                      <th className="pb-1 pr-3 font-normal">Alasan</th>
                      <th className="pb-1 pr-3 text-right font-normal">Return</th>
                    </tr>
                  </thead>
                  <tbody className="tabular">
                    {result.ouZone.trades.slice(-8).reverse().map((t, i) => (
                      <tr key={i} className="border-t border-border/50">
                        <td className={`py-1 pr-3 ${t.direction === "LONG" ? "text-bull" : "text-bear"}`}>
                          {t.direction}
                        </td>
                        <td className="py-1 pr-3">${t.entryPrice.toFixed(2)}</td>
                        <td className="py-1 pr-3">${t.exitPrice.toFixed(2)}</td>
                        <td className="py-1 pr-3 text-muted-foreground">{t.reason}</td>
                        <td className={`py-1 pr-3 text-right ${t.retPct >= 0 ? "text-bull" : "text-bear"}`}>
                          {fmtPct(t.retPct, 2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          <p className="text-[11px] text-muted-foreground">
            Backtest historis tidak menjamin performa masa depan. Fold OOS di atas membagi
            equity curve walk-forward secara kronologis untuk melihat apakah hasil bergantung
            pada satu sub-periode. Fold ini bukan validation set independen dan bukan bukti
            adanya edge di masa depan. Bobot skor komposit & parameter OU dipakai apa adanya
            dari kode live (tidak dioptimasi ulang khusus untuk periode ini).
          </p>
        </div>
      )}
    </Panel>
  );
}
