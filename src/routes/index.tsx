import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useServerFn } from "@tanstack/react-start";
import { useMemo, useState } from "react";
import { Moon, Sun } from "lucide-react";

import { getMarketData } from "@/lib/market.functions";
import { useTheme } from "@/lib/theme";
import type { Interval } from "@/lib/market.server";
import { BARS_PER_YEAR } from "@/lib/market.server";
import {
  alignedReturns,
  analyzeSymbol,
  fmtPct,
  fmtPrice,
  HMM_STATES,
  type SymbolAnalysis,
} from "@/lib/quant/analysis";
import { analyzePortfolio, pca } from "@/lib/quant/portfolio";
import { Panel, Stat } from "@/components/dashboard/Panel";
import { ProbBar, Sparkline } from "@/components/dashboard/Sparkline";
import { BacktestPanel } from "@/components/dashboard/BacktestPanel";
import { Plot3D } from "@/components/viz/Plot3D";
import { usePalette } from "@/lib/viz/palette";
import {
  frontierScene,
  hmmScene,
  monteCarloScene,
  pcaScene,
  volSurfaceScene,
} from "@/lib/viz/scenes";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Quant Terminal — Analisa Crypto HMM, GBM & Markowitz" },
      {
        name: "description",
        content:
          "Dashboard monitoring crypto real-time dengan deteksi regime HMM, simulasi Monte Carlo GBM, permukaan volatilitas, optimasi portofolio Markowitz, PCA, dan sinyal entry Z-score/RSI/MACD.",
      },
      { property: "og:title", content: "Quant Terminal — Analisa Crypto Berbasis Model Matematika" },
      {
        property: "og:description",
        content:
          "Regime HMM, Monte Carlo GBM, efficient frontier, dan timing entry dalam satu dashboard 3D interaktif.",
      },
    ],
  }),
  component: Dashboard,
});

const INTERVALS: Interval[] = ["1h", "4h", "1d"];
const REGIME_LABELS = ["Bear", "Sideways", "Bull"];
const VIEWS = [
  { id: "mc", label: "Monte Carlo Cone" },
  { id: "vol", label: "Volatility Surface" },
  { id: "hmm", label: "Hidden State Space" },
  { id: "frontier", label: "Efficient Frontier" },
  { id: "pca", label: "Eigen-space Cloud" },
] as const;
type ViewId = (typeof VIEWS)[number]["id"];

function Dashboard() {
  const [interval, setIntervalValue] = useState<Interval>("4h");
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [view, setView] = useState<ViewId>("mc");
  const palette = usePalette();
  const { theme, toggleTheme } = useTheme();
  const fetchMarket = useServerFn(getMarketData);

  const { data, isLoading, isError, error, isFetching, refetch } = useQuery({
    queryKey: ["market", interval],
    queryFn: () => fetchMarket({ data: { interval, limit: 500 } }),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  const analyses = useMemo<SymbolAnalysis[]>(() => {
    if (!data) return [];
    return data.series
      .filter((s) => s.candles.length > 120)
      .map((s) => analyzeSymbol(s.symbol, s.candles, data.interval));
  }, [data]);

  const active = analyses.find((a) => a.symbol === symbol) ?? analyses[0];

  const portfolio = useMemo(() => {
    if (!data || data.series.length < 2) return null;
    const { R, labels } = alignedReturns(data.series);
    if (R.length < 30) return null;
    return {
      ...analyzePortfolio(R, BARS_PER_YEAR[data.interval]),
      labels,
      pcaResult: pca(R),
    };
  }, [data]);

  const scene = useMemo(() => {
    if (!active) return null;
    switch (view) {
      case "mc":
        return monteCarloScene(active.mc, active.price, palette);
      case "vol":
        return volSurfaceScene(active.surface, palette);
      case "hmm":
        return hmmScene(active.statePoints, palette, HMM_STATES);
      case "frontier":
        return portfolio
          ? frontierScene(portfolio.frontier, portfolio.labels, palette)
          : null;
      case "pca":
        return portfolio
          ? pcaScene(
              portfolio.pcaResult.scores3d.slice(-Math.min(600, portfolio.pcaResult.scores3d.length)),
              active.hmm.viterbi.slice(-600),
              palette,
              HMM_STATES,
            )
          : null;
      default:
        return null;
    }
  }, [active, view, palette, portfolio]);

  const viewMeta: Record<ViewId, { formula: string; desc: string }> = {
    mc: {
      formula: "S_{t+Δ} = S_t · exp((μ − σ²/2)Δ + σ√Δ·Z)",
      desc: "600 jalur GBM eksak; pita P05/P50/P95 membentuk kerucut probabilitas.",
    },
    vol: {
      formula: "σ(x,T) = σ_ATM(T)·[1 + (s/6)z + (k/24)(z² − 1)], x = ln(K/S)",
      desc: "Skew/smile diturunkan dari skewness & excess kurtosis realized (ekspansi Corrado-Su).",
    },
    hmm: {
      formula: "α_t(j) = [Σ_i α_{t-1}(i)·A_ij]·B_j(o_t)",
      desc: "Baum-Welch + Viterbi; tiap titik diwarnai sesuai regime tersembunyi.",
    },
    frontier: {
      formula: "min wᵀΣw s.t. wᵀμ = r*, 1ᵀw = 1  ⇒  w* = Σ⁻¹(λ1 + γμ)",
      desc: "Permukaan trade-off risk–return–bobot aset; garis amber = efficient frontier.",
    },
    pca: {
      formula: "Σv = λv",
      desc: "Return terstandarisasi diproyeksikan ke 3 principal component utama.",
    },
  };

  return (
    <main className="mx-auto min-h-screen w-full max-w-[1500px] px-4 py-6 lg:px-8">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="mono-label">Quant Terminal · Crypto</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight lg:text-3xl">
            Analisa & Entry Pasar Crypto — Model Matematika
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            HMM (regime) → Stochastic calculus (volatilitas & risiko) → Aljabar linear
            (alokasi) → Z-score/RSI/MACD (timing entry) → position sizing.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex overflow-hidden rounded-md border border-border">
            {INTERVALS.map((iv) => (
              <button
                key={iv}
                type="button"
                onClick={() => setIntervalValue(iv)}
                className={`tabular px-3 py-1.5 text-xs transition-colors ${
                  interval === iv
                    ? "bg-primary text-primary-foreground"
                    : "bg-surface text-muted-foreground hover:text-foreground"
                }`}
              >
                {iv}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => refetch()}
            className="rounded-md border border-border bg-surface px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            {isFetching ? "memuat…" : "refresh"}
          </button>
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Ganti ke mode terang" : "Ganti ke mode gelap"}
            title={theme === "dark" ? "Mode terang" : "Mode gelap"}
            className="grid size-[30px] place-items-center rounded-md border border-border bg-surface text-muted-foreground transition-colors hover:text-foreground"
          >
            {theme === "dark" ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
          </button>
        </div>
      </header>

      <div className="mb-5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="size-1.5 rounded-full bg-bull" />
          Data diambil di sisi server (server function), bukan dari browser — tidak
          terpengaruh blokir ISP Indonesia.
        </span>
        {data ? (
          <span className="tabular">
            sumber: {data.sources.join(" + ") || "-"} · {data.series.length} pair ·{" "}
            {new Date(data.fetchedAt).toLocaleTimeString("id-ID")}
            {data.failed.length ? ` · gagal: ${data.failed.join(", ")}` : ""}
          </span>
        ) : null}
      </div>

      {isLoading ? (
        <div className="panel grid h-64 place-items-center text-sm text-muted-foreground">
          Mengambil data pasar & mem-fit model…
        </div>
      ) : isError || !active ? (
        <div className="panel p-6">
          <h2 className="text-sm font-semibold text-bear">Data pasar tidak tersedia</h2>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            Server gagal menghubungi Binance maupun Bitget. Semua request memang sudah
            dijalankan dari server (bukan browser Anda), jadi blokir ISP lokal bukan
            penyebabnya — kemungkinan bursa membatasi IP server atau jaringan sedang
            bermasalah. Coba refresh beberapa saat lagi.
          </p>
          <p className="tabular mt-2 text-xs text-muted-foreground">
            {error instanceof Error ? error.message : ""}
          </p>
        </div>
      ) : (
        <div className="grid gap-5 xl:grid-cols-[300px_1fr]">
          {/* Watchlist */}
          <aside className="panel h-fit p-2">
            <div className="mono-label px-2 py-2">Watchlist · sinyal komposit</div>
            <ul className="space-y-1">
              {analyses.map((a) => {
                const activeRow = a.symbol === active.symbol;
                return (
                  <li key={a.symbol}>
                    <button
                      type="button"
                      onClick={() => setSymbol(a.symbol)}
                      className={`w-full rounded-md border px-3 py-2 text-left transition-colors ${
                        activeRow
                          ? "border-primary/60 bg-surface-2"
                          : "border-transparent hover:bg-surface-2/60"
                      }`}
                    >
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-sm font-semibold">
                          {a.symbol.replace("USDT", "")}
                        </span>
                        <span className="tabular text-xs text-muted-foreground">
                          ${fmtPrice(a.price)}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center gap-2">
                        <div className="h-[34px] flex-1">
                          <Sparkline
                            values={a.closes}
                            tone={a.changePct >= 0 ? "bull" : "bear"}
                          />
                        </div>
                        <div className="text-right">
                          <div
                            className={`tabular text-[11px] font-semibold ${
                              a.action === "LONG"
                                ? "text-bull"
                                : a.action === "SHORT"
                                  ? "text-bear"
                                  : "text-muted-foreground"
                            }`}
                          >
                            {a.action}
                          </div>
                          <div className="tabular text-[10px] text-muted-foreground">
                            {a.regime} · {a.score.toFixed(2)}
                          </div>
                        </div>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          </aside>

          <div className="space-y-5">
            {/* Ringkasan aset aktif */}
            <Panel
              title={`${active.symbol.replace("USDT", "")}/USDT · ${interval}`}
              subtitle={`Regime saat ini: ${active.regime} · persistensi A_ii = ${active.regimePersistence.toFixed(3)} · durasi ekspektasi ≈ ${Number.isFinite(active.expectedDurationBars) ? active.expectedDurationBars.toFixed(1) : "∞"} bar`}
              right={
                <div className="text-right">
                  <div className="tabular text-2xl font-semibold">
                    ${fmtPrice(active.price)}
                  </div>
                  <div
                    className={`tabular text-xs ${active.changePct >= 0 ? "text-bull" : "text-bear"}`}
                  >
                    {active.changePct >= 0 ? "+" : ""}
                    {active.changePct.toFixed(2)}% / bar
                  </div>
                </div>
              }
            >
              <div className="mb-4">
                <ProbBar probs={active.regimeProbs} labels={REGIME_LABELS} />
              </div>
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-6">
                <Stat
                  label="σ tahunan"
                  value={`${(active.gbm.sigma * 100).toFixed(1)}%`}
                  hint={`σ pendek ${(active.sigmaShort * 100).toFixed(1)}%`}
                />
                <Stat
                  label="μ tahunan"
                  value={fmtPct(active.gbm.mu, 0)}
                  tone={active.gbm.mu >= 0 ? "bull" : "bear"}
                  hint="drift + koreksi Itô"
                />
                <Stat
                  label="VaR 95% "
                  value={fmtPct(active.mc.var95)}
                  tone="bear"
                  hint={`CVaR ${fmtPct(active.mc.cvar95)}`}
                />
                <Stat
                  label="P(naik)"
                  value={`${(active.mc.probUp * 100).toFixed(1)}%`}
                  tone={active.mc.probUp >= 0.5 ? "bull" : "bear"}
                  hint={`horizon ${(active.mc.horizonYears * 365).toFixed(0)} hari`}
                />
                <Stat
                  label="Z-score"
                  value={active.z.toFixed(2)}
                  tone={active.z <= active.entryThreshold ? "bull" : "default"}
                  hint={`ambang optimal ${active.entryThreshold.toFixed(2)}`}
                />
                <Stat
                  label="Quarter-Kelly"
                  value={fmtPct(active.kellyFraction, 0)}
                  tone="accent"
                  hint="f* = (μ − r)/σ² × 0.25"
                />
                <Stat label="RSI(14)" value={active.rsi.toFixed(1)} />
                <Stat
                  label="MACD hist"
                  value={active.macdHist.toFixed(4)}
                  tone={active.macdHist >= 0 ? "bull" : "bear"}
                />
                <Stat
                  label="OU θ"
                  value={active.ou.theta.toFixed(2)}
                  hint={`half-life ${Number.isFinite(active.ou.halfLifeBars) ? active.ou.halfLifeBars.toFixed(1) : "∞"} bar`}
                />
                <Stat label="Skewness" value={active.skew.toFixed(2)} />
                <Stat
                  label="Excess kurtosis"
                  value={active.excessKurtosis.toFixed(2)}
                  tone={active.excessKurtosis > 1 ? "bear" : "default"}
                  hint="fat tail"
                />
                <Stat
                  label="Skor komposit"
                  value={active.score.toFixed(2)}
                  tone={
                    active.action === "LONG"
                      ? "bull"
                      : active.action === "SHORT"
                        ? "bear"
                        : "default"
                  }
                  hint={active.action}
                />
              </div>
            </Panel>

            {/* Zona Entry */}
            {(() => {
              const zone =
                active.action === "LONG"
                  ? active.longZone
                  : active.action === "SHORT"
                    ? active.shortZone
                    : null;
              const triggered =
                zone &&
                (zone.direction === "LONG"
                  ? active.price <= zone.entry
                  : active.price >= zone.entry);
              return (
                <Panel
                  title="Zona Entry (mean-reversion)"
                  subtitle={
                    zone
                      ? `Arah ${zone.direction} · R:R ${zone.riskReward.toFixed(2)}x · ${triggered ? "harga sudah di zona" : "menunggu harga mencapai zona"}`
                      : "Belum ada sinyal LONG/SHORT aktif, atau harga tidak bersifat mean-reverting (θ ≤ 0) saat ini."
                  }
                  formula="P(z) = exp(μ_roll + z·σ_z) — z dari optimal stopping OU"
                >
                  {zone ? (
                    <div className="grid grid-cols-3 gap-2">
                      <Stat
                        label="Entry"
                        value={`$${fmtPrice(zone.entry)}`}
                        tone={zone.direction === "LONG" ? "bull" : "bear"}
                        hint={triggered ? "aktif" : "pending"}
                      />
                      <Stat
                        label="Stop-loss"
                        value={`$${fmtPrice(zone.stop)}`}
                        tone="bear"
                        hint={`${(Math.abs(zone.entry - zone.stop) / zone.entry * 100).toFixed(1)}% dari entry`}
                      />
                      <Stat
                        label="Target"
                        value={`$${fmtPrice(zone.target)}`}
                        tone="bull"
                        hint={`${(Math.abs(zone.target - zone.entry) / zone.entry * 100).toFixed(1)}% dari entry`}
                      />
                    </div>
                  ) : (
                    <div className="grid h-16 place-items-center text-sm text-muted-foreground">
                      Tidak ada zona entry untuk ditampilkan saat ini.
                    </div>
                  )}
                  <p className="mt-3 text-[11px] text-muted-foreground">
                    Snapshot berbasis rata-rata & deviasi log-price bergulir saat ini — bukan
                    proyeksi harga masa depan, dan akan bergeser saat data di-refresh. Bukan
                    nasihat keuangan.
                  </p>
                </Panel>
              );
            })()}

            {/* Visual 3D */}
            <Panel
              title={VIEWS.find((v) => v.id === view)!.label}
              subtitle={viewMeta[view].desc}
              formula={viewMeta[view].formula}
              right={
                <span className="mono-label hidden sm:block">drag = rotasi · scroll = zoom</span>
              }
            >
              <div className="mb-3 flex flex-wrap gap-1.5">
                {VIEWS.map((v) => (
                  <button
                    key={v.id}
                    type="button"
                    onClick={() => setView(v.id)}
                    className={`rounded-md border px-2.5 py-1 text-[11px] transition-colors ${
                      view === v.id
                        ? "border-primary/60 bg-surface-2 text-foreground"
                        : "border-border bg-surface text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {v.label}
                  </button>
                ))}
              </div>
              {scene ? (
                <Plot3D scene={scene} height={380} />
              ) : (
                <div className="grid h-[380px] place-items-center text-sm text-muted-foreground">
                  Butuh minimal 2 aset dengan riwayat cukup untuk visual ini.
                </div>
              )}
            </Panel>

            {/* Portofolio */}
            {portfolio ? (
              <div className="grid gap-5 lg:grid-cols-2">
                <Panel
                  title="Alokasi Portofolio (Markowitz)"
                  subtitle={`Long-only (w ≥ 0, Σw = 1), r_f = ${(portfolio.riskFree * 100).toFixed(1)}% · Sharpe ${portfolio.longOnly.sharpe.toFixed(2)}${portfolio.tangency ? ` · tangency unconstrained Sharpe ${portfolio.maxSharpe.toFixed(2)}` : " · tangency unconstrained tidak eksis (semua excess return ≤ 0)"}`}
                  formula="max wᵀ(μ − r_f)/√(wᵀΣw) s.t. w ≥ 0, 1ᵀw = 1"
                >
                  <ul className="space-y-1.5">
                    {portfolio.labels.map((label, i) => {
                      const w = portfolio.longOnly.weights[i] ?? 0;
                      return (
                        <li key={label} className="flex items-center gap-3">
                          <span className="tabular w-12 text-xs">{label}</span>
                          <div className="relative h-2 flex-1 rounded-full bg-surface-2">
                            <div
                              className={`absolute top-0 h-2 rounded-full ${w >= 0 ? "bg-primary" : "bg-bear"}`}
                              style={{
                                left: "50%",
                                width: `${Math.min(Math.abs(w) * 50, 50)}%`,
                                transform: w >= 0 ? "none" : `translateX(-${Math.min(Math.abs(w) * 50, 50)}%)`,
                              }}
                            />
                          </div>
                          <span className="tabular w-16 text-right text-xs text-muted-foreground">
                            {(w * 100).toFixed(1)}%
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                  <p className="mt-3 text-[11px] text-muted-foreground">
                    Portofolio varians minimum global (GMV): return{" "}
                    {fmtPct(portfolio.gmv.ret, 0)} · risk{" "}
                    {(portfolio.gmv.risk * 100).toFixed(1)}% · expected return long-only{" "}
                    {fmtPct(portfolio.longOnly.ret, 0)} pada risk{" "}
                    {(portfolio.longOnly.risk * 100).toFixed(1)}%.
                  </p>
                </Panel>

                <Panel
                  title="Faktor Risiko (PCA)"
                  subtitle="Eigenvalue matriks korelasi return — PC1 biasanya “market beta” terhadap BTC."
                  formula="Σv = λv"
                >
                  <ul className="space-y-2">
                    {portfolio.pcaResult.explained.slice(0, 5).map((e, i) => (
                      <li key={i} className="flex items-center gap-3">
                        <span className="tabular w-12 text-xs">PC{i + 1}</span>
                        <div className="h-2 flex-1 rounded-full bg-surface-2">
                          <div
                            className="h-2 rounded-full bg-accent"
                            style={{ width: `${e * 100}%` }}
                          />
                        </div>
                        <span className="tabular w-16 text-right text-xs text-muted-foreground">
                          {(e * 100).toFixed(1)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                  <p className="mt-3 text-[11px] text-muted-foreground">
                    λ₁ = {portfolio.pcaResult.eigenvalues[0]?.toFixed(2)} dari{" "}
                    {portfolio.labels.length} aset — makin dominan, makin kecil manfaat
                    diversifikasi.
                  </p>
                </Panel>
              </div>
            ) : null}

            {/* Backtest */}
            {(() => {
              const series = data?.series.find((s) => s.symbol === active.symbol);
              return series ? (
                <BacktestPanel symbol={active.symbol} candles={series.candles} interval={interval} />
              ) : null;
            })()}

            {/* Catatan */}
            <Panel
              title="Catatan Keterbatasan Model di Pasar Crypto"
              subtitle="Dipakai sebagai input risk management, bukan rumus ajaib."
            >
              <ul className="grid gap-2 text-xs text-muted-foreground md:grid-cols-2">
                <li>
                  <b className="text-foreground">Fat tails</b> — excess kurtosis aset ini{" "}
                  {active.excessKurtosis.toFixed(2)} (normal = 0), jadi GBM &
                  Black-Scholes cenderung meremehkan risiko ekor.
                </li>
                <li>
                  <b className="text-foreground">Non-stationarity</b> — parameter di-fit
                  ulang tiap refresh; regime bisa berubah drastis akibat hack, regulasi,
                  atau depeg.
                </li>
                <li>
                  <b className="text-foreground">Liquidity gap</b> — model mengasumsikan
                  eksekusi di harga wajar; slippage altcoin bisa besar.
                </li>
                <li>
                  <b className="text-foreground">Reflexivity</b> — sinyal populer rentan
                  jadi self-fulfilling atau target stop hunting.
                </li>
              </ul>
              <p className="mt-3 text-[11px] text-muted-foreground">
                Semua estimasi memakai data historis apa adanya (tanpa lookahead):
                indikator dihitung hanya dari bar sampai t. Bukan nasihat keuangan.
              </p>
            </Panel>
          </div>
        </div>
      )}
    </main>
  );
}
