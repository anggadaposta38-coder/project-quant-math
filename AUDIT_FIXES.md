# Quant Math Audit & Bug Fixes

Audit date: 2026-08-10

## Fixed

1. **Backtest transaction costs**
   - Fee + slippage are per side.
   - Round-trip cost is now exactly `2 * (feeBps + slippageBps) / 10000`.
   - Equity curve charges one transaction cost on entry and one on exit.

2. **HMM posterior consistency**
   - `gamma` is recomputed from the final fitted/sorted parameters.
   - Final log-likelihood now corresponds to the final parameters.

3. **Markowitz return convention**
   - Portfolio mean/covariance now use simple returns (`expm1(logReturn)`) before annualization.
   - This avoids mixing GBM continuous drift with arithmetic portfolio returns.

4. **Small covariance matrices**
   - `inverse()` now uses a scale-relative pivot tolerance.
   - `ridge()` now has a minimum regularization scale, so tiny but valid crypto return covariance matrices do not become numerically singular.

5. **Portfolio input validation**
   - Validates matrix dimensions, finite values, sample length, annualization factor, risk-free rate and frontier point count.

6. **Backtest input validation**
   - Validates config values and OHLC integrity/timestamp monotonicity before running.

7. **Timestamp alignment**
   - `alignedReturns()` now joins returns by candle timestamp instead of blindly aligning equal-length suffixes.

8. **Monte Carlo VaR/CVaR convention**
   - `var95` and `cvar95` are now positive loss magnitudes, matching the UI labels “VaR 95%” and “CVaR”.

9. **Quant model parameter validation**
   - Added finite/positive input checks for GBM, OU, Monte Carlo and HMM parameters.

10. **Test tooling**
    - Added `vitest` dependency and `test` / `test:watch` scripts.
    - Added regression tests for VaR/CVaR, HMM posterior consistency, timestamp alignment and transaction-cost behavior.

## Not changed deliberately

The following are model-design limitations rather than deterministic coding bugs and should be treated separately:

- OU entry threshold is still a heuristic/numerical threshold, not a closed-form optimal-stopping solution.
- OU validity is not yet backed by an ADF/unit-root significance test or confidence intervals.
- GBM drift/Kelly remains estimation-sensitive; position caps and uncertainty-aware sizing are still recommended.
- The volatility surface is a model surface from realized moments, not a market implied-volatility surface.
- OHLC backtest fills still assume the specified limit level is filled when the bar touches it; queue/partial-fill/liquidity modeling is not implemented.

## Verification

- TypeScript transpile/syntax checks passed for all modified source and test files.
- A runtime smoke test passed for Monte Carlo VaR/CVaR, HMM posterior normalization, ridge/inverse behavior and portfolio analysis.
- Full Vitest execution could not be completed in the audit environment because the configured package registry did not provide the `vitest` package and the project dependencies were not installed.

## Priority-high audit pass 2

- Closed-candle filter: live market data now excludes the currently forming candle.
- Market data integrity: returned series must have valid OHLCV and exact interval spacing.
- Composite backtest execution: signal at close `t` is executed at close `t+1`, eliminating same-close signal/execution optimism.
- OU-zone backtest: the entry candle is now checked for stop/target after a limit fill; if both are touched, stop wins as the conservative OHLC assumption.
- UI fee label corrected to show true round-trip cost.

## Prioritas Tinggi #3 — execution / intrabar accounting
- Trade return ledger now compounds entry and exit transaction costs exactly instead of subtracting costs linearly.
- OU-zone backtest no longer exits on the same OHLC candle as a limit fill, because tick ordering is unknowable from OHLC.
- If both LONG and SHORT entry limits are touched on the same candle, the ambiguous candle is skipped rather than selecting a side with an arbitrary distance heuristic.
- Gap-through stop/target exits use candle OPEN when the market has already crossed the requested level at the open.
- Added regression tests for strictly increasing equity timestamps, exit-after-entry, and exact compounded transaction-cost accounting.


## Audit Prioritas Tinggi #5 — annualization/timeframe

- Annualization crypto dipusatkan pada 365 hari/tahun dan diturunkan dari interval, bukan angka duplikat.
- Faktor menjadi 8,760 bar/tahun untuk 1h, 2,190 untuk 4h, dan 365 untuk 1d.
- Backtest sekarang menolak timestamp yang melompati/merapatkan interval; sebelumnya hanya memeriksa monotonic timestamp sehingga CAGR dan annualized metrics dapat salah bila ada gap.
- Sharpe memakai risk-free periodic rate yang dikompaun dari annual simple rate: `(1 + rf)^(1/barsPerYear) - 1`, bukan `rf/barsPerYear`.
- Annualized volatility tetap `sd_bar * sqrt(barsPerYear)`, yaitu konvensi standar untuk return periodik dengan interval reguler.
- Monte Carlo/GBM tetap memakai `dt = 1 / barsPerYear`, sehingga horizon dan volatility konsisten dengan timeframe.
- Maturity pada volatility surface tetap berbasis hari kalender/365 karena maturity adalah waktu kalender, bukan jumlah bar.

## Prioritas Tinggi #6 — SHORT accounting

- Ekstrak `directionalRawReturn()` agar LONG/SHORT menggunakan rumus PnL yang sama secara simetris.
- Ekstrak `netTradeReturn()` agar return trade dan equity accounting memakai formula biaya yang sama.
- Verifikasi gap stop/target SHORT: stop fill menggunakan `max(stop, open)` dan target menggunakan `min(target, open)`.
- Candle entry OU sekarang dihitung sebagai exposed bar setelah limit fill diketahui.
- Candle ambigu yang menyentuh long dan short entry sekaligus tetap diabaikan sebagai entry, tetapi sekarang timestamp candle tetap dicatat ke equity curve sehingga tidak ada gap artifisial pada exposure timeline.
- Regression tests ditambahkan untuk simetri LONG/SHORT dan gap exit SHORT.

## Audit #7 — Position sizing / risk limits

- Added `riskPerTradePct` (default 1%) and `maxPositionFraction` (default 25%) to backtest config.
- Added `positionFractionFromStop()` for stop-distance risk sizing.
- Added `positionFractionFromKelly()` with direction alignment: a negative Kelly value cannot silently reverse a LONG into a SHORT (and vice versa).
- Composite backtest now sizes notional from aligned quarter-Kelly and applies transaction costs proportionally to notional.
- OU-zone backtest sizes from stop distance, capped at 25% equity, and applies price PnL/fees proportionally to notional.
- Each trade records `positionFraction` for auditability.
- Backtest no longer implicitly assumes 100% capital allocation / 1x notional for every trade.
- Added regression tests for sizing caps and recorded position fractions.

Note: position sizing is a risk-control layer, not a guarantee of profitability. The 1% risk and 25% max-notional defaults are conservative engineering defaults and should be validated against the intended strategy/account constraints.

## Prioritas Tinggi #8 — NaN/Infinity propagation
- Core statistical functions now reject non-finite inputs instead of silently propagating them.
- `logReturns()` no longer silently skips invalid prices; invalid price data fails fast.
- GBM, Monte Carlo, Black-Scholes, OU and volatility-surface outputs are validated for finite values.
- Signal computation rejects invalid RSI/MACD/Z-score/volatility outputs after warm-up.
- Portfolio inverse covariance and final portfolio outputs are checked for NaN/Infinity.
- PCA now validates its input and output.
- Quantile validates probability/data inputs.


## Prioritas Tinggi #9 — HMM label switching

- State HMM sekarang memiliki canonical ordering eksplisit berdasarkan emission mean (mu), dengan sigma/original index sebagai tie-breaker deterministik.
- Posterior dan Viterbi dihitung setelah canonicalization sehingga state IDs yang dipakai UI konsisten dengan parameter final.
- Ditambahkan ambiguity guard: bila mean state berdekatan (< 0.25 pooled standard deviation), regime dianggap tidak stabil untuk keputusan trading. Ini mencegah tiny EM noise membuat label Bear/Sideways/Bull tampak berubah-ubah antar walk-forward refit.
- Regression tests ditambahkan untuk canonicalization dan kestabilan label.
- Catatan: canonicalization berdasarkan mean membuat label SEMANTIK (Bear → Bull), bukan mempertahankan ID numerik mentah EM. Jika mean dua regime benar-benar bertukar secara ekonomi, label boleh berubah karena memang karakter regime berubah; itu bukan label-switching numerik.


## Audit #10 — covariance / Markowitz stability
- covariance input now validates rectangular finite data
- correlation matrix handles zero-variance assets explicitly and keeps unit diagonal
- added nearestPsd eigenvalue flooring before Markowitz inversion
- Markowitz rejects degenerate A/D scalars instead of silently producing unstable frontier weights
- frontier return range made symmetric around GMV with a robust scale
- regression tests added for PSD projection and correlation invariants


## Audit #11 — PCA & eigen-decomposition
- PCA sekarang menggunakan return yang benar-benar distandarisasi dengan standard deviation sampel, tanpa artificial variance floor.
- Asset dengan variance nol/nyaris nol ditolak agar tidak menciptakan correlation/eigenstructure artifisial.
- Explained variance selalu non-negatif dan dinormalisasi tepat ke 1.
- Tanda eigenvector dibuat deterministik berdasarkan komponen dengan magnitude terbesar.
- PCA scores dihitung langsung dari standardized observations dan canonical loadings.
- Regression tests ditambahkan untuk orthonormality, finite scores, dan constant-asset rejection.

## Prioritas Tinggi #13 — OU mean-reversion / half-life
- `fitOu()` sekarang mengembalikan diagnostics `rSquared`, `dfStatistic`, `stationary`, dan `reliable` untuk membedakan fit OU numerik dari fit yang cukup kuat untuk signal.
- Ditambahkan simple Dickey-Fuller statistic (tanpa lagged differences) sebagai diagnostic; dokumentasi sengaja menyebutnya DF sederhana, bukan full ADF test.
- OU signal sekarang mensyaratkan minimal 60 observasi, bukti mean-reversion pada diagnostic DF 5% approximation, dan half-life tidak lebih dari 50% panjang sample. Ini mencegah model dengan persistence sangat dekat 1 menghasilkan zona entry seolah-olah mean-reverting.
- `optimalEntryThreshold()` sekarang memvalidasi `sigmaZ` dan risk-free rate sebelum melakukan optimisasi.
- Half-life ekstrem/infinite tidak lagi dianggap OU valid untuk signal, walaupun parameter hasil regresinya masih tersedia untuk audit.
- Regression tests ditambahkan untuk menolak random-walk-like sample dan menerima sample AR mean-reverting yang jelas.
- Tidak diklaim sebagai ADF penuh: jika diperlukan rigor lebih lanjut, tahap berikutnya dapat menambahkan lagged-difference ADF, critical values MacKinnon, confidence interval parameter OU, dan out-of-sample validation.

## Prioritas Tinggi #14 — Volatility / Return Estimation

- `variance()` sekarang menggunakan Welford untuk mengurangi catastrophic cancellation pada data dengan level besar dan deviasi kecil.
- Ditambahkan `annualizedVolatility()` sebagai satu konvensi terpusat untuk annualisasi sample volatility.
- GBM volatility dan rolling volatility memakai helper annualisasi yang sama.
- `sigmaShort` memakai `barsPerYear` secara eksplisit, menghindari konversi `dt` yang tersebar.
- Ditambahkan regression tests untuk numerical stability dan konsistensi annualization.
- Estimator tetap **sample variance (n-1)**. Portfolio covariance dan realized-volatility estimation menggunakan estimator yang sama; tidak dicampur dengan population variance.
- Skewness/kurtosis tetap memakai adjusted Fisher-Pearson / unbiased excess-kurtosis estimator; tidak diubah menjadi population moments karena itu akan mengubah definisi statistik yang sudah dipakai UI.


## Prioritas Tinggi #16 — trade ledger vs equity curve reconciliation
- Fixed position notional at entry: `positionFraction` is now explicitly a fraction of equity at entry, rather than being implicitly rebalanced against current equity on every bar.
- Composite and OU mark-to-market now use `entryEquity + fixedNotional * rawReturn`, with entry/exit fees charged on actual entry/exit notional.
- Corrected `netTradeReturn()` to match cash accounting for entry + exit fees: `(1 + rawReturn) * (1 - oneWayCostRate) - 1 - oneWayCostRate`, where the exit fee is applied to the exit notional.
- Exit fee is calculated from exit notional, so gap exits and profitable/losing trades reconcile with the trade ledger.
- Added deterministic ledger/equity reconciliation helpers/tests covering zero-return round trips, fixed-notional PnL, fee accounting, and LONG/SHORT symmetry.
- This removes the previous mismatch where the equity curve dynamically re-sized a position every bar while the trade ledger described a fixed fraction of entry equity.

## Prioritas Menengah #1 — mathematical audit HMM / GBM / OU
- GBM sekarang mengekspos `rawLogDrift` dan `driftShrinkageWeight` agar drift regularisasi tidak keliru dianggap sebagai MLE murni. `mu`/`logDrift` tetap memakai shrinkage yang sudah menjadi desain model.
- OU innovation volatility diperbaiki dari sample residual SD (penyebut n-2) menjadi conditional Gaussian MLE `sqrt(SSE/(n-1))`; ini konsisten dengan estimasi transisi AR(1) yang dipakai untuk menurunkan sigma OU.
- OU sekarang mengekspos `innovationStd` untuk auditability.
- HMM memvalidasi parameter akhir EM (probabilitas non-negatif, jumlah pi/row transition = 1, sigma positif dan finite) sebelum posterior final diekspos.
- HMM/OU tetap merupakan estimator Gaussian/AR(1) dengan asumsi model tertentu; simple Dickey-Fuller yang ada bukan ADF penuh, dan OU `reliable` tetap merupakan heuristic trading gate, bukan bukti kausal bahwa harga wajib mean-revert.
- Tidak mengubah shrinkage drift GBM menjadi MLE murni karena shrinkage adalah regularisasi yang disengaja untuk mengurangi estimator drift crypto yang sangat noisy; perubahan hanya membuat perbedaan MLE vs regularized estimate eksplisit.

## Prioritas Menengah #2 — Composite score & directional Kelly

- Exposed normalized score components (`regime`, `z`, `rsi`, `macd`) in `SignalState` for auditability.
- Documented composite weights as heuristic, not calibrated probabilities.
- Fixed directional Kelly mismatch: `kellyFraction` is now signed according to the selected action and is zero when WAIT or when model Kelly points against the selected direction.
- Added regression coverage for normalized component bounds and Kelly/action sign consistency.


## v23 — Portfolio timestamp/candle-gap alignment
- `alignedReturns()` now requires exact expected candle spacing.
- Missing candles no longer become multi-period returns that are aligned with another asset's single-period return.
- Portfolio route passes the selected interval explicitly.
- Regression tests cover timestamp intersection and multi-bar gaps.

## Prioritas Menengah #7 — Exchange candle normalization

- Raw candle order is normalized chronologically.
- Exact duplicate timestamps are collapsed; conflicting duplicate timestamps are rejected.
- Open candles are removed before model use.
- OHLCV and timestamp alignment are validated, including UTC interval boundaries.
- Missing intervals are rejected rather than interpolated or silently bridged.
- Binance is preferred; Bitget is used only as a complete-source fallback, never stitched into Binance candles.
- A source must provide at least 60 closed contiguous candles before it is returned.


## Prioritas Menengah #8 — HMM EM convergence & likelihood

- Baum-Welch sekarang menyimpan `logLikelihoodHistory` untuk audit convergence, bukan hanya nilai akhir.
- EM memeriksa bahwa log-likelihood tidak turun melampaui toleransi numerik; penurunan yang lebih besar dianggap kegagalan numerik dan pipeline fail-closed.
- `HmmFit.converged` membedakan fit yang benar-benar mencapai tolerance dari fit yang berhenti karena `maxIter`.
- `gamma` dan `logLikelihood` tetap dihitung ulang dari parameter final setelah M-step/canonicalization, sehingga diagnostics dan output final berasal dari model yang sama.
- Ditambahkan regression test untuk monotonicity likelihood dan convergence flag.
- Ini tidak mengklaim global optimum: Gaussian HMM dengan Baum-Welch tetap dapat berhenti pada local optimum; random restarts/multi-start belum dipaksakan karena akan mengubah reproducibility dan biaya komputasi.

## Prioritas Menengah #9 — Walk-forward robustness
- Walk-forward core sudah memiliki future-data firewall: checkpoint hanya mem-fit data `0..i`, eksekusi dimulai `i+1`.
- Ditambahkan `summarizeWalkForwardFolds()` untuk memecah periode OOS hasil backtest secara kronologis menjadi beberapa sub-periode dan mendeteksi ketergantungan performa pada satu regime/periode.
- Fold diagnostics melaporkan return, max drawdown, jumlah trade, win rate, dan exposure tanpa melakukan refit menggunakan data fold masa depan.
- Dokumentasi memperjelas bahwa fold robustness bukan pengganti holdout independen; parameter strategy tetap harus ditetapkan sebelum evaluasi OOS.


## Prioritas Menengah #10 — end-to-end quant pipeline consistency
- `computeSignal()` sekarang mendefinisikan `pBear`/`pBull` secara eksplisit dari HMM state canonical (mean terendah/tertinggi), sehingga composite regime score tidak bergantung pada identifier yang tidak didefinisikan.
- Regime gate sekarang juga mensyaratkan HMM EM benar-benar `converged`; fit yang hanya berhenti di `maxIter` tidak boleh menghasilkan regime-driven trading signal.
- Ditemukan dan diperbaiki mismatch penting pada `netTradeReturn()`: equity ledger memang membebankan biaya entry dan exit, tetapi helper trade return sebelumnya hanya merepresentasikan satu biaya. Formula sekarang merekonsiliasi kedua biaya: `(1 + rawReturn) * (1 - c) - 1 - c`.
- Regression expectation untuk zero-move round trip diperbaiki menjadi `-2c`, konsisten dengan entry fee + exit fee.
- End-to-end contract sekarang lebih konsisten dari HMM posterior → composite score → action → directional Kelly → position fraction → fixed-notional equity → trade return → performance metrics.
- Slippage tetap dimodelkan sebagai per-side proportional execution cost (digabung dengan fee dalam `oneWayCost`); ini merupakan model biaya konservatif, bukan simulasi microstructure/price-impact.


## Prioritas Rendah #1 — Backtest performance / duplicate model fits
- Composite dan OU walk-forward memakai checkpoint yang sama, tetapi sebelumnya masing-masing strategi memanggil `computeSignal()` sendiri sehingga HMM/GBM/OU di-fit ulang dua kali pada checkpoint yang identik.
- Ditambahkan `buildCheckpointSignals()` untuk menghitung satu `SignalState` per checkpoint dan membaginya ke kedua strategi.
- Perubahan ini tidak mengubah parameter, urutan checkpoint, atau execution timing; hanya menghilangkan duplicate computation.
- Future-data firewall tetap sama: setiap signal dibangun dari `candles[0..checkpoint]` dan baru digunakan mulai `checkpoint + 1`.
- Ini terutama mengurangi CPU dan allocation pada backtest panjang; optimasi tidak mengubah hasil numerik yang diharapkan.

## Prioritas Rendah #3 — Bundle/build efficiency

- `Plot3D` sekarang dimuat dengan `React.lazy()` sehingga engine 3D tidak masuk ke initial dashboard chunk.
- `BacktestPanel` sekarang dimuat dengan `React.lazy()` sehingga modul walk-forward backtest yang relatif berat hanya diunduh saat bagian tersebut dirender.
- Keduanya memakai `Suspense` fallback yang ringan agar initial render tetap responsif saat chunk tambahan dimuat.
- Tidak ada perubahan pada model quant, signal, accounting, atau parameter trading.
- Tidak menambahkan dependency baru.


## Prioritas Rendah #8 — loading state & perceived performance
- React Query now uses `keepPreviousData` when the interval/query key changes, so the last valid market snapshot remains visible while the new timeframe is loading.
- `isFetching` is exposed as a non-blocking status banner instead of replacing the dashboard with an empty loading panel when usable data already exists.
- Fatal API errors still use the existing error state when no data is available; a refresh/retry does not blank an otherwise usable dashboard.
- Added `aria-live="polite"` to the refresh status so screen readers receive the background update without an intrusive announcement.
- Quant calculations and 3D scene inputs are unchanged; this is a data-loading UX/lifecycle change only.

## Low #10 — Code quality / dead logic / consistency

- Removed unused `roundTripCost` helper from `quant/backtest.ts`.
- Promoted signal-policy tuning knobs to named constants in `quant/analysis.ts` (`SIGNAL_WEIGHTS`, `SIGNAL_ACTION_THRESHOLD`, `QUARTER_KELLY`, `REGIME_STABLE_MIN`) to avoid scattered magic numbers and make audit/review safer.
- Added a regression test that enforces signal weights sum to 1 and keeps the action threshold/Kelly policy within expected bounds.
- Kept generated `routeTree.gen.ts` and the reusable UI component library intact rather than deleting generated/template files merely because they are not currently imported; this avoids breaking future route/UI generation.
