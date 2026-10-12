// Builds the project report (.docx) from outputs/results.json + figures/*.png
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, Header, Footer,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs/results.json"), "utf8"));
const CODE = fs.readFileSync(path.join(ROOT, "src/tat_eda_dl.py"), "utf8").split("\n");
const FONT = "Times New Roman";
const f = (x, d = 3) => Number(x).toFixed(d);
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const DEV = ["MacBookPro1", "MacBookPro2", "VM", "RasberryPi"];

// ---------- helpers
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, ...o });
function para(parts, o = {}) {
  const kids = (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x));
  return new Paragraph({ children: kids, spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...o });
}
const b = (t) => run(t, { bold: true });
const i = (t) => run(t, { italics: true });
const ph = (t) => { const r = run(t, { highlight: "yellow" }); r.options = { text: t }; return r; };
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], keepNext: true });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })], keepNext: true });
const bullet = (parts) => new Paragraph({
  numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 },
  children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)),
});
const num = (parts, ref = "num") => new Paragraph({
  numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 276 },
  children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? run(x) : x)),
});
const bd = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
function table(widths, header, rows, o = {}) {
  const total = widths.reduce((a, c) => a + c, 0);
  const mk = (txt, w, hdr, align) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: hdr ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: align, children: [typeof txt === "object" ? new TextRun({ text: txt.options ? txt.options.text : "", font: FONT, size: 19, highlight: "yellow" }) : run(String(txt), { size: 19, bold: hdr })] })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, k) => mk(h, widths[k], true, AlignmentType.CENTER)) }),
      ...rows.map((r) => new TableRow({
        cantSplit: true,
        children: r.map((c, k) => mk(c, widths[k], false, (o.left || []).includes(k) ? AlignmentType.LEFT : AlignmentType.CENTER)),
      })),
    ],
  });
}
const spacer = () => new Paragraph({ children: [], spacing: { after: 100 } });
function figure(file, caption, wpx = 560) {
  const buf = fs.readFileSync(path.join(ROOT, "figures", file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: Math.round((wpx * h) / w) },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [run(caption, { italics: true, size: 20 })] }),
  ];
}
// "What it shows / Why / Insights" block used for every visual
function viz(what, why, ins) {
  return [
    para([b("What it shows. "), what]),
    para([b("Why it was created. "), why]),
    para([b("Insights obtained. "), ins]),
  ];
}
const tableCaption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 80 }, children: [run(t, { bold: true, size: 20 })] });

// ---------- numbers
const rs = R.raw_stats, sk = R.skew_kurt, ol = R.outliers, ci = R.ci;
const rm = R.reg_metrics, M = R.metrics, mlp = rm["MLP (proposed)"];
const dsc = R.discrete;
const chi = R.chi2_prev_outlier.table;
const pAfter = chi[1][1] / (chi[1][0] + chi[1][1]), pBase = chi[0][1] / (chi[0][0] + chi[0][1]);
const cm = R.cm;

// ---------- abstract (word count is checked below)
const abstractText =
  "Mobile edge computing offloads image-recognition tasks to nearby servers, and the turnaround time (TAT) of each task decides whether offloading pays off. " +
  "This study analyses 4,000 TAT measurements recorded on four platforms: two MacBook Pro laptops, a virtual machine and a Raspberry Pi. " +
  "After cleaning and feature engineering (24 fields), univariate, distribution-fitting, bivariate and multivariate analyses were performed using ten visualisations. " +
  `Median TAT ranged from ${f(rs.MacBookPro1.median)} s to ${f(rs.RasberryPi.median)} s; every distribution was right-skewed with heavy tails, and slow outliers arrived in bursts (dispersion index ${f(dsc.dispersion, 1)}). ` +
  `A multi-layer perceptron predicting log-TAT reached an MAE of ${f(mlp.mae)} s, but device identity explained most of this accuracy, and a classifier for slow tasks performed near chance (AUC ${f(M.MLP.auc, 2)}).`;
const abstractWords = abstractText.split(/\s+/).length;
console.log("Abstract words:", abstractWords);

// ---------- content
const C = [];

// ===== Title page
C.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 }, children: [run("KPR Institute of Engineering and Technology", { bold: true, size: 32 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [run("Department of Artificial Intelligence and Data Science", { bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "1F3864", space: 8 } }, children: [run("U21ADP05 – Exploratory Data Analysis and Visualization", { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [run("ASSIGNMENT II – CASE STUDY REPORT", { bold: true, size: 26, color: "1F3864" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 300, after: 300 }, children: [run("Exploratory Data Analysis and Visualization of Task Turnaround Times in Mobile Edge Computing, with Deep Learning-Based Prediction", { bold: true, size: 40 })] }),
  spacer(),
  table([3000, 6026], ["Item", "Details"], [
    ["Student Name", ph("[Your Name]")],
    ["Roll No.", ph("[Your Roll Number]")],
    ["Department", "Artificial Intelligence and Data Science"],
    ["Year / Semester / Batch", "III Year / V Semester / 2023 – 2027"],
    ["Academic Year", "2026 – 2027 (Odd Semester)"],
    ["Course Coordinator", "Mr. Rushikesh Kadam"],
    ["Date of Submission", "15 / 10 / 2026"],
    ["GitHub Repository", ph("[Paste your public GitHub link]")],
  ], { left: [0, 1] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ===== 2. Abstract
C.push(h1("1. Abstract"), para(abstractText), para([b("Keywords: "), "exploratory data analysis, mobile edge computing, turnaround time, distribution fitting, multi-layer perceptron, data visualization."]));

// ===== 3. Introduction
C.push(
  h1("2. Introduction and Objectives"),
  h2("2.1 Background and problem statement"),
  para("Mobile devices increasingly delegate compute-heavy work, such as image recognition, to nearby edge servers. Whether a task should be offloaded, and to which server, depends on how long the round trip will take. This round-trip time, from sending the image until the prediction is received, is called the turnaround time (TAT) [6], [7]. A scheduler that understands the typical TAT of each server, how variable it is, and how often it spikes can make far better offloading decisions than one that assumes a fixed latency."),
  para("This case study applies exploratory data analysis (EDA) [1] and the seven-stage visualization process [2] to a real TAT data set measured on four different edge servers, and then asks a practical question: can a deep learning model predict the TAT of the next task, or warn about a slow one, from the recent history of that server?"),
  h2("2.2 Objectives"),
  num("Load, parse, clean and transform the raw measurements, and engineer a feature set of at least 15 fields."),
  num("Carry out univariate, distribution-based, bivariate and multivariate analysis, supported by at least five well-labelled visualizations that pass through all seven visualization stages."),
  num("Identify and test probability distributions (continuous and discrete) that describe TAT and the occurrence of slow tasks."),
  num("Build and tune a multi-layer perceptron (MLP) for TAT prediction (regression) and for slow-task warning (classification), and visualize its performance honestly against simple baselines."),
  num("Summarise findings, limitations and future scope."),
  h2("2.3 Related work"),
  para("Tukey's exploratory approach [1] stresses looking at the data, through summaries and plots, before modelling; Fry's seven stages (acquire, parse, filter, mine, represent, refine, interact) [2] provide a repeatable workflow for visualization. Surveys of mobile edge computing [6] and edge computing challenges [7] identify latency, and especially its variability, as the central quantity for computation offloading. For modelling, feed-forward networks trained with the Adam optimiser [3], [8] are a standard starting point for tabular prediction; scikit-learn [4], Matplotlib [5] and Seaborn [9] were used for implementation and plotting."),
);

// ===== 4. Dataset
C.push(
  h1("3. Dataset Description"),
  h2("3.1 Source and collection"),
  para(["The data set is the ", i("Execution Times for Image Recognition Task Offloading in Mobile Edge Computing"), " collection supplied with the assignment (folder TATDescriptionDataset, accompanied by a description PDF). An edge mobile node sent one image per second to an edge server; the server ran an image-recognition model (the Python imageai.Prediction library) and returned the prediction. The TAT is the time from transferring the image until the result is back at the mobile node. The experiment was run separately on four servers on 26 November 2020, each for between about 18 and 34 minutes at different times of day (13:05 to 17:03)."]),
  tableCaption("Table 1. Edge servers and collection windows"),
  table([2000, 1000, 3000, 3026], ["Edge server", "Records", "Collection window (26 Nov 2020)", "Duration"],
    DEV.map((d) => {
      const v = R.devices[d]; const s = new Date(v.start.replace(" ", "T")), e = new Date(v.end.replace(" ", "T"));
      const mins = ((e - s) / 60000).toFixed(1);
      return [d, v.n, v.start.slice(11) + " – " + v.end.slice(11), mins + " min"];
    })),
  spacer(),
  h2("3.2 Fields and the 15-feature requirement"),
  para(["Each raw file holds only two columns: ", b("Time"), " (a timestamp with 1-second resolution) and ", b("Execution Time"), " (TAT in seconds). The four files were combined (4,000 rows) with an added ", b("device"), " column. Because the assignment asks for at least 15 fields, additional fields were ", i("derived"), " from the timestamps and from each server's own history. The engineered table has ", b(String(R.n_fields)), " columns, of which ", b(String(R.n_model_features)), " are model inputs. This is a limitation of the source data and it is stated openly: the engineered fields describe the same measurement process rather than independent physical sensors (see Section 8.2)."]),
  tableCaption("Table 2. Fields of the engineered data set"),
  table([2000, 1900, 5126], ["Field", "Type", "Description"], [
    ["time_str, time", "Raw / datetime", "Original timestamp string and parsed datetime"],
    ["tat", "Numeric (s)", "Turnaround time – the main measurement"],
    ["device", "Categorical", "Edge server: MacBookPro1, MacBookPro2, VM, RasberryPi"],
    ["log_tat", "Derived", "Natural log of TAT (reduces right skew)"],
    ["gap_s", "Derived", "Seconds since the previous task of the same server"],
    ["elapsed_s, task_index", "Derived", "Seconds since start; running task number"],
    ["hour, minute, second", "Derived", "Time-of-day components"],
    ["lag1, lag2, lag3", "Derived", "TAT of the previous 1, 2 and 3 tasks"],
    ["diff1", "Derived", "lag1 − lag2 (most recent change)"],
    ["roll_mean5, roll_std5, roll_max5", "Derived", "Mean / std / max of the previous 5 TATs (shifted, no leakage)"],
    ["roll_mean20", "Derived", "Mean of the previous 20 TATs"],
    ["prev_gap", "Derived", "gap_s of the previous task"],
    ["lag1_ratio", "Derived", "lag1 divided by roll_mean20 (relative spike indicator)"],
    ["iqr_outlier", "Binary", "1.5 × IQR outlier flag (per device)"],
    ["split", "Categorical", "train / val / test (chronological per device)"],
    ["is_slow", "Binary target", "TAT above the device's own training 90th percentile"],
  ], { left: [0, 2] }),
  spacer(),
  h2("3.3 Basic statistics"),
  tableCaption("Table 3. Descriptive statistics of TAT (seconds)"),
  table([1500, 1000, 1000, 1000, 1000, 1000, 1000, 1526],
    ["Server", "Mean", "Std", "Median", "Max", "Skewness", "IQR outliers", "95% CI of mean"],
    DEV.map((d) => [d, f(rs[d].mean), f(rs[d].std), f(rs[d].median), f(rs[d].max, 2), f(sk[d].skew, 1),
      `${ol[d].sum} (${pct(ol[d].mean)})`, `${f(ci[d][0])}–${f(ci[d][1])}`])),
  spacer(),
  para(`Mean TAT increases from ${f(rs.MacBookPro1.mean)} s (MacBookPro1) to ${f(rs.RasberryPi.mean)} s (Raspberry Pi), roughly a ten-fold difference. Every server has a mean above its median and skewness between ${f(sk.MacBookPro2.skew, 1)} and ${f(sk.VM.skew, 1)}, which already signals right-skewed, heavy-tailed behaviour. The virtual machine's maximum of ${f(rs.VM.max, 1)} s is about 50 times its median.`),
);

// ===== 5. EDA & Preprocessing
C.push(
  h1("4. EDA and Preprocessing"),
  h2("4.1 Methods used"),
  tableCaption("Table 4. Data-quality checks and treatments"),
  table([2200, 3100, 3726], ["Task", "Finding", "Treatment"], [
    ["Missing values", `${R.raw_missing} missing values in the 4,000 raw rows`, "None needed"],
    ["Duplicates", `${R.raw_dup_rows} duplicate rows; ${R.raw_dup_ts} duplicate timestamps within a server`, "Duplicate check kept in the pipeline; nothing removed"],
    ["Data types", "Timestamp stored as text (e.g. 'Thu Nov 26 14:47:38 2020')", "Parsed to datetime; each file sorted by time"],
    ["Outliers", `IQR rule flags ${Object.values(ol).reduce((a, c) => a + c.sum, 0)} of 4,000 points (${DEV.map((d) => ol[d].sum).join(" / ")} per server)`, "Flagged, not deleted: they are genuine slow executions that matter for offloading. For modelling, inputs were winsorised at the training 99.5th percentile"],
    ["Skewed values", "Skewness 5 – 31", "Log transform (log TAT for the regression target; log1p for lag / rolling inputs)"],
    ["Feature engineering", "Only 2 raw columns", "22 derived columns (Table 2); the first 5 rows of each server lack history and were dropped → 3,980 rows"],
    ["Categorical encoding", "device (4 levels)", "One-hot encoding (4 dummy columns)"],
    ["Scaling", "Inputs on very different scales", "Z-score standardisation using training-set mean / std only"],
    ["Train / validation / test", "Data is a time series", `Chronological 70 / 15 / 15 split inside each server: ${R.split_counts.train} / ${R.split_counts.val} / ${R.split_counts.test} rows (no shuffling, so no look-ahead)`],
  ], { left: [0, 1, 2] }),
  spacer(),
  h2("4.2 Insights from EDA and pre-processing"),
  bullet([b("Clean but not stationary. "), "There are no missing values or duplicates, yet behaviour changes inside a run: MacBookPro2 turns slower and spikier after roughly task 830, and the Raspberry Pi has a cluster of spikes between tasks 430 and 780 (Fig. 3). The share of 'slow' tasks therefore rises from " + pct(R.slow_rate.train) + " in training to " + pct(R.slow_rate.test) + " in the test period."]),
  bullet([b("Target leakage was found and removed. "), "The timestamp is written when the result returns, so the gap between two log entries contains the current task's own TAT. In a first model run this feature appeared as the most important predictor; it was then excluded and only the previous gap (prev_gap) is used."]),
  bullet([b("Timestamps are coarse. "), "With 1-second resolution, gaps are mostly 1 s or 2 s, so gap features are only a rough signal."]),
  bullet([b("Device identity dominates. "), "TAT levels differ so much between servers that any pooled statistic mostly reflects 'which server', which is why per-device and within-device views are used below."]),
);

// ===== 6. Visualization
C.push(
  h1("5. Data Visualization"),
  para(`All visuals were produced in Python (Matplotlib and Seaborn) and cover ten figures, above the minimum of five. The data is numerical (time-series), so histograms, box plots, line plots, correlation heat-maps, scatter plots, a pair plot and ECDFs are used, as suggested by the assignment.`),
  h2("5.1 The seven stages of visualization"),
  tableCaption("Table 5. How each stage of Fry's process was applied"),
  table([1500, 3600, 3926], ["Stage", "What was done", "Where"], [
    ["1. Acquire", "Four CSV files obtained from the supplied archive; description PDF read for context", "Section 3"],
    ["2. Parse", "Timestamps parsed to datetime; files merged with a device label; numeric types checked", "Section 4, Table 4"],
    ["3. Filter", "Duplicate / missing check; first rows lacking history removed; outliers flagged; extreme values winsorised for modelling", "Section 4"],
    ["4. Mine", "Summary statistics, skewness, Spearman correlation, K–S and AIC distribution fitting, Kruskal–Wallis, Levene, chi-square tests", "Sections 5.2 – 5.5"],
    ["5. Represent", "Histogram, box plot, line plot, heat-map, scatter, pair plot, ECDF, bar charts chosen to match each question", "Figs. 1 – 10"],
    ["6. Refine", "Log scales, clipping the x-axis at the 99.5th percentile with the maximum annotated, fixed colour per device, outliers highlighted, legends moved off the data", "All figures"],
    ["7. Interact", "A self-contained interactive HTML explorer (outputs/interactive_tat.html): toggle servers, choose a window, switch log / linear scale", "Section 5.6"],
  ], { left: [0, 1, 2] }),
  spacer(),

  h2("5.2 Univariate analysis"),
  h2("Figure 1 – Histogram with KDE"),
  ...figure("fig1_histograms.png", "Fig. 1  Histogram and kernel density of TAT for each server (x-axis clipped at the 99.5th percentile)"),
  ...viz("The shape of the TAT distribution of each server, with mean (dashed) and median (dotted).",
    "To see centre, spread and skew before choosing any statistical model.",
    `MacBookPro1 is a tight, almost bell-shaped distribution around ${f(rs.MacBookPro1.median)} s (std ${f(rs.MacBookPro1.std)} s). MacBookPro2 is clearly bimodal, with modes near 0.30 s and 0.40 s, so its mean (${f(rs.MacBookPro2.mean)} s) lies between the modes and describes almost no actual task. The Raspberry Pi shows two close peaks near 0.93 s and 1.02 s. The VM concentrates at about 0.41 s with a small low-TAT cluster below 0.25 s. All servers have a right tail, and the VM's single extreme value is ${f(rs.VM.max, 1)} s.`),
  h2("Figure 2 – Box plots"),
  ...figure("fig2_boxplot.png", "Fig. 2  Box plots of TAT by server (log scale)", 480),
  ...viz("Median, interquartile range (IQR) and outliers for every server on a logarithmic scale.",
    "To compare four servers whose TATs differ by an order of magnitude, and to quantify outliers with an objective rule.",
    `The medians are ordered ${f(rs.MacBookPro1.median, 2)} s < ${f(rs.MacBookPro2.median, 2)} s < ${f(rs.VM.median, 2)} s < ${f(rs.RasberryPi.median, 2)} s and the boxes do not overlap except that MacBookPro2 and the VM are close. The VM has the narrowest box yet the most outliers (${ol.VM.sum}, ${pct(ol.VM.mean)}), meaning it is stable most of the time but unpredictable in the tail. MacBookPro1 is the most consistent (${ol.MacBookPro1.sum} outliers, ${pct(ol.MacBookPro1.mean)}). The Kruskal–Wallis test confirms that the servers differ (H = ${f(R.kruskal.H, 0)}, p < 0.001).`),
  h2("Figure 3 – TAT over time"),
  ...figure("fig3_timeseries.png", "Fig. 3  TAT versus task sequence for each server (black dots = IQR outliers)", 540),
  ...viz("The raw TAT of every task in order, with outliers marked.",
    "Distribution plots hide order; a line plot reveals trends, regimes and bursts.",
    "There is no trend in the baseline level, but outliers are not spread evenly. They cluster: the Raspberry Pi has a dense group between tasks 430 and 780, MacBookPro2 becomes erratic after task 830, and the VM shows one 20.8 s freeze near task 600. This clustering is quantified in Section 5.3."),
  h2("Figure 7 – Empirical cumulative distribution"),
  ...figure("fig7_ecdf.png", "Fig. 7  Empirical CDF of TAT for each server (log x-axis)", 480),
  ...viz("The probability that a task finishes within a given time, for each server.",
    "An ECDF reads percentiles directly, which is what a service-level target such as 'finish within x seconds' needs.",
    "The curves of MacBookPro1 and the Raspberry Pi are completely separated (the slowest MacBookPro1 task, 0.23 s, is faster than the fastest Raspberry Pi task, 0.85 s), while MacBookPro2 and the VM overlap in the 0.27 – 0.5 s range. The MacBookPro2 curve rises in two steps (about 60% of tasks finish by 0.31 s, the rest by about 0.45 s), which is the bimodality seen in Fig. 1. The VM curve has a small early shelf (about 3% of tasks finish between 0.17 s and 0.3 s) and reaches 100% only at its 20.8 s maximum, which is why a high percentile (for example the 99th) is a far better planning figure than the maximum."),

  h2("5.3 Discrete and continuous probability distributions"),
  h2("Figure 8 – Continuous distribution fitting"),
  ...figure("fig8_distfit.png", "Fig. 8  Normal, log-normal and gamma fits to the central 99% of TAT (top) and log-normal Q–Q plots of all data (bottom)", 560),
  ...viz("Fitted normal, log-normal and gamma densities against each server's histogram, and Q–Q plots of log(TAT) against a normal distribution.",
    "To identify which continuous model describes TAT, using AIC (lower is better) and the Kolmogorov–Smirnov (K–S) test.",
    `Fits used the central 99% of values so that a single freeze does not distort the parameters. For MacBookPro1 the log-normal is the best model (AIC ${f(R.fit.find((r) => r.device === "MacBookPro1" && r.dist === "lognorm").aic, 0)}, K–S p = ${f(R.fit.find((r) => r.device === "MacBookPro1" && r.dist === "lognorm").ks_p, 2)}), so it is not rejected at the 5% level. For MacBookPro2 and the Raspberry Pi the log-normal also has the lowest AIC, and for the VM the normal has, but the K–S test rejects every candidate (p < 0.001) because those distributions are multi-modal. The Q–Q plots are straight in the middle but bend upward at the right: the upper tail is heavier than log-normal. TAT is therefore better described as a mixture of regimes plus a heavy tail than by a single textbook distribution.`),
  h2("Figure 9 – Discrete distribution of outlier counts"),
  ...figure("fig9_discrete.png", "Fig. 9  Number of outliers per block of 50 tasks versus Binomial and Poisson models", 480),
  ...viz(`The observed distribution of IQR outliers in ${dsc.blocks} non-overlapping blocks of 50 tasks, with fitted Binomial(50, p = ${f(dsc.p_hat, 3)}) bars and a Poisson(λ = ${f(dsc.lam, 2)}) curve.`,
    "If slow tasks were independent random events, their count per block would follow a Binomial (or approximately Poisson) law; checking this shows whether spikes are random or bursty.",
    `The mean count is ${f(dsc.lam, 2)} per block but the variance is ${f(dsc.var, 2)}, a dispersion index of ${f(dsc.dispersion, 1)} (it would be about 1 for a Poisson process and slightly below 1 for a Binomial). Both models therefore underestimate the number of blocks with no outliers and with many outliers: the data are over-dispersed. A chi-square test supports this: after an outlier, the chance that the next task is also an outlier is ${pct(pAfter)}, against ${pct(pBase)} otherwise (χ² p < 0.001). Slow tasks arrive in bursts, so a negative-binomial or clustered-event model would be more appropriate.`),

  h2("5.4 Bivariate and multivariate analysis"),
  h2("Figure 4 – Correlation heat-map"),
  ...figure("fig4_heatmap.png", "Fig. 4  Spearman correlation among TAT and engineered features (all servers pooled)", 500),
  ...viz("Spearman rank correlations among TAT, its lags, rolling statistics, gaps and task index.",
    "To find which engineered features are related to TAT, and which are redundant. Spearman is used because the data are skewed.",
    `TAT correlates strongly with lag1 (ρ = ${f(R.corr_tat.lag1, 2)}), roll_mean5 (${f(R.corr_tat.roll_mean5, 2)}) and roll_mean20 (${f(R.corr_tat.roll_mean20, 2)}), and these features are almost perfectly correlated with each other (0.92 – 0.97), i.e. redundant. The relative or change-type features (diff1, lag1_ratio) and task_index are uncorrelated with TAT (|ρ| ≤ 0.05), so there is no time trend. Important caution: pooling the four servers inflates these correlations, because most of the variance is between servers. Within a single server the lag-1 correlation is only ${DEV.map((d) => d + " " + f(R.lag1_spearman[d], 2)).join(", ")}. The gap_s correlation (${f(R.corr_tat.gap_s, 2)}) is partly the leakage described in Section 4.2.`),
  h2("Figure 5 – Scatter plot of current vs previous TAT"),
  ...figure("fig5_lagscatter.png", "Fig. 5  Current TAT against previous TAT for each server (log–log scale)", 480),
  ...viz("Each point is a task: previous TAT on the x-axis, current TAT on the y-axis.",
    "To test whether a slow task tends to be followed by another slow task, which is the basis for any history-based prediction.",
    "The clouds are round, not diagonal. Within a server, successive TATs are almost independent (Spearman 0.05 – 0.18), with MacBookPro2 showing the strongest dependence, consistent with its slow regime after task 830. A few points along the edges are the burst events discussed above."),
  h2("Figure 6 – Pair plot"),
  ...figure("fig6_pairplot.png", "Fig. 6  Pair plot of log TAT and four engineered features, coloured by server", 540),
  ...viz("All pairwise scatter plots (and per-server densities on the diagonal) for log TAT, gap, rolling mean, rolling standard deviation and the lag-1 ratio.",
    "To examine several variables jointly and see how the servers separate.",
    "The servers form four distinct clusters in log TAT and in the rolling mean, so those variables mostly encode device identity. Inside a cluster there is little slope, supporting the weak within-server dependence. The rolling standard deviation is larger for slower servers and during burst periods, indicating that variability scales with the level, as in a multiplicative (log-normal-like) process."),
  h2("Figure 10 – Inter-task gap"),
  ...figure("fig10_gap.png", "Fig. 10  Proportion of tasks by gap (seconds) since the previous task", 480),
  ...viz("For each server, the share of tasks that follow the previous task after 1 s, 2 s, 3 s or more.",
    "To understand the pacing of the experiment and whether timing itself is a usable feature.",
    "The Raspberry Pi, whose TAT is about 1 s, is almost always logged 2 s after the previous task (about 92%), while the fast MacBookPro1 is mostly 1 s apart (about 89%). Timestamps have only 1-second resolution, so gaps partly reflect rounding. The gap tracks TAT through the logging process, which is why the current gap cannot be used as a predictor."),

  h2("5.5 Statistical tests"),
  tableCaption("Table 6. Summary of statistical tests"),
  table([2900, 3300, 2826], ["Question", "Test and result", "Conclusion"], [
    ["Do servers differ in TAT?", `Kruskal–Wallis H = ${f(R.kruskal.H, 0)}, p < 0.001`, "Yes – strongly"],
    ["Do servers differ in spread (log TAT)?", `Levene W = ${f(R.levene.W, 1)}, p < 0.001`, "Yes – unequal variance"],
    ["Are outliers independent?", `Chi-square on previous/next outlier: ${pct(pAfter)} vs ${pct(pBase)}, p < 0.001`, "No – they cluster"],
    ["Is count per block Poisson/Binomial?", `Dispersion index ${f(dsc.dispersion, 1)}`, "No – over-dispersed"],
    ["Is TAT normally / log-normally distributed?", "K–S tests on fitted models (Fig. 8)", "Only MacBookPro1 is consistent with log-normal"],
    ["Does the previous gap indicate a slow task?", `Mann–Whitney p = ${f(R.gap_vs_slow.mw_p, 2)}`, "No – prev_gap carries no signal"],
  ], { left: [0, 1, 2] }),
  spacer(),
  h2("5.6 Interactive stage"),
  para("The file outputs/interactive_tat.html is a stand-alone page (no server or internet needed) in which the viewer can switch servers on and off, slide a window across the 1,000 tasks, choose window size and toggle between linear and logarithmic scale. This completes the seventh visualization stage by letting a reader explore bursts and regimes directly."),
);

// ===== 7. DL model
const bg = R.best_hp_reg, bc = R.best_hp;
C.push(
  h1("6. Deep Learning Model"),
  h2("6.1 Problem formulation"),
  bullet([b("Task A (main) – TAT regression: "), "predict log(TAT) of the next task from the server identity and the recent history of that server. This is the quantity an offloading scheduler needs."]),
  bullet([b("Task B (secondary) – slow-task warning: "), `binary classification of whether the next task's TAT exceeds its own server's 90th percentile (thresholds ${DEV.map((d) => d + " " + f(R.thresholds[d], 3) + " s").join(", ")}).`]),
  para(`Inputs are the ${R.n_model_features} features listed in Section 4: prev_gap, lag1–lag3, diff1, roll_mean5, roll_std5, roll_max5, roll_mean20, lag1_ratio, second-of-minute and four one-hot device columns. Current-gap and elapsed-time features were deliberately excluded.`),
  h2("6.2 Feature analysis"),
  para("Section 5 showed that device identity and the recent level (lag and rolling features) carry the signal, that those level features are highly redundant, and that relative-change features are uninformative. A Ridge-regression ablation (Table 10) and permutation importance (Fig. 15) quantify this on the test data."),
  h2("6.3 Architecture and training parameters"),
  tableCaption("Table 7. MLP architecture and training settings"),
  table([3000, 6026], ["Item", "Regression model (Task A)"], [
    ["Framework", "scikit-learn MLPRegressor, trained epoch by epoch with partial_fit so that loss curves can be recorded"],
    ["Architecture", `Input (${R.n_model_features}) → Dense(${bg.hidden[0]}, ReLU) → Output(1, linear); ${R.n_params_reg} trainable parameters`],
    ["Loss", "Mean squared error on log(TAT)"],
    ["Optimiser", `Adam, learning rate ${bg.lr}, mini-batch 64`],
    ["Regularisation", `L2 penalty α = ${bg.alpha}; early stopping on validation loss (patience 12 epochs, maximum 80)`],
    ["Epochs", `${bg.epochs_run} run, best epoch = ${bg.best_epoch} (weights restored)`],
    ["Classifier (Task B)", `Same design: Dense(${bc.hidden[0]}, ReLU) → sigmoid output, cross-entropy loss, lr ${bc.lr}, α ${bc.alpha}, best epoch ${bc.best_epoch}; decision threshold ${f(R.thr_opt, 3)} chosen on the validation set to maximise F1`],
    ["Baselines", "Device median, persistence (previous TAT), 5-task rolling mean, Ridge regression; logistic regression and random forest for Task B"],
  ], { left: [0, 1] }),
  spacer(),
  new Paragraph({ heading: HeadingLevel.HEADING_2, pageBreakBefore: true, children: [new TextRun({ text: "6.4 Hyperparameter tuning", font: FONT })] }),
  para("A basic grid search covered three architectures (one, two and three hidden layers), two learning rates and two L2 penalties (12 settings). Each was trained with early stopping and ranked by validation loss; the test set was not used."),
  tableCaption("Table 8a. Hyperparameter grid for the regression MLP (sorted by validation MSE of log TAT)"),
  table([2200, 1300, 1300, 1200, 1500, 1526], ["Hidden layers", "Learning rate", "L2 (α)", "Epochs", "Val. MSE (log)", "Val. MAE (s)"],
    R.grid_reg.map((r) => [r.hidden.replace(",)", ")"), r.lr, r.alpha, r.epochs, f(r.val_mse_log, 4), f(r.val_mae_s, 4)])),
  spacer(),
  para(`The differences between the best settings are small (validation MSE between ${f(R.grid_reg[0].val_mse_log, 4)} and ${f(R.grid_reg[R.grid_reg.length - 1].val_mse_log, 4)}); the shallow 16-unit network with learning rate 0.01 was selected. Deeper networks did not help, which is expected when there are only about 2,800 training rows and a weak signal.`),
);

// ===== 8. Results
const bestBase = Object.entries(rm);
C.push(
  h1("7. Result Visualization and Interpretation"),
  h2("7.1 Regression results (Task A)"),
  tableCaption("Table 9. Test-set performance for TAT prediction (600 tasks, last 15% of every server)"),
  table([2700, 1100, 1100, 1250, 1300, 1576], ["Model", "MAE (s)", "RMSE (s)", "Median AE (s)", "R² (log TAT)", "R² (seconds)"],
    bestBase.map(([k, v]) => [k, f(v.mae, 4), f(v.rmse, 4), f(v.medae, 4), f(v.r2_log, 3), f(v.r2, 3)]), { left: [0] }),
  spacer(),
  para(["The MLP reaches an MAE of ", b(`${f(mlp.mae, 4)} s`), ` and R² of ${f(mlp.r2_log, 3)} on the log scale. However, the honest comparison is with the baselines: the device-median predictor, which uses no history at all, gets MAE ${f(rm["Device median (train)"].mae, 4)} s and R² ${f(rm["Device median (train)"].r2_log, 3)}, and Ridge regression is marginally better than the MLP (MAE ${f(rm["Ridge regression"].mae, 4)} s). The network is therefore competitive but not superior. Persistence (using the previous TAT) is the worst, because one spike is copied into the next prediction.`]),
  tableCaption("Table 10. Ridge ablation – which information matters (test MAE, seconds)"),
  table([5026, 2000], ["Inputs", "MAE (s)"], Object.entries(R.ablation_mae).map(([k, v]) => [k, f(v, 4)]), { left: [0] }),
  spacer(),
  para("Using device identity alone gives almost the same error as using everything; using history alone (no device label) is about twice as bad. In other words, nearly all the predictable information is 'which server is it', and the recent history adds very little."),
  ...figure("fig11_loss.png", "Fig. 11  Training and validation loss (MSE of log TAT) versus epoch", 460),
  para([b("What it conveys. "), `Both curves fall steeply during the first five epochs and then flatten with small fluctuations; the validation loss reaches its minimum at epoch ${bg.best_epoch} and training stops 12 epochs later. The validation loss lies below the training loss, which is not a sign of a bug: the training window contains the 20.8 s VM freeze and more extreme events, which inflate the squared error, while the validation window is calmer. There is no over-fitting (validation loss does not rise). The early plateau shows that the model learns the per-server level within a few epochs and has little left to learn.`]),
  ...figure("fig12_acc.png", "Fig. 12  R² (accuracy measure for regression) versus epoch", 460),
  para([b("What it conveys. "), `For regression, R² plays the role of accuracy. Training and validation R² rise from about 0.88 – 0.90 in epoch 1 to a plateau near 0.97 by epoch 10, consistent with Fig. 11 (validation above training for the same reason as above). Because R² is computed on the pooled data, this high value is mostly the between-server differences: when R² is computed inside each server on the test period, it is negative (about ${f(mlp.within_device_r2_log, 2)} on average, and similarly negative for every baseline), meaning that within a server the model cannot beat simply knowing that server's average level.`]),
  ...figure("fig13_predactual.png", "Fig. 13  Predicted versus actual TAT on the test set (log–log, dashed line = perfect prediction)", 440),
  para([b("What it conveys. "), `The four servers form four tight clusters on the diagonal – the model knows the level of each server. Inside each cluster the points are horizontal, i.e. the prediction hardly varies while the actual TAT does. The large errors are the high points on the MacBookPro2 row: actual TATs of 0.8 – 1.6 s predicted at about 0.3 – 0.5 s, which are the burst events in its late slow regime.`]),
  ...figure("fig14_errors.png", "Fig. 14  Distribution of prediction errors (left, absolute; right, relative by server)", 560),
  para([b("What it conveys. "), `The residual (actual − predicted) is centred on zero (median ${f(R.res_stats.median, 4)} s, mean ${f(R.res_stats.mean, 4)} s) with a standard deviation of ${f(R.res_stats.std, 3)} s; 90% of errors fall between ${f(R.res_stats.p5, 3)} s and +${f(R.res_stats.p95, 3)} s. The right-hand panel shows the median relative error per server: MacBookPro1 ${pct(R.reg_per_device_rel.MacBookPro1)}, VM ${pct(R.reg_per_device_rel.VM)}, Raspberry Pi ${pct(R.reg_per_device_rel.RasberryPi)}, MacBookPro2 ${pct(R.reg_per_device_rel.MacBookPro2)}. MacBookPro2 has a wide, right-skewed error distribution, caused by its bimodal behaviour and late bursts, which a single smooth model cannot anticipate.`]),
  ...figure("fig15_importance.png", "Fig. 15  Permutation feature importance of the MLP regressor", 480),
  para([b("What it conveys. "), `Shuffling a feature and measuring the rise in error shows what the model relies on. The most important inputs are lag1 (${f(R.importance_reg[0].imp, 2)}), roll_max5 (${f(R.importance_reg[1].imp, 2)}), lag2, roll_mean5 and the device indicators (dev_VM, dev_MacBookPro2). Because lag and rolling features are strongly correlated and mostly encode the server's level, their importance should be read jointly with the device dummies, not as evidence of short-term predictability. diff1 and lag1_ratio contribute almost nothing, which agrees with the heat-map (Fig. 4).`]),
  h2("7.2 Classification results (Task B – slow-task warning)"),
  ...figure("fig16_cls_loss.png", "Fig. 16  Classifier: training and validation loss versus epoch", 420),
  ...figure("fig17_cls_acc.png", "Fig. 17  Classifier: accuracy and validation AUC versus epoch", 420),
  para([b("What they convey. "), `The training loss falls slowly (0.34 to 0.32) while the validation loss stays flat and noisy at 0.40 – 0.42, with its lowest value at epoch ${bc.best_epoch}: the network fits the training data a little better but learns nothing that carries over. Validation AUC fluctuates between 0.50 and 0.56. The accuracy curves are perfectly flat (about 0.90 on training and 0.86 on validation) because at the default 0.5 cut-off the model predicts 'normal' for every task, which is already right for the ~90% of tasks that are not slow; accuracy is therefore a misleading measure here and AUC, precision and recall are reported instead.`]),
  tableCaption("Table 11. Test-set performance for slow-task warning (105 slow tasks out of 600)"),
  table([2600, 1200, 1300, 1000, 1000, 1000, 926], ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "AP"],
    Object.entries(M).map(([k, v]) => [k, f(v.accuracy, 3), f(v.precision, 3), f(v.recall, 3), f(v.f1, 3), f(v.auc, 3), f(v.ap, 3)]), { left: [0] }),
  spacer(),
  ...figure("fig18_cm.png", `Fig. 18  Confusion matrix of the MLP classifier on the test set (threshold ${f(R.thr_opt, 2)})`, 340),
  para([b("What it conveys. "), `Of ${R.test_pos} truly slow tasks the MLP finds ${cm[1][1]} (recall ${pct(cm[1][1] / (cm[1][0] + cm[1][1]), 0)}), but it also raises ${cm[0][1]} false alarms, so only ${pct(cm[1][1] / (cm[0][1] + cm[1][1]), 0)} of its warnings are correct (precision). The low threshold selected on the validation set trades accuracy (${f(M.MLP.accuracy, 2)}) for recall, and is barely better than guessing.`]),
  ...figure("fig19_roc.png", "Fig. 19  ROC curves with AUC for MLP, logistic regression and random forest (test set)", 420),
  para([b("What it conveys. "), `The MLP's ROC curve follows the diagonal (AUC ${f(M.MLP.auc, 3)}), meaning no ability to rank slow tasks above normal ones. Logistic regression (${f(M["Logistic Regression"].auc, 3)}) and random forest (${f(M["Random Forest"].auc, 3)}) are somewhat above chance, and the random forest, which can use burst indicators such as roll_max5, shows a weak signal, but none of the models is good enough for operational use. Per-server AUC for the MLP lies between ${f(Math.min(...Object.values(R.per_device_auc)), 2)} and ${f(Math.max(...Object.values(R.per_device_auc)), 2)}.`]),
  h2("7.3 Overall interpretation"),
  para("The deep learning experiment gives a clear and useful negative result. Knowing which server will execute the task predicts the TAT well; recent history adds only a little; and individual spikes cannot be forecast from history because they are rare, clustered events caused by factors that are not in the data (background load, network congestion, thermal throttling). The model is therefore best used as a calibrated per-server baseline, with the burst finding in Section 5.3 used as a risk rule."),
);

// ===== 9. Conclusion
C.push(
  h1("8. Conclusion and Future Scope"),
  h2("8.1 What was learned"),
  bullet([b("About the data. "), `Four servers produce non-overlapping TAT levels (median ${f(rs.MacBookPro1.median, 2)} – ${f(rs.RasberryPi.median, 2)} s). All distributions are right-skewed with heavy tails; only MacBookPro1 is consistent with a log-normal law, while the others are multi-modal.`]),
  bullet([b("Key patterns. "), `Slow outliers are bursty, not random: dispersion index ${f(dsc.dispersion, 1)}, and the probability of an outlier is ${pct(pAfter)} right after one versus ${pct(pBase)} otherwise. The behaviour of two servers changes within a single run (MacBookPro2 after task 830, Raspberry Pi between tasks 430 and 780).`]),
  bullet([b("Model performance. "), `The MLP predicts TAT with MAE ${f(mlp.mae, 3)} s (R² ${f(mlp.r2_log, 2)} on the log scale) but is on par with a Ridge model and only marginally better than the per-server median. Slow-task classification is near chance (AUC ${f(M.MLP.auc, 2)}).`]),
  bullet([b("Practical recommendation. "), "An offloading scheduler can rely on per-server median TAT for planning, add a safety margin that grows after any recent outlier, and avoid relying on short-term prediction of single spikes."]),
  h2("8.2 Challenges and limitations"),
  bullet("The source files contain only two columns; the 15-field requirement was met with derived features, which describe the same process rather than independent measurements."),
  bullet("Each server was measured at a different time of day, so server and time-of-day effects are confounded."),
  bullet("The data are non-stationary and the test period contains more slow tasks (17.5%) than training (10.1%), which lowers test performance."),
  bullet("A target-leakage trap (the current inter-task gap) had to be detected and removed."),
  bullet("PyTorch and TensorFlow could not be installed in the working environment, so the MLP was built with scikit-learn and trained epoch by epoch. The same architecture can be reproduced in Keras or PyTorch."),
  bullet("Timestamps have 1-second resolution, and a single freeze (20.8 s) dominates the VM's variance."),
  h2("8.3 Future scope"),
  bullet([b("Richer features: "), "CPU and memory load, network round-trip time and bandwidth, image size and resolution, queue length and device temperature, which are the likely causes of spikes."]),
  bullet([b("Better data collection: "), "interleave servers in time, log both start and end timestamps with millisecond resolution, and collect longer runs under controlled load."]),
  bullet([b("Sequence and probabilistic models: "), "LSTM / GRU or Transformer models on longer windows; quantile or mixture-density networks to predict the 90th / 99th percentile instead of the mean."]),
  bullet([b("Burst modelling: "), "self-exciting (Hawkes) or negative-binomial models, and change-point detection for regime shifts such as MacBookPro2's late slow phase."]),
  bullet([b("Online learning and drift detection: "), "update the predictor as conditions change and trigger a retrain when error rises."]),
);

// ===== 10. References
C.push(
  h1("9. References"),
  ...[
    "J. W. Tukey, Exploratory Data Analysis. Reading, MA, USA: Addison-Wesley, 1977.",
    "B. Fry, Visualizing Data: Exploring and Explaining Data with the Processing Environment. Sebastopol, CA, USA: O'Reilly Media, 2007.",
    "I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA, USA: MIT Press, 2016.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "J. D. Hunter, \"Matplotlib: A 2D graphics environment,\" Computing in Science & Engineering, vol. 9, no. 3, pp. 90–95, 2007.",
    "Y. Mao, C. You, J. Zhang, K. Huang, and K. B. Letaief, \"A survey on mobile edge computing: The communication perspective,\" IEEE Communications Surveys & Tutorials, vol. 19, no. 4, pp. 2322–2358, 2017.",
    "W. Shi, J. Cao, Q. Zhang, Y. Li, and L. Xu, \"Edge computing: Vision and challenges,\" IEEE Internet of Things Journal, vol. 3, no. 5, pp. 637–646, 2016.",
    "D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" in Proc. 3rd International Conference on Learning Representations (ICLR), 2015. arXiv:1412.6980.",
    "M. L. Waskom, \"seaborn: statistical data visualization,\" Journal of Open Source Software, vol. 6, no. 60, p. 3021, 2021.",
    "Execution Times for Image Recognition Task Offloading in Mobile Edge Computing (TAT data set and experimental description), supplied with the course assignment, 2020.",
  ].map((t, k) => new Paragraph({ spacing: { after: 80, line: 276 }, indent: { left: 540, hanging: 540 }, children: [run(`[${k + 1}]  `), run(t)] })),
);

// ===== 11. Appendix
C.push(
  new Paragraph({ children: [new PageBreak()] }),
  h1("10. Appendix – Code Section"),
  h2("A. Repository structure (to be uploaded to GitHub)"),
  ...["project/", "├── README.md                         short description, how to run, results", "├── requirements.txt", "├── data/                             four CSV files + description PDF", "├── src/tat_eda_dl.py                 complete, commented pipeline (code below)", "├── figures/                          fig1 … fig21 (PNG)", "├── outputs/                          results.json, engineered dataset, tuning tables, interactive_tat.html", "└── report/                           this report (.docx) and build script"].map((t) =>
    new Paragraph({ spacing: { after: 0 }, children: [run(t, { font: "Courier New", size: 18 })] })),
  spacer(),
  h2("B. Complete Python source (src/tat_eda_dl.py)"),
  para("The script covers data loading, EDA, pre-processing, feature engineering, visualization, model creation and training, hyperparameter search, evaluation and result plots. Run it from the project root with: python src/tat_eda_dl.py"),
  ...CODE.map((ln) => new Paragraph({ spacing: { after: 0, line: 220 }, children: [new TextRun({ text: ln === "" ? " " : ln, font: "Courier New", size: 14 })] })),
);

// ---------- document
const doc = new Document({
  creator: "Student", title: "EDA and Visualization of Edge-Server TAT – Assignment II",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: "2F5496" }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { titlePage: true, page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1300, left: 1440 } } },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "999999", space: 4 } },
        children: [run("U21ADP05 · EDA & Visualization · Assignment II", { size: 18, color: "666666" })] })] }),
      first: new Header({ children: [new Paragraph({ children: [] })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run("Page ", { size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, "report", "EDA_Case_Study_Report.docx");
  fs.writeFileSync(out, buf); console.log("written", out, (buf.length / 1e6).toFixed(1) + " MB");
});
