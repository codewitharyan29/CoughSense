const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  ImageRun, Header, Footer, PageNumber, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, PageBreak
} = require("docx");

const FIG = "reports/figures";
const img = (p) => fs.readFileSync(p);

const ACCENT = "2E9968";
const DARK = "0D1614";
const GREY = "5A6B64";

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 140 },
    children: [new TextRun({ text, bold: true, size: 30, font: "Calibri", color: ACCENT })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 220, after: 100 },
    children: [new TextRun({ text, bold: true, size: 25, font: "Calibri", color: "1A1A1A" })],
  });
}
function body(runs) {
  const children = Array.isArray(runs) ? runs : [new TextRun({ text: runs, size: 22, font: "Calibri" })];
  return new Paragraph({ spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, children });
}
function bullet(text) {
  return new Paragraph({
    bullet: { level: 0 },
    spacing: { after: 60, line: 264 },
    children: [new TextRun({ text, size: 22, font: "Calibri" })],
  });
}
function figure(path, caption, widthPx) {
  const w = widthPx || 520;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: img(path), transformation: { width: w, height: Math.round(w * 0.66) } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 180 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: GREY, font: "Calibri" })],
    }),
  ];
}

// Results table
function resultsTable() {
  const header = ["Model", "CV Accuracy", "Std Dev", "Type"];
  const rows = [
    ["Random Forest", "83.5%", "±6.9%", "Classical ML"],
    ["XGBoost", "84.4%", "±6.9%", "Classical ML (best)"],
    ["CNN (augmented)", "72.8%", "±7.6%", "Deep Learning"],
  ];
  const widths = [2600, 2200, 1800, 2400];
  const mkCell = (text, isHeader, colIdx) => new TableCell({
    width: { size: widths[colIdx], type: WidthType.DXA },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: ACCENT } : { type: ShadingType.CLEAR, fill: "F2F7F5" },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({
      text, bold: isHeader, size: 20, font: "Calibri",
      color: isHeader ? "FFFFFF" : "1A1A1A",
    })] })],
  });
  const tableRows = [
    new TableRow({ tableHeader: true, children: header.map((t, i) => mkCell(t, true, i)) }),
    ...rows.map(r => new TableRow({ children: r.map((t, i) => mkCell(t, false, i)) })),
  ];
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    rows: tableRows,
  });
}

const doc = new Document({
  creator: "Team Auscultate",
  title: "CoughSense Technical Report",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1200, bottom: 1200, left: 1440, right: 1440 } } },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "CoughSense — Technical Report", size: 16, color: GREY, font: "Calibri" })],
        border: { bottom: { color: "D5DED9", space: 6, style: BorderStyle.SINGLE, size: 4 } },
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: GREY, font: "Calibri" })],
      })] }),
    },
    children: [
      // ---- Title block ----
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 400, after: 60 },
        children: [new TextRun({ text: "CoughSense", bold: true, size: 56, font: "Calibri", color: ACCENT })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 200 },
        children: [new TextRun({ text: "AI-Based Cough Acoustic Analysis for Early Respiratory Disease Screening", bold: true, size: 26, font: "Calibri", color: "1A1A1A" })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: [new TextRun({ text: "Team Auscultate", size: 22, font: "Calibri", color: "1A1A1A", bold: true })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: [new TextRun({ text: "Aryan Verma (B.Tech AI)  ·  Arfa Alam (B.Tech Civil Engineering)", size: 20, font: "Calibri", color: GREY })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 260 },
        border: { bottom: { color: "D5DED9", space: 10, style: BorderStyle.SINGLE, size: 4 } },
        children: [new TextRun({ text: "A machine learning and deep learning system for respiratory pre-screening from cough audio", italics: true, size: 20, font: "Calibri", color: GREY })],
      }),

      // ---- Abstract ----
      h1("Abstract"),
      body("Respiratory illnesses such as COVID-19 produce subtle but measurable changes in the acoustic signature of a cough. CoughSense is a screening system that analyzes short cough recordings and flags likely respiratory conditions, providing a fast, low-cost first-pass check requiring only a smartphone microphone. The system deliberately implements and compares two complementary approaches: classical machine learning on hand-crafted audio features (Mel-Frequency Cepstral Coefficients, spectral descriptors, zero-crossing rate) and a convolutional neural network trained on mel-spectrograms. Evaluated with 5-fold cross-validation on a clinically-labeled dataset, an XGBoost classifier achieved 84.4% accuracy, outperforming both a Random Forest (83.5%) and the CNN (72.8%). We use SHAP analysis to explain model decisions, confirming that MFCC-based timbral features carry the strongest diagnostic signal — consistent with clinical intuition about cough quality. The result is a transparent, deployable screening aid rather than a diagnostic black box."),

      // ---- 1. Problem & Motivation ----
      h1("1. Problem and Motivation"),
      body("Access to respiratory screening remains limited in low-resource settings: clinics are distant, tests cost money, and results take time. Yet the cough itself carries information. Clinicians have long used cough character — wet versus dry, productive versus barking — as a diagnostic cue. If a machine can learn those same acoustic patterns, screening becomes as accessible as a phone call."),
      body("CoughSense targets this gap. It is not intended to replace a clinical diagnosis; it is intended to answer a narrower, high-value question: should this person seek further testing? That framing keeps the tool honest and clinically defensible while still delivering real utility."),
      body([
        new TextRun({ text: "Why this problem is a strong project choice: ", bold: true, size: 22, font: "Calibri" }),
        new TextRun({ text: "audio biomarkers are substantially less explored than image or text classification at the student-project level, despite solid published clinical grounding. This combination — real-world relevance, an underexplored modality, and a defensible scope — is what makes the work stand out.", size: 22, font: "Calibri" }),
      ]),

      // ---- 2. Dataset ----
      h1("2. Dataset"),
      body("We use the Virufy clinical dataset: cough recordings collected in a hospital setting with confirmed COVID-19 status labels. After segmenting to isolated cough events, the working set comprises 121 clips — 48 COVID-positive and 73 healthy. This is a small dataset, and we treat that constraint as a first-class design consideration throughout, rather than hiding it."),
      body([
        new TextRun({ text: "Split strategy. ", bold: true, size: 22, font: "Calibri" }),
        new TextRun({ text: "Files are partitioned into training, validation, and test sets at the file level before any augmentation, ensuring that augmented copies of a clip can never leak across the split boundary — a subtle but critical guard against inflated accuracy.", size: 22, font: "Calibri" }),
      ]),

      // ---- 3. Methodology ----
      h1("3. Methodology"),
      h2("3.1 Feature Extraction"),
      body("Each clip is resampled to 22.05 kHz and standardized to a fixed duration. Two parallel representations are then computed:"),
      bullet("Statistical feature vector (for classical ML): 13 MFCCs with means and standard deviations, plus spectral centroid, spectral rolloff, spectral bandwidth, zero-crossing rate, RMS energy, and chroma — 37 features total."),
      bullet("Log-scaled mel-spectrogram (for deep learning): a 128 × 87 time-frequency image capturing the raw acoustic texture the CNN learns from directly."),

      h2("3.2 Models"),
      bullet("Random Forest — 300 trees, balanced class weights to handle the 40/60 class skew; interpretable and fast."),
      bullet("XGBoost — gradient-boosted trees, the strongest performer, with tuned depth and learning rate."),
      bullet("Convolutional Neural Network — three convolutional blocks with batch normalization and adaptive pooling, deliberately compact to resist overfitting on limited data."),

      h2("3.3 Data Augmentation"),
      body("To give the CNN more to learn from, the training split alone is expanded four-fold using waveform-level augmentation: additive noise, pitch shifting, and time shifting. Each simulates realistic real-world variation (different microphones, voices, and timing) without altering the underlying label. Validation and test sets remain untouched originals."),

      new Paragraph({ children: [new PageBreak()] }),

      // ---- 4. Results ----
      h1("4. Results"),
      body("All models are evaluated with 5-fold stratified cross-validation. Reporting the mean and standard deviation across folds — rather than a single split — is essential at this dataset size, where one unlucky fold can swing accuracy by 15–20 points."),
      resultsTable(),
      new Paragraph({ spacing: { after: 160 }, children: [] }),
      body([
        new TextRun({ text: "XGBoost is the clear winner ", bold: true, size: 22, font: "Calibri" }),
        new TextRun({ text: "at 84.4% ± 6.9%. The tree-based models outperform the CNN by roughly 11 points — a result we do not hide but explain in Section 5.", size: 22, font: "Calibri" }),
      ]),

      ...figure(`${FIG}/roc_curves.png`, "Figure 1. ROC curves. Both tree models show strong class discrimination, with XGBoost achieving the highest area under the curve.", 440),
      ...figure(`${FIG}/confusion_matrices.png`, "Figure 2. Confusion matrices on the held-out test split. Both models correctly identify all healthy samples; the few errors are COVID cases predicted as healthy.", 520),

      // ---- 5. Why the CNN underperforms ----
      h1("5. Interpreting the ML vs DL Gap"),
      body("A convolutional network losing to gradient-boosted trees is not a failure — it is an expected and well-documented consequence of dataset scale. With roughly a hundred training clips, a CNN cannot learn robust spectrogram patterns the way it would given thousands of samples. Hand-crafted MFCC features, by contrast, inject decades of audio-engineering domain knowledge, letting tree models generalize from far less data."),
      body("Augmentation narrowed the CNN's variance (its fold-to-fold consistency improved) but could not manufacture the underlying diversity a larger corpus would provide. This tradeoff — classical ML dominating on small structured problems, deep learning pulling ahead only at scale — is one of the most practically important lessons in applied machine learning, and CoughSense demonstrates it cleanly on real clinical data."),

      // ---- 6. Explainability ----
      h1("6. Explainability"),
      body("A screening tool that cannot explain itself is hard to trust. We apply SHAP (SHapley Additive exPlanations) to the XGBoost model to quantify each feature's contribution to individual predictions."),
      ...figure(`${FIG}/shap_summary.png`, "Figure 3. SHAP summary. Each point is a test sample; horizontal position shows how strongly that feature pushed the prediction toward COVID or healthy. MFCC coefficients dominate.", 460),
      body("The analysis confirms that MFCC-based timbral features — the mathematical encoding of a cough's texture — carry the strongest signal. This aligns with clinical intuition: the wet-versus-dry quality a doctor listens for is precisely what these coefficients capture. The model is learning something real, not a spurious artifact."),

      // ---- 7. System & Deployment ----
      h1("7. System and Deployment"),
      body("The trained models are served through a FastAPI inference endpoint that accepts an audio upload and returns predictions with confidence scores from both the classical and deep-learning models, alongside an explicit medical disclaimer. A browser-based interface provides two views: a live screening tool (record or upload a cough, see an instant result) and an analytics dashboard visualizing dataset composition, per-fold performance, confusion matrices, and feature importances. The entire stack runs locally with no paid services."),

      // ---- 8. Limitations & Future Work ----
      h1("8. Limitations and Future Work"),
      bullet("Small, single-source dataset — the model has not been validated across populations, recording devices, or other respiratory conditions; results are a proof of concept for the approach, not a clinical-grade claim."),
      bullet("Binary scope — currently COVID-versus-healthy; extending to asthma, bronchitis, and other conditions requires additional labeled audio."),
      bullet("Future directions — larger training corpora to close the CNN gap, multimodal fusion with self-reported symptoms, Grad-CAM visualization of spectrogram regions, and integration with a guidance chatbot that translates a screening result into clear next steps."),

      // ---- 9. Conclusion ----
      h1("9. Conclusion"),
      body("CoughSense demonstrates a complete, honest, and explainable machine-learning pipeline for respiratory pre-screening from cough audio. By implementing classical and deep-learning approaches side by side, rigorously cross-validating them, and explaining the winning model with SHAP, the project delivers not just a working prototype but a clear scientific narrative: on small clinical datasets, well-engineered features and gradient-boosted trees remain the pragmatic choice, and transparency is as important as accuracy. The system is deployable today and designed to scale toward genuine clinical utility."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("reports/CoughSense_Technical_Report.docx", buf);
  console.log("Report created");
});
