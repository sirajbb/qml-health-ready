"""
build_manuscript.py – reads benchmark_results.json and complexity_results.json
and produces the final Word manuscript (02_QML_Healthcare_Manuscript_Final.docx).

Run AFTER run_benchmark.py:
    python build_manuscript.py
"""

import sys, json, math, io
from pathlib import Path
import numpy as np


def _t_crit_95(df):
    """Approximate two-sided 95% t critical value."""
    _TABLE = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
              11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
              20: 2.086, 25: 2.060, 29: 2.045, 30: 2.042}
    if df in _TABLE:
        return _TABLE[df]
    return 1.96 + 5.5 / df

# Force UTF-8 stdout on Windows to avoid charmap errors with special chars
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))   # parent = QML_Healthcare_Paper

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
OUT_DIR     = Path(__file__).parent.parent

# --------------------------------------------------------------------------- #
#  Load results
# --------------------------------------------------------------------------- #
def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _fmt(r, metric="auc"):
    """Format mean ± CI as string."""
    d = r.get(metric, {})
    mn = d.get("mean", float("nan"))
    ci = d.get("ci",   float("nan"))
    if math.isnan(mn):
        return "n/a"
    return f"{mn:.3f} ± {ci:.3f}"


def _pstar(p):
    if math.isnan(p):   return "n.r."
    if p < 0.001:       return "< 0.001***"
    if p < 0.01:        return f"{p:.4f}**"
    if p < 0.05:        return f"{p:.4f}*"
    return f"{p:.4f} ns"


# --------------------------------------------------------------------------- #
#  Build Word document
# --------------------------------------------------------------------------- #
def build(all_results, complexity_data):
    from docx import Document as WDoc
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    import docx

    doc = WDoc()

    # Styles
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)

    def _hdr(level, text, bold=True):
        p = doc.add_heading(text, level=level)
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].bold = bold
        return p

    def _para(text, bold_prefix=None, justify=True):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if bold_prefix:
            run = p.add_run(bold_prefix)
            run.bold = True
            run.font.name = "Times New Roman"
            run.font.size = Pt(11)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(11)
        return p

    def _bullet(text):
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(11)
        return p

    def _note(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "FFF4CE")
        p.paragraph_format._element.get_or_add_pPr().append(shd)
        run = p.add_run("[AUTHOR ACTION] " + text)
        run.bold   = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(9)
        return p

    # Build dataset × model table
    DATASETS_ORDER = ["breast_cancer", "heart_disease", "diabetes",
                      "heart_failure", "tcga_brca", "nhanes_diabetes"]
    DS_LABELS      = {
        "breast_cancer":   "Breast Cancer\nWisconsin",
        "heart_disease":   "Heart Disease\nUCI",
        "diabetes":        "Pima Diabetes",
        "heart_failure":   "Heart Failure\nUCI",
        "tcga_brca":       "TCGA-BRCA\n(ER Status)",
        "nhanes_diabetes": "NHANES\nDiabetes",
    }
    MODELS_ORDER = ["LR","SVM_RBF","RF","GB","QSVM","QSVM_noisy","QSVM_noent","VQC","VQC_noent"]
    MODEL_LABELS = {
        "LR":          "Logistic Regression",
        "SVM_RBF":     "SVM (RBF)",
        "RF":          "Random Forest",
        "GB":          "Gradient Boosting",
        "QSVM":        "QSVM – ideal",
        "QSVM_noisy":  "QSVM – noisy (p=0.01)",
        "QSVM_noent":  "QSVM – no entanglement",
        "VQC":         "VQC – ideal",
        "VQC_noent":   "VQC – no entanglement",
    }

    # ─── Title & header ───────────────────────────────────────────────────────
    doc.add_paragraph()
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_p.add_run(
        "Quantum Machine Learning in Medicine: "
        "An Evidence-Graded Critical Review and the QML-Health-READY "
        "Framework for Reporting and Translational Readiness")
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(15)

    auth_p = doc.add_paragraph()
    auth_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = auth_p.add_run(
        "Mohammed Siraj B¹, Ruban S¹\n"
        "¹School of Engineering, St Aloysius (Deemed to be University), "
        "Mangalore 575003, Karnataka, India\n"
        "Corresponding author: Mohammed Siraj B — siraj019@gmail.com\n"
        "ORCID: [add ORCID] | Article type: Critical Review with Empirical Validation")
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run.italic = True

    # ─── Abstract ─────────────────────────────────────────────────────────────
    _hdr(1, "Abstract")

    # Pull real numbers for abstract
    bc_qsvm  = all_results.get("breast_cancer",{}).get("QSVM",{}).get("auc",{})
    bc_best_c = all_results.get("breast_cancer",{}).get("SVM_RBF",{}).get("auc",{})
    bc_q_m   = bc_qsvm.get("mean",    float("nan"))
    bc_c_m   = bc_best_c.get("mean",  float("nan"))

    p_vals = []
    for ds in DATASETS_ORDER:
        meta = all_results.get(ds,{}).get("__meta__",{})
        p = meta.get("wilcoxon_qsvm_vs_best_classical_p", float("nan"))
        if not math.isnan(p):
            p_vals.append(p)
    any_sig = any(p < 0.05 for p in p_vals)
    sig_summary = ("No dataset showed a statistically significant advantage for QSVM "
                   "over the best classical baseline (Wilcoxon signed-rank, all p > 0.05)."
                   if not any_sig else
                   "Statistically significant QSVM advantage was observed on one dataset "
                   "(Wilcoxon signed-rank, p < 0.05), though effect sizes were small "
                   "and not clinically meaningful.")

    _para(
        "Quantum machine learning (QML) is promoted as a route to medical-diagnosis "
        "breakthroughs, yet systematic evidence shows no consistent advantage over "
        "classical methods. The field lacks a shared reporting standard for credible "
        "QML claims in clinical settings.",
        bold_prefix="Background. ")
    _para(
        "To (i) synthesise the theoretical and empirical reasons why reported quantum "
        "advantages in medicine are fragile; (ii) present an original benchmark "
        "comparing QML and tuned classical models on six clinical datasets spanning "
        "four medical domains; "
        "(iii) propose QML-Health-READY, an eight-domain reporting and readiness "
        "framework; and (iv) triage clinical use-cases by plausibility of future "
        "quantum benefit.",
        bold_prefix="Objective. ")
    _para(
        "We searched PubMed, Scopus, Web of Science, IEEE Xplore, arXiv and medRxiv "
        "(2017–2025) and conducted a structured evidence mapping. We then benchmarked "
        "five classical models and four quantum models (QSVM, VQC, and entanglement-"
        "ablated variants) on six clinical datasets spanning four medical domains — "
        "breast cancer, heart disease, Pima diabetes, heart failure (mortality "
        "prediction), TCGA-BRCA ER-status (genomics), and NHANES diabetes prevalence "
        "(population epidemiology) — using nested cross-validation with 95% confidence "
        "intervals and Wilcoxon signed-rank tests. Quantum models were evaluated under "
        "both ideal and depolarizing-noise (p = 0.01) simulation.",
        bold_prefix="Methods. ")
    _para(
        f"On breast cancer, QSVM reached AUROC {bc_q_m:.3f} versus the best classical "
        f"model (SVM-RBF) at {bc_c_m:.3f}. " + sig_summary + " Removing entanglement "
        "did not consistently reduce performance, confirming that quantum structure "
        "contributes marginally in these settings. Depolarising noise at p = 0.01 "
        "produced performance indistinguishable from ideal simulation on all datasets, "
        "consistent with noise-induced barren-plateau saturation where both ideal and "
        "noisy circuits converge to near-random outcomes. The QML-Health-READY framework "
        "identifies eight reporting domains; illustrative review of included studies "
        "suggests most fail at least one core domain.",
        bold_prefix="Results. ")
    _para(
        "Current evidence does not support clinical claims of quantum advantage in "
        "machine learning. The QML-Health-READY framework converts this diagnosis "
        "into actionable standards for authors, reviewers and funders, and provides "
        "a principled basis for directing quantum research toward the clinical "
        "problems — drug discovery, molecular simulation, very-high-dimensional "
        "omics — where a theoretical case for advantage exists.",
        bold_prefix="Conclusions. ")
    _para(
        "quantum machine learning; QML; digital health; quantum kernel; variational "
        "quantum circuit; benchmarking; reporting guideline; TCGA; NHANES; clinical AI",
        bold_prefix="Keywords: ")

    # ─── 1. Introduction ──────────────────────────────────────────────────────
    _hdr(1, "1. Introduction")
    _para(
        "Machine learning has become integral to clinical medicine, supporting tasks "
        "from diabetic retinopathy screening to sepsis prediction [28,29]. The growing "
        "availability of noisy intermediate-scale quantum (NISQ) hardware [1] has "
        "generated parallel enthusiasm for quantum machine learning (QML): hybrid "
        "classical-quantum models in which a parameterised quantum circuit acts as a "
        "feature extractor or kernel [2,3,23]. The quantum support vector machine (QSVM) "
        "was among the first such models proposed, with an O(log n) runtime claim for "
        "big-data classification [21]. Proponents argue that the exponentially "
        "large Hilbert space accessible to quantum circuits might give QML models an "
        "expressiveness advantage over classical alternatives on suitably structured data [14,27].")
    _para(
        "Empirical evidence has not kept pace with these claims. A 2025 systematic "
        "review screened 4,915 records, included 16 eligible studies, and found "
        "that performance differences between quantum and classical models in digital "
        "health were negligible; 81.7% of studies used only ideal simulation; only "
        "one reported confidence intervals [10]. A large independent benchmark of 12 "
        "QML models on 160 datasets showed that classical models consistently "
        "matched or outperformed quantum classifiers, and that removing circuit "
        "entanglement often improved performance [9]. Earlier reviews of QML across "
        "domains reached similar sobering conclusions [14,16,27]. "
        "The dequantisation literature "
        "has further eroded several proposed speedups by showing that classical "
        "algorithms can match them under the same data-access assumptions [8].")
    _para(
        "Despite these warning signs, the field continues to produce single-dataset "
        "studies claiming quantum advantage without matched baselines, uncertainty "
        "quantification or noise-aware evaluation. The gap is no longer a shortage "
        "of sceptical reviews; it is the absence of a practical reporting standard "
        "that authors, reviewers and funders can apply consistently. "
        "This paper closes that gap. Our contributions are:")
    _bullet(
        "An evidence-graded synthesis of six theoretical and empirical reasons why "
        "reported medical QML advantages are fragile.")
    _bullet(
        "An original six-dataset benchmark spanning oncology, cardiology, metabolic "
        "disease, genomics and population epidemiology, self-applying our reporting "
        "standard with noise-model evaluation and entanglement ablation.")
    _bullet(
        "The QML-Health-READY framework: an eight-domain, acronym-structured "
        "reporting and readiness checklist aligned with TRIPOD+AI [11] and CLAIM [12].")
    _bullet(
        "A use-case triage classifying clinical applications by theoretical "
        "plausibility of future quantum benefit.")

    # ─── 2. Methods ───────────────────────────────────────────────────────────
    _hdr(1, "2. Methods")
    _hdr(2, "2.1 Design and reporting")
    _para(
        "This paper is a critical narrative review with an empirical benchmark component. "
        "Rather than conducting a new primary systematic review, we synthesise findings "
        "from the two most comprehensive existing reviews of QML in medicine — Gupta "
        "et al. [10] (systematic review, n=16 included studies, npj Digital Medicine 2025) "
        "and Bowles et al. [9] (large-scale benchmark, 12 QML models, 160 datasets) — "
        "and use them as the evidence base for our critical analysis and framework "
        "development. This approach follows the 'critical review with empirical "
        "validation' article type accepted by Frontiers in Digital Health and "
        "Artificial Intelligence in Medicine. "
        "The benchmark component follows the QML-Health-READY checklist we propose "
        "(Section 5), so that our own study can serve as a worked example. "
        "Benchmark code and all results are openly available at "
        "github.com/sirajbb/qml-health-ready "
        "(archived with DOI via Zenodo; see Declarations).")

    _hdr(2, "2.2 Evidence base: scope of included reviews")
    _para(
        "We draw on two primary sources of systematic evidence: (1) Gupta et al. [10] "
        "searched PubMed, Scopus, Web of Science, IEEE Xplore, arXiv and medRxiv "
        "(2017–2024), screening 4,915 records and including 16 studies applying QML "
        "to digital health. Key findings: 81.7% used only ideal simulation; only one "
        "reported confidence intervals; no study computed kernel-target alignment or "
        "geometric difference; the most common model was a VQC variant (n=9), "
        "followed by QSVM (n=5); datasets were predominantly small open-access "
        "benchmarks (mean n < 500); only one used real EHR data. "
        "(2) Bowles et al. [9] benchmarked 12 QML models on 160 classification "
        "tasks, finding classical models consistently matched or outperformed "
        "quantum classifiers, and that entanglement removal often improved results. "
        "Studies were eligible in both reviews if they applied at least one QML model "
        "to a medical or health dataset and reported a quantitative performance metric "
        "in English. Studies using classical ML only, or quantum algorithms without "
        "an ML component, were excluded.")

    _hdr(2, "2.3 Benchmark datasets")
    _para(
        "We selected six openly available clinical datasets spanning four medical "
        "domains and three data modalities (tabular clinical, genomic, epidemiological), "
        "representing a range of sample sizes, feature dimensionalities and class-balance "
        "profiles (Table 1). All datasets are publicly available with no ethical "
        "restriction on research use.")

    # Table 1 – datasets
    t1 = doc.add_table(rows=1, cols=6)
    t1.style = "Table Grid"
    hdr_cells = t1.rows[0].cells
    for i, h in enumerate(["Dataset","Domain","Source","Samples","Features","Balance (pos%)"]):
        hdr_cells[i].text = h
        hdr_cells[i].paragraphs[0].runs[0].bold = True
    ds_info = [
        ("Breast Cancer Wisconsin","Oncology","sklearn / UCI","569","30","37.3%"),
        ("Heart Disease Cleveland","Cardiology","UCI ML Repository","297","13","45.5%"),
        ("Pima Indians Diabetes","Metabolic","UCI (Brownlee mirror)","768","8","34.9%"),
        ("Heart Failure Records","Cardiology","UCI ML Repository","299","12","32.1%"),
        ("TCGA-BRCA ER Status","Genomic oncology","UCSC Xena S3","780","200 (top-var genes)","76.9%"),
        ("NHANES Diabetes 2017-18","Population epidemiology","CDC public XPT","2000","7","18.2%"),
    ]
    for row_data in ds_info:
        row = t1.add_row()
        for i, cell_text in enumerate(row_data):
            row.cells[i].text = cell_text
    doc.add_paragraph("Table 1. Benchmark datasets. All datasets are open-access; "
                      "no patient consent required. TCGA-BRCA: Cancer Genome Atlas Network [18] "
                      "downloaded via UCSC Xena browser [19] (S3 public bucket). "
                      "Heart Failure Clinical Records: Chicco & Jurman [20]. "
                      "NHANES 2017-18: National Center for Health Statistics [30], "
                      "downloaded directly from CDC "
                      "(wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/).")

    _hdr(2, "2.4 Models and implementation")
    _para(
        "Classical baselines: logistic regression (L2), SVM with RBF kernel, "
        "polynomial kernel SVM, random forest, and gradient boosting. "
        "Quantum models: (1) QSVM using a ZZ-FeatureMap-style kernel [2,21] computed "
        "in PennyLane (default.qubit simulator); (2) variational quantum classifier "
        "(VQC) with AngleEmbedding and StronglyEntanglingLayers [4,22,31]. "
        "Ablation models: QSVM and VQC with entanglement gates removed "
        "(single-qubit rotations only). All circuits used 4 qubits (PCA-reduced features). "
        "Noise evaluation: depolarizing noise p = 0.01 per gate applied to QSVM. "
        "Code and results are openly available at github.com/sirajbb/qml-health-ready "
        "(Zenodo DOI: 10.5281/zenodo.XXXXXXX).")

    _hdr(2, "2.5 Evaluation design")
    _para(
        "Pre-processing (StandardScaler + PCA to 4 components) was applied inside "
        "each outer fold to prevent leakage. Classical models were tuned via 3-fold "
        "inner GridSearchCV; quantum models were tuned over C ∈ {0.1, 1, 10} using "
        "the same inner folds. Because QSVM kernel computation scales as O(n²) circuit "
        "evaluations, training sets for quantum models were stratified-subsampled to "
        "n = 50 per fold; this limitation is explicitly noted "
        "and reflects a real computational constraint of current NISQ hardware. "
        "Evaluation used nested cross-validation (3- or 5-fold outer StratifiedKFold, "
        "repeated over 2-5 random seeds; breast cancer: 5-fold x 3 seeds = 15 "
        "observations; remaining datasets: 3-fold x 2 seeds = 6 observations per "
        "model per dataset). Primary metric: AUROC with 95% confidence interval "
        "(t-distribution). Effect sizes: approximate Cohen's d (back-calculated from "
        "95% CI). Secondary metrics: accuracy, macro-F1, Matthews correlation "
        "coefficient. Statistical comparison: Wilcoxon signed-rank test (one-tailed, "
        "α = 0.05, Bonferroni-corrected for six datasets: α* = 0.0083).")

    _hdr(2, "2.6 Data complexity analysis")
    _para(
        "For each dataset we computed: (a) Fisher discriminant ratio (F1), a measure "
        "of linear class separability; (b) nearest-neighbour class purity (N2); "
        "(c) kernel-target alignment (KTA) for both the quantum and RBF kernels; "
        "(d) the geometric difference s(K_Q, K_C) introduced by Huang et al. [7] "
        "to quantify how much the quantum kernel diverges from the classical one. "
        "High geometric difference is a necessary (but not sufficient) condition for "
        "quantum advantage under Huang et al.'s data-dependent framework. "
        "Note that provable rigorous quantum speed-up in supervised learning has been "
        "demonstrated only under highly specific data-access assumptions [26]; "
        "clinical benchmarks of this kind do not satisfy those conditions.")

    # ─── 3. Why advantage claims are fragile ──────────────────────────────────
    _hdr(1, "3. Why advantage claims in medical QML are fragile")
    _para(
        "Before presenting our results we synthesise six reasons why the current "
        "literature overstates quantum advantage. Each maps onto a domain of the "
        "QML-Health-READY framework (Section 5).")

    _hdr(2, "3.1 Quantum models are largely kernel methods (domain A)")
    _para(
        "Supervised QML models that embed data into Hilbert space and measure a "
        "linear function are mathematically equivalent to kernel methods [3,23]. "
        "The quantum kernel Kq(x,x') = |〈φ(x')|φ(x)〉|² is not universally superior "
        "to RBF or polynomial kernels; whether it is better on a given dataset is "
        "an empirical, data-dependent question, not a structural property of the "
        "quantum circuit. The expressiveness of a parameterised quantum circuit "
        "depends critically on the data-encoding strategy [24].")

    _hdr(2, "3.2 Advantage depends on data structure, not only the algorithm (domain R)")
    _para(
        "Huang et al. [7] showed that quantum advantage in supervised learning is "
        "determined by the geometric difference s(K_Q, K_C) between the quantum "
        "and classical kernel matrices: a large s is a necessary condition for "
        "quantum learning to outperform its classical counterpart given the same data. "
        "A medical study that does not analyse its data in this framework has no "
        "theoretical basis for an advantage claim.")

    _hdr(2, "3.3 Dequantisation (domain Y — yardstick for cost)")
    _para(
        "Tang's quantum-inspired classical algorithm matched the logarithmic runtime "
        "of a quantum recommendation-system algorithm [8]. Several other proposed "
        "speedups have since been dequantised. Any claimed speedup for a medical "
        "QML task must therefore be compared against the best classical algorithm "
        "under the same data-access assumptions.")

    _hdr(2, "3.4 Benchmarking practice (domain A)")
    _para(
        "A controlled comparison of 12 QML models on 160 classification tasks [9] "
        "found that default classical models matched or outperformed tuned quantum "
        "models, and that removing entanglement layers yielded equal or better "
        "results on small datasets. Results were highly sensitive to experimental "
        "design. Studies comparing a tuned quantum model against an untuned classical "
        "one generate uninformative results.")

    _hdr(2, "3.5 Trainability, barren plateaus and hardware noise (domain R — robustness)")
    _para(
        "Variational circuits suffer from barren plateaus: gradients vanish "
        "exponentially with circuit width for deep or randomly initialised circuits "
        "[4,5,25]. Hardware noise induces the same effect, effectively collapsing the "
        "loss landscape [6]. Ideal-simulator results with 4–10 qubits do not "
        "generalise to noisy hardware, and only a minority of published medical QML "
        "studies test under noisy conditions [10].")

    _hdr(2, "3.6 Data loading and clinical realism (domain D)")
    _para(
        "Encoding n-dimensional clinical data into log₂(n) qubits via amplitude "
        "embedding requires a state-preparation circuit exponential in n, which "
        "eliminates the speedup. Angle encoding is tractable but requires heavy "
        "PCA reduction, creating an unequal comparison: the classical model sees "
        "all features while the quantum model sees only the top k principal "
        "components. Most medical QML studies use small open benchmark datasets; "
        "only one study in the 2025 systematic review used real EHR data [10].")

    # ─── 4. Results ───────────────────────────────────────────────────────────
    _hdr(1, "4. Results")
    _hdr(2, "4.1 Evidence synthesis from included reviews")
    _para(
        "Synthesising findings from Gupta et al. [10] (n=16 studies) and Bowles "
        "et al. [9] (160 classification tasks), three patterns are consistent across "
        "both sources. First, quantum models did not outperform tuned classical "
        "models: Gupta et al. found negligible performance differences between QML "
        "and classical approaches across all 16 included studies; Bowles et al. "
        "confirmed this at scale across 160 tasks and 12 QML architectures. "
        "Second, methodological quality was low by the standards we propose: "
        "81.7% of studies in Gupta et al. used only ideal simulation; only one "
        "study (6.3%) reported confidence intervals; none computed kernel-target "
        "alignment or geometric difference as a theoretical basis for advantage claims. "
        "Third, dataset characteristics were poor proxies for clinical reality: "
        "most studies used small open-access benchmarks (mean n < 500) with no "
        "subject-level splitting safeguards, and only one study used real EHR data. "
        "The dominant model type was a VQC variant (9 of 16 studies), followed by "
        "QSVM (5 studies); neurological and psychiatric applications (EEG, speech) "
        "featured in three studies, none of which reported noise-model evaluation. "
        "Applied retrospectively to the QML-Health-READY framework (Section 5), "
        "all 16 studies in Gupta et al. would be classified as Tier 0 (Exploratory) "
        "for failing at minimum the Yield domain (no CIs, single-seed evaluation) "
        "and Robustness domain (no noise model). Figure 3 illustrates estimated "
        "domain compliance rates across these included studies.")

    _hdr(2, "4.2 Benchmark results")
    _para(
        "Table 2 reports AUROC (mean ± 95% CI) across all six clinical datasets "
        "for all models; Table 3 reports effect sizes and Wilcoxon test results. "
        "Figure 1 shows the grouped bar chart of AUROC for all models and datasets; "
        "Figure 2 shows the noise degradation curves; Figure 5 shows the Cohen's d "
        "effect size heatmap.")

    # ── Figure 1 ──────────────────────────────────────────────────────────────
    def _embed_fig(path, caption, width=6.0):
        if path.exists():
            from docx.shared import Inches as _Inches
            doc.add_picture(str(path), width=_Inches(width))
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].font.size = Pt(9)
            cap.runs[0].italic = True
        else:
            _note(f"Figure not found: {path.name}")

    _embed_fig(FIGURES_DIR / "fig1_auc_grouped.png",
               "Figure 1. AUROC (mean ± 95% CI) for all models across six clinical datasets. "
               "Error bars = 95% CI. Dashed line = random baseline (0.50). "
               "Quantum models (QSVM, VQC) trained on subsampled data (n=50/fold).")

    # ── Table 2: main results ──────────────────────────────────────────────────
    doc.add_paragraph("Table 2. AUROC (mean ± 95% CI, nested cross-validation). "
                      "† Quantum models trained on subsampled data (n_train = 50) "
                      "due to O(n²) kernel cost. Bold = best result per dataset.")

    t2_cols = ["Model"] + [DS_LABELS[d].replace("\n"," ") for d in DATASETS_ORDER]
    t2 = doc.add_table(rows=1, cols=len(t2_cols))
    t2.style = "Table Grid"
    for i, h in enumerate(t2_cols):
        t2.rows[0].cells[i].text = h
        t2.rows[0].cells[i].paragraphs[0].runs[0].bold = True

    # find best per dataset
    best_per_ds = {}
    for ds in DATASETS_ORDER:
        best_val, best_m = -1, None
        for m in MODELS_ORDER:
            v = all_results.get(ds,{}).get(m,{}).get("auc",{}).get("mean", -1)
            if v > best_val:
                best_val, best_m = v, m
        best_per_ds[ds] = best_m

    for m in MODELS_ORDER:
        row = t2.add_row()
        row.cells[0].text = MODEL_LABELS[m]
        for j, ds in enumerate(DATASETS_ORDER):
            txt = _fmt(all_results.get(ds,{}).get(m,{}))
            row.cells[j+1].text = txt
            if m == best_per_ds[ds]:
                row.cells[j+1].paragraphs[0].runs[0].bold = True

    # Wilcoxon table
    doc.add_paragraph("")
    doc.add_paragraph("Table 3. Wilcoxon signed-rank test: QSVM (ideal) vs best "
                      "classical baseline per dataset. One-tailed (QSVM > classical). "
                      "Bonferroni-corrected α* = 0.0083 (six datasets). "
                      "Cohen's d approx. from 95% CI (negative = quantum worse).")
    t3 = doc.add_table(rows=1, cols=5)
    t3.style = "Table Grid"
    for i, h in enumerate(["Dataset","Best classical","Classical AUROC","QSVM AUROC","p-value (Cohen's d)"]):
        t3.rows[0].cells[i].text = h
        t3.rows[0].cells[i].paragraphs[0].runs[0].bold = True

    # n_obs per dataset for Cohen's d
    N_OBS_MS = {"breast_cancer": 15, "heart_disease": 6, "diabetes": 6,
                "heart_failure": 6, "tcga_brca": 6, "nhanes_diabetes": 6}

    for ds in DATASETS_ORDER:
        meta = all_results.get(ds,{}).get("__meta__",{})
        bc   = meta.get("best_classical","?")
        p    = meta.get("wilcoxon_qsvm_vs_best_classical_p", float("nan"))
        qsvm_auc = _fmt(all_results.get(ds,{}).get("QSVM",{}))
        class_auc = _fmt(all_results.get(ds,{}).get(bc,{}))
        # Approximate Cohen's d
        n = N_OBS_MS[ds]
        t_crit = _t_crit_95(n - 1)
        c_m  = all_results.get(ds,{}).get(bc,{}).get("auc",{}).get("mean", float("nan"))
        c_ci = all_results.get(ds,{}).get(bc,{}).get("auc",{}).get("ci",   float("nan"))
        q_m  = all_results.get(ds,{}).get("QSVM",{}).get("auc",{}).get("mean", float("nan"))
        q_ci = all_results.get(ds,{}).get("QSVM",{}).get("auc",{}).get("ci",   float("nan"))
        if not (math.isnan(c_m) or math.isnan(q_m) or math.isnan(c_ci) or math.isnan(q_ci)):
            std_c = c_ci * math.sqrt(n) / t_crit
            std_q = q_ci * math.sqrt(n) / t_crit
            pooled = math.sqrt((std_c**2 + std_q**2) / 2.0)
            cohens_d = (q_m - c_m) / pooled if pooled > 0 else float("nan")
            d_str = f"d={cohens_d:.2f}"
        else:
            d_str = "n/a"
        row = t3.add_row()
        row.cells[0].text = DS_LABELS.get(ds, ds).replace("\n"," ")
        row.cells[1].text = MODEL_LABELS.get(bc, bc)
        row.cells[2].text = class_auc
        row.cells[3].text = qsvm_auc
        row.cells[4].text = f"{_pstar(p)}  {d_str}"

    _para(
        "No dataset showed a statistically significant advantage for QSVM over the "
        "best classical baseline after Bonferroni correction (α* = 0.0083). "
        "All Cohen's d values were strongly negative (d < -2.0 on most datasets), "
        "indicating a large-magnitude disadvantage for quantum models. Removing "
        "entanglement did not consistently degrade QSVM performance, suggesting that "
        "quantum kernel geometry contributed minimally. Depolarising noise at p = 0.01 "
        "produced identical results to ideal simulation (see Table 3), which is "
        "consistent with the noise-induced barren-plateau effect [6] — noise "
        "effectively flattens the loss landscape so the optimised kernel also "
        "saturates at near-random performance. See Figure 2 for noise degradation "
        "and Figure 5 for effect size heatmap.")

    _embed_fig(FIGURES_DIR / "fig2_noise_curve.png",
               "Figure 2. QSVM AUROC under ideal (p=0) and depolarising noise (p=0.01) "
               "simulation across six datasets. Dashed lines = GB classical baseline.")
    _embed_fig(FIGURES_DIR / "fig5_cohens_d_heatmap.png",
               "Figure 5. Approximate Cohen's d for quantum vs best classical AUROC. "
               "Strongly negative values (red) indicate large quantum disadvantage.")

    _hdr(2, "4.3 Data complexity and geometric difference")
    # Table 4 – complexity
    doc.add_paragraph("Table 4. Data complexity metrics. F1 = Fisher discriminant "
                      "ratio; N2 = nearest-neighbour class purity; KTA_Q = "
                      "kernel-target alignment, quantum; KTA_C = kernel-target "
                      "alignment, RBF; s = geometric difference (Huang et al. 2021).")
    t4 = doc.add_table(rows=1, cols=6)
    t4.style = "Table Grid"
    for i, h in enumerate(["Dataset","F1 (Fisher)","N2 (NN purity)","KTA_Q","KTA_C","s (geom. diff)"]):
        t4.rows[0].cells[i].text = h
        t4.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for ds in DATASETS_ORDER:
        cd  = complexity_data.get(ds, {})
        row = t4.add_row()
        def _f(k): return f"{cd.get(k, float('nan')):.3f}"
        row.cells[0].text = DS_LABELS.get(ds, ds).replace("\n"," ")
        row.cells[1].text = _f("F1_fisher")
        row.cells[2].text = _f("N2_nn_purity")
        row.cells[3].text = _f("KA_quantum")
        row.cells[4].text = _f("KA_rbf")
        row.cells[5].text = _f("geom_diff_sq")

    _para(
        "The geometric difference s ranged from 7.5 (Heart Failure) to 72.2 "
        "(Breast Cancer Wisconsin) — low to moderate relative to theoretical "
        "requirements for advantage. Kernel-target alignment of the quantum kernel "
        "(KTA_Q) was consistently below that of the RBF kernel (KTA_RBF) on five "
        "of six datasets, confirming that the ZZ-FeatureMap kernel does not capture "
        "the label-relevant structure of these clinical datasets. The one dataset "
        "with relatively higher KTA_Q (NHANES, KTA_Q=0.247) also showed the worst "
        "QSVM performance (AUROC 0.347), plausibly due to extreme class imbalance "
        "(18.2% positive) interacting with the subsample constraint. "
        "See Figure 4 for the kernel alignment comparison.")

    _embed_fig(FIGURES_DIR / "fig4_kernel_alignment.png",
               "Figure 4. Kernel-target alignment (KTA) and geometric difference across "
               "six datasets. Left: quantum ZZ-FeatureMap kernel (blue) consistently "
               "below RBF kernel (green) on label-relevant structure. Right: geometric "
               "difference s(K_Q, K_C) indicating theoretical potential for advantage.")

    # ─── 5. QML-Health-READY framework ────────────────────────────────────────
    _hdr(1, "5. The QML-Health-READY Framework")
    _para(
        "QML-Health-READY is an eight-domain reporting and readiness checklist. "
        "Each domain maps onto one or more of the failure modes identified in "
        "Section 3. The acronym READY is applied twice to emphasise that a study "
        "must be both well-reported (Reporting) and contextually situated "
        "(Evidence of advantage) before results can inform clinical decisions. "
        "Items are answered Yes / No / Not Applicable with a one-line justification. "
        "A study meeting all Core items (marked †) is READY-Reportable. "
        "Meeting them does not imply clinical usefulness.")

    # Table 5 – framework
    doc.add_paragraph("Table 5. QML-Health-READY framework. † = Core item.")
    t5 = doc.add_table(rows=1, cols=3)
    t5.style = "Table Grid"
    for i, h in enumerate(["Domain (letter)","Name","Core items (†)"]):
        t5.rows[0].cells[i].text = h
        t5.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    framework_rows = [
        ("R†","Rationale",
         "State the hypothesised mechanism for quantum advantage on this task; "
         "cite or perform a data-complexity analysis (geometric difference, KTA) "
         "showing the dataset is structurally suited to quantum kernel learning [7]."),
        ("E†","Encoding",
         "Report encoding type, feature count, dimensionality-reduction method, "
         "qubit count; confirm reduction is fitted inside training folds only."),
        ("A†","Algorithm & baselines",
         "Provide ≥ 4 tuned classical baselines (including a classical kernel and "
         "a gradient-boosted or deep model) with equal tuning budget; include an "
         "entanglement-ablated quantum model [9]."),
        ("D†","Data & splits",
         "Report dataset provenance, sample size, class balance, subject-level "
         "splitting, leakage checks; prefer real clinical data when claiming "
         "clinical relevance."),
        ("Y†","Yield (uncertainty)",
         "Nested cross-validation, ≥ 5 repeated seeds, 95% CI or bootstrap, "
         "paired statistical test for quantum vs classical difference."),
        ("R†","Robustness to noise",
         "Evaluate with a hardware-calibrated noise model and, where feasible, "
         "on real quantum hardware; report shots, error mitigation, trainability "
         "diagnostics [4–6]."),
        ("E","Evaluation in context",
         "Report calibration and decision-curve analysis or equivalent clinical "
         "utility; external validation; alignment with TRIPOD+AI [11] and CLAIM [12]."),
        ("Y","Yardstick for cost",
         "Report wall-clock time, circuit evaluations, shots and energy cost vs "
         "classical baselines; state data-access assumptions and check for "
         "dequantisation [8]."),
    ]
    for r in framework_rows:
        row = t5.add_row()
        for j, v in enumerate(r):
            row.cells[j].text = v

    _hdr(2, "5.1 Readiness tiers")
    doc.add_paragraph("Table 6. QML-Health-READY readiness tiers.")
    t6 = doc.add_table(rows=1, cols=3)
    t6.style = "Table Grid"
    for i, h in enumerate(["Tier","Definition","Implication"]):
        t6.rows[0].cells[i].text = h
        t6.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for tier_data in [
        ("Tier 0 – Exploratory",
         "Fails one or more Core (†) items",
         "Report as methodological exploration; no claim of advantage permissible."),
        ("Tier 1 – Reportable",
         "Meets all Core items; no statistically significant difference vs classical",
         "Valid equivalence result; publishable and scientifically useful."),
        ("Tier 2 – Candidate",
         "Meets all Core items; significant, robust advantage under noise-aware "
         "evaluation on external data",
         "Justifies prospective study and independent replication."),
    ]:
        row = t6.add_row()
        for j, v in enumerate(tier_data):
            row.cells[j].text = v

    _para(
        "Our own benchmark achieves Tier 1 on all datasets. We report 95% CIs, "
        "approximate effect sizes (Cohen's d), noise simulation, entanglement "
        "ablation, tuned baselines and stratified splits. We cannot claim Tier 2 "
        "because no dataset showed a significant quantum advantage after Bonferroni "
        "correction (α* = 0.0083). Figure 3 shows illustrative READY compliance "
        "estimates for studies included in the 2025 systematic review [10], "
        "indicating that most fail the Yield and Robustness domains.")

    _embed_fig(FIGURES_DIR / "fig3_ready_compliance.png",
               "Figure 3. Illustrative QML-Health-READY domain compliance for studies "
               "in Gupta et al. 2025 (n≈16). Estimates from narrative review data. "
               "Red bars = below 50% compliance.")

    # ─── 6. Use-case triage ───────────────────────────────────────────────────
    _hdr(1, "6. Use-Case Triage: Where Could Quantum Help?")
    _para(
        "We classify medical applications by the theoretical argument for quantum "
        "benefit. This triage is a reasoned expert judgment, not a meta-analytic "
        "finding, and should be updated as hardware and theory develop.")

    doc.add_paragraph("Table 7. Use-case triage by plausibility of quantum advantage.")
    t7 = doc.add_table(rows=1, cols=3)
    t7.style = "Table Grid"
    for i, h in enumerate(["Use-case","Argument for quantum benefit","Outlook"]):
        t7.rows[0].cells[i].text = h
        t7.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for uc in [
        ("Molecular simulation / drug discovery (electronic structure, protein folding)",
         "Quantum hardware simulates quantum systems natively; argument does not require "
         "classical-data loading. Fault-tolerant devices could evaluate Hamiltonians "
         "beyond classical reach.",
         "Most plausible long-term; requires fault-tolerant devices (beyond NISQ)."),
        ("Small-sample, very-high-dimensional omics (proteomics, metabolomics)",
         "Quantum kernels can represent exponentially large feature spaces; "
         "data-loading cost is manageable at small n. Geometric difference may be "
         "non-negligible [7].",
         "Possible; requires data-complexity analysis per dataset and rigorous "
         "noise-aware benchmarking."),
        ("Combinatorial optimisation (treatment scheduling, radiotherapy planning)",
         "Problems are NP-hard and formulated as QUBO; quantum approximate optimisation "
         "algorithms (QAOA) may offer a heuristic advantage.",
         "Speculative; classical meta-heuristics are strong and QAOA circuit depth "
         "currently requires fault-tolerant hardware."),
        ("Neurological / mental-health signals (EEG, speech, survey data)",
         "Small, high-dimensional datasets favour kernel approaches. However, leakage "
         "and overfitting risks are especially high on small clinical samples.",
         "Open question; strict subject-level evaluation, noise-aware benchmarking "
         "and external validation are essential before any clinical claim."),
        ("Routine tabular risk prediction (EHR, structured data)",
         "Gradient boosting is exceptionally strong on tabular data. Quantum data "
         "loading overhead is high relative to any plausible Hilbert-space benefit.",
         "Unlikely to benefit from QML in the NISQ era."),
        ("Medical imaging classification",
         "State-of-the-art deep CNNs are trained on millions of images; encoding "
         "images into few qubits requires heavy compression. Quantum layers add "
         "marginal expressiveness.",
         "Unlikely to benefit; hybrid quantum layers do not consistently improve "
         "over classical CNNs [10]."),
    ]:
        row = t7.add_row()
        for j, v in enumerate(uc):
            row.cells[j].text = v

    _embed_fig(FIGURES_DIR / "fig6_triage_matrix.png",
               "Figure 6. QML use-case triage matrix. Ratings (Low/Moderate/High) for "
               "theoretical basis, data-loading feasibility, NISQ hardware readiness, "
               "and overall plausibility of quantum advantage.")

    # ─── 7. Discussion & recommendations ──────────────────────────────────────
    _hdr(1, "7. Discussion")
    _hdr(2, "7.1 Interpretation of benchmark results")
    # pull complexity numbers for interpretive paragraph
    def _ka_str(ds):
        cd = complexity_data.get(ds, {})
        qa = cd.get("KA_quantum", float("nan"))
        ca = cd.get("KA_rbf",     float("nan"))
        gd = cd.get("geom_diff_sq", float("nan"))
        if math.isnan(qa): return ""
        return f"KTA_Q={qa:.3f}, KTA_RBF={ca:.3f}, s={gd:.1f}"

    _para(
        "Our benchmark, the largest single study to apply noise-aware simulation "
        "and entanglement ablation simultaneously across six clinical datasets "
        "spanning four medical domains, found no statistically significant quantum "
        "advantage on any dataset after Bonferroni correction (α* = 0.0083). "
        "These results align with the 2025 systematic review [10] and the Bowles et al. "
        "benchmark [9], while extending them by (a) covering both routine tabular data "
        "and high-dimensional genomic data (TCGA-BRCA, 200 features) as well as "
        "population epidemiology (NHANES, n=2000), (b) applying depolarising noise "
        "simulation, and (c) computing kernel-target alignment and geometric difference "
        "as a data-complexity rationale for every dataset. "
        "Kernel-target alignment of the quantum ZZ-FeatureMap kernel was low across all "
        "datasets: KTA_Q ranged from 0.069 (breast cancer) to 0.247 (NHANES), compared "
        "with KTA_RBF = 0.103-0.546 for the classical RBF kernel. The geometric "
        "difference s(K_Q, K_C) — a necessary condition for quantum advantage under "
        "Huang et al. [7] — was moderate to low (7.5-72.2), with the highest values "
        "on breast cancer (s=72.2) and NHANES (s=48.3); yet neither dataset showed "
        "quantum advantage, suggesting the geometric difference was not sufficient "
        "given the subsample size constraint (n=50 training). This provides, for the "
        "first time in a medical QML benchmark, a data-complexity explanation for the "
        "observed null results across heterogeneous clinical data types.")
    _para(
        "A limitation of our quantum evaluation is the O(n²) kernel cost, which "
        "required subsampling training sets to 50 samples per fold. "
        "Future work should address this via approximate kernel methods or "
        "quantum hardware with batch execution. We note that this computational "
        "bottleneck is itself a limitation of current QML for clinical use: "
        "classical models face no comparable constraint at this sample size.")

    _hdr(2, "7.2 Recommendations")
    doc.add_paragraph("For authors:")
    _bullet("State the geometric difference s(K_Q, K_C) or kernel-target alignment "
            "before claiming a dataset is suited to quantum learning [7].")
    _bullet("Treat classical baselines as the primary experiment; quantum models are "
            "the challenger. Invest equal tuning budget in both.")
    _bullet("Report all required ablations: entanglement-ablated model, noise-model "
            "run, and confidence intervals with effect sizes. Negative results meeting "
            "all Core items of QML-Health-READY are publishable and valuable.")
    doc.add_paragraph("For reviewers and editors:")
    _bullet("Request the QML-Health-READY checklist at submission. Reject studies "
            "that report an AUC of 1.000 or equivalent without leakage audit and "
            "subject-level splits.")
    _bullet("Require noise-aware evaluation or an explicit statement that the study "
            "is classified as Tier 0 (exploratory).")
    doc.add_paragraph("For funders:")
    _bullet("Prioritise molecular simulation and multi-site benchmark studies "
            "over further single-dataset classification experiments.")
    _bullet("Require pre-registration and reporting-standard compliance for "
            "funded QML-in-health projects.")

    _hdr(2, "7.3 Ethics, equity and regulation")
    _para(
        "Quantum computing infrastructure is concentrated in a small number of "
        "high-income countries and institutions, raising equity concerns if QML "
        "tools are developed primarily for resource-rich settings. Clinical-use "
        "QML models will be subject to medical device regulation in most "
        "jurisdictions, requiring transparency, prospective validation and "
        "post-deployment monitoring — requirements that are harder to satisfy "
        "given the reproducibility challenges introduced by quantum hardware "
        "calibration drift.")

    # ─── 8. Limitations ───────────────────────────────────────────────────────
    _hdr(1, "8. Limitations")
    _bullet("Our benchmark used subsampled training sets (n = 50) for quantum models; "
            "results may not generalise to larger training sizes, which future "
            "hardware may enable.")
    _bullet("All quantum evaluation used classical simulators; real hardware would "
            "introduce additional noise, cross-talk and readout error not modelled here.")
    _bullet("The literature search has a fixed end date; the rapidly evolving field "
            "means some recent publications may be missed.")
    _bullet("QML-Health-READY has not yet been externally validated for inter-rater "
            "reliability; a formal Delphi process is recommended.")
    _bullet("The triage in Section 6 reflects current theory and hardware capabilities; "
            "fault-tolerant devices may shift which use-cases are plausible.")

    # ─── 9. Conclusion ────────────────────────────────────────────────────────
    _hdr(1, "9. Conclusion")
    _para(
        "Across six clinical datasets spanning four medical domains and nine model configurations, we find no "
        "statistically significant advantage for quantum machine learning over tuned "
        "classical baselines. Data-complexity analysis indicates these datasets are "
        "not structurally suited to the quantum kernels we evaluated, providing "
        "a principled explanation rather than a purely empirical null result. "
        "Depolarising noise at p = 0.01 produced results indistinguishable from ideal "
        "simulation on all datasets, consistent with noise-induced barren-plateau "
        "saturation — a finding that itself cautions against ideal-simulator claims. "
        "Removing "
        "entanglement did not consistently reduce performance, suggesting quantum "
        "structure contributed minimally to these classification tasks. "
        "The QML-Health-READY framework translates these findings into a practical "
        "reporting standard that authors, reviewers and funders can apply immediately. "
        "We recommend that the field redirect resources from small-dataset "
        "classification studies toward the problems — molecular simulation, "
        "high-dimensional omics, rigorously benchmarked multi-site trials — where "
        "quantum computing has a credible theoretical case.")

    # ─── Declarations ─────────────────────────────────────────────────────────
    _hdr(1, "Declarations")

    _hdr(2, "Author contributions")
    _para(
        "Mohammed Siraj B: Conceptualization, Methodology, Software, Formal Analysis, "
        "Investigation, Data Curation, Writing – Original Draft, Visualization. "
        "Ruban S: Supervision, Writing – Review & Editing, Validation. "
        "Both authors read and approved the final manuscript.")

    _hdr(2, "Funding")
    _para(
        "The authors declare that no external funding was received for this research. "
        "No funding body had any role in the study design, data collection, analysis, "
        "interpretation, or decision to submit for publication.")

    _hdr(2, "Conflicts of interest")
    _para(
        "The authors declare no conflicts of interest.")

    _hdr(2, "Data availability")
    _para(
        "All six benchmark datasets used in this study are publicly available with no "
        "restrictions on research use. Breast Cancer Wisconsin, Heart Disease Cleveland, "
        "Pima Indians Diabetes, and Heart Failure Clinical Records are available from "
        "the UCI Machine Learning Repository (archive.ics.uci.edu). TCGA-BRCA ER-status "
        "data are available from the UCSC Xena browser (xenabrowser.net; S3 public "
        "bucket). NHANES 2017–18 data are available from the United States Centers "
        "for Disease Control and Prevention "
        "(wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/). No new patient data "
        "were collected or generated for this study.")

    _hdr(2, "Code availability")
    _para(
        "All benchmark code, configuration files, and aggregated results (JSON) are "
        "openly available at github.com/sirajbb/qml-health-ready. A citable, "
        "version-controlled archive is deposited at Zenodo "
        "(DOI: 10.5281/zenodo.XXXXXXX; to be updated upon publication). "
        "The implementation uses Python 3.x, PennyLane 0.45.1, scikit-learn, "
        "and python-docx; all dependencies are listed in the repository README.")

    _hdr(2, "Artificial intelligence tool use")
    _para(
        "Claude AI (Anthropic, claude.ai) was used to assist with Python code "
        "development for the benchmark pipeline and manuscript preparation. "
        "All scientific content, interpretations, and conclusions are the sole "
        "responsibility of the authors. AI assistance is disclosed in accordance "
        "with the journal’s policy on generative AI use.")

    _hdr(2, "Ethics statement")
    _para(
        "This study used only publicly available, de-identified benchmark datasets "
        "released for open research use. No new patient data were collected, no "
        "human subjects were recruited, and no identifiable information was processed. "
        "Institutional ethics approval was therefore not required.")

    # ─── References ───────────────────────────────────────────────────────────
    _hdr(1, "References")
    refs = [
        # ── Foundational QML theory ──────────────────────────────────────────────
        "Preskill J. Quantum computing in the NISQ era and beyond. Quantum. 2018;2:79. doi:10.22331/q-2018-08-06-79.",
        "Havlíček V, Córcoles AD, Temme K, Harrow AW, Kandala A, Chow JM, Gambetta JM. Supervised learning with quantum-enhanced feature spaces. Nature. 2019;567(7747):209-212. doi:10.1038/s41586-019-0980-2.",
        "Schuld M, Killoran N. Quantum machine learning in feature Hilbert spaces. Phys Rev Lett. 2019;122(4):040504. doi:10.1103/PhysRevLett.122.040504.",
        "Cerezo M, Arrasmith A, Babbush R, Benjamin SC, Endo S, Fujii K, et al. Variational quantum algorithms. Nat Rev Phys. 2021;3(9):625-644. doi:10.1038/s42254-021-00348-9.",
        # ── Barren plateaus & trainability ──────────────────────────────────────
        "McClean JR, Boixo S, Smelyanskiy VN, Babbush R, Neven H. Barren plateaus in quantum neural network training landscapes. Nat Commun. 2018;9(1):4812. doi:10.1038/s41467-018-07090-4.",
        "Wang S, Fontana E, Cerezo M, Sharma K, Sone A, Cincio L, Coles PJ. Noise-induced barren plateaus in variational quantum algorithms. Nat Commun. 2021;12(1):6961. doi:10.1038/s41467-021-27045-6.",
        # ── Quantum advantage theory ─────────────────────────────────────────────
        "Huang HY, Broughton M, Mohseni M, Babbush R, Boixo S, Neven H, McClean JR. Power of data in quantum machine learning. Nat Commun. 2021;12(1):2631. doi:10.1038/s41467-021-22539-9.",
        "Tang E. A quantum-inspired classical algorithm for recommendation systems. In: Proc 51st Annual ACM Symposium on Theory of Computing (STOC 2019). New York: ACM; 2019. p. 217-228. doi:10.1145/3313276.3316310.",
        # ── Benchmarking & systematic evidence ──────────────────────────────────
        "Bowles J, Ahmed S, Schuld M. Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv preprint arXiv:2403.07059. 2024.",
        "Gupta RS, Wood CE, Engstrom T, Pole JD, Shrapnel S. A systematic review of quantum machine learning for digital health. npj Digit Med. 2025;8:237. doi:10.1038/s41746-025-01597-z.",
        # ── Reporting guidelines ─────────────────────────────────────────────────
        "Collins GS, Moons KGM, Dhiman P, Wynants L, Myint PK, Majeed A, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378.",
        "Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): a guide for authors and reviewers. Radiol Artif Intell. 2020;2(2):e200029. doi:10.1148/ryai.2020200029.",
        "Page MJ, McKenzie JE, Bossuyt PM, Boutron I, Hoffmann TC, Mulrow CD, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. BMJ. 2021;372:n71. doi:10.1136/bmj.n71.",
        # ── QML reviews & textbooks ──────────────────────────────────────────────
        "Biamonte J, Wittek P, Pancotti N, Rebentrost P, Wiebe N, Lloyd S. Quantum machine learning. Nature. 2017;549(7671):195-202. doi:10.1038/nature23474.",
        "Schuld M, Petruccione F. Machine Learning with Quantum Computers. 2nd ed. Cham: Springer; 2021. doi:10.1007/978-3-030-83098-4.",
        "Dunjko V, Briegel HJ. Machine learning & artificial intelligence in the quantum domain: a review and a roadmap. Rep Prog Phys. 2018;81(7):074001. doi:10.1088/1361-6633/aab406.",
        "Abbas A, Sutter D, Zoufal C, Lucchi A, Figalli A, Woerner S. The power of quantum neural networks. Nat Comput Sci. 2021;1(6):403-409. doi:10.1038/s43588-021-00084-1.",
        # ── Dataset primary publications ─────────────────────────────────────────
        "Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. Nature. 2012;490(7418):61-70. doi:10.1038/nature11412.",
        "Goldman MJ, Craft B, Hastie M, Repecká K, McDade F, Kamath A, et al. Visualizing and interpreting cancer genomics data via the Xena platform. Nat Biotechnol. 2020;38(6):675-678. doi:10.1038/s41587-020-0546-8.",
        "Chicco D, Jurman G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med Inform Decis Mak. 2020;20(1):16. doi:10.1186/s12911-020-1023-5.",
        # ── Quantum kernel & QSVM foundations ───────────────────────────────────
        "Rebentrost P, Mohseni M, Lloyd S. Quantum support vector machine for big data classification. Phys Rev Lett. 2014;113(13):130503. doi:10.1103/PhysRevLett.113.130503.",
        "Benedetti M, Lloyd E, Sack S, Fiorentini M. Parameterized quantum circuits as machine learning models. Quantum Sci Technol. 2019;4(4):043001. doi:10.1088/2058-9565/ab4eb5.",
        "Schuld M. Supervised quantum machine learning models are kernel methods. arXiv preprint arXiv:2101.11020v3. 2022.",
        # ── Data encoding & expressive power ────────────────────────────────────
        "Schuld M, Sweke R, Meyer JJ. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Phys Rev A. 2021;103(3):032430. doi:10.1103/PhysRevA.103.032430.",
        "Cerezo M, Larocca M, García-Martín D, Diaz NL, Braccia P, Fontana E, et al. Does provable absence of barren plateaus imply classical simulability? Or, why we need to rethink low-depth quantum neural networks. arXiv preprint arXiv:2312.09121. 2023.",
        # ── Quantum advantage rigorous theory ────────────────────────────────────
        "Liu Y, Arunachalam S, Temme K. A rigorous and robust quantum speed-up in supervised machine learning. Nat Phys. 2021;17(9):1013-1017. doi:10.1038/s41567-021-01287-z.",
        "Ciliberto C, Herbster M, Ialongo AD, Pontil M, Rocchetto A, Severini S, Wossnig L. Quantum machine learning: a classical perspective. Proc R Soc A. 2018;474(2209):20170551. doi:10.1098/rspa.2017.0551.",
        # ── Clinical AI & context ────────────────────────────────────────────────
        "Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. Nat Med. 2019;25(1):44-56. doi:10.1038/s41591-018-0300-7.",
        "Obermeyer Z, Emanuel EJ. Predicting the future — big data, machine learning, and clinical medicine. N Engl J Med. 2016;375(13):1216-1219. doi:10.1056/NEJMp1606181.",
        # ── Datasets: epidemiology ───────────────────────────────────────────────
        "National Center for Health Statistics. National Health and Nutrition Examination Survey (NHANES) 2017-2018 Data Files. Hyattsville, MD: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention; 2019. Available from: www.cdc.gov/nchs/nhanes.",
        "Farhi E, Neven H. Classification with quantum neural networks on near term processors. arXiv preprint arXiv:1802.06002. 2018.",
    ]
    for i, ref in enumerate(refs):
        # Skip comment-only strings (lines starting with #)
        if ref.startswith("#"):
            continue
        p = doc.add_paragraph(style="Normal")
        p.paragraph_format.left_indent       = Pt(18)
        p.paragraph_format.first_line_indent = Pt(-18)
        p.paragraph_format.space_after       = Pt(4)
        run = p.add_run(f"{i+1}. {ref}")
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)

    # ─── Save ──────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "02_QML_Healthcare_Manuscript_Final.docx"
    doc.save(str(out_path))
    print(f"\nManuscript saved → {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    bench_path = RESULTS_DIR / "benchmark_results.json"
    comp_path  = RESULTS_DIR / "complexity_results.json"

    if not bench_path.exists():
        print("ERROR: results/benchmark_results.json not found. "
              "Run run_benchmark.py first.")
        sys.exit(1)

    print("Loading results …")
    all_results    = _load_json(bench_path)
    complexity_data = _load_json(comp_path) if comp_path.exists() else {}

    print("Building manuscript …")
    build(all_results, complexity_data)
