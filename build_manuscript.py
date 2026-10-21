"""
build_manuscript.py (v3) - builds the manuscript and supplementary material from
results/benchmark_results.json (v2 or v3 schema). All statistics are recomputed
from the per-split AUROC values; citations are numbered in order of first appearance.

    python build_manuscript.py
"""

import sys, io, re, json, math
from pathlib import Path
from collections import Counter

import numpy as np


def _betacf(a, b, x):
    """Continued fraction for the regularised incomplete beta (Numerical Recipes)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


class stats:
    """Student-t distribution without SciPy (SciPy DLLs are blocked on this machine)."""

    class t:
        @staticmethod
        def sf(x, df):
            tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + x * x))
            return tail if x >= 0 else 1.0 - tail

        @staticmethod
        def cdf(x, df):
            return 1.0 - stats.t.sf(x, df)

        @staticmethod
        def ppf(q, df):
            lo, hi = -50.0, 50.0
            for _ in range(200):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if stats.t.cdf(mid, df) < q else (lo, mid)
            return (lo + hi) / 2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results" / "benchmark_results.json"
FIG = HERE / "figures"
OUT_MAIN = HERE.parent / "02_QML_Healthcare_Manuscript_Final.docx"
OUT_SUPP = HERE.parent / "02_QML_Healthcare_Supplementary.docx"

TITLE = ("Matched but Not Better: An Equivalence-Tested Benchmark of Quantum Machine "
         "Learning on Six Clinical Datasets and the QML-Health-READY Reporting Framework")

DATASETS = ["breast_cancer", "heart_disease", "diabetes",
            "heart_failure", "tcga_brca", "nhanes_diabetes"]
DS = {"breast_cancer": "Breast Cancer Wisconsin", "heart_disease": "Heart Disease Cleveland",
      "diabetes": "Pima Indians Diabetes", "heart_failure": "Heart Failure",
      "tcga_brca": "TCGA-BRCA", "nhanes_diabetes": "NHANES 2017–18"}
CLASSICAL = ["LR", "SVM_RBF", "SVM_POLY", "RF", "GB"]
MODELS = CLASSICAL + ["QSVM", "QSVM_noisy", "QSVM_noent", "VQC", "VQC_noent"]
ML = {"LR": "Logistic regression", "SVM_RBF": "SVM (RBF)", "SVM_POLY": "SVM (polynomial)",
      "RF": "Random forest", "GB": "Gradient boosting", "QSVM": "QSVM",
      "QSVM_noisy": "QSVM (noisy, p = 0.01)", "QSVM_noent": "QSVM (product-state kernel)",
      "VQC": "VQC", "VQC_noent": "VQC (no entanglement)",
      "QSVM_noisy_shots": "QSVM (noisy + 1,024 shots)"}
MARGINS = [0.01, 0.02, 0.05]

REFS = {
    "topol": "Topol EJ. High-performance medicine: the convergence of human and artificial intelligence. Nat Med. 2019;25(1):44-56. doi:10.1038/s41591-018-0300-7.",
    "obermeyer": "Obermeyer Z, Emanuel EJ. Predicting the future — big data, machine learning, and clinical medicine. N Engl J Med. 2016;375(13):1216-1219. doi:10.1056/NEJMp1606181.",
    "preskill": "Preskill J. Quantum computing in the NISQ era and beyond. Quantum. 2018;2:79. doi:10.22331/q-2018-08-06-79.",
    "havlicek": "Havlíček V, Córcoles AD, Temme K, Harrow AW, Kandala A, Chow JM, Gambetta JM. Supervised learning with quantum-enhanced feature spaces. Nature. 2019;567(7747):209-212. doi:10.1038/s41586-019-0980-2.",
    "schuld_killoran": "Schuld M, Killoran N. Quantum machine learning in feature Hilbert spaces. Phys Rev Lett. 2019;122(4):040504. doi:10.1103/PhysRevLett.122.040504.",
    "schuld_book": "Schuld M, Petruccione F. Machine Learning with Quantum Computers. 2nd ed. Cham: Springer; 2021. doi:10.1007/978-3-030-83098-4.",
    "schuld_kernel": "Schuld M. Supervised quantum machine learning models are kernel methods. arXiv preprint arXiv:2101.11020. 2021.",
    "rebentrost": "Rebentrost P, Mohseni M, Lloyd S. Quantum support vector machine for big data classification. Phys Rev Lett. 2014;113(13):130503. doi:10.1103/PhysRevLett.113.130503.",
    "biamonte": "Biamonte J, Wittek P, Pancotti N, Rebentrost P, Wiebe N, Lloyd S. Quantum machine learning. Nature. 2017;549(7671):195-202. doi:10.1038/nature23474.",
    "ciliberto": "Ciliberto C, Herbster M, Ialongo AD, Pontil M, Rocchetto A, Severini S, Wossnig L. Quantum machine learning: a classical perspective. Proc R Soc A. 2018;474(2209):20170551. doi:10.1098/rspa.2017.0551.",
    "gupta": "Gupta RS, Wood CE, Engstrom T, Pole JD, Shrapnel S. A systematic review of quantum machine learning for digital health. npj Digit Med. 2025;8:237. doi:10.1038/s41746-025-01597-z.",
    "bowles": "Bowles J, Ahmed S, Schuld M. Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv preprint arXiv:2403.07059. 2024. doi:10.48550/arXiv.2403.07059.",
    "dunjko": "Dunjko V, Briegel HJ. Machine learning & artificial intelligence in the quantum domain: a review and a roadmap. Rep Prog Phys. 2018;81(7):074001. doi:10.1088/1361-6633/aab406.",
    "tang": "Tang E. A quantum-inspired classical algorithm for recommendation systems. In: Proc 51st Annual ACM Symposium on Theory of Computing (STOC 2019). New York: ACM; 2019. p. 217-228. doi:10.1145/3313276.3316310.",
    "sahebi": "Sahebi M, Barthe A, Suzuki Y, Holmes Z, Grossi M. On dequantization of supervised quantum machine learning via random Fourier features. arXiv preprint arXiv:2505.15902. 2025. doi:10.48550/arXiv.2505.15902.",
    "kakavand": "Kakavand S, Strohmeyer C, Schlotter M. Benchmarking quantum kernel support vector machines against classical baselines on tabular data: a rigorous empirical study with hardware validation. arXiv preprint arXiv:2604.18837. 2026. doi:10.48550/arXiv.2604.18837.",
    "leither": "Leither S, Lubinski T, Kubal M, Johri S. Benchmarking quantum and classical machine learning models on oncological data. arXiv preprint arXiv:2608.11373. 2026. doi:10.48550/arXiv.2608.11373.",
    "schnabel": "Schnabel J, Roth M. Quantum kernel methods under scrutiny: a benchmarking study. Quantum Mach Intell. 2025;7(1):58. doi:10.1007/s42484-025-00273-5.",
    "liu_hw": "Liu H, Zhang Z, Zheng K, Liang Z, Chen CS, Carothers C, Hahn J. Quantum machine learning for biomedical classification problems: a feasibility study on real quantum hardware. Ann Biomed Eng. 2026. doi:10.1007/s10439-026-04204-8.",
    "priyadharshini": "Priyadharshini M, Raju BD, Banu AF, Kumar PJ, Murugesh V, Rybin O. A quantum machine learning framework for predicting drug sensitivity in multiple myeloma using proteomic data. Sci Rep. 2025;15(1):26553. doi:10.1038/s41598-025-06544-2.",
    "setiawan": "Setiawan AE, Rustad S, Syukur A, Soeleman MA, Shidik GF, Mahargya IL, et al. A systematic literature review of quantum machine learning for medical: trends, datasets, topics, and methods. Int J Cogn Comput Eng. 2026;7:609-630. doi:10.1016/j.ijcce.2026.05.002.",
    "thanasilp": "Thanasilp S, Wang S, Cerezo M, Holmes Z. Exponential concentration in quantum kernel methods. Nat Commun. 2024;15:5200. doi:10.1038/s41467-024-49287-w.",
    "shaydulin": "Shaydulin R, Wild SM. Importance of kernel bandwidth in quantum machine learning. Phys Rev A. 2022;106(4):042407. doi:10.1103/PhysRevA.106.042407.",
    "canatar": "Canatar A, Peters E, Pehlevan C, Wild SM, Shaydulin R. Bandwidth enables generalization in quantum kernel models. Trans Mach Learn Res. 2023. arXiv:2206.06686.",
    "florez": "Flórez-Ablan R, Roth M, Schnabel J. On the similarity of bandwidth-tuned quantum kernels and classical kernels. Quantum Sci Technol. 2025;10(3):035051. doi:10.1088/2058-9565/ade7ad.",
    "miroszewski": "Miroszewski A, Fellous Asiani M, Mielczarek J, Le Saux B, Nalepa J. In search of quantum advantage: estimating the number of shots in quantum kernel methods. arXiv preprint arXiv:2407.15776. 2024. doi:10.48550/arXiv.2407.15776.",
    "kubler": "Kübler J, Buchholz S, Schölkopf B. The inductive bias of quantum kernels. Adv Neural Inf Process Syst. 2021;34:12661-12673.",
    "tripod": "Collins GS, Moons KGM, Dhiman P, Wynants L, Myint PK, Majeed A, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378.",
    "claim": "Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): a guide for authors and reviewers. Radiol Artif Intell. 2020;2(2):e200029. doi:10.1148/ryai.2020200029.",
    "futureai": "Lekadir K, Frangi AF, Porras AR, Glocker B, Cintas C, Langlotz CP, et al. FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare. BMJ. 2025;388:e081554. doi:10.1136/bmj-2024-081554.",
    "abbas": "Abbas A, Sutter D, Zoufal C, Lucchi A, Figalli A, Woerner S. The power of quantum neural networks. Nat Comput Sci. 2021;1(6):403-409. doi:10.1038/s43588-021-00084-1.",
    "schuld_encoding": "Schuld M, Sweke R, Meyer JJ. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Phys Rev A. 2021;103(3):032430. doi:10.1103/PhysRevA.103.032430.",
    "huang": "Huang HY, Broughton M, Mohseni M, Babbush R, Boixo S, Neven H, McClean JR. Power of data in quantum machine learning. Nat Commun. 2021;12(1):2631. doi:10.1038/s41467-021-22539-9.",
    "thabet": "Thabet S, Monbroussou L, Mamon EZ, Landman J. When quantum and classical models disagree: learning beyond minimum norm least square. npj Quantum Inf. 2026;12(1):81. doi:10.1038/s41534-026-01217-y.",
    "cerezo_vqa": "Cerezo M, Arrasmith A, Babbush R, Benjamin SC, Endo S, Fujii K, et al. Variational quantum algorithms. Nat Rev Phys. 2021;3(9):625-644. doi:10.1038/s42254-021-00348-9.",
    "mcclean": "McClean JR, Boixo S, Smelyanskiy VN, Babbush R, Neven H. Barren plateaus in quantum neural network training landscapes. Nat Commun. 2018;9(1):4812. doi:10.1038/s41467-018-07090-4.",
    "wang_nibp": "Wang S, Fontana E, Cerezo M, Sharma K, Sone A, Cincio L, Coles PJ. Noise-induced barren plateaus in variational quantum algorithms. Nat Commun. 2021;12(1):6961. doi:10.1038/s41467-021-27045-6.",
    "cerezo_bp": "Cerezo M, Larocca M, García-Martín D, Diaz NL, Braccia P, Fontana E, et al. Does provable absence of barren plateaus imply classical simulability? Nat Commun. 2025;16(1):7907. doi:10.1038/s41467-025-63099-6.",
    "liu": "Liu Y, Arunachalam S, Temme K. A rigorous and robust quantum speed-up in supervised machine learning. Nat Phys. 2021;17(9):1013-1017. doi:10.1038/s41567-021-01287-z.",
    "chicco": "Chicco D, Jurman G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Med Inform Decis Mak. 2020;20(1):16. doi:10.1186/s12911-020-1023-5.",
    "tcga": "Cancer Genome Atlas Network. Comprehensive molecular portraits of human breast tumours. Nature. 2012;490(7418):61-70. doi:10.1038/nature11412.",
    "xena": "Goldman MJ, Craft B, Hastie M, Repecká K, McDade F, Kamath A, et al. Visualizing and interpreting cancer genomics data via the Xena platform. Nat Biotechnol. 2020;38(6):675-678. doi:10.1038/s41587-020-0546-8.",
    "nhanes": "National Center for Health Statistics. National Health and Nutrition Examination Survey (NHANES) 2017-2018 Data Files. Hyattsville, MD: US Department of Health and Human Services, Centers for Disease Control and Prevention; 2019. www.cdc.gov/nchs/nhanes.",
    "benedetti": "Benedetti M, Lloyd E, Sack S, Fiorentini M. Parameterized quantum circuits as machine learning models. Quantum Sci Technol. 2019;4(4):043001. doi:10.1088/2058-9565/ab4eb5.",
    "farhi": "Farhi E, Neven H. Classification with quantum neural networks on near term processors. arXiv preprint arXiv:1802.06002. 2018.",
    "pennylane": "Bergholm V, Izaac J, Schuld M, Gogolin C, Ahmed S, Ajith V, et al. PennyLane: automatic differentiation of hybrid quantum-classical computations. arXiv preprint arXiv:1811.04968. 2018.",
    "sklearn": "Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830.",
    "nadeau": "Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn. 2003;52(3):239-281. doi:10.1023/A:1024068626366.",
    "bouckaert": "Bouckaert RR, Frank E. Evaluating the replicability of significance tests for comparing learning algorithms. In: Advances in Knowledge Discovery and Data Mining (PAKDD 2004). Lecture Notes in Computer Science, vol 3056. Berlin: Springer; 2004. p. 3-12. doi:10.1007/978-3-540-24775-3_3.",
    "schuirmann": "Schuirmann DJ. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. J Pharmacokinet Biopharm. 1987;15(6):657-680. doi:10.1007/BF01068419.",
    "lakens": "Lakens D. Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. Soc Psychol Personal Sci. 2017;8(4):355-362. doi:10.1177/1948550617697177.",
    "dersimonian": "DerSimonian R, Laird N. Meta-analysis in clinical trials. Control Clin Trials. 1986;7(3):177-188. doi:10.1016/0197-2456(86)90046-2.",
    "inthout": "IntHout J, Ioannidis JPA, Borm GF. The Hartung-Knapp-Sidik-Jonkman method for random effects meta-analysis is straightforward and considerably outperforms the standard DerSimonian-Laird method. BMC Med Res Methodol. 2014;14:25. doi:10.1186/1471-2288-14-25.",
    "holm": "Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65-70.",
    "cortes": "Cortes C, Mohri M, Rostamizadeh A. Algorithms for learning kernels based on centered alignment. J Mach Learn Res. 2012;13:795-828.",
}


class Citer:
    def __init__(self):
        self.order = []

    def __call__(self, text):
        def rep(m):
            nums = []
            for k in m.group(1).split(";"):
                k = k.strip().lstrip("@")
                if k not in REFS:
                    raise KeyError(f"unknown reference key {k}")
                if k not in self.order:
                    self.order.append(k)
                nums.append(self.order.index(k) + 1)
            nums = sorted(set(nums))
            parts, i = [], 0
            while i < len(nums):
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                    j += 1
                parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else
                             ",".join(str(n) for n in nums[i:j + 1]))
                i = j + 1
            return "[" + ",".join(parts) + "]"
        return re.sub(r"\[(@[^\]]+)\]", rep, text)


# ------------------------------------------------------------------ statistics
def paired(q, c, ratio):
    d = np.asarray(q) - np.asarray(c)
    J, m, s = len(d), float(d.mean()), float(d.std(ddof=1))
    se = s * math.sqrt(1 / J + ratio)
    tc = stats.t.ppf(0.975, J - 1)
    out = {"mean_diff": m, "se": se, "ci95": (m - tc * se, m + tc * se),
           "dz": m / s if s else float("nan"),
           "p_t": float(2 * stats.t.sf(abs(m / se), J - 1)) if se else 1.0}
    for M in MARGINS:
        out[M] = float(max(stats.t.sf((m + M) / se, J - 1),
                           stats.t.cdf((m - M) / se, J - 1))) if se else 0.0
    return out


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def _d(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "−")


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _of6(n):
    return {0: "none of the six datasets", 6: "all six datasets"}.get(n, f"{n} of 6 datasets")


def _sci(x):
    m, e = f"{x:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m} × 10{str(int(e)).translate(sup)}"


def _vgrid_of(R):
    grid = R["breast_cancer"]["__meta__"].get("vqc_tuning", {}).get("grid", [])
    return "{" + ", ".join(f"({layers}, {lr})" for layers, lr in grid) + "}"


def _pct(x):
    return f"{100 * x:.0f}%"


def _names(dss):
    dss = list(dss)
    if not dss:
        return "none"
    names = [DS[d] for d in dss]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def analyse(R):
    M = {d: R[d]["__meta__"] for d in DATASETS}
    v3 = "QSVM_fault_b1" in R["breast_cancer"]
    ratio = {d: M[d].get("test_train_ratio", 0.25) for d in DATASETS}
    best = {d: max(CLASSICAL, key=lambda k: R[d][k]["auc"]["mean"]) for d in DATASETS}
    val = lambda d, k: R[d][k]["auc"]["values"]
    km = "SVM_RBF_df" if "SVM_RBF_df" in R["breast_cancer"] else "SVM_RBF"
    vq = "VQC_tuned" if "VQC_tuned" in R["breast_cancer"] else "VQC"
    pairs = {
        "QSVM_vs_SVM_RBF": lambda d: ("QSVM", km),
        "QSVM_vs_best": lambda d: ("QSVM", best[d]),
        "VQC_vs_best": lambda d: (vq, best[d]),
        "QSVM_noent_vs_best": lambda d: ("QSVM_noent", best[d]),
        "QSVM_vs_QSVM_noent": lambda d: ("QSVM", "QSVM_noent"),
        "QSVM_vs_QSVM_noisy": lambda d: ("QSVM", "QSVM_noisy"),
        "VQC_vs_VQC_noent": lambda d: ("VQC", "VQC_noent"),
    }
    best_full = None
    if v3:
        best_full = {d: max([c + "_full" for c in CLASSICAL], key=lambda k: R[d][k]["auc"]["mean"])
                     for d in DATASETS}
        pairs.update({
            "QSVM_vs_QSVM_noisy_shots": lambda d: ("QSVM", "QSVM_noisy_shots"),
            "QSVM_noisy_shots_vs_best": lambda d: ("QSVM_noisy_shots", best[d]),
            "best_full_vs_best": lambda d: (best_full[d], best[d]),
            "QSVM_vs_best_full": lambda d: ("QSVM", best_full[d]),
        })
    if vq == "VQC_tuned":
        pairs["VQC_untuned_vs_best"] = lambda d: ("VQC", best[d])
    S = {}
    for name, f in pairs.items():
        S[name] = {d: paired(val(d, f(d)[0]), val(d, f(d)[1]), ratio[d]) for d in DATASETS}
        adj_t = holm([S[name][d]["p_t"] for d in DATASETS])
        adj_m = {M_: holm([S[name][d][M_] for d in DATASETS]) for M_ in MARGINS}
        for i, d in enumerate(DATASETS):
            s = S[name][d]
            s["p_t_holm"] = float(adj_t[i])
            for M_ in MARGINS:
                s[f"tost_holm_{M_}"] = float(adj_m[M_][i])
            for M_ in MARGINS:
                if s["p_t_holm"] < 0.05:
                    s[f"verdict_{M_}"] = "Worse" if s["mean_diff"] < 0 else "Better"
                elif s[f"tost_holm_{M_}"] < 0.05:
                    s[f"verdict_{M_}"] = "Equivalent"
                else:
                    s[f"verdict_{M_}"] = "Inconclusive"
    pooled = [c for c in ("QSVM_vs_SVM_RBF", "QSVM_vs_best", "VQC_vs_best",
                          "QSVM_vs_QSVM_noisy_shots") if c in S]
    P = {c: pool([S[c][d] for d in DATASETS]) for c in pooled}
    return M, v3, best, best_full, S, P


def pool(stats_list):
    """Random-effects pooled difference across datasets: DerSimonian-Laird tau^2 with the
    Hartung-Knapp-Sidik-Jonkman variance, floored at the fixed-weight variance (conservative)."""
    y = np.array([s["mean_diff"] for s in stats_list])
    v = np.array([s["se"] for s in stats_list]) ** 2
    k = len(y)
    w = 1 / v
    q = float(np.sum(w * (y - np.sum(w * y) / w.sum()) ** 2))
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * y) / ws.sum())
    var = max(np.sum(ws * (y - mu) ** 2) / ((k - 1) * ws.sum()), 1 / ws.sum())
    se = math.sqrt(var)
    t975, t95 = stats.t.ppf(0.975, k - 1), stats.t.ppf(0.95, k - 1)
    out = {"mean_diff": mu, "se": se, "ci95": (mu - t975 * se, mu + t975 * se),
           "ci90": (mu - t95 * se, mu + t95 * se), "tau": math.sqrt(tau2),
           "I2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0,
           "p": float(2 * stats.t.sf(abs(mu / se), k - 1)), "k": k}
    for M_ in MARGINS:
        out[f"equiv_{M_}"] = -M_ < out["ci90"][0] and out["ci90"][1] < M_
    return out


def count(S, comp, verdict, margin):
    return [d for d in DATASETS if S[comp][d][f"verdict_{margin}"] == verdict]


# ------------------------------------------------------------------ document helpers
class Doc:
    def __init__(self, cite):
        from docx import Document
        from docx.shared import Pt
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(11)
        self.cite = cite

    def h(self, level, text):
        p = self.doc.add_heading(text, level=level)
        for r in p.runs:
            r.font.name = "Times New Roman"
        return p

    def p(self, text, bold_prefix=None, size=11, cite=True):
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if bold_prefix:
            r = par.add_run(bold_prefix)
            r.bold = True
            r.font.size = Pt(size)
        r = par.add_run(self.cite(text) if cite else text)
        r.font.size = Pt(size)
        return par

    def bullet(self, text):
        from docx.shared import Pt
        par = self.doc.add_paragraph(style="List Bullet")
        r = par.add_run(self.cite(text))
        r.font.size = Pt(11)

    def caption(self, text):
        from docx.shared import Pt
        par = self.doc.add_paragraph()
        r = par.add_run(self.cite(text))
        r.font.size = Pt(9.5)
        r.italic = True
        return par

    def table(self, headers, rows, size=8.5, bold=(), widths=None):
        from docx.shared import Pt, Inches
        t = self.doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        if widths:
            t.autofit = False
        for i, h in enumerate(headers):
            t.rows[0].cells[i].text = h
        for row in rows:
            cells = t.add_row().cells
            for j, v in enumerate(row):
                cells[j].text = self.cite(str(v))
        for ri, row in enumerate(t.rows):
            for ci, cell in enumerate(row.cells):
                if widths:
                    cell.width = Inches(widths[ci])
                for par in cell.paragraphs:
                    for r in par.runs:
                        r.font.size = Pt(size)
                        if ri == 0 or (ri - 1, ci) in bold:
                            r.bold = True
        self.doc.add_paragraph()
        return t

    def figure(self, name, caption, width=6.3):
        from docx.shared import Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        path = FIG / name
        if not path.exists():
            raise FileNotFoundError(path)
        self.doc.add_picture(str(path), width=Inches(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.caption(caption)


# ------------------------------------------------------------------ build
def build(R):
    import plots
    M, v3, best, best_full, S, P = analyse(R)
    dfp = "SVM_RBF_df" in R["breast_cancer"]
    vt = "VQC_tuned" in R["breast_cancer"]
    _vgrid = _vgrid_of(R)
    mm = "QSVM_fault_minmax" in R["breast_cancer"]
    cfg = M["breast_cancer"]["config"]
    import os
    if not os.environ.get("QML_ALLOW_FAST"):
        assert cfg["fast_mode"] is False and cfg["cv_outer"] == 5 and len(cfg["seeds"]) == 5
    J = M["breast_cancer"]["n_observations"]
    gkey = (lambda c: c["geom_diff_CQ"]["0.001"]) if v3 else (lambda c: c["geom_diff_CQ"])
    comp = {d: M[d]["complexity"] for d in DATASETS}

    # ---------------- derived numbers
    mt_eq05 = count(S, "QSVM_vs_SVM_RBF", "Equivalent", 0.05)
    mt_eq02 = count(S, "QSVM_vs_SVM_RBF", "Equivalent", 0.02)
    mt_worse = count(S, "QSVM_vs_SVM_RBF", "Worse", 0.05)
    qb_eq05 = count(S, "QSVM_vs_best", "Equivalent", 0.05)
    qb_eq02 = count(S, "QSVM_vs_best", "Equivalent", 0.02)
    qb_worse = count(S, "QSVM_vs_best", "Worse", 0.05)
    vb_eq05 = count(S, "VQC_vs_best", "Equivalent", 0.05)
    vb_eq02 = count(S, "VQC_vs_best", "Equivalent", 0.02)
    vb_worse = count(S, "VQC_vs_best", "Worse", 0.05)
    eq01_any = [d for c in ("QSVM_vs_SVM_RBF", "QSVM_vs_best", "VQC_vs_best")
                for d in count(S, c, "Equivalent", 0.01)]
    better = [(c, d) for c in S for d in DATASETS
              if S[c][d]["verdict_0.05"] == "Better" and c in ("QSVM_vs_SVM_RBF", "QSVM_vs_best",
                                                               "VQC_vs_best", "QSVM_noisy_shots_vs_best")]
    pos_point = [(c, d) for c in ("QSVM_vs_best", "VQC_vs_best") for d in DATASETS
                 if S[c][d]["mean_diff"] > 0]
    prod = {d: S["QSVM_noent_vs_best"][d]["mean_diff"] for d in DATASETS}
    prod_close = [d for d in DATASETS if abs(prod[d]) < 0.01]
    ent_sig = [d for d in DATASETS if S["QSVM_vs_QSVM_noent"][d]["p_t_holm"] < 0.05]
    ent_noent_ge = [d for d in DATASETS if S["QSVM_vs_QSVM_noent"][d]["mean_diff"] <= 0]
    vqc_ent_sig = [d for d in DATASETS if S["VQC_vs_VQC_noent"][d]["p_t_holm"] < 0.05]
    noise01 = max(abs(S["QSVM_vs_QSVM_noisy"][d]["mean_diff"]) for d in DATASETS)
    noise_sig = [d for d in DATASETS if S["QSVM_vs_QSVM_noisy"][d]["p_t_holm"] < 0.05]
    p_max = list(M["breast_cancer"]["noise_curve"].keys())[-1]
    drop10 = {d: M[d]["noise_curve"]["0.0"] - M[d]["noise_curve"][p_max] for d in DATASETS}
    chosen = [bw for d in DATASETS for bw, _ in M[d]["QSVM_chosen_bandwidth_C"]]
    bw_min = min(cfg["bandwidths"])
    frac_min = sum(b == bw_min for b in chosen) / len(chosen)
    frac_le02 = sum(b <= 0.02 for b in chosen) / len(chosen)
    frac_ge05 = sum(b >= 0.5 for b in chosen) / len(chosen)
    modal = Counter(chosen).most_common(1)[0][0]
    g_vals = [gkey(comp[d]) for d in DATASETS]
    kta_below = [d for d in DATASETS if comp[d]["KTA_quantum"] < comp[d]["KTA_rbf"]]
    worst_kta = min(DATASETS, key=lambda d: comp[d]["KTA_quantum"])
    n_tr = {d: M[d].get("n_train_mean", 0.8 * M[d]["n"]) for d in DATASETS}
    n_te = {d: M[d].get("n_test_mean", 0.2 * M[d]["n"]) for d in DATASETS}
    circ = {d: len(cfg["bandwidths"]) * (n_tr[d] * (n_tr[d] + 1) / 2 + n_te[d] * n_tr[d])
            for d in DATASETS}
    t_split = lambda m: {d: M[d]["time_sec"][m] / J for d in DATASETS}
    if v3:
        shot_pen = {d: S["QSVM_vs_QSVM_noisy_shots"][d]["mean_diff"] for d in DATASETS}
        pairs_bw = [(a[0], b[0]) for d in DATASETS for a, b in
                    zip(M[d]["QSVM_chosen_bandwidth_C"], M[d]["QSVM_noisy_shots_chosen_bandwidth_C"])]
        frac_shot_larger = sum(b > a for a, b in pairs_bw) / len(pairs_bw)
        spread = {d: math.sqrt(comp[d]["kq_offdiag_var"]) for d in DATASETS}
        compressed = sorted(DATASETS, key=lambda d: spread[d])[:3]
        sweep_gap = {d: M[d]["noise_curve"]["0.0"] - M[d]["shot_curve"][max(M[d]["shot_curve"], key=int)]
                     for d in DATASETS}
        max_shots = max(int(k) for k in M["breast_cancer"]["shot_curve"])

    # ---------------- figures
    print("Figures ...")
    plots.fig_auroc_dots(R, {d: R[d][best_full[d]]["auc"]["mean"] for d in DATASETS} if v3 else None)
    plots.fig_forest({c: S[c] for c in ("QSVM_vs_SVM_RBF", "QSVM_vs_best", "VQC_vs_best")},
                     pooled=P)
    plots.fig_noise(R, {d: M[d]["shot_curve"] for d in DATASETS} if v3 else None)
    plots.fig_kernels(comp, gkey)
    plots.fig_triage()
    if v3:
        plots.fig_graphical_abstract(R, S["QSVM_vs_SVM_RBF"])
        plots.fig_fault_ladder(R)
        plots.fig_qubit_scaling(R, [d for d in DATASETS if M[d].get("qubit_scaling")])

    cite = Citer()
    D = Doc(cite)
    doc = D.doc
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    # Continuous line numbers and a centred page number, for reviewers
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    sp = doc.sections[0]._sectPr
    cols = sp.find(qn("w:cols"))
    cols.addprevious(ln) if cols is not None else sp.append(ln)  # schema order: before w:cols
    fp = doc.sections[0].footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        el = OxmlElement("w:fldChar" if tag else "w:instrText")
        if tag:
            el.set(qn("w:fldCharType"), tag)
        else:
            el.text = text
        fp.add_run()._r.append(el)

    # ---------------- title block
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = tp.add_run(TITLE)
    r.bold = True
    r.font.size = Pt(15)
    ap = doc.add_paragraph()
    ap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = ap.add_run("Mohammed Siraj Bᵃ*, Ruban Sᵃ\n"
                   "ᵃSchool of Engineering, St Aloysius (Deemed to be University), "
                   "Mangalore 575003, Karnataka, India\n"
                   "*Corresponding author: Mohammed Siraj B, siraj019@gmail.com | ORCID: [add ORCID]\n"
                   "Article type: Full Length Article")
    r.italic = True
    r.font.size = Pt(10)

    # ---------------- abstract (<= 250 words)
    D.h(1, "Abstract")
    abstract = [
        ("Background: ", "Quantum machine learning (QML) is promoted for clinical prediction, but "
         "its comparisons with classical models are rarely paired or tested for equivalence."),
        ("Methods: ", f"On six public clinical datasets, we compared a quantum-kernel support vector "
         f"machine (QSVM; 4 qubits, tuned bandwidth) and a variational quantum classifier (VQC) "
         f"with five tuned classical models, using identical four-component inputs and {J} "
         f"identical cross-validation splits. Paired differences in the area under the ROC curve "
         f"(AUROC) were assessed with a corrected resampled t-test, Holm-adjusted equivalence tests "
         f"(margins ±0.02 and ±0.05) and random-effects pooling. Ablation, noise"
         f"{', finite-shot and implementation-fault' if v3 else ''} analyses followed. We "
         f"propose QML-Health-READY, an eight-domain reporting framework."),
        ("Results: ", (f"No quantum model significantly outperformed any classical comparator. "
                       if not better else "") +
         f"Pooled across datasets, QSVM differed from the kernel-matched radial basis function "
         f"(RBF) SVM by {_d(P['QSVM_vs_SVM_RBF']['mean_diff'])} AUROC (95% CI "
         f"{_d(P['QSVM_vs_SVM_RBF']['ci95'][0])} to {_d(P['QSVM_vs_SVM_RBF']['ci95'][1])}"
         + ("; equivalent within ±0.02" if P['QSVM_vs_SVM_RBF']['equiv_0.02'] else "") +
         f"), and the VQC differed from the best classical model by "
         f"{_d(P['VQC_vs_best']['mean_diff'])} ({_d(P['VQC_vs_best']['ci95'][0])} to "
         f"{_d(P['VQC_vs_best']['ci95'][1])}). Per dataset, QSVM was equivalent to the RBF SVM "
         f"within ±0.05 on {len(mt_eq05)} of 6"
         + (f" but significantly worse than the best classical model on "
            f"{_names(qb_worse)} ({_d(S['QSVM_vs_best'][qb_worse[0]]['mean_diff'])})" if qb_worse else "") +
         f". A classically trivial product-state kernel performed as well as the entangling kernel."
         + (f" With p = 0.01 depolarising noise and 1,024 shots per kernel entry, QSVM AUROC fell "
            f"by a pooled {P['QSVM_vs_QSVM_noisy_shots']['mean_diff']:.3f} "
            f"({P['QSVM_vs_QSVM_noisy_shots']['ci95'][0]:.3f}–"
            f"{P['QSVM_vs_QSVM_noisy_shots']['ci95'][1]:.3f}); an untuned bandwidth "
            f"gave near-chance AUROC "
            f"({min(R[d]['QSVM_fault_b1']['auc']['mean'] for d in DATASETS):.2f}–"
            f"{max(R[d]['QSVM_fault_b1']['auc']['mean'] for d in DATASETS):.2f})."
            if v3 else "")),
        ("Conclusions: ", "Correctly configured quantum classifiers approached, but did not exceed, "
         "tuned classical models"
         + ("; finite-shot estimation, not gate noise, was their main practical cost, and "
            "configuration errors alone could reverse the conclusion. " if v3 else
            ", and implementation errors alone can reverse the conclusion. ") +
         "QML-Health-READY specifies the reporting that credible claims require."),
    ]
    highlights = [
        "Quantum classifiers matched, but never beat, tuned classical models on clinical data",
        f"QSVM matched an RBF SVM within ±0.05 AUROC on {len(mt_eq05)} of 6 clinical datasets",
        "A classically trivial product-state kernel performed as well as the entangled one",
    ] + ([
        "Finite measurement shots, not gate noise, were the binding cost for quantum kernels",
        "Configuration faults alone turned parity into near-chance quantum performance",
    ] if v3 else [])
    assert all(len(h) <= 85 for h in highlights), [h for h in highlights if len(h) > 85]
    words = 0
    for pre, txt in abstract:
        D.p(txt, bold_prefix=pre)
        words += len(txt.split())
    assert words <= 250, f"abstract has {words} words (limit 250)"
    D.p("quantum machine learning; quantum kernel; equivalence testing; benchmarking; "
        "reporting guideline; clinical prediction", bold_prefix="Keywords: ")
    H = Doc(cite)
    H.h(1, "Highlights")
    for h in highlights:
        H.bullet(h)
    H.doc.save(str(OUT_MAIN.with_name("Highlights.docx")))

    # ---------------- 1 Introduction
    D.h(1, "1. Introduction")
    D.p("Machine learning is now embedded in clinical research and practice [@topol;@obermeyer]. "
        "Noisy intermediate-scale quantum (NISQ) hardware [@preskill] has prompted parallel "
        "interest in quantum machine learning (QML), in which a parameterised quantum circuit "
        "serves as a feature map or kernel [@havlicek;@schuld_killoran;@schuld_book]. The "
        "quantum support vector machine (QSVM) was proposed with a logarithmic-runtime claim "
        "[@rebentrost], and the exponentially large Hilbert space has been argued to offer "
        "expressive power unavailable to classical models [@biamonte;@ciliberto]. For clinical "
        "prediction the question is practical: whether a quantum model discriminates better than "
        "a well-tuned classical one, at a cost a health system could bear. Overstated claims can "
        "divert research effort and funding from methods that already work.")
    D.p("The evidence has not followed. A 2025 systematic review identified 4,915 records "
        "(2015 to June 2024), evaluated 46 studies of QML for digital health in detail after "
        "excluding many for insufficient rigour, found that only 16 tested algorithms under "
        "realistic conditions (quantum hardware or noisy circuits), and concluded that there is "
        "not yet clear evidence of an empirical advantage [@gupta]. A benchmark of 12 QML models "
        "on 160 datasets found that out-of-the-box classical models outperformed the quantum "
        "classifiers [@bowles], earlier reviews were similarly cautious [@dunjko], and "
        "dequantisation results show that classical algorithms can often match proposed quantum "
        "speed-ups or trained quantum models [@tang;@sahebi]. Independent benchmarks from 2025 "
        "and 2026 point the same way: quantum-kernel SVMs on nine tabular datasets, validated on "
        "IBM hardware, showed no significant advantage in any of 29 comparisons [@kakavand], and a "
        "multi-modal oncology benchmark found no evidence of quantum advantage [@leither]. Applied "
        "medical studies nevertheless continue to appear, on hardware [@liu_hw] and in simulation "
        "[@priyadharshini], and recent reviews document highly heterogeneous methods [@setiawan].")
    D.p("Several mechanisms behind these results are now understood. Quantum fidelity kernels "
        "concentrate exponentially with qubit count [@thanasilp]; performance depends strongly on "
        "a bandwidth hyperparameter [@shaydulin;@canatar], and once that bandwidth is tuned the "
        "kernels become close to classical RBF or low-order polynomial kernels [@florez]; and "
        "concentration and spread effects inflate the number of measurement shots that kernel "
        "estimation requires [@miroszewski]. These results come mostly from physics and "
        "machine-learning benchmarks. What is missing for clinical prediction is evidence of "
        "three kinds: whether quantum and classical models are equivalent, rather than merely "
        "“not significantly different”, under identical inputs and splits; how large the "
        "shot cost is on clinical data once bandwidth is tuned; and how far implementation choices "
        "alone can move the conclusion. Our contributions are: (i) a synthesis of seven reasons why "
        "medical QML advantage claims are fragile; (ii) QML-Health-READY, an eight-domain reporting "
        "and readiness framework aligned with TRIPOD+AI, CLAIM and FUTURE-AI "
        "[@tripod;@claim;@futureai]; (iii) a matched, paired, equivalence-tested benchmark on six "
        "clinical datasets with a kernel-matched classical comparator"
        + ("; (iv) an empirical quantification of the finite-shot penalty and of its interaction "
           "with bandwidth tuning; and (v) an implementation-sensitivity experiment showing how "
           "plausible configuration faults reverse the conclusion" if v3 else "") + ".")

    # ---------------- 2 Fragility
    D.h(1, "2. Why advantage claims in medical QML are fragile")
    items = [
        ("2.1 Quantum models are kernel methods (domain A).",
         "Supervised QML models that embed data in Hilbert space and measure a linear observable "
         "are kernel methods [@schuld_killoran;@schuld_kernel]. Whether a quantum kernel beats an "
         "RBF or polynomial kernel is an empirical, data-dependent question, and expressive power "
         "depends on the encoding [@abbas;@schuld_encoding]."),
        ("2.2 Advantage depends on data structure (domain R).",
         "A quantum learning advantage requires a large geometric difference between quantum and "
         "classical kernels on the data at hand [@huang]; without such an analysis an advantage "
         "claim has no theoretical basis."),
        ("2.3 Dequantisation (domain Y).",
         "Quantum-inspired classical algorithms have matched several proposed speed-ups [@tang], "
         "random-Fourier-feature models can approximate trained quantum kernel and variational "
         "models under stated conditions [@sahebi], and the regimes in which quantum and classical "
         "models make different predictions are being characterised [@thabet]."),
        ("2.4 Benchmarking practice (domain A).",
         "Results are highly sensitive to experimental design [@bowles;@schnabel]. A tuned quantum "
         "model compared with an untuned classical model, or one given different inputs, yields "
         "uninformative results."),
        ("2.5 Trainability and noise (domain R).",
         "Variational circuits can exhibit barren plateaus [@cerezo_vqa;@mcclean], noise induces "
         "similar effects [@wang_nibp], and many models that provably avoid barren plateaus can be "
         "simulated classically once data are collected from the device [@cerezo_bp]."),
        ("2.6 Data loading and clinical realism (domain D).",
         "Amplitude encoding of an n-dimensional vector into log₂ n qubits needs on the order "
         "of n gates — exponential in the number of qubits — which removes logarithmic "
         "speed-ups [@liu]. Angle encoding is tractable but forces strong dimensionality "
         "reduction, so classical comparators must receive the same reduced inputs or the "
         "asymmetry must be reported."),
        ("2.7 Kernel scaling, bandwidth and concentration (domain E).",
         "Fidelity kernels concentrate exponentially with qubit count, making Gram matrices nearly "
         "diagonal [@thanasilp]. Performance depends strongly on a bandwidth hyperparameter "
         "[@shaydulin;@canatar]; at the small bandwidths that generalise well, quantum kernels "
         "become close to RBF or low-order polynomial kernels [@florez;@kubler], and estimating "
         "nearly constant kernel entries demands many measurement shots [@miroszewski]. Untuned "
         "encodings can therefore produce spurious failures, and tuned encodings can produce "
         "results that are strong but classically approximable and costly to measure."),
    ]
    for head, txt in items:
        D.p(txt, bold_prefix=head + " ")

    # ---------------- 3 Framework
    D.h(1, "3. The QML-Health-READY framework")
    D.p("QML-Health-READY has eight domains (R, E, A, D, Y; R, E, Y), each mapped to the failure "
        "modes of Section 2 (Table 1). Items are answered Yes, Partial, No or Not applicable with a "
        "one-line justification. Studies meeting all Core items (†) are READY-Reportable; "
        "readiness tiers are defined in Supplementary Table S1, and our own self-assessment is in "
        "Supplementary Table S6. Meeting the Core items is necessary for credible reporting and "
        "does not imply clinical usefulness."
        + (" Section 5.9 reports, for each domain, an effect measured in this study that the "
           "corresponding items would have detected." if v3 else ""))
    D.caption("Table 1. QML-Health-READY domains. † Core item.")
    D.table(["Domain", "Name", "Items"], [
        ("R†", "Rationale", "Hypothesised mechanism for advantage; data-complexity analysis "
         "(geometric difference, kernel alignment) [@huang;@cortes]."),
        ("E†", "Encoding", "Encoding, feature scaling, kernel bandwidth and its selection "
         "procedure (checking grid edges), dimensionality reduction fitted within training folds, "
         "qubit count [@shaydulin;@florez]."),
        ("A†", "Algorithm & baselines", "≥ 4 tuned classical baselines including a "
         "kernel-matched classical model, given the same inputs and tuning budget; entanglement-"
         "ablated quantum model [@bowles]."),
        ("D†", "Data & splits", "Provenance, sample size, class balance, subject-level "
         "splitting, leakage audit (including post-baseline predictors)."),
        ("Y†", "Yield", "Repeated nested cross-validation with identical splits for all models; "
         "paired tests corrected for split dependence [@nadeau;@bouckaert]; equivalence testing "
         "with a pre-specified margin when parity is claimed [@schuirmann;@lakens]; multiplicity "
         "control."),
        ("R†", "Robustness", "Explicit noise model (verified to be active) and finite-shot "
         "estimation; hardware-calibrated noise or real hardware where feasible [@wang_nibp;@thanasilp]."),
        ("E", "Evaluation in context", "Calibration, threshold-dependent metrics, decision-curve "
         "analysis, external validation; alignment with TRIPOD+AI, CLAIM and FUTURE-AI "
         "[@tripod;@claim;@futureai]."),
        ("Y", "Yardstick for cost", "Wall-clock time, circuit executions and shots per fit versus "
         "classical baselines; data-access assumptions; dequantisation check [@tang;@sahebi]."),
    ], size=9, widths=[0.6, 1.3, 4.1])

    # ---------------- 4 Methods
    D.h(1, "4. Methods")
    D.h(2, "4.1 Datasets")
    nh = M["nhanes_diabetes"].get("data_notes", {}).get("eligible_adults_complete")
    D.p("We used six public datasets (Table 2): Breast Cancer Wisconsin (benign coded as class 1, "
        "as distributed; AUROC is invariant to this choice); Heart Disease Cleveland "
        "(processed UCI file, incomplete rows removed, any disease positive); Pima Indians Diabetes"
        + (" (zero values for glucose, blood pressure, skin thickness, insulin and BMI treated as "
           "missing and median-imputed within each training fold)" if v3 else "") +
        "; Heart Failure Clinical Records [@chicco]"
        + (" with follow-up time removed because it is recorded after baseline" if v3 else "") +
        "; TCGA-BRCA oestrogen-receptor status [@tcga;@xena] from Illumina HiSeq expression "
        "percentiles, restricted to primary tumours with one sample per patient (all 780 "
        "ER-labelled samples met this criterion) and to the 200 highest-variance genes, a "
        "label-free filter; and NHANES 2017–18 adults [@nhanes], with diabetes defined as "
        "HbA1c ≥ 6.5% or a physician diagnosis and HbA1c excluded from the predictors to "
        "avoid target leakage" + (f" (a stratified random sample of 2,000 of {nh:,} eligible adults "
                                  f"with complete data; survey weights not applied)." if nh else
                                  " (a stratified random sample of 2,000 eligible adults; survey "
                                  "weights not applied)."))
    ds_meta = [("Oncology", "scikit-learn / UCI", "benign tumour"),
               ("Cardiology", "UCI", "any heart disease"),
               ("Metabolic", "UCI (mirror)", "diabetes"),
               ("Cardiology", "UCI", "death during follow-up"),
               ("Genomic oncology", "UCSC Xena", "ER-positive"),
               ("Epidemiology", "CDC NHANES", "diabetes")]
    D.caption("Table 2. Datasets. Prevalence is the fraction of class 1, the positive class "
              "for AUROC.")
    D.table(["Dataset", "Domain", "Source", "n", "Predictors", "Class 1", "Prevalence"],
            [(DS[d], a, b, f"{M[d]['n']:,}", M[d]["n_features"], c, _pct(M[d]["pos_rate"]))
             for d, (a, b, c) in zip(DATASETS, ds_meta)], size=9)

    D.h(2, "4.2 Models")
    D.p(f"Within each outer training fold we fitted"
        + (" median imputation," if v3 else "") +
        f" standardisation, principal component analysis (PCA) to {cfg['n_qubits']} components "
        f"and re-standardisation of the "
        f"components; all models received exactly these four inputs"
        + (", and the classical models were additionally fitted on all standardised predictors "
           "to quantify the cost of the reduction" if v3 else "") +
        ". Classical models (L2 logistic regression, support vector machines (SVMs) with radial "
        "basis function (RBF) and polynomial kernels, random forest, "
        "gradient boosting) were tuned by 3-fold inner grid search on AUROC [@sklearn] "
        "(grids in Supplementary Table S5).")
    D.p(f"The QSVM uses the ZZ feature map [@havlicek] with {cfg['zz_reps']} repetitions on a "
        f"linear chain: Hadamard and Rz(2bz_i) on each qubit, then "
        f"CNOT–Rz(2(π−bz_i)(π−bz_{{i+1}}))–CNOT, where z are the "
        f"standardised components and b is a bandwidth [@shaydulin;@canatar]. The fidelity kernel "
        f"|⟨φ(x′)|φ(x)⟩|² was computed by exact state-vector simulation in "
        f"PennyLane {cfg['pennylane']} [@pennylane]. Bandwidth "
        f"({', '.join(str(b) for b in cfg['bandwidths'])}) and C ({', '.join(str(c) for c in cfg['qsvm_C'])}) "
        f"were tuned by 3-fold inner cross-validation on the AUROC of the SVM decision function, "
        f"and the SVM was refitted on the full training fold. The product-state ablation removes "
        f"the CNOT blocks. The noisy QSVM applies a depolarising channel (p = {cfg['noise_eval']}) to "
        f"every qubit acted on by every gate (density-matrix simulation) and uses "
        f"Tr(ρ(x)ρ(x′)) as the kernel, re-tuning b and C"
        + (f"; a finite-shot arm additionally estimates every kernel entry from "
           f"{cfg['shots_eval']:,} shots and projects the training Gram matrix onto the positive "
           f"semidefinite cone. Shots were modelled as binomial draws with success probability equal "
           f"to the exact kernel value; this is exact for the noiseless compute–uncompute "
           f"estimator but only approximate for the noisy kernel, and two-qubit gates received "
           f"independent single-qubit channels rather than a two-qubit channel" if v3 else "") +
        ". Because global depolarising noise maps the kernel to a·K + c, which an SVM with a "
        "bias term and re-tuned C largely absorbs, exact noisy kernels are expected to change "
        "AUROC little; finite-shot estimation is the more demanding test [@thanasilp]. A noise "
        "sweep (p = 0–0.1) used the first seed’s folds at the bandwidth and C selected "
        "for the noiseless QSVM" + (", as did a shot sweep." if v3 else "."))
    D.p(f"The VQC applies AngleEmbedding (Rx) and {cfg['vqc_layers']} StronglyEntanglingLayers "
        f"[@cerezo_vqa;@benedetti;@farhi], reads out "
        + ("⟨Z⟩ on all qubits through a trainable linear layer" if v3 else
           "⟨Z₀⟩ through a trainable affine layer") +
        f", and was trained by full-batch Adam (learning rate {cfg['vqc_lr']}, {cfg['vqc_epochs']} "
        f"epochs, cross-entropy)"
        + (f"; its number of layers and learning rate were tuned by 3-fold inner cross-validation "
           f"over {_vgrid}, as for the QSVM, and the fixed configuration ({cfg['vqc_layers']} layers, "
           f"learning rate {cfg['vqc_lr']}) is reported as a sensitivity analysis. "
           if vt else " without hyperparameter tuning. ") +
        f"Its ablation replaces the "
        f"entangling layers with single-qubit rotations"
        + ("; with all-qubit readout this removes entanglement only." if v3 else
           "; because readout is on qubit 0 only, it also removes access to three of the four "
           "inputs, so it is not a clean test of entanglement."))
    if v3:
        D.p("To quantify implementation sensitivity we re-ran the QSVM with three cumulative "
            "configuration faults, all present in an earlier version of our own pipeline and "
            "plausible elsewhere: bandwidth fixed at 1; plus unscaled principal components as "
            "circuit inputs; plus a CZ–Rz–CZ entangling block, which equals a single-qubit Rz "
            "because diagonal gates commute, so the circuit creates no entanglement"
            + (". We also ran a widely used tutorial-style configuration, with inputs min–max "
               "scaled to [0, π] on each training fold and no bandwidth" if mm else "") +
            ". A qubit-scaling experiment (4, 8 and 12 qubits; first-seed folds) on "
            "TCGA-BRCA and breast cancer compared the QSVM with an RBF SVM on the same components "
            "and tracked kernel concentration at bandwidth 1.")

    D.h(2, "4.3 Statistical analysis")
    D.p(f"Five seeds × 5-fold stratified cross-validation gave {J} outer splits per dataset, "
        f"identical for every model, so all comparisons are paired. The primary metric is the area "
        f"under the receiver operating characteristic curve (AUROC). "
        f"Because overlapping training sets make the {J} differences dependent, we used the "
        f"corrected resampled t-test (variance inflated by 1/J + n_test/n_train) [@nadeau;@bouckaert] "
        f"and, for parity claims, two one-sided tests (TOST) [@schuirmann;@lakens] with the same "
        f"variance at margins of ±0.01, ±0.02 and ±0.05 AUROC. Within each comparison "
        f"family, p-values across the six datasets were Holm-adjusted [@holm]. A comparison is "
        f"Equivalent at a margin if its adjusted TOST p < 0.05, Worse or Better if its adjusted "
        f"t-test p < 0.05, and otherwise Inconclusive. To summarise across datasets, the "
        f"per-dataset differences and their corrected standard errors were pooled with a "
        f"random-effects model (DerSimonian–Laird heterogeneity; Hartung–Knapp–Sidik–Jonkman "
        f"confidence interval with k − 1 degrees of freedom) [@dersimonian;@inthout], and pooled "
        f"equivalence was declared when the 90% confidence interval lay inside the margin. With six "
        f"datasets these pooled estimates summarise the datasets studied rather than all clinical "
        f"prediction tasks. The primary QSVM comparison is with the RBF "
        f"SVM, which differs only in the kernel"
        + (" and, like the QSVM, was scored by its decision function" if dfp else "") +
        f"; comparisons with the best classical model per dataset (selected post hoc, which "
        f"favours the classical side) and for the VQC are secondary. Accuracy, macro-F1"
        + (", balanced accuracy" if v3 else "") +
        " and the Matthews correlation coefficient (MCC) are reported in Supplementary Table S4; "
        "thresholds differ between models "
        "(probability 0.5 versus decision-function sign), so these are descriptive only.")
    D.p("The analysis was not pre-registered, and parts of it were revised after an initial "
        "analysis of an earlier version of the pipeline (Supplementary Table S8). The ±0.05 "
        "margin was specified before any results were seen. The ±0.01 and ±0.02 margins, "
        "the designation of the RBF SVM as the primary comparator, and the finite-shot, "
        "all-predictor, implementation-sensitivity and qubit-scaling analyses were added "
        "afterwards. All margins are conventional rather than clinically derived, and all "
        "comparisons that were run are reported. Because this is a methodological benchmark rather "
        "than the development of a clinical prediction tool, we do not claim TRIPOD+AI compliance; "
        "the items it covers that apply here (data sources, outcome and predictor definitions, "
        "missing data, validation design and performance measures) are reported in Sections "
        "4.1–4.3."
        + (" A complete independent re-run of the benchmark reproduced every per-split AUROC "
           "exactly except those of SVMs scored with Platt-scaled probabilities, which rely on an "
           "internal random split and differed by at most 0.001 per split; this is why the "
           "kernel-matched comparison scores the RBF SVM by its decision function." if dfp else ""))

    D.h(2, "4.4 Kernel and data-complexity analysis")
    D.p("On a stratified sample of up to 300 observations we computed the maximum per-feature "
        "Fisher ratio, 1-nearest-neighbour label agreement, centred kernel–target alignment "
        f"[@cortes] for the quantum{', product-state' if v3 else ''} and RBF kernels, and the "
        "geometric difference g(K_C‖K_Q) [@huang] from trace-normalised kernels with "
        "regularisation λ = 10⁻³"
        + (" (and 10⁻⁴ to 10⁻¹ as a sensitivity analysis), together with the "
           "centred alignment between the quantum kernel and the best-matching RBF kernel" if v3
           else "") +
        ". √n is used as a reference scale: Huang et al. relate a potential advantage to g "
        "approaching √n, but g is not bounded by √n under regularisation. Provable "
        "speed-ups in supervised learning require data-access assumptions that clinical datasets "
        "do not satisfy [@liu].")

    # ---------------- 5 Results
    D.h(1, "5. Results")
    D.h(2, "5.1 Discrimination")
    best_pm = {d: max(MODELS, key=lambda m: R[d][m]["auc"]["mean"]) for d in DATASETS}
    bc_auc = {d: R[d][best[d]]["auc"]["mean"] for d in DATASETS}
    q_auc = {d: R[d]["QSVM"]["auc"]["mean"] for d in DATASETS}
    top_q = [d for d in DATASETS if best_pm[d] not in CLASSICAL]
    win_ct = Counter(best[d] for d in DATASETS).most_common()
    D.p(f"Discrimination varied widely across datasets (Table 3; Figure 1). The best classical "
        f"model reached AUROC {min(bc_auc.values()):.3f} ({DS[min(bc_auc, key=bc_auc.get)]}) to "
        f"{max(bc_auc.values()):.3f} ({DS[max(bc_auc, key=bc_auc.get)]}), and no single classical "
        f"model was best everywhere ("
        + "; ".join(f"{ML[m].lower() if m in ('LR', 'RF', 'GB') else ML[m]} on {n}" for m, n in win_ct) +
        f"). The noiseless QSVM ranged from {min(q_auc.values()):.3f} to {max(q_auc.values()):.3f} "
        f"and tracked the classical models closely on every dataset except NHANES 2017–18. "
        + (f"Among four-component models, a quantum variant had the highest mean AUROC on "
           f"{_names(top_q)} ({', '.join(ML[best_pm[d]] for d in top_q)}), but by at most "
           f"{max(R[d][best_pm[d]]['auc']['mean'] - bc_auc[d] for d in top_q):.3f} AUROC over the "
           f"best classical model, far inside the confidence intervals." if top_q else
           "No quantum variant had the highest mean AUROC on any dataset."))
    D.caption(f"Table 3. AUROC, mean ± 95% CI over {J} paired splits (CIs treat splits as "
              f"independent and are therefore optimistic). Bold: highest mean per dataset among "
              f"four-component models." + (" The last row gives the best classical model on all "
                                            "predictors." if v3 else ""))
    rows = [[ML[m]] + [f"{R[d][m]['auc']['mean']:.3f} ± {R[d][m]['auc']['ci']:.3f}"
                       for d in DATASETS] for m in MODELS]
    bold = {(i, j + 1) for i, m in enumerate(MODELS) for j, d in enumerate(DATASETS)
            if best_pm[d] == m}
    if v3:
        k = MODELS.index("QSVM_noisy") + 1
        rows.insert(k, [ML["QSVM_noisy_shots"]] + [
            f"{R[d]['QSVM_noisy_shots']['auc']['mean']:.3f} ± "
            f"{R[d]['QSVM_noisy_shots']['auc']['ci']:.3f}" for d in DATASETS])
        bold = {(i + (1 if i >= k else 0), j) for i, j in bold}
        if vt:
            rows.append(["VQC (tuned)"] + [
                f"{R[d]['VQC_tuned']['auc']['mean']:.3f} ± {R[d]['VQC_tuned']['auc']['ci']:.3f}"
                for d in DATASETS])
        if dfp:
            rows.append(["SVM (RBF), decision-function score"] + [
                f"{R[d]['SVM_RBF_df']['auc']['mean']:.3f} ± {R[d]['SVM_RBF_df']['auc']['ci']:.3f}"
                for d in DATASETS])
        rows.append(["Best classical, all predictors"] + [
            f"{R[d][best_full[d]]['auc']['mean']:.3f} ({ML[best_full[d][:-5]]})" for d in DATASETS])
    D.table(["Model"] + [DS[d] for d in DATASETS], rows, size=7.5, bold=bold, widths=[1.2] + [0.8] * 6)
    D.figure("fig1_auroc.png",
             f"Figure 1. AUROC (mean ± 95% CI, {J} splits) for classical (blue) and quantum "
             f"(red) models on identical four-component inputs."
             + (" Dashed line: best classical model on all predictors." if v3 else ""), width=6.4)

    D.h(2, "5.2 Paired comparisons and equivalence")
    D.caption("Table 4. Paired AUROC differences (quantum − classical) with corrected 95% CIs, "
              "Holm-adjusted p-values for the corrected t-test and for equivalence (TOST) at "
              "±0.02 and ±0.05, and verdicts at each margin (E = equivalent, I = "
              "inconclusive, W = worse, B = better).")
    t4 = []
    lab = {"QSVM_vs_SVM_RBF": "QSVM − SVM (RBF)", "QSVM_vs_best": "QSVM − best",
           "VQC_vs_best": "VQC (tuned) − best" if vt else "VQC − best"}
    for d in DATASETS:
        for c in ("QSVM_vs_SVM_RBF", "QSVM_vs_best", "VQC_vs_best"):
            s = S[c][d]
            comp_name = lab[c] + (f" ({ML[best[d]]})" if c != "QSVM_vs_SVM_RBF" else "")
            t4.append((DS[d], comp_name,
                       f"{_d(s['mean_diff'])} [{_d(s['ci95'][0])}, {_d(s['ci95'][1])}]",
                       _p(s["p_t_holm"]), _p(s["tost_holm_0.02"]), _p(s["tost_holm_0.05"]),
                       f"{s['verdict_0.02'][0]} / {s['verdict_0.05'][0]}"))
    D.table(["Dataset", "Comparison", "Δ AUROC [95% CI]", "p(t)", "p(TOST ±0.02)",
             "p(TOST ±0.05)", "Verdict ±0.02 / ±0.05"], t4, size=7.5,
            widths=[0.95, 1.45, 1.3, 0.5, 0.6, 0.6, 0.6])
    D.p((f"No quantum model was significantly better than a classical comparator"
         if not better else
         f"A quantum model was significantly better in {len(better)} "
         f"{'comparison' if len(better) == 1 else 'comparisons'}") +
        (f", and the quantum point estimate exceeded the classical one in only "
         f"{len(pos_point)} of 12 secondary comparisons. " if pos_point else
         ", and no quantum point estimate exceeded the best classical model. ") +
        f"Against the kernel-matched RBF SVM, the QSVM was equivalent within ±0.05 on "
        f"{len(mt_eq05)} datasets ({_names(mt_eq05)}) and within ±0.02 on {len(mt_eq02)} "
        f"({_names(mt_eq02)})"
        + (f", and significantly worse on {_names(mt_worse)}" if mt_worse else
           ", and was not significantly worse on any dataset") +
        f". Against the best classical model, the QSVM was equivalent within ±0.05 on "
        f"{len(qb_eq05)} and within ±0.02 on {len(qb_eq02)} datasets"
        + (f" and significantly worse on {_names(qb_worse)}" if qb_worse else "") +
        f"; the VQC was equivalent to the best classical model on {len(vb_eq05)} datasets at "
        f"±0.05 and on {len(vb_eq02)} at ±0.02"
        + (f" and significantly worse on {_names(vb_worse)}" if vb_worse else "") +
        ". " + ("No comparison was equivalent within ±0.01. " if not eq01_any else
                f"Equivalence within ±0.01 held in only {len(eq01_any)} of the 18 "
                f"comparisons. ") +
        ("Dataset by dataset, parity therefore holds at the conventional ±0.05 margin but is "
         "rarely established at tighter margins."))

    def _eq(p):
        return ("equivalent within ±0.02" if p["equiv_0.02"] else
                "equivalent within ±0.05 but not ±0.02" if p["equiv_0.05"] else
                "not equivalent within ±0.05")

    def _pt(p):
        return (f"{_d(p['mean_diff'])} AUROC (95% CI {_d(p['ci95'][0])} to {_d(p['ci95'][1])}; "
                f"τ = {p['tau']:.3f}, I² = {100 * p['I2']:.0f}%)")

    pk, pb, pv = P["QSVM_vs_SVM_RBF"], P["QSVM_vs_best"], P["VQC_vs_best"]
    D.p(f"Pooled across the six datasets (Figure 2, bottom rows), the QSVM differed from the "
        f"kernel-matched RBF SVM by {_pt(pk)}, {_eq(pk)}. Its pooled difference from the best "
        f"classical model was {_pt(pb)}, {_eq(pb)}, and the "
        f"{'tuned ' if 'VQC_tuned' in R['breast_cancer'] else ''}VQC differed from the best "
        f"classical model by {_pt(pv)}, {_eq(pv)}. "
        + ("No pooled difference significantly favoured a quantum model, and every pooled point "
           "estimate was negative." if all(p["mean_diff"] < 0 and p["ci95"][0] < 0
                                            for p in (pk, pb, pv)) else ""))
    if vt:
        tdiff = [R[d]["VQC_tuned"]["auc"]["mean"] - R[d]["VQC"]["auc"]["mean"] for d in DATASETS]
        n_fixed = len(count(S, "VQC_untuned_vs_best", "Equivalent", 0.05))
        D.p(f"Tuning the VQC changed its AUROC by {_d(min(tdiff))} to {_d(max(tdiff))} relative to "
            f"the fixed configuration. The fixed-configuration VQC was equivalent to the best "
            f"classical model within ±0.05 on {n_fixed} of 6 datasets, against {len(vb_eq05)} for "
            f"the tuned VQC"
            + (", so the VQC conclusions do not depend on tuning." if n_fixed == len(vb_eq05) else
               "; tuning therefore changed the per-dataset equivalence verdicts, and the tuned "
               "results are the primary ones."))
    worst_best = qb_worse[0] if qb_worse else None
    if worst_best and worst_best not in mt_worse:
        s1, s2 = S["QSVM_vs_best"][worst_best], S["QSVM_vs_SVM_RBF"][worst_best]
        D.p(f"The largest deficit against the best classical model, on {DS[worst_best]} "
            f"({_d(s1['mean_diff'])}), shrank to {_d(s2['mean_diff'])} and was not significant "
            f"against the RBF SVM. Every SVM performed poorly on this dataset relative to "
            f"{ML[best[worst_best]].lower()}, so the deficit appears to reflect the learner more "
            f"than the quantum kernel.")
    D.figure("fig2_paired_differences.png",
             "Figure 2. Paired AUROC differences with corrected 95% CIs for each dataset; diamonds "
             "show random-effects pooled estimates with Hartung–Knapp 95% CIs. Shaded bands: "
             "±0.02 and ±0.05 equivalence margins.", width=5.8)

    D.h(2, "5.3 Entanglement and the product-state kernel")
    D.p(f"Removing entanglement did not significantly change QSVM AUROC on "
        f"{'any dataset' if not ent_sig else 'most datasets'}; the product-state kernel had the "
        f"higher or equal point estimate on {len(ent_noent_ge)} of 6. This kernel is a product of "
        f"single-qubit overlaps and is trivially computed classically, yet it came within 0.01 "
        f"AUROC of the best classical model on {len(prod_close)} of 6 datasets ({_names(prod_close)}). "
        f"The QSVM's performance on these data therefore did not depend on entanglement. "
        + ((f"With all-qubit readout, removing entanglement from the VQC "
            + ("did not significantly reduce AUROC on any dataset, and the unentangled VQC had the "
               f"higher or equal point estimate on "
               f"{sum(S['VQC_vs_VQC_noent'][d]['mean_diff'] <= 0 for d in DATASETS)} of 6."
               if not vqc_ent_sig else
               f"significantly reduced AUROC on {_names(vqc_ent_sig)}.")) if v3 else
           f"For the VQC, removing entanglement reduced AUROC significantly on {len(vqc_ent_sig)} "
           f"datasets, but this ablation also removed three of four inputs from the readout and "
           f"cannot be attributed to entanglement.") +
        " Details are in Supplementary Table S2.")

    D.h(2, "5.4 Noise" + (" and finite shots" if v3 else ""))
    D.p(f"Depolarising noise at p = 0.01 changed QSVM AUROC by at most {noise01:.3f} "
        f"({'not significant on any dataset' if not noise_sig else 'significant on ' + _names(noise_sig)}), "
        f"as expected from the near-affine effect of depolarising noise on the kernel. Larger noise "
        f"reduced AUROC by up to {max(drop10.values()):.3f} at p = {p_max} ({DS[max(drop10, key=drop10.get)]}) "
        f"(Figure 3A)."
        + ((" With 1,024 shots per kernel entry on top of p = 0.01 noise, the QSVM changed by "
            f"{_d(min(S['QSVM_vs_QSVM_noisy_shots'][d]['mean_diff'] * -1 for d in DATASETS))} to "
            f"{_d(max(S['QSVM_vs_QSVM_noisy_shots'][d]['mean_diff'] * -1 for d in DATASETS))} AUROC "
            f"relative to the noiseless QSVM, and was equivalent to the best classical model within "
            f"±0.05 on {_of6(len(count(S, 'QSVM_noisy_shots_vs_best', 'Equivalent', 0.05)))} "
            f"(Figure 3B shows the shot sweep).") if v3 else
           " Shot noise was not modelled in this analysis."))
    D.figure("fig3_noise.png",
             "Figure 3. Change in QSVM AUROC relative to the noiseless kernel (first-seed folds, "
             "bandwidth and C fixed at the noiseless selection). A: per-gate depolarising "
             "probability." + (" B: shots per kernel entry." if v3 else ""), width=6.2)

    D.h(2, "5.5 Kernel regime and data complexity")
    D.p(f"Inner cross-validation chose the grid minimum ({bw_min}) in {_pct(frac_min)} of QSVM "
        f"fits"
        + (f", a bandwidth of 0.02 or less in {_pct(frac_le02)}," if bw_min < 0.02 else "")
        + (" and never 0.5 or more." if frac_ge05 == 0 else f" and 0.5 or more in {_pct(frac_ge05)}.") +
        f" At the modal bandwidth ({modal}) the quantum kernel had lower "
        f"alignment with the labels than the RBF kernel on {_of6(len(kta_below))}, with the "
        f"lowest alignment on {DS[worst_kta]}; the geometric difference was "
        f"{min(g_vals):.2f}–{max(g_vals):.2f} at λ = 10⁻³, small relative to the "
        f"√n reference scale ({min(math.sqrt(comp[d]['n_used']) for d in DATASETS):.1f}–"
        f"{max(math.sqrt(comp[d]['n_used']) for d in DATASETS):.1f})"
        + (f", and the quantum kernel's centred alignment with the best-matching RBF kernel was "
           f"{min(comp[d]['kq_vs_rbf_alignment_best'] for d in DATASETS):.2f}–"
           f"{max(comp[d]['kq_vs_rbf_alignment_best'] for d in DATASETS):.2f}" if v3 else "") +
        " (Figure 4; Supplementary Table S3). Alignment and g were computed at the modal bandwidth "
        "and a fixed RBF γ of 1/4, not at each fold’s selected settings, so they are "
        "descriptive. The tuned quantum kernels thus operated in the smooth, small-bandwidth regime "
        "in which they approximate classical kernels [@canatar;@florez], which is consistent with "
        "parity and indicates that a classical kernel can match the quantum one on these data "
        "[@huang].")
    if v3:
        sc_ds = [d for d in DATASETS if M[d].get("qubit_scaling")]
        msg = []
        for d in sc_ds:
            sc = M[d]["qubit_scaling"]
            qs = list(sc.keys())
            msg.append(f"{DS[d]}: off-diagonal kernel variance at bandwidth 1 fell from "
                       f"{_sci(sc[qs[0]]['offdiag_var_b1'])} ({qs[0]} qubits) to "
                       f"{_sci(sc[qs[-1]]['offdiag_var_b1'])} ({qs[-1]} qubits), and tuned QSVM "
                       f"AUROC was {sc[qs[-1]]['qsvm_auc']:.3f} versus {sc[qs[-1]]['svm_rbf_auc']:.3f} "
                       f"for the RBF SVM at {qs[-1]} qubits")
        D.p("Increasing the number of qubits reproduced exponential concentration of the "
            "untuned kernel [@thanasilp]: " + "; ".join(msg) + " (Supplementary Figure S2).")
    D.figure("fig4_kernels.png",
             "Figure 4. A: centred kernel–target alignment. B: geometric difference g "
             "(λ = 10⁻³) with the √n reference scale (not an upper bound).",
             width=6.3)

    if v3:
        D.h(2, "5.6 Implementation sensitivity")
        f1 = [R[d]["QSVM_fault_b1"]["auc"]["mean"] for d in DATASETS]
        f3 = [R[d]["QSVM_fault_v1map"]["auc"]["mean"] for d in DATASETS]
        D.p(f"Fixing the bandwidth at 1 reduced QSVM AUROC from "
            f"{min(R[d]['QSVM']['auc']['mean'] for d in DATASETS):.2f}–"
            f"{max(R[d]['QSVM']['auc']['mean'] for d in DATASETS):.2f} to "
            f"{min(f1):.2f}–{max(f1):.2f}; adding unscaled inputs and the non-entangling "
            f"CZ–Rz–CZ block left it at {min(f3):.2f}–{max(f3):.2f} (Figure 5)."
            + (f" The tutorial-style configuration (min–max scaling to [0, π], no bandwidth) "
               f"was less extreme but still poor, at "
               f"{min(R[d]['QSVM_fault_minmax']['auc']['mean'] for d in DATASETS):.2f}–"
               f"{max(R[d]['QSVM_fault_minmax']['auc']['mean'] for d in DATASETS):.2f}, between "
               f"{min(R[d]['QSVM']['auc']['mean'] - R[d]['QSVM_fault_minmax']['auc']['mean'] for d in DATASETS):.2f} "
               f"and {max(R[d]['QSVM']['auc']['mean'] - R[d]['QSVM_fault_minmax']['auc']['mean'] for d in DATASETS):.2f} "
               f"AUROC below the tuned QSVM on every dataset."
               if mm else "") +
            f" A study using the faulty settings would conclude that quantum models fail, and one "
            f"using the tutorial-style settings that they lag behind classical models on most datasets, whereas the "
            f"corrected pipeline gives the results in Table 3; the difference comes from "
            f"implementation, not from quantum computing, which is why the "
            f"Encoding and Robustness domains of QML-Health-READY are Core items.")
        D.figure("fig5_fault_ladder.png",
                 "Figure 5. QSVM AUROC under cumulative implementation faults (mean ± 95% CI, "
                 f"{J} splits). Dotted line: chance.", width=6.3)
        D.h(2, "5.7 Effect of dimensionality reduction")
        bf = S["best_full_vs_best"]
        sig_full = [d for d in DATASETS if bf[d]["p_t_holm"] < 0.05]
        D.p(f"Classical models on all predictors changed AUROC by "
            f"{_d(min(bf[d]['mean_diff'] for d in DATASETS))} to "
            f"{_d(max(bf[d]['mean_diff'] for d in DATASETS))} relative to four components "
            + (f"(significant after Holm adjustment on {_names(sig_full)}). " if sig_full else
               "(not significant on any dataset after Holm adjustment). ") +
            f"Against the best all-predictor model, the QSVM "
            f"was equivalent within ±0.05 on "
            f"{len(count(S, 'QSVM_vs_best_full', 'Equivalent', 0.05))} of 6 datasets.")

    D.h(2, "5.8 Computational cost" if v3 else "5.6 Computational cost")
    tq, tn, tl = t_split("QSVM"), t_split("QSVM_noisy"), t_split("LR")
    D.p(f"Per outer split, including tuning, the noiseless QSVM took {min(tq.values()):.1f}–"
        f"{max(tq.values()):.1f} s and the noisy QSVM {min(tn.values()):.0f}–"
        f"{max(tn.values()):.0f} s of simulation, versus {min(tl.values()):.2f}–"
        f"{max(tl.values()):.2f} s for logistic regression. Simulation time understates hardware "
        f"cost: bandwidth tuning on one outer split requires "
        f"{min(circ.values()):,.0f}–{max(circ.values()):,.0f} distinct kernel-entry circuits "
        f"(|grid| × [n(n+1)/2 training + n_test·n test entries]), each executed for "
        f"hundreds to thousands of shots, whereas every classical model fits in seconds.")

    v2_path = HERE / "results" / "superseded" / "v2_benchmark_results.json"
    if v3 and v2_path.exists():
        v2 = json.loads(v2_path.read_text())
        hf_leak = [v2["heart_failure"][k]["auc"]["mean"] - R["heart_failure"][k]["auc"]["mean"]
                   for k in CLASSICAL]
        b1_drop = [R[d]["QSVM"]["auc"]["mean"] - R[d]["QSVM_fault_b1"]["auc"]["mean"]
                   for d in DATASETS]
        nh = "nhanes_diabetes"
        ps = P["QSVM_vs_QSVM_noisy_shots"]
        D.h(2, "5.9 Empirical support for the framework domains")
        D.p("Each QML-Health-READY domain corresponds to an effect measured in this study, and "
            "several were large enough to change a conclusion (Table 5). The heart-failure comparison is a clean test of the "
            "leakage-audit item: version 2 of our pipeline (Supplementary Table S8) used the same "
            "splits and models but kept follow-up time as a predictor, and every classical model "
            f"scored {min(hf_leak):.2f}–{max(hf_leak):.2f} AUROC higher than in the reported "
            "analysis. These observations support the framework’s items; they do not replace the "
            "consensus and inter-rater validation that a reporting guideline requires.")
        D.caption("Table 5. Effects measured in this study that each QML-Health-READY domain would "
                  "detect.")
        D.table(["Domain", "Item tested", "Measured effect in this study"], [
            ("R† Rationale", "Data-complexity analysis",
             f"Geometric difference {min(g_vals):.1f}–{max(g_vals):.1f} against a √n scale of "
             f"≈17, and lower quantum than RBF kernel–target alignment on "
             f"{_of6(len(kta_below))}: no data-driven advantage was expected, and none was seen."),
            ("E† Encoding", "Bandwidth tuning and input scaling",
             f"Fixing the bandwidth at 1 lowered QSVM AUROC by {min(b1_drop):.2f}–"
             f"{max(b1_drop):.2f}, to near chance."),
            ("A† Algorithm & baselines", "Kernel-matched comparator; entanglement ablation",
             f"The NHANES deficit was {_d(S['QSVM_vs_best'][nh]['mean_diff'])} against the best "
             f"classical model but {_d(S['QSVM_vs_SVM_RBF'][nh]['mean_diff'])} (not significant) "
             f"against the RBF SVM; the product-state kernel matched the entangling one."),
            ("D† Data & splits", "Leakage audit (post-baseline predictors)",
             f"Including heart-failure follow-up time inflated classical AUROC by "
             f"{min(hf_leak):.2f}–{max(hf_leak):.2f}."),
            ("Y† Yield", "Pre-specified equivalence margin",
             f"QSVM–RBF SVM equivalence held on {len(mt_eq05)} of 6 datasets at ±0.05 but "
             f"{len(mt_eq02)} at ±0.02 and {len(count(S, 'QSVM_vs_SVM_RBF', 'Equivalent', 0.01))} "
             f"at ±0.01."),
            ("R† Robustness", "Explicit noise and finite shots",
             f"Exact depolarising noise (p = 0.01) changed AUROC by at most {noise01:.3f}; 1,024 "
             f"shots lowered it by a pooled {ps['mean_diff']:.3f}."),
            ("E Evaluation in context", "Threshold-dependent metrics",
             f"On NHANES the QSVM reached AUROC {R[nh]['QSVM']['auc']['mean']:.2f} but MCC "
             f"{R[nh]['QSVM']['mcc']['mean']:.3f} and balanced accuracy "
             f"{R[nh]['QSVM']['bacc']['mean']:.2f}: at its default threshold it predicted almost "
             f"only the majority class."),
            ("Y Yardstick for cost", "Circuit executions per fit",
             f"{min(circ.values()) / 1e6:.2f}–{max(circ.values()) / 1e6:.1f} million kernel-entry "
             f"circuits per tuned outer split, each repeated for many shots."),
        ], size=8.5, widths=[1.05, 1.45, 3.5])

    # ---------------- 6 Discussion
    D.h(1, "6. Discussion")
    D.h(2, "6.1 Principal findings")
    D.p("In a matched, paired design, correctly configured quantum classifiers approached tuned "
        "classical models but did not exceed them, and parity was supported only at the "
        "conventional ±0.05 margin. Three observations are consistent with this. First, inner "
        "cross-validation drove the quantum kernel to small bandwidths, where it behaves like an "
        "RBF or low-order polynomial kernel [@canatar;@florez]. Second, the classically trivial "
        "product-state kernel performed as well as the entangling kernel. Third, the geometric "
        "difference was small, so Huang et al.’s necessary condition for a data-driven "
        "advantage was not met [@huang].")
    D.h(2, "6.2 Relation to previous work")
    D.p("Our results agree with recent independent benchmarks: "
        "Kakavand et al. found no significant advantage in 29 comparisons and quantum-kernel "
        "spectra that were too flat or too concentrated, with the one competitive configuration "
        "costing about 2,000 times more computation [@kakavand]; Leither et al. found no "
        "advantage on oncological data [@leither]; and Schnabel and Roth showed, across more than "
        "20,000 quantum-kernel models, how strongly hyperparameters drive performance [@schnabel]. "
        "Several of our mechanistic findings were anticipated by theory or by benchmarks outside "
        "medicine: the classical character of bandwidth-tuned kernels [@florez], exponential "
        "concentration [@thanasilp] and the shot cost of kernel estimation [@miroszewski]. What "
        "this study adds is their confirmation on clinical data under a matched, "
        "equivalence-tested design; a kernel-matched comparison showing that apparent quantum "
        "deficits can belong to the learner rather than the kernel; a measured shot penalty and "
        "its interaction with bandwidth tuning; and an experiment showing how far configuration "
        "faults alone move the conclusion.")
    if v3:
        D.h(2, "6.3 Measurement cost")
        D.p(f"Finite-shot estimation, not depolarising noise, was the binding constraint. Exact "
            f"noise at p = 0.01 changed AUROC by at most {noise01:.3f}, whereas adding 1,024 shots "
            f"per kernel entry lowered it by {min(shot_pen.values()):.3f}–{max(shot_pen.values()):.3f}. "
            f"The penalty in the shot sweep was largest on the datasets whose tuned kernels were "
            f"most compressed ({_names(compressed)}; off-diagonal standard deviation "
            f"{min(spread[d] for d in compressed):.3f}–{max(spread[d] for d in compressed):.3f}), "
            f"where, at the bandwidth selected without shot noise, even {max_shots:,} shots left "
            f"AUROC {min(sweep_gap[d] for d in compressed):.2f}–"
            f"{max(sweep_gap[d] for d in compressed):.2f} below the noiseless value. When bandwidth "
            f"was re-tuned under shots, as in the main finite-shot arm, inner cross-validation chose "
            f"a larger bandwidth than in the noiseless case in {_pct(frac_shot_larger)} of fits, "
            f"recovering part of the loss. This is consistent with a trade-off "
            f"between bandwidth and measurement cost: the small bandwidths that help quantum kernels "
            f"generalise leave differences between kernel entries that finite sampling must "
            f"resolve [@thanasilp;@canatar;@miroszewski]. Exact state-vector simulation hides this cost, so "
            f"finite-shot evaluation belongs in the Robustness domain of any QML benchmark.")
    svm_weak = [d for d in DATASETS
                if max(R[d]["SVM_RBF"]["auc"]["mean"], R[d]["SVM_POLY"]["auc"]["mean"])
                < min(R[d][k]["auc"]["mean"] for k in ("LR", "RF", "GB"))]
    D.h(2, f"6.{4 if v3 else 3} Comparator choice and implementation sensitivity")
    D.p("The comparison with the kernel-matched RBF SVM matters. Against the best of five "
        "classical models, a quantum kernel can appear inferior because SVMs as a class are "
        "unsuited to a dataset"
        + (f", as on {_names(svm_weak)}, where logistic regression and both tree ensembles beat "
           f"every SVM" if svm_weak else "") +
        ". We recommend that QSVM studies report both comparisons."
        + (" The implementation-sensitivity experiment shows that an untuned bandwidth or "
           "unscaled inputs alone produce near-chance results, so published reports of quantum "
           "models failing should be read with the encoding and bandwidth in mind. Because "
           "configuration choices also determine which classical comparators look strong, the "
           "same caution applies to reported successes." if v3 else
           " In an earlier, superseded version of our own pipeline, an untuned encoding and an "
           "inactive noise channel produced near-chance QSVM results on the same data, illustrating "
           "how implementation errors can masquerade as findings."))
    D.h(2, f"6.{5 if v3 else 4} Scope and clinical implications")
    D.p("These conclusions apply to 4-qubit simulations of small, public tabular datasets"
        + (" (with a 12-qubit check on two datasets)" if v3 else "") +
        ". They say nothing about barren plateaus, which such small circuits cannot exhibit "
        "[@mcclean;@cerezo_bp], or about fault-tolerant devices. Where quantum methods may help is "
        "summarised in Supplementary Table S7 and Figure S1: molecular simulation is the most "
        "plausible long-term application, high-dimensional omics a possible one requiring "
        "per-dataset complexity analysis, and routine tabular risk prediction and imaging "
        "unlikely beneficiaries in the NISQ era. Clinical use of any QML model would also be "
        "subject to the principles of trustworthy healthcare AI [@futureai] and to medical-device "
        "regulation, and quantum infrastructure is concentrated in a few high-income settings.")
    D.h(2, f"6.{6 if v3 else 5} Recommendations and future work")
    D.p("Authors should give quantum and classical models identical inputs, splits and tuning "
        "budgets; report a kernel-matched classical comparator; report and check bandwidth "
        "selection; verify that noise channels are active and model finite shots; use "
        "split-corrected paired tests with equivalence margins; and complete the QML-Health-READY "
        "checklist. Reviewers should request it, and funders should prioritise multi-site and "
        "molecular-simulation studies over further single-dataset classification.")
    D.p("Four extensions would test how far these findings generalise. The first is evaluation on "
        "quantum hardware with calibrated noise and error mitigation, where shot budgets are real "
        "rather than simulated. The second is a larger collection of clinical datasets, including "
        "electronic health record cohorts with external validation, which would allow "
        "cross-dataset inference rather than per-dataset tests. The third is higher-dimensional "
        "omics data with more qubits, the setting our triage rates as most plausible for a "
        "data-driven advantage. The fourth is validation of QML-Health-READY itself, through a "
        "Delphi consensus process and independent dual-rater application to published studies.")

    # ---------------- 7 Limitations
    D.h(1, "7. Limitations")
    lim = [
        "All quantum results come from classical simulation of 4-qubit circuits"
        + (" (12 qubits in a secondary experiment)" if v3 else "") +
        ". Noise was generic depolarising noise"
        + (" with a binomial shot model" if v3 else " with exact expectation values") +
        "; hardware-calibrated noise, readout error and real hardware were not studied.",
        "Restricting every model to four principal components ensures a matched comparison but is "
        "clinically artificial"
        + ("; all-predictor classical results are reported for context." if v3 else
           "; classical models on all predictors were not evaluated."),
        "Equivalence margins are conventional rather than clinically derived; the best classical "
        "model is selected post hoc; single-model confidence intervals treat dependent splits as "
        "independent.",
        ("The VQC used one initialisation per split and a small tuning grid (four "
         "configurations), and its entanglement ablation used the fixed configuration." if vt else
         "The VQC was not tuned and used one initialisation per split"
         + ("." if v3 else ", and its ablation also removed three of four inputs from the readout.")),
        (f"The selected bandwidth frequently lay at the lower edge of the grid ({_pct(frac_min)} of "
         f"fits at {bw_min})." if frac_min >= 0.2 else
         f"The bandwidth grid was searched down to {bw_min}; {_pct(frac_min)} of fits chose the "
         f"minimum, so the optimum is unlikely to lie outside the grid."),
        "The datasets are small public sets without external validation, calibration or "
        "decision-curve analysis; NHANES survey weights were not applied"
        + ("." if v3 else "; Pima contains zero-coded missing values and the heart-failure data "
           "include follow-up time, a post-baseline predictor."),
        "The literature synthesis relies on published reviews and benchmarks rather than a new "
        "systematic search; QML-Health-READY has not been validated for inter-rater reliability, "
        "and a Delphi process and pilot application to published studies are needed.",
    ]
    for t in lim:
        D.bullet(t)

    # ---------------- 8 Conclusion
    D.h(1, "8. Conclusion")
    D.p("On six clinical datasets, correctly configured quantum classifiers matched tuned "
        "classical models within a conventional margin but never outperformed them, and their "
        "performance did not depend on entanglement."
        + (" Their main practical cost was measurement: estimating kernel entries from a "
           "realistic number of shots erased the parity seen in exact simulation." if v3 else "") +
        " Implementation choices, not quantum effects, determined whether they appeared to fail "
        "or to succeed. QML-Health-READY sets out the reporting needed to tell these apart.")

    # ---------------- declarations
    D.h(1, "Declarations")
    for head, txt in [
        ("CRediT authorship contribution statement", "Mohammed Siraj B: Conceptualization, "
         "Methodology, Software, Formal analysis, Investigation, Data curation, Visualization, "
         "Writing – original draft. Ruban S: Supervision, Validation, Writing – review and "
         "editing. Both authors approved the final manuscript."),
        ("Funding", "This research did not receive any specific grant from funding agencies in the "
         "public, commercial, or not-for-profit sectors."),
        ("Declaration of competing interest", "The authors declare that they have no known "
         "competing financial interests or personal relationships that could have appeared to "
         "influence the work reported in this paper."),
        ("Data availability", "All datasets are public: UCI Machine Learning Repository "
         "(archive.ics.uci.edu), UCSC Xena (xenabrowser.net) and CDC NHANES "
         "(wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/)."),
        ("Code availability", "Code and per-split results: github.com/sirajbb/qml-health-ready; "
         "archived at Zenodo [new version DOI to be inserted]."),
        ("Declaration of generative AI and AI-assisted technologies in the manuscript preparation "
         "process", "During the preparation of this work the authors used Claude (Anthropic) to "
         "assist with code development, code review and drafting of the manuscript. After using "
         "this tool, the authors reviewed and edited the content as needed and take full "
         "responsibility for the content of the published article."),
        ("Ethics statement", "Only public, de-identified data were used; ethics approval was not "
         "required."),
    ]:
        D.h(2, head)
        D.p(txt)
    D.h(1, "Acknowledgements")
    D.p("None.")

    # ---------------- references (order of first citation)
    D.h(1, "References")
    unused = [k for k in REFS if k not in cite.order]
    if unused:
        raise RuntimeError(f"Uncited references: {unused}")
    for i, k in enumerate(cite.order, 1):
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Pt(18)
        par.paragraph_format.first_line_indent = Pt(-18)
        r = par.add_run(f"{i}. {REFS[k]}")
        r.font.size = Pt(9.5)
    doc.save(str(OUT_MAIN))
    print(f"Main -> {OUT_MAIN}  (abstract {words} words, {len(cite.order)} references)")

    build_supplement(R, M, v3, best, best_full, S, cfg, comp, gkey, cite)
    if v3:
        build_cover_letter(mt_eq05, mt_eq02, shot_pen, R)


def build_cover_letter(mt_eq05, mt_eq02, shot_pen, R):
    from docx import Document
    from docx.shared import Pt, RGBColor
    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(11)

    def para(text, action=False):
        r = doc.add_paragraph().add_run(text)
        if action:
            r.bold = True
            r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    fault = [R[d]["QSVM_fault_b1"]["auc"]["mean"] for d in DATASETS]
    para("[Date]", action=True)
    para("The Editor-in-Chief\nComputers in Biology and Medicine")
    para("Dear Editor,")
    para(f"We are pleased to submit our manuscript, “{TITLE}”, for consideration as a "
         f"Full Length Article in Computers in Biology and Medicine.")
    para("Quantum machine learning is increasingly proposed for clinical prediction, yet published "
         "comparisons with classical models are rarely paired, seldom test equivalence, and depend "
         "on implementation choices that are seldom reported. We compared a quantum-kernel support "
         "vector machine and a variational quantum classifier with five tuned classical models on "
         "six public clinical datasets, using identical inputs and identical cross-validation "
         "splits, split-corrected paired tests and Holm-adjusted equivalence tests. We also propose "
         "QML-Health-READY, an eight-domain reporting framework.")
    para(f"The main findings are as follows. No quantum model significantly outperformed a "
         f"classical comparator. Against a kernel-matched classical SVM, the quantum-kernel model "
         f"was equivalent within ±0.05 AUROC on {len(mt_eq05)} of six datasets, but within "
         f"±0.02 on only {len(mt_eq02)}. A classically trivial product-state kernel performed as "
         f"well as the entangling one. Estimating kernel entries from 1,024 measurement shots "
         f"lowered AUROC by {min(shot_pen.values()):.2f}–{max(shot_pen.values()):.2f}, whereas "
         f"gate noise had little effect. Finally, common configuration faults alone reduced the "
         f"quantum model to near-chance performance ({min(fault):.2f}–{max(fault):.2f}), showing "
         f"how implementation can reverse a study’s conclusion.")
    para("We believe the work suits Computers in Biology and Medicine because it addresses how "
         "computational methods for biomedical data should be evaluated and reported, and it "
         "provides a fully reproducible benchmark that other groups can extend. All code and "
         "per-split results are openly available on GitHub and archived at Zenodo. The analysis "
         "history, including corrections made after an internal review of an earlier version of "
         "our own pipeline, is reported in full in the supplementary material.")
    para("This manuscript is original, has not been published previously, and is not under "
         "consideration elsewhere. Both authors have approved the submission. The authors declare "
         "no competing interests and received no specific funding. Generative AI assistance is "
         "declared in the manuscript in accordance with the journal’s policy.")
    para("[If you post a preprint, add: “A preprint is available at <server and DOI>.” "
         "Otherwise delete this paragraph.]", action=True)
    para("Thank you for considering our manuscript.")
    para("Yours sincerely,\n\nMohammed Siraj B (corresponding author), on behalf of both authors\n"
         "School of Engineering, St Aloysius (Deemed to be University), Mangalore 575003, "
         "Karnataka, India\nsiraj019@gmail.com")
    path = OUT_MAIN.with_name("Cover_Letter.docx")
    doc.save(str(path))
    print(f"Cover letter -> {path}")


def build_supplement(R, M, v3, best, best_full, S, cfg, comp, gkey, cite):
    D = Doc(cite)
    D.h(0, "Supplementary Material")
    D.p(TITLE, cite=False)

    D.caption("Table S1. QML-Health-READY readiness tiers.")
    D.table(["Tier", "Definition", "Implication"], [
        ("0 – Exploratory", "Fails one or more Core items", "No claim of advantage permissible."),
        ("1 – Reportable", "Meets all Core items; no significant advantage",
         "Valid parity or null result; publishable."),
        ("2 – Candidate", "Meets all Core items; significant, robust advantage under noise-aware "
         "evaluation on external data", "Justifies prospective study and independent replication."),
    ], size=9)

    D.caption("Table S2. Paired AUROC differences for ablations (corrected 95% CI; Holm-adjusted "
              "corrected-t p).")
    cols = [("QSVM_vs_QSVM_noisy", "QSVM: noiseless − noisy (p = 0.01)"),
            ("QSVM_vs_QSVM_noent", "QSVM: entangling − product-state"),
            ("VQC_vs_VQC_noent", "VQC: entangling − rotations only")]
    if v3:
        cols.insert(1, ("QSVM_vs_QSVM_noisy_shots", "QSVM: noiseless − noisy + shots"))
    D.table(["Dataset"] + [c[1] for c in cols],
            [[DS[d]] + [f"{_d(S[c][d]['mean_diff'])} [{_d(S[c][d]['ci95'][0])}, "
                        f"{_d(S[c][d]['ci95'][1])}], "
                        + ("p < 0.001" if S[c][d]['p_t_holm'] < 0.001 else f"p = {_p(S[c][d]['p_t_holm'])}")
                        for c, _ in cols]
             for d in DATASETS], size=8)

    D.caption("Table S3. Kernel and data-complexity metrics (stratified sample of up to 300).")
    hdr = ["Dataset", "Fisher ratio (max)", "1-NN agreement", "KTA quantum", "KTA RBF", "g (λ=10⁻³)",
           "√n", "Mean off-diagonal K_Q"]
    rows = []
    for d in DATASETS:
        c = comp[d]
        rows.append([DS[d], f"{c['F1_fisher_max']:.2f}", f"{c['N2_nn_purity']:.2f}",
                     f"{c['KTA_quantum']:.3f}", f"{c['KTA_rbf']:.3f}", f"{gkey(c):.2f}",
                     f"{math.sqrt(c['n_used']):.1f}", f"{c['kq_offdiag_mean']:.2f}"])
    if v3:
        hdr += ["g (λ=10⁻⁴ / 10⁻² / 10⁻¹)", "K_Q–RBF alignment"]
        for row, d in zip(rows, DATASETS):
            g = comp[d]["geom_diff_CQ"]
            row += [f"{g['0.0001']:.1f} / {g['0.01']:.1f} / {g['0.1']:.1f}",
                    f"{comp[d]['kq_vs_rbf_alignment_best']:.2f}"]
    D.table(hdr, rows, size=7.5)

    metrics = ["acc", "f1", "mcc"] + (["bacc"] if v3 else [])
    mname = {"acc": "Accuracy", "f1": "Macro-F1", "mcc": "MCC", "bacc": "Balanced accuracy"}
    D.caption("Table S4. Threshold-dependent metrics (mean over splits). Thresholds differ between "
              "models, so values are descriptive.")
    rows = []
    for d in DATASETS:
        for m in dict.fromkeys((best[d], "SVM_RBF", "QSVM", "VQC")):
            rows.append([DS[d], ML[m]] + [f"{R[d][m][k]['mean']:.3f}" for k in metrics])
    D.table(["Dataset", "Model"] + [mname[k] for k in metrics], rows, size=8)

    D.caption("Table S5. Hyperparameter grids (3-fold inner cross-validation on AUROC).")
    D.table(["Model", "Grid"], [
        ("Logistic regression", "C ∈ {0.01, 0.1, 1, 10}"),
        ("SVM (RBF)", "C ∈ {0.1, 1, 10, 100}; γ ∈ {scale, auto, 0.01, 0.1}"),
        ("SVM (polynomial)", "C ∈ {0.1, 1, 10}; degree ∈ {2, 3}; coef0 ∈ {0, 1}"),
        ("Random forest", "trees ∈ {100, 200}; depth ∈ {None, 5, 10}; min leaf ∈ {1, 3}"),
        ("Gradient boosting", "trees ∈ {100, 200}; learning rate ∈ {0.05, 0.1, 0.2}; depth ∈ {3, 5}"),
        ("QSVM (all variants)", f"bandwidth ∈ {{{', '.join(str(b) for b in cfg['bandwidths'])}}}; "
                                f"C ∈ {{{', '.join(str(c) for c in cfg['qsvm_C'])}}}"),
        ("VQC", (f"(layers, Adam learning rate) ∈ {_vgrid_of(R)}; {cfg['vqc_epochs']} epochs; "
                 f"fixed configuration for sensitivity and ablation: {cfg['vqc_layers']} layers, "
                 f"lr {cfg['vqc_lr']}") if "VQC_tuned" in R["breast_cancer"] else
         f"fixed: {cfg['vqc_layers']} layers, Adam lr {cfg['vqc_lr']}, {cfg['vqc_epochs']} epochs"),
    ], size=9)

    D.caption("Table S6. QML-Health-READY self-assessment of this study.")
    D.table(["Domain", "Status", "Justification"], [
        ("R† Rationale", "Yes", "Geometric difference, kernel alignment and Fisher/1-NN metrics."),
        ("E† Encoding", "Yes", "Encoding, scaling, bandwidth grid and selection reported; "
                                   "grid-edge selections disclosed."),
        ("A† Algorithm & baselines", "Yes", "Five tuned classical models on identical inputs, "
                                                "kernel-matched comparator, entanglement ablation."),
        ("D† Data & splits", "Yes", "Provenance, prevalence and leakage handling reported; TCGA "
                                        "restricted to primary tumours, one per patient."
         + (" Post-baseline heart-failure predictor removed." if v3 else
            " Heart-failure follow-up time retained (limitation).")),
        ("Y† Yield", "Yes", "25 identical paired splits, corrected tests, TOST at three margins, "
                                 "Holm adjustment."),
        ("R† Robustness", "Partial", "Depolarising noise verified active"
         + (" with a finite-shot model" if v3 else "") + "; no hardware-calibrated noise or hardware."),
        ("E Evaluation in context", "No", "No calibration, decision-curve analysis or external validation."),
        ("Y Yardstick for cost", "Partial", "Simulation time and circuit counts; no hardware cost."),
    ], size=9)
    D.p("Because the Robustness Core item is only partly met, this study is Tier 0 by a strict "
        "reading and would reach Tier 1 with a hardware-calibrated noise model.")

    D.caption("Table S7. Use-case triage (authors’ reasoned judgement, not a meta-analytic finding).")
    D.table(["Use-case", "Argument for quantum benefit", "Outlook"], [
        ("Molecular simulation, drug discovery", "Native simulation of quantum systems; no "
         "classical-data loading.", "Most plausible; needs fault tolerance."),
        ("High-dimensional omics", "Large feature spaces at small n; geometric difference may be "
         "non-negligible.", "Possible; needs per-dataset complexity analysis."),
        ("Combinatorial optimisation", "QUBO formulations; heuristic QAOA.", "Speculative."),
        ("Neurological and mental-health signals", "Small, high-dimensional data; high leakage risk.",
         "Open; strict subject-level evaluation needed."),
        ("Routine tabular risk prediction", "Strong classical models; loading overhead; parity at "
         "best in this study.", "Unlikely in the NISQ era."),
        ("Medical imaging", "Heavy compression into few qubits.", "Unlikely near term."),
    ], size=9)
    D.figure("figS1_triage.png", "Figure S1. Use-case triage matrix (authors’ judgement).", width=5.5)
    if v3 and any(M[d].get("qubit_scaling") for d in DATASETS):
        D.figure("figS2_qubit_scaling.png",
                 "Figure S2. QSVM (tuned bandwidth) and RBF SVM AUROC against the number of "
                 "qubits/principal components (first-seed folds).", width=6.0)

    D.caption("Table S8. Analysis history. Results from versions 1 and 2 are superseded and not "
              "reported in the main text; the code of every version is preserved in the "
              "repository history, and version 1 modules are kept in legacy_v1/.")
    hist = [
        ("1 (withdrawn)", "Initial pipeline", "Faults found on code review: CZ–Rz–CZ block "
         "(no entanglement); noise argument ignored (noisy = ideal); no feature scaling or bandwidth; "
         "quantum models trained on 50 (QSVM) or 25 (VQC) samples; one-sided Wilcoxon on unpaired "
         "test sets; HbA1c used as an NHANES predictor (target leakage)."),
        ("2 (superseded)", "Corrected pipeline", "Entangling ZZ map; density-matrix noise; tuned "
         "bandwidth (0.05–1); full training folds; identical paired splits; corrected t-test and "
         "TOST with a pre-specified ±0.05 margin. Remaining issues: heart-failure follow-up "
         "time as a predictor; Pima zeros not treated as missing; VQC ablation confounded by "
         "single-qubit readout; bandwidth optimum at the grid edge."),
        ("3 (reported)", "Revised after internal review", "Follow-up time removed; Pima imputation; "
         "bandwidth grid to 0.005; all-qubit VQC readout; margins ±0.01 and ±0.02 and the "
         "kernel-matched comparator added; finite-shot arm and sweep; all-predictor classical "
         "models; implementation-sensitivity and qubit-scaling experiments; TCGA restricted to "
         "primary tumours (no change: all samples qualified)."),
    ]
    if "SVM_RBF_df" in R["breast_cancer"] or "QSVM_fault_minmax" in R["breast_cancer"]:
        hist.append(("3b (reported)", "Supplementary run", "RBF SVM additionally scored by its "
                     "decision function for the kernel-matched comparison; tutorial-style "
                     "min–max [0, π] QSVM configuration added to the fault experiment."))
    D.table(["Version", "Description", "Changes and known issues"], hist, size=8.5)
    D.doc.save(str(OUT_SUPP))
    print(f"Supplement -> {OUT_SUPP}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS
    R = json.loads(path.read_text())
    if len(sys.argv) > 2:
        OUT_MAIN = Path(sys.argv[2])
        OUT_SUPP = OUT_MAIN.with_name(OUT_MAIN.stem + "_Supplementary.docx")
    build(R)
