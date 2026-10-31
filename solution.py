"""
Machine Learning Assignment 2 — Bagging, Boosting, Bias–Variance
Roll number: m26csa004

Run:
    python solution.py
Dependencies:
    numpy, scipy, scikit-learn, matplotlib, pandas

The script downloads/fetches Olivetti Faces on first run if it is not cached.
All reported metrics are computed from the experiment; none are hard-coded.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROLL_NO = "m26csa004"
SEED = int("".join(ch for ch in ROLL_NO if ch.isdigit()))
OUT = Path("results_m26csa004")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(SEED)
DEPTHS = [2, 4, 8, 16, None]
LEAF_SIZES = [1, 2, 4]
B_VALUES = [1, 5, 10, 25, 50, 100, 200]
BOOST_DEPTHS = [1, 2, 3, 4]
BOOST_ROUNDS = [10, 25, 50, 100]
N_CLASSES = 40
CLASS_NAMES = np.arange(N_CLASSES)

def save_json(path, obj):
    def convert(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, default=convert))

def accuracy(y, pred):
    return float(np.mean(np.asarray(y) == np.asarray(pred)))

def make_split():
    data = fetch_olivetti_faces(shuffle=False)
    Xall = data.data.astype(np.float32)  # raw flattened 64x64 pixels
    yall = data.target.astype(int)
    rng = np.random.default_rng(SEED)
    split = {"train": [], "val": [], "test": []}
    # Exactly 10 images per identity; split six/two/two within each identity.
    for c in range(N_CLASSES):
        ids = np.flatnonzero(yall == c).copy()
        rng.shuffle(ids)
        split["train"].extend(ids[:6])
        split["val"].extend(ids[6:8])
        split["test"].extend(ids[8:10])
    for key in split:
        split[key] = np.array(sorted(split[key]), dtype=int)
    np.savez(OUT / "split_indices.npz", **split)
    return Xall, yall, split

def load_data():
    split_file = OUT / "split_indices.npz"
    data = fetch_olivetti_faces(shuffle=False)
    Xall, yall = data.data.astype(np.float32), data.target.astype(int)
    if split_file.exists():
        z = np.load(split_file)
        split = {k: z[k] for k in ("train", "val", "test")}
    else:
        # make_split reuses the same cached dataset and deterministic seed
        Xall, yall, split = make_split()
    Xtr, ytr = Xall[split["train"]], yall[split["train"]]
    Xv, yv = Xall[split["val"]], yall[split["val"]]
    Xt, yt = Xall[split["test"]], yall[split["test"]]
    Xmasked = Xt.copy()
    # centred 16x16 block, rows/columns 24..39 inclusive, set to zero
    imgs = Xmasked.reshape(-1, 64, 64)
    imgs[:, 24:40, 24:40] = 0.0
    Xmasked = imgs.reshape(len(Xmasked), -1)
    return Xtr, ytr, Xv, yv, Xt, yt, Xmasked, split, Xall, yall

def tree(depth, leaf):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)

def leaf_statistics(model, X):
    leaf_ids = model.apply(X)
    _, counts = np.unique(leaf_ids, return_counts=True)
    return {
        "n_leaves": int(model.get_n_leaves()),
        "mean_samples_per_leaf": float(len(X) / model.get_n_leaves()),
        "fraction_singleton_leaves": float(np.mean(counts == 1)),
    }

def vote_predictions(pred_matrix):
    # pred_matrix: (n_models, n_samples)
    out = np.empty(pred_matrix.shape[1], dtype=int)
    conf = np.empty(pred_matrix.shape[1], dtype=float)
    counts_all = []
    for i in range(pred_matrix.shape[1]):
        counts = np.bincount(pred_matrix[:, i], minlength=N_CLASSES)
        order = np.argsort(counts)
        out[i] = order[-1]
        conf[i] = counts[out[i]] / pred_matrix.shape[0]
        counts_all.append(counts)
    return out, conf, np.asarray(counts_all)

def pairwise_disagreement(pred_matrix):
    n = pred_matrix.shape[0]
    if n < 2: return 0.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(np.mean(pred_matrix[i] != pred_matrix[j]))
    return float(np.mean(vals))

def vote_margin(pred_matrix):
    margins = []
    for i in range(pred_matrix.shape[1]):
        cnt = np.bincount(pred_matrix[:, i], minlength=N_CLASSES) / pred_matrix.shape[0]
        s = np.sort(cnt)
        margins.append(s[-1] - s[-2])
    return float(np.mean(margins))

def fit_bagging(X, y, B, depth, leaf, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    models, samples = [], []
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        m = tree(depth, leaf)
        m.fit(X[idx], y[idx])
        models.append(m)
        samples.append(idx)
    return models, samples

def bag_predict(models, X):
    return np.vstack([m.predict(X).astype(int) for m in models])

def oob_predictions(models, samples, X, n_train):
    # For each original training item, use only trees that omitted it.
    pred_lists = [[] for _ in range(n_train)]
    for m, idx in zip(models, samples):
        in_bag = np.zeros(n_train, dtype=bool)
        in_bag[np.unique(idx)] = True
        oob_idx = np.flatnonzero(~in_bag)
        if len(oob_idx):
            p = m.predict(X[oob_idx]).astype(int)
            for i, pi in zip(oob_idx, p):
                pred_lists[i].append(int(pi))
    pred = np.full(n_train, -1, dtype=int)
    for i, vals in enumerate(pred_lists):
        if vals:
            pred[i] = np.bincount(vals, minlength=N_CLASSES).argmax()
    valid = pred >= 0
    return float(np.mean(pred[valid] == YTR[valid])) if np.any(valid) else float("nan")

# Global only used by the OOB helper so that its signature stays simple.
YTR = None

def plot_lines(df, x, ys, filename, title, ylabel):
    plt.figure(figsize=(7.2, 4.5))
    for y in ys:
        plt.plot(df[x], df[y], marker="o", label=y.replace("_", " "))
    plt.xlabel(x); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180); plt.close()

def run_q1(Xtr, ytr, Xv, yv, Xt, yt, Xmasked):
    global YTR
    YTR = ytr
    # Q1(a): depth/leaf grid; select validation accuracy only.
    rows, models = [], {}
    for d in DEPTHS:
        for leaf in LEAF_SIZES:
            m = tree(d, leaf); m.fit(Xtr, ytr)
            key = (d, leaf); models[key] = m
            rows.append({
                "max_depth": "None" if d is None else d,
                "min_samples_leaf": leaf,
                "train_accuracy": accuracy(ytr, m.predict(Xtr)),
                "validation_accuracy": accuracy(yv, m.predict(Xv)),
                "test_accuracy": accuracy(yt, m.predict(Xt)),
                **leaf_statistics(m, Xtr),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "q1a_tree_grid.csv", index=False)
    chosen_row = df.sort_values(["validation_accuracy", "train_accuracy"], ascending=False).iloc[0]
    d = None if chosen_row.max_depth == "None" else int(chosen_row.max_depth)
    leaf = int(chosen_row.min_samples_leaf)
    base = models[(d, leaf)]
    save_json(OUT / "selected_tree.json", {"max_depth": d, "min_samples_leaf": leaf,
        "validation_accuracy": float(chosen_row.validation_accuracy)})

    # Q1(b,c): fit up to 200 trees once; report prefix ensembles.
    bag_models, bag_samples = fit_bagging(Xtr, ytr, max(B_VALUES), d, leaf, SEED + 101)
    distinct = [len(np.unique(idx)) for idx in bag_samples]
    bag_rows, bag_val_preds = [], {}
    for B in B_VALUES:
        tr_pm = bag_predict(bag_models[:B], Xtr)
        va_pm = bag_predict(bag_models[:B], Xv)
        te_pm = bag_predict(bag_models[:B], Xt)
        tr_pred, _, _ = vote_predictions(tr_pm)
        va_pred, _, _ = vote_predictions(va_pm)
        te_pred, _, _ = vote_predictions(te_pm)
        bag_val_preds[B] = va_pred
        oob = oob_predictions(bag_models[:B], bag_samples[:B], Xtr, len(ytr))
        bag_rows.append({
            "B": B, "train_accuracy": accuracy(ytr, tr_pred),
            "validation_accuracy": accuracy(yv, va_pred),
            "test_accuracy": accuracy(yt, te_pred),
            "mean_distinct_bootstrap_examples": float(np.mean(distinct[:B])),
            "theoretical_expected_distinct": float(len(ytr) * (1 - (1 - 1/len(ytr))**len(ytr))),
            "oob_accuracy": oob,
            "pairwise_disagreement": pairwise_disagreement(va_pm),
            "mean_vote_margin": vote_margin(va_pm),
        })
    bdf = pd.DataFrame(bag_rows); bdf.to_csv(OUT / "q1b_c_bagging.csv", index=False)
    plot_lines(bdf, "B", ["train_accuracy", "validation_accuracy", "test_accuracy"],
               "q1_accuracy_vs_B.png", "Bagging accuracy vs ensemble size", "Accuracy")
    plot_lines(bdf, "B", ["oob_accuracy", "pairwise_disagreement", "mean_vote_margin"],
               "q1_oob_disagreement_margin.png", "OOB, disagreement and margin vs B", "Metric")
    best_B = int(bdf.sort_values(["validation_accuracy", "B"], ascending=[False, True]).iloc[0].B)
    save_json(OUT / "selected_bagging.json", {"B": best_B, "max_depth": d, "min_samples_leaf": leaf})
    final_models = bag_models[:best_B]

    # Q1(d): clean/masked evaluation, confidence, and illustrative face cases.
    base_clean = base.predict(Xt)
    base_masked = base.predict(Xmasked)
    bag_clean_pm = bag_predict(final_models, Xt)
    bag_mask_pm = bag_predict(final_models, Xmasked)
    bag_clean, bag_clean_conf, bag_clean_counts = vote_predictions(bag_clean_pm)
    bag_mask, bag_mask_conf, bag_mask_counts = vote_predictions(bag_mask_pm)
    robust = pd.DataFrame([
        {"model":"single_tree","test_type":"clean","accuracy":accuracy(yt, base_clean),
         "mean_winning_confidence":float(np.mean(np.max(base.predict_proba(Xt), axis=1)))},
        {"model":"bagging","test_type":"clean","accuracy":accuracy(yt, bag_clean),"mean_winning_confidence":float(np.mean(bag_clean_conf))},
        {"model":"single_tree","test_type":"masked","accuracy":accuracy(yt, base_masked),
         "mean_winning_confidence":float(np.mean(np.max(base.predict_proba(Xmasked), axis=1)))},
        {"model":"bagging","test_type":"masked","accuracy":accuracy(yt, bag_mask),"mean_winning_confidence":float(np.mean(bag_mask_conf))},
    ])
    robust.to_csv(OUT / "q1d_robustness.csv", index=False)

    # Select first two cases bagging corrects, and first two where both are wrong.
    corrected = np.flatnonzero((base_clean != yt) & (bag_clean == yt))[:2]
    both_wrong = np.flatnonzero((base_clean != yt) & (bag_clean != yt))[:2]
    cases = []
    for kind, ids in [("bagging_corrects_tree", corrected), ("both_wrong", both_wrong)]:
        for i in ids:
            cases.append({"case":kind,"test_position":int(i),"original_dataset_index":int(SPLIT["test"][i]),
                "true_label":int(yt[i]),"tree_prediction":int(base_clean[i]),
                "bagging_prediction":int(bag_clean[i]),"bagging_vote_counts":bag_clean_counts[i].tolist()})
    save_json(OUT / "q1d_face_cases.json", cases)
    return base, final_models, best_B, (d, leaf), bag_rows

SPLIT = None

class SAMME:
    """Multiclass SAMME AdaBoost with decision trees and explicit random-weak-learner handling."""
    def __init__(self, depth, n_estimators, random_state=SEED):
        self.depth, self.n_estimators, self.random_state = depth, n_estimators, random_state
        self.models, self.alphas, self.history = [], [], []
        self.classes_ = CLASS_NAMES

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n, K = len(y), len(self.classes_)
        w = np.full(n, 1/n, dtype=float)
        self.models, self.alphas, self.history = [], [], []
        for t in range(self.n_estimators):
            m = DecisionTreeClassifier(max_depth=self.depth, random_state=self.random_state + t)
            m.fit(X, y, sample_weight=w)
            pred = m.predict(X).astype(int)
            err = float(np.sum(w * (pred != y)))
            threshold = 1.0 - 1.0/K
            if err >= threshold - 1e-12:
                self.history.append({"round":t+1,"weighted_error":err,"alpha":0.0,
                    "max_sample_weight":float(w.max()),
                    "top10_weight":float(np.sort(w)[-min(10,n):].sum()),
                    "status":"stopped_not_better_than_random"})
                break
            # SAMME coefficient: log((1-error)/error) + log(K-1).
            if err <= 1e-15:
                alpha = float(np.log((1.0 - 1e-15)/1e-15) + np.log(K-1))
            else:
                alpha = float(np.log((1.0 - err)/err) + np.log(K-1))
            self.models.append(m); self.alphas.append(alpha)
            # Numerically stable SAMME weight update
            misclassified = pred != y

            log_w = np.full_like(w, -np.inf, dtype=float)
            positive = w > 0
            log_w[positive] = np.log(w[positive])

            log_w[misclassified] += alpha

            # Normalize in log space to avoid underflow/overflow.
            finite = np.isfinite(log_w)
            if not np.any(finite):
                raise ValueError("SAMME sample weights became invalid.")

            log_w -= np.max(log_w[finite])
            w = np.exp(log_w)
            w_sum = w.sum()

            if not np.isfinite(w_sum) or w_sum <= 0:
                raise ValueError("SAMME sample weights cannot be normalized.")

            w /= w_sum
            self.history.append({"round":t+1,"weighted_error":err,"alpha":alpha,
                "max_sample_weight":float(w.max()),
                "top10_weight":float(np.sort(w)[-min(10,n):].sum()),"status":"fit"})
            if err <= 1e-15:
                break
        return self

    def scores(self, X):
        S = np.zeros((len(X), N_CLASSES), dtype=float)
        for m, a in zip(self.models, self.alphas):
            p = m.predict(X).astype(int)
            S[np.arange(len(X)), p] += a
        return S

    def predict(self, X):
        return self.scores(X).argmax(axis=1)

    def confidence(self, X):
        S = self.scores(X)
        denom = sum(self.alphas)
        return S / denom if denom > 0 else np.zeros_like(S)

def tiny_samme_example():
    # 9 points, three classes; one complete round logged exactly.
    X = np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1],[3,0],[3,1],[4,0]], dtype=float)
    y = np.array([0,0,0,1,1,1,2,2,2], dtype=int)
    w = np.full(len(y), 1/len(y))
    stump = DecisionTreeClassifier(max_depth=1, random_state=SEED)
    stump.fit(X, y, sample_weight=w)
    pred = stump.predict(X).astype(int)
    err = float(np.sum(w * (pred != y)))
    alpha = float(np.log((1-err)/err) + np.log(2)) if 0 < err < 2/3 else 0.0
    updated = w * np.exp(alpha * (pred != y))
    updated /= updated.sum()
    save_json(OUT / "q2a_tiny_samme_round.json", {
        "X":X.tolist(),"y":y.tolist(),"initial_weights":w.tolist(),
        "weak_learner_predictions":pred.tolist(),"weighted_error":err,
        "random_threshold":2/3,"alpha":alpha,"updated_normalised_weights":updated.tolist(),
        "note":"If this stump's error is >= 2/3, alpha is set to zero for the illustration; the face-data SAMME implementation stops in that case."
    })

def run_q2(Xtr, ytr, Xv, yv, Xt, yt, Xmasked, base, bag_models, best_B):
    tiny_samme_example()
    # Fit every depth/round candidate independently; select validation accuracy only.
    candidates, fitted = [], {}
    for depth in BOOST_DEPTHS:
        for rounds in BOOST_ROUNDS:
            start = time.perf_counter()
            model = SAMME(depth, rounds, SEED + 2000 + depth*100 + rounds).fit(Xtr, ytr)
            elapsed = time.perf_counter() - start
            val_acc = accuracy(yv, model.predict(Xv))
            candidates.append({"depth":depth,"requested_rounds":rounds,"actual_rounds":len(model.models),
                "train_accuracy":accuracy(ytr, model.predict(Xtr)),"validation_accuracy":val_acc,
                "test_accuracy":accuracy(yt, model.predict(Xt)),"training_time_seconds":elapsed})
            fitted[(depth, rounds)] = model
    cdf = pd.DataFrame(candidates); cdf.to_csv(OUT / "q2b_boost_grid.csv", index=False)
    chosen = cdf.sort_values(["validation_accuracy","requested_rounds"], ascending=[False, True]).iloc[0]
    depth, rounds = int(chosen.depth), int(chosen.requested_rounds)
    boost = fitted[(depth, rounds)]
    save_json(OUT / "selected_boosting.json", {"depth":depth,"requested_rounds":rounds,
        "actual_rounds":len(boost.models),"validation_accuracy":float(chosen.validation_accuracy)})
    hdf = pd.DataFrame(boost.history); hdf.to_csv(OUT / "q2b_selected_round_history.csv", index=False)
    if len(hdf):
        plot_lines(hdf, "round", ["weighted_error","alpha","max_sample_weight","top10_weight"],
                   "q2b_boosting_diagnostics.png", "Selected SAMME diagnostics", "Value")

    # Q2(c): comparison at 25/50/100. Bagging uses same frozen tree setting.
    comp = []
    for n in [25, 50, 100]:
        bstart = time.perf_counter()
        bm, _ = fit_bagging(Xtr, ytr, n, *SELECTED_TREE, seed=SEED + 5000 + n)
        btime = time.perf_counter() - bstart
        bp = vote_predictions(bag_predict(bm, Xtr))[0]
        vp = vote_predictions(bag_predict(bm, Xv))[0]
        tp = vote_predictions(bag_predict(bm, Xt))[0]
        start = time.perf_counter()
        am = SAMME(depth, n, SEED + 6000 + n).fit(Xtr, ytr)
        atime = time.perf_counter() - start
        comp.append({"learners":n,"model":"bagging","train_accuracy":accuracy(ytr,bp),
            "validation_accuracy":accuracy(yv,vp),"clean_test_accuracy":accuracy(yt,tp),
            "training_time_seconds":btime})
        comp.append({"learners":n,"model":"boosting","train_accuracy":accuracy(ytr,am.predict(Xtr)),
            "validation_accuracy":accuracy(yv,am.predict(Xv)),"clean_test_accuracy":accuracy(yt,am.predict(Xt)),
            "training_time_seconds":atime})
    pd.DataFrame(comp).to_csv(OUT / "q2c_bagging_vs_boosting.csv", index=False)

    # Training face with largest final normalized boosting weight: replay predictions by each learner.
    if boost.models:
        # Reconstruct terminal weights by applying the same updates round-by-round.
        w = np.full(len(ytr), 1/len(ytr))
        for m, a in zip(boost.models, boost.alphas):
            pred = m.predict(Xtr).astype(int)
            w *= np.exp(a * (pred != ytr)); w /= w.sum()
        ix = int(np.argmax(w))
        hist = []
        for j, m in enumerate(boost.models):
            hist.append({"round":j+1,"prediction":int(m.predict(Xtr[ix:ix+1])[0]),
                         "true_label":int(ytr[ix]),"correct":bool(m.predict(Xtr[ix:ix+1])[0] == ytr[ix]),
                         "alpha":float(boost.alphas[j])})
        save_json(OUT / "q2c_high_weight_training_face.json", {
            "training_position":ix,"original_dataset_index":int(SPLIT["train"][ix]),
            "true_label":int(ytr[ix]),"final_weight":float(w[ix]),"prediction_history":hist})
    return boost, (depth, rounds)

def class_vector_predictions(model, X):
    # Tree probabilities mapped into the fixed 40-class order.
    proba = model.predict_proba(X)
    out = np.zeros((len(X), N_CLASSES), dtype=float)
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = proba[:, j]
    return out

def bag_vectors(models, X):
    # Hard-vote fraction vector.
    pm = bag_predict(models, X)
    out = np.zeros((len(X), N_CLASSES), dtype=float)
    for i in range(len(X)):
        out[i] = np.bincount(pm[:, i], minlength=N_CLASSES) / len(models)
    return out

def boost_vectors(model, X):
    return model.confidence(X)

def bias_variance(pred_vectors):
    # pred_vectors shape: (repetitions, test examples, classes)
    mean_p = pred_vectors.mean(axis=0)
    y_onehot = np.eye(N_CLASSES)[YT_FIXED]
    bias2 = float(np.mean(np.sum((mean_p - y_onehot)**2, axis=1)))
    variance = float(np.mean(np.sum((pred_vectors - mean_p[None,:,:])**2, axis=2)))
    return bias2, variance

YT_FIXED = None
SELECTED_TREE = (8, 1)

def resample_indices(ytr, rng):
    # For each identity, draw six samples with replacement from its six original train images.
    indices = []
    for c in range(N_CLASSES):
        ids = np.flatnonzero(ytr == c)
        indices.extend(rng.choice(ids, size=6, replace=True).tolist())
    return np.asarray(indices, dtype=int)

def run_q3(Xtr, ytr, Xt, yt, selected_base, selected_bag, selected_boost, tree_setting, boost_setting):
    global YT_FIXED
    YT_FIXED = yt
    reps = 30
    rng = np.random.default_rng(SEED + 9000)
    # Q3(a): selected models, 30 stratified bootstrap refits.
    vectors = {"single_tree":[], "bagging":[], "boosting":[]}
    accuracies = {"single_tree":[], "bagging":[], "boosting":[]}
    for r in range(reps):
        ix = resample_indices(ytr, rng)
        Xr, yr = Xtr[ix], ytr[ix]
        m = tree(*tree_setting).fit(Xr, yr)
        p = class_vector_predictions(m, Xt); vectors["single_tree"].append(p)
        accuracies["single_tree"].append(accuracy(yt, p.argmax(axis=1)))
        # Refit selected bagging with 100 trees when selected ensemble was smaller;
        # otherwise use the selected B, as Q3 asks for the final selected bagged model.
        B = max(1, len(selected_bag))
        bm, _ = fit_bagging(Xr, yr, B, *tree_setting, seed=SEED + 10000 + r)
        p = bag_vectors(bm, Xt); vectors["bagging"].append(p)
        accuracies["bagging"].append(accuracy(yt, p.argmax(axis=1)))
        bd, br = boost_setting
        am = SAMME(bd, br, SEED + 11000 + r).fit(Xr, yr)
        p = boost_vectors(am, Xt); vectors["boosting"].append(p)
        accuracies["boosting"].append(accuracy(yt, am.predict(Xt)))
    q3rows = []
    for name in vectors:
        b2, var = bias_variance(np.asarray(vectors[name]))
        q3rows.append({"model":name,"bias_squared":b2,"variance":var,
            "mean_test_accuracy":float(np.mean(accuracies[name])),
            "std_test_accuracy":float(np.std(accuracies[name], ddof=1))})
    pd.DataFrame(q3rows).to_csv(OUT / "q3a_bias_variance.csv", index=False)

    # Q3(b): shallow, selected, much deeper trees and 100-tree bags at each depth.
    selected_depth = tree_setting[0]
    if selected_depth is None: selected_depth = 1000
    depths = sorted(set([2, int(selected_depth), 32]))
    depth_rows = []
    depth_vectors = {}
    for depth in depths:
        for kind in ["single_tree", "bagging100"]:
            vecs, accs = [], []
            for r in range(reps):
                ix = resample_indices(ytr, rng); Xr, yr = Xtr[ix], ytr[ix]
                if kind == "single_tree":
                    m = tree(depth, tree_setting[1]).fit(Xr, yr)
                    p = class_vector_predictions(m, Xt)
                else:
                    bm, _ = fit_bagging(Xr, yr, 100, depth, tree_setting[1], seed=SEED+12000+r+depth)
                    p = bag_vectors(bm, Xt)
                vecs.append(p); accs.append(accuracy(yt, p.argmax(axis=1)))
            b2, var = bias_variance(np.asarray(vecs))
            depth_rows.append({"depth":depth,"model":kind,"bias_squared":b2,"variance":var,
                "mean_test_accuracy":float(np.mean(accs)),"std_test_accuracy":float(np.std(accs,ddof=1))})
            depth_vectors[(depth,kind)] = (b2,var,float(np.mean(accs)))
    ddf = pd.DataFrame(depth_rows); ddf.to_csv(OUT / "q3b_depth_bias_variance.csv", index=False)

    # Q2(d)/Q3(d): clean and masked fixed test set, confidence and summary table.
    Xmasked = Xt.reshape(-1,64,64).copy()
    Xmasked[:,24:40,24:40] = 0.0
    Xmasked = Xmasked.reshape(len(Xt),-1)
    # Note: selected_bag may be the selected ensemble prefix from Q1.
    bclean = vote_predictions(bag_predict(selected_bag, Xt))
    bmask = vote_predictions(bag_predict(selected_bag, Xmasked))
    rows = []
    for name, pred_clean, pred_mask, conf_clean, conf_mask in [
        ("single_tree", selected_base.predict(Xt), selected_base.predict(Xmasked),
         np.max(class_vector_predictions(selected_base, Xt),axis=1),
         np.max(class_vector_predictions(selected_base, Xmasked),axis=1)),
        ("bagging", bclean[0], bmask[0], bclean[1], bmask[1]),
        ("boosting", selected_boost.predict(Xt), selected_boost.predict(Xmasked),
         np.max(boost_vectors(selected_boost,Xt),axis=1),np.max(boost_vectors(selected_boost,Xmasked),axis=1))
    ]:
        rows.append({"model":name,"clean_test_accuracy":accuracy(yt,pred_clean),
            "masked_test_accuracy":accuracy(yt,pred_mask),"clean_mean_winning_vote_or_probability":float(np.mean(conf_clean)),
            "masked_mean_winning_vote_or_probability":float(np.mean(conf_mask)),
            "bias_squared":float(next(r["bias_squared"] for r in q3rows if r["model"]==name)),
            "variance":float(next(r["variance"] for r in q3rows if r["model"]==name))})
    pd.DataFrame(rows).to_csv(OUT / "q2d_q3d_final_summary.csv", index=False)

def main():
    global SPLIT, SELECTED_TREE
    Xtr, ytr, Xv, yv, Xt, yt, Xmasked, split, Xall, yall = load_data()
    SPLIT = split
    save_json(OUT / "run_config.json", {"roll_no":ROLL_NO,"seed":SEED,"features":"raw flattened 64x64 pixels",
        "train_per_identity":6,"validation_per_identity":2,"test_per_identity":2,
        "n_train":len(ytr),"n_validation":len(yv),"n_test":len(yt)})
    print(f"Roll={ROLL_NO}; seed={SEED}; train/val/test={len(ytr)}/{len(yv)}/{len(yt)}")
    base, bag_models, best_B, tree_setting, _ = run_q1(Xtr,ytr,Xv,yv,Xt,yt,Xmasked)
    SELECTED_TREE = tree_setting
    boost, boost_setting = run_q2(Xtr,ytr,Xv,yv,Xt,yt,Xmasked,base,bag_models,best_B)
    run_q3(Xtr,ytr,Xt,yt,base,bag_models[:best_B],boost,tree_setting,boost_setting)
    print(f"Done. Results, plots, logs, split and JSON configs are in: {OUT.resolve()}")
    print("Inspect CSV/JSON outputs before writing the report; all reported values are measured.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
