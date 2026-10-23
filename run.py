"""Reproduce LAIS innovation prediction and illustrative review-cost experiments."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 20261023
OUT = Path(__file__).resolve().parent
FEATURES = ["country", "year", "dictexp", "drdintexp", "dIPexp"]
DATA = pd.read_csv(os.environ.get("LAIS_CSV", "lais-public.csv"), usecols=FEATURES + ["dnewpd"], low_memory=False)
DATA = DATA.loc[DATA["dnewpd"].isin([0, 1])].copy()
X, y = DATA[FEATURES], DATA["dnewpd"].astype(int)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=SEED)

def pipeline(model):
    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["country", "year"]),
        ("numeric", SimpleImputer(strategy="most_frequent", add_indicator=True), ["dictexp", "drdintexp", "dIPexp"]),
    ], sparse_threshold=0)
    return Pipeline([("preprocessing", prep), ("classifier", model)])

candidates = {
    "Logistic regression": LogisticRegression(max_iter=500, random_state=SEED),
    "Histogram boosting": HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15, l2_regularization=2, random_state=SEED),
}
rows = []
for name, estimator in candidates.items():
    fit = pipeline(estimator).fit(X_train, y_train)
    prob = fit.predict_proba(X_val)[:, 1]
    rows.append(dict(model=name, validation_auc=roc_auc_score(y_val, prob),
                     validation_pr_auc=average_precision_score(y_val, prob),
                     validation_brier=brier_score_loss(y_val, prob)))
table1 = pd.DataFrame(rows)
table1.to_csv(OUT / "table1_model_validation.csv", index=False)
best = table1.sort_values("validation_brier").iloc[0]["model"]
fit = pipeline(candidates[best]).fit(X_dev, y_dev)
p = fit.predict_proba(X_test)[:, 1]
truth = y_test.to_numpy()
MISS, FALSE_ALARM, REVIEW = 5., 1., .2

def decisions(prob, budget, policy):
    cost0, cost1 = prob * MISS, (1 - prob) * FALSE_ALARM
    action = np.where(cost0 <= cost1, 0, 1)
    count = int(np.floor(len(prob) * budget))
    if policy == "confidence":
        chosen = np.argsort(np.abs(prob - .5), kind="stable")[:count]
    else:
        gain = np.minimum(cost0, cost1) - REVIEW
        eligible = np.flatnonzero(gain > 0)
        chosen = eligible[np.argsort(-gain[eligible], kind="stable")[:count]]
    action[chosen] = 2
    return action

def loss(action):
    return np.where(action == 2, REVIEW, np.where(action == 0, truth * MISS, (1 - truth) * FALSE_ALARM))

records = []
for budget in np.arange(0, .401, .05):
    for policy in ("confidence", "knowledge"):
        action = decisions(p, budget, policy)
        records.append(dict(budget=round(float(budget), 2), policy=policy,
                            review_rate=float(np.mean(action == 2)),
                            simulated_loss=float(loss(action).mean()),
                            review_count=int(np.sum(action == 2))))
table2 = pd.DataFrame(records)
table2.to_csv(OUT / "table2_policy_scenarios.csv", index=False)
summary = dict(total_rows=119900, outcome_observed_n=len(DATA), outcome_positive_rate=float(y.mean()),
               train_n=len(X_train), validation_n=len(X_val), test_n=len(X_test), selected_model=best,
               test_auc=float(roc_auc_score(truth, p)), test_pr_auc=float(average_precision_score(truth, p)),
               test_brier=float(brier_score_loss(truth, p)), costs=dict(miss=MISS, false_alarm=FALSE_ALARM, review=REVIEW),
               variables=FEATURES, outcome="dnewpd")
(OUT / "results.json").write_text(json.dumps(summary, indent=2))
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig, ax = plt.subplots(figsize=(9, 2.8))
names = ["LAIS firm-level\ninnovation indicators", "Observed product-\ninnovation outcome",
         "Validated predictive\nmodel", "Hypothetical cost +\nreview-capacity rules"]
for i, name in enumerate(names):
    ax.text(i, 0, name, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=.55", facecolor="#ecf2f8", edgecolor="#315d83"))
    if i:
        ax.annotate("", xy=(i-.37,0), xytext=(i-.63,0),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#315d83"))
ax.set(xlim=(-.5,3.5), ylim=(-.65,.65))
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "figure1_framework.png", dpi=240, bbox_inches="tight")
plt.close(fig)
fig, ax = plt.subplots(figsize=(7.4,4.2))
for policy, subset in table2.groupby("policy"):
    ax.plot(subset.review_rate * 100, subset.simulated_loss, "o-", lw=2, label=policy.title())
ax.set(xlabel="Actually reviewed firms (%)", ylabel="Simulated mean loss per firm",
       title="Illustrative decision-cost scenario (perfect review)")
ax.grid(alpha=.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figure2_policy_tradeoff.png", dpi=240)
plt.close(fig)
print(json.dumps(summary, indent=2))
print(table1.to_string(index=False))
print(table2.loc[table2.budget == .2].to_string(index=False))
