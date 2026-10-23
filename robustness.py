"""Reproduce subgroup, paired bootstrap and imperfect-review sensitivity checks."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 20261023
OUT = Path(__file__).resolve().parent
FEATURES = ["country", "year", "dictexp", "drdintexp", "dIPexp"]
data = pd.read_csv(os.environ.get("LAIS_CSV", "lais-public.csv"),
                   usecols=FEATURES + ["dnewpd"], low_memory=False)
data = data.loc[data.dnewpd.isin([0,1])].copy()
y = data.dnewpd.astype(int)
X = data[FEATURES]
dev, test = train_test_split(np.arange(len(data)), test_size=.2, stratify=y, random_state=SEED)
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["country","year"]),
    ("num", SimpleImputer(strategy="most_frequent", add_indicator=True), ["dictexp","drdintexp","dIPexp"])
], sparse_threshold=0)
model = Pipeline([
    ("prep", prep),
    ("model", HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15,
                                             l2_regularization=2, random_state=SEED))
])
model.fit(X.iloc[dev], y.iloc[dev])
p = model.predict_proba(X.iloc[test])[:,1]
truth = y.iloc[test].to_numpy()
countries = data.iloc[test].country.to_numpy()
MISS, FALSE_ALARM, REVIEW = 5., 1., .2

def decisions(prob, budget, policy):
    action = np.where(prob * MISS <= (1-prob) * FALSE_ALARM, 0, 1)
    n = int(np.floor(len(prob) * budget))
    if policy == "confidence":
        chosen = np.argsort(np.abs(prob - .5), kind="stable")[:n]
    else:
        gain = np.minimum(prob*MISS, (1-prob)*FALSE_ALARM) - REVIEW
        eligible = np.flatnonzero(gain > 0)
        chosen = eligible[np.argsort(-gain[eligible], kind="stable")[:n]]
    action[chosen] = 2
    return action

def loss(action):
    return np.where(action == 2, REVIEW,
                    np.where(action == 0, truth*MISS, (1-truth)*FALSE_ALARM))

ac = decisions(p,.2,"confidence")
ak = decisions(p,.2,"knowledge")
lc, lk = loss(ac), loss(ak)
delta = lc-lk
rng = np.random.default_rng(SEED)
boot = np.empty(1500)
for i in range(len(boot)):
    idx = rng.integers(0,len(delta),size=len(delta))
    boot[i] = delta[idx].mean()
summary = dict(n_test=len(truth), test_auc=float(roc_auc_score(truth,p)),
               test_brier=float(brier_score_loss(truth,p)), confidence_loss=float(lc.mean()),
               knowledge_loss=float(lk.mean()), absolute_loss_reduction=float(delta.mean()),
               relative_reduction_pct=float(100*delta.mean()/lc.mean()),
               paired_bootstrap_95ci=list(map(float,np.quantile(boot,[.025,.975]))),
               knowledge_review_rate=float(np.mean(ak==2)),
               confidence_review_rate=float(np.mean(ac==2)))
(OUT/"robustness_summary.json").write_text(json.dumps(summary,indent=2))
rows = []
for country in np.unique(countries):
    selected = countries == country
    if selected.sum() >= 30 and len(np.unique(truth[selected])) == 2:
        rows.append(dict(country=str(country), test_n=int(selected.sum()),
                         positive_rate=float(truth[selected].mean()),
                         roc_auc=float(roc_auc_score(truth[selected],p[selected])),
                         brier=float(brier_score_loss(truth[selected],p[selected])),
                         policy_loss_difference=float(delta[selected].mean())))
pd.DataFrame(rows).to_csv(OUT/"country_subgroup_audit.csv",index=False)
sensitivity = []
for accuracy in (.6,.7,.8,.9,.95,1.):
    for name, action in (("confidence",ac),("knowledge",ak)):
        selected = action == 2
        expected = np.where(selected, REVIEW+(1-accuracy)*np.where(truth==1,MISS,FALSE_ALARM), loss(action))
        sensitivity.append(dict(accuracy=accuracy,policy=name,expected_loss=float(expected.mean())))
pd.DataFrame(sensitivity).to_csv(OUT/"reviewer_accuracy_sensitivity.csv",index=False)
print(json.dumps(summary,indent=2))
