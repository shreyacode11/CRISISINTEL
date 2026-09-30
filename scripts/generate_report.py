"""Builds an accuracy comparison report + chart from the saved model results.
Run by Jenkins after the tests; output goes to reports/ and is archived as a build artifact.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed on a CI server
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports")
os.makedirs(OUT, exist_ok=True)

cyc = pd.read_csv(os.path.join(ROOT, "Backend/CYCLONE_BACKEND/outputs/model_accuracy_comparison.csv"))
eq = pd.read_csv(os.path.join(ROOT, "Backend/EARTHQAUKE_BACKEND/earthquake_outputs/algorithm_accuracy_comparison.csv"))
cyc["Accuracy_Percentage"] = cyc["Accuracy"] * 100
eq["Accuracy_Percentage"] = eq["Accuracy"] * 100

rows = []
for name, df in (("Cyclone", cyc), ("Earthquake", eq)):
    for _, r in df.iterrows():
        rows.append({"Model": name, "Algorithm": r["Algorithm"], "Accuracy_%": round(r["Accuracy_Percentage"], 2)})
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT, "accuracy_summary.csv"), index=False)

order = ["Random Forest", "SVM", "Logistic Regression"]
pivot = summary.pivot(index="Algorithm", columns="Model", values="Accuracy_%").reindex(order)
ax = pivot.plot(kind="bar", figsize=(7, 4), rot=0)
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 110)
ax.set_title("CrisisIntel - model accuracy comparison")
ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "accuracy_comparison.png"), dpi=120)
print(summary.to_string(index=False))
print("Report written to", OUT)
