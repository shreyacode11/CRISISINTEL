# CrisisIntel: setup guide (Jenkins, MATLAB, LaTeX)

## 0. Put the files in your repo

```
CrisisIntel/                 <- repo root (next to app.py)
├── Jenkinsfile
├── requirements-ci.txt
├── tests/test_models.py
├── scripts/generate_report.py
├── ml_services/             (your existing code)
├── Backend/                 (your existing code, incl. the accuracy CSVs)
├── crisisintel_accuracy_report.m
└── report/                  (optional: crisisintel_report.tex + accuracy_comparison.png)
```

Make sure both accuracy CSVs are committed to Git, otherwise the CI report step
cannot find them:
`Backend/CYCLONE_BACKEND/outputs/model_accuracy_comparison.csv`
`Backend/EARTHQAUKE_BACKEND/earthquake_outputs/algorithm_accuracy_comparison.csv`
The saved model files (.pkl / .joblib) that `ml_services/` loads must be in the repo too.

## 1. Test locally first (5 minutes)

```bash
python3.10 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-ci.txt
pytest tests/ -v          # expect 5 passed
python scripts/generate_report.py
```

## 2. Jenkins (Docker route, easiest)

1. Install Docker Desktop.
2. Run Jenkins with Docker access:
   ```bash
   docker run -d --name jenkins -p 8080:8080 -p 50000:50000 \
     -v jenkins_home:/var/jenkins_home \
     -v /var/run/docker.sock:/var/run/docker.sock \
     jenkins/jenkins:lts
   docker logs jenkins        # copy the initial admin password
   ```
3. Open http://localhost:8080, paste the password, choose "Install suggested plugins".
4. Manage Jenkins > Plugins > install **Docker Pipeline** (needed for `agent { docker }`).
5. Inside the Jenkins container, install the Docker CLI (or use an agent that already has it):
   `docker exec -u root jenkins bash -c "apt-get update && apt-get install -y docker.io"`
6. New Item > **Pipeline** > Pipeline script from SCM > Git > your repo URL > Script Path `Jenkinsfile` > Save.
7. Build Now. Green build = 5 tests passed and `reports/` archived (accuracy_summary.csv, accuracy_comparison.png, coverage.xml).

Troubleshooting:
- `docker: not found` -> step 5 not done.
- `No such file ... .csv` -> the CSVs are not committed.
- scikit-learn install fails -> the agent is not Python 3.10 (keep `python:3.10-slim`).

## 3. MATLAB

1. Go to matlab.mathworks.com (MATLAB Online, free tier with a MathWorks account).
2. Upload the repo folder (or at least the two CSVs in the same folder structure plus the .m file).
3. Set the current folder to the project root, then run `crisisintel_accuracy_report`.
4. Expect: a table in the Command Window, `matlab_accuracy_comparison.png`,
   `matlab_validation_report.csv`, and "VALIDATION PASSED" (threshold 60%; the lowest model, Earthquake Logistic Regression, is 66.5%).
5. Take MATLAB Onramp for the certificate: mathworks.com/learn/tutorials/matlab-onramp.html (free, ~2 h).

## 4. LaTeX (Overleaf)

1. overleaf.com > New Project > Upload Project > upload `crisisintel_report.tex` and `accuracy_comparison.png`.
2. Recompile. To use your own chart, replace the PNG with `reports/accuracy_comparison.png` from Jenkins.

## 5. Fix before you show this to anyone

`ml_services/cyclone_predictor.py` overwrites the model confidence with `random.uniform(90.0, 99.0)`.
Replace that with the real probability:

```python
proba = model.predict_proba(X)[0]
confidence = float(proba.max())          # 0..1, matches the other predictors
```
(or return `None` if the model has no `predict_proba`, which the tests already accept).

## Free learning links (checked 30 Sep 2026)

- MATLAB Onramp: https://www.mathworks.com/learn/tutorials/matlab-onramp.html (free, certificate)
- Learn LaTeX in 30 minutes: https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes
- Jenkins docs and tutorials: https://www.jenkins.io/doc/
- Atlassian University, Jira Fundamentals (free, badge): https://university.atlassian.com
