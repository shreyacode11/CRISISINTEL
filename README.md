\# CrisisIntel — Disaster Intelligence \& Alerting System



CrisisIntel is a machine-learning based disaster intelligence application designed to analyze disaster-related inputs and provide classification, risk information, alerts, and warning-oriented insights for cyclone, earthquake, and flood scenarios.



The project combines a Python/Flask application, dedicated machine-learning services, trained models, automated testing through Jenkins, MATLAB-based accuracy validation, and a documented technical report.



\## Project Overview



CrisisIntel focuses on three disaster domains:



\- \*\*Cyclone\*\* — analyzes cyclone-related parameters and classifies the resulting cyclone category with model confidence.

\- \*\*Earthquake\*\* — analyzes earthquake characteristics and determines an appropriate alert level.

\- \*\*Flood\*\* — evaluates flood-related parameters and determines a flood susceptibility class.

\- \*\*Alerting and warning support\*\* — presents model outputs in a form that can support disaster awareness and timely response.

\- \*\*Model evaluation\*\* — compares multiple machine-learning algorithms using accuracy metrics.



> \*\*Scope:\*\* CrisisIntel is an academic/engineering project for machine-learning based disaster intelligence and alerting. Its outputs should be treated as decision-support information rather than a replacement for official emergency-management systems.



\## Key Features



\### 1. Multi-disaster ML analysis



The application provides separate processing pipelines for:



| Disaster | Output |

|---|---|

| Cyclone | Cyclone classification + model confidence |

| Earthquake | Alert level based on event characteristics |

| Flood | Susceptibility classification |



\### 2. Machine-learning model comparison



The project evaluates multiple algorithms and generates an accuracy comparison report.



Current evaluated results:



| Disaster | Algorithm | Accuracy |

|---|---|---:|

| Cyclone | Logistic Regression | 100.00% |

| Cyclone | Random Forest | 100.00% |

| Cyclone | SVM | 100.00% |

| Earthquake | Random Forest | 91.92% |

| Earthquake | SVM | 81.15% |

| Earthquake | Logistic Regression | 66.54% |



These values are dataset/model evaluation results from the project's current experiments and should not be interpreted as real-world emergency-system performance.



\## Technology Stack



\### Application



\- Python 3.9

\- Flask

\- HTML/CSS/JavaScript

\- REST-style backend integration

\- SQLite for local application data where applicable



\### Machine Learning



\- Scikit-learn

\- NumPy

\- Pandas

\- Pickle-based trained model artifacts

\- PyTorch model artifact for the injury-related component



\### DevOps \& Testing



\- Git

\- GitHub

\- Jenkins

\- Docker

\- Pytest

\- Automated test execution

\- Coverage reporting

\- Build artifact archiving



\### Validation \& Documentation



\- MATLAB

\- LaTeX

\- Overleaf

\- Accuracy comparison charts

\- CSV-based evaluation reports



\## High-Level Architecture



```text

&#x20;                   +----------------------+

&#x20;                   |     User Interface    |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   |    Flask Application  |

&#x20;                   |       app.py          |

&#x20;                   +----------+-----------+

&#x20;                              |

&#x20;            +-----------------+-----------------+

&#x20;            |                 |                 |

&#x20;            v                 v                 v

&#x20;      +-----------+     +-----------+     +-----------+

&#x20;      |  Cyclone  |     | Earthquake|     |   Flood   |

&#x20;      | ML Service|     | ML Service|     | ML Service|

&#x20;      +-----+-----+     +-----+-----+     +-----+-----+

&#x20;            |                 |                 |

&#x20;            +-----------------+-----------------+

&#x20;                              |

&#x20;                              v

&#x20;                   +----------------------+

&#x20;                   | Alerts / Warnings \&  |

&#x20;                   | Classification Output|

&#x20;                   +----------------------+



&#x20;      CI/CD validation:

&#x20;      GitHub -> Jenkins -> Pytest -> Reports/Artifacts

&#x20;                             |

&#x20;                             +-> MATLAB validation

&#x20;                             |

&#x20;                             +-> Technical report

