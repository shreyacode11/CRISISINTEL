// Jenkinsfile for CrisisIntel (place in the ROOT of the repo, next to app.py)
// Pipeline: checkout -> install -> test -> generate report -> archive results.
// Uses Python 3.10 because the models were trained with scikit-learn 1.0.2.
pipeline {
    agent {
        docker { image 'python:3.10-slim' }   // needs Docker on the Jenkins server
    }

    options {
        timestamps()
        timeout(time: 15, unit: 'MINUTES')
    }

    triggers {
        pollSCM('H/5 * * * *')   // or use a GitHub webhook for instant runs
    }

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }

        stage('Install dependencies') {
            steps {
                sh '''
                    python -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements-ci.txt
                '''
            }
        }

        stage('Run tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest tests/ -v --junitxml=reports/test-results.xml \
                           --cov=ml_services --cov-report=xml:reports/coverage.xml
                '''
            }
            post {
                always { junit 'reports/test-results.xml' }   // shows pass/fail in Jenkins UI
            }
        }

        stage('Generate accuracy report') {
            steps {
                sh '''
                    . .venv/bin/activate
                    python scripts/generate_report.py
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'reports/**', fingerprint: true
        }
        success { echo 'Build passed: all model tests green, report archived.' }
        failure { echo 'Build FAILED - check the test results above.' }
    }
}
