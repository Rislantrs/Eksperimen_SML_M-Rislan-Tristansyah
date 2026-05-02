# File ini adalah duplikat dari .github/workflows/mlflow_ci.yml
# GitHub Actions hanya membaca file di dalam folder .github/workflows/
# File ini dibuat untuk memenuhi kriteria struktur folder tugas.

name: MLflow CI Workflow

on:
  push:
    paths:
      - 'Workflow-CI/MLProject/**'
  workflow_dispatch:

jobs:
  train-and-dockerize:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install MLflow & Dependencies
        run: |
          pip install mlflow
          pip install pandas numpy scikit-learn

      - name: Run MLflow Project
        run: |
          cd Workflow-CI/MLProject
          mlflow run . --env-manager=local

      - name: Save Artifacts to GitHub (Advance Criteria)
        uses: actions/upload-artifact@v4
        with:
          name: mlruns-artifacts
          path: Workflow-CI/MLProject/mlruns/

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Build Docker Image with MLflow
        run: |
          cd Workflow-CI/MLProject
          RUN_ID=$(cat run_id.txt)
          mlflow models build-docker -m "runs:/$RUN_ID/model" -n ${{ secrets.DOCKER_USERNAME }}/mental-health-model:latest
          
      - name: Push Docker Image to Docker Hub
        run: |
          docker push ${{ secrets.DOCKER_USERNAME }}/mental-health-model:latest
