---
title: CoughSense
emoji: 🫁
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# CoughSense — Cough-Based Respiratory Screening

Custom FastAPI backend + hand-built frontend (live screening tool + analytics dashboard),
served together in one Docker container. Three models (Random Forest, XGBoost, CNN) predict
COVID vs. healthy from a cough clip. Best model XGBoost: **85.1% ± 6.2% (5-fold CV)** on the
Virufy clinical dataset.

By **Team Auscultate** — Aryan Verma & Arfa Alam.

⚠️ Research prototype and screening aid — **not a medical diagnosis**.

Full project: https://github.com/codewitharyan29/CoughSense
