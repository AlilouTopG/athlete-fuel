# 🏋️‍♂️ Athlete Fuel AI

> **A high-performance, asynchronous AI web application designed to compute precise athletic biometrics (BMR/TDEE) and generate macro-optimized meal plans aligned with daily training intensity.**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integrated-1C2C5E.svg?style=for-the-badge&logo=chainlink&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)

---

## 🌟 Overview

**Athlete Fuel AI** addresses a fundamental challenge for athletes and fitness enthusiasts: static diet plans fail to adapt to changing daily workloads. By leveraging **FastAPI**, **LangChain**, and **Gemini Flash**, the system instantly generates personalized, science-backed nutrition plans using real-time NDJSON streaming.

Initially prototyped to test prompt orchestration, the platform was overhauled into a production-ready, ultra-low latency architecture deployed on **Render**.

---

## 🚀 Key Features

* **⚡ Real-Time NDJSON Streaming:** Asynchronous responses ensure immediate feedback and a fluid typing interface for instant UX.
* **🎯 Precision Biometrics Engine:** Accurate calculations for BMR (Basal Metabolic Rate) and TDEE (Total Daily Energy Expenditure) adjusted for workout intensity.
* **🌐 Web-Enabled AI Orchestration:** Powered by **LangChain** and **Tavily Search** to inject up-to-date nutritional insights into response chains.
* **🌍 Bilingual Core (AR/EN):** Dynamic, seamless toggle between Arabic and English with full LTR/RTL layout support.
* **🎨 Dark Athletic UI:** Sleek, responsive, zero-dependency frontend built with Tailwind CSS, Vanilla JS, and secure Markdown rendering (`DOMPurify` + `Marked.js`).

---

## 🏗️ Architecture & Tech Stack

- **Frontend:** Tailwind CSS + Vanilla JS
- **Streaming:** NDJSON Async Stream
- **Backend:** FastAPI Gateway
- **AI Engine:** LangChain + Gemini Flash + Tavily Search API

---

## 🛠️ Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/AlilouTopG/athlete-fuel.git](https://github.com/AlilouTopG/athlete-fuel.git)
   cd athlete-fuel