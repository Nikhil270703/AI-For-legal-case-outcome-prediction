# AI for Legal Case Outcome Prediction ⚖️🤖

An advanced, end-to-end Legal AI decision-support tool designed to modernize the Indian Judicial System. This system digests lengthy, unstructured court judgment PDFs, bypasses token limitations, and utilizes a fine-tuned **InLegalBERT** model to predict appellate outcomes with high accuracy.

Crucially, it breaks the "Black Box" of Deep Learning by integrating **Explainable AI (SHAP)**, visually highlighting the exact legal statutes, facts, and phrases that drove the AI's prediction.

---

## 🌟 Key Features
* **📄 Automated PDF Parsing:** Utilizes PyMuPDF and custom Regex to strip away irrelevant administrative metadata (Coram, headers) and extract pure legal facts.
* **🧠 Dual-Inference Pipeline:** Fine-tuned InLegalBERT models predict both the Jurisdiction (Civil/Criminal/Constitutional) and the Appellate Outcome (Allowed/Dismissed).
* **⚖️ Domain-Specific NLP:** Outperforms generic LLMs (like GPT-3.5/LLaMA) by understanding deep Indian legal semantics (e.g., IPC Sections, Bail, FIR).
* **🔍 Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) generates interactive visual heatmaps, proving mathematically why the AI made its decision.
* **💻 Modern UI:** Built with Streamlit for a fast, intuitive, "drag-and-drop" user experience.

---

## 🤗 Models on Hugging Face
To enable seamless team collaboration and bypass GitHub storage limits, our fine-tuned models are hosted on Hugging Face:
* **Model Repository:** [Nick1027/legal-case-outcome-predictor](https://huggingface.co/Nick1027/legal-case-outcome-predictor)

The system is configured to automatically pull these weights during the first run.

---

## 🏗️ System Architecture
1.  **Ingestion:** User uploads a court Judgment/Appeal (PDF/TXT).
2.  **Preprocessing:** Dynamic chunking algorithm slices long legal texts (>4000 words) into overlapping 512-token segments.
3.  **Inference:** Text chunks are processed bidirectionally by InLegalBERT.
4.  **Explainability:** SHAP engine calculates the marginal contribution of every token.
5.  **Output:** Streamlit Dashboard renders Jurisdiction, Outcome (with confidence %), and the SHAP Force Plot.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Nikhil270703/AI-For-legal-case-outcome-prediction.git](https://github.com/Nikhil270703/AI-For-legal-case-outcome-prediction.git)
cd AI-For-legal-case-outcome-prediction
