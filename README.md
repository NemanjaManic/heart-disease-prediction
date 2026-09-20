# Heart Disease Prediction API

An end-to-end Machine Learning deployment project that predicts the risk of heart disease based on clinical parameters. This project bridges biomedical engineering data analysis with production-ready software development using a high-performance REST API.

---

## Research & Methodology
The complete theoretical background, feature engineering process, and model evaluation metrics for this project are documented in a formal research paper.
* **Format:** IEEE Style
* **Location:** You can access the full PDF in the repository under `docs/Heart_Disease_Prediction_Paper.pdf`.

---

## Tech Stack & Libraries
* **Language:** Python
* **API Framework:** FastAPI (with Uvicorn ASGI server)
* **Data Validation:** Pydantic
* **Machine Learning:** Scikit-Learn 
* **Data Manipulation:** Pandas
* **Model Serialization:** Joblib

---

## Features & Data Structure
The underlying model utilizes **15 distinct clinical features** after undergoing proper One-Hot Encoding during the data preprocessing stage to handle categorical variables cleanly and avoid multi-collinearity:

* `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`
* `Sex_M` (Binary encoded)
* `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA`
* `RestingECG_Normal`, `RestingECG_ST`
* `ExerciseAngina_Y`
* `ST_Slope_Flat`, `ST_Slope_Up`

---

## Installation & Local Setup

To clone and run this API locally, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/HeartDiseasePrediction.git
cd HeartDiseasePrediction
```

### 2. Set up a Virtual Environment
```bash
python -m venv .venv
```
* **Activate on Windows:**
  ```bash
  .venv\Scripts\activate
  ```
* **Activate on macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Server
```bash
uvicorn src.main:app --reload
```

---

## API Usage & Interactive Documentation

Once the server is running, FastAPI automatically generates interactive Swagger UI documentation. 

* **Interactive Interface:** Open your browser and navigate to `http://127.0.0.1:8000/docs`
* **Health Check:** `GET http://127.0.0.1:8000/`

### Example Request Body (`POST /predikcija`)
```json
{
  "Age": 50,
  "RestingBP": 140,
  "Cholesterol": 289,
  "FastingBS": 0,
  "MaxHR": 172,
  "Oldpeak": 0.0,
  "Sex_M": 1,
  "ChestPainType_ATA": 1,
  "ChestPainType_NAP": 0,
  "ChestPainType_TA": 0,
  "RestingECG_Normal": 1,
  "RestingECG_ST": 0,
  "ExerciseAngina_Y": 0,
  "ST_Slope_Flat": 0,
  "ST_Slope_Up": 1
}
```

### Example Response
```json
{
  "status": "Uspesno",
  "rizik_od_bolesti": 1,
  "verovatnoca_bolesti": 0.9999
}
```

---

## Model Comparison: LogisticRegression vs. Jev (TypeSafe AI)

As an experiment, the deployed LogisticRegression model was compared against
**Jev**, TypeSafe AI's fast "System One" decision/classification model
(accessed directly via the TypeSafe API), on the exact same classification
task.

### Methodology

* **Test set:** the same 92 patients (10% held-out split, `random_state=42`)
  used to evaluate the deployed model.
* **Zero-shot for Jev:** Jev received only the raw clinical values of a
  single patient per call. It had **no access to the training set, no
  few-shot examples, and no fine-tuning** — it decides purely from its own
  general knowledge and the typed question schema. This is an intentional
  asymmetry versus LogisticRegression (which *is* trained on the training
  set); the comparison is fair in the sense that both models see the same
  test instances and neither sees the answers in advance, not in the sense
  of an identical learning approach.
* **Identical metrics:** the same precision/accuracy/recall/F1 computation
  is applied to both models' predictions.
* **Latency:** average wall-clock time per single prediction — local
  in-process inference (LogisticRegression) vs. a network API call (Jev).

A second experiment gave Jev a **few-shot** version of the same task: 30
labeled patients sampled from the training set (15 with heart disease, 15
without, fixed seed) are prepended to the state as reference examples
before each test patient is classified. Jev still never sees the *test*
labels, and there is no fine-tuning — the training data only ever appears
as in-context text, the same training set LogisticRegression was fit on.

### Results

| Model                         | Precision | Accuracy | Recall | F1   | Avg. latency |
|-------------------------------|-----------|----------|--------|------|--------------|
| LogisticRegression             |      0.93 |     0.87 |   0.82 | 0.88 |      ~0.15ms |
| Jev, zero-shot                 |      0.75 |     0.79 |   0.94 | 0.83 |     ~310ms   |
| Jev, few-shot (30 examples)    |      0.85 |     0.84 |   0.86 | 0.85 |     ~340ms   |

### Takeaways

* **LogisticRegression is still the most balanced and by far the fastest**
  (roughly 2000x faster than either Jev variant), since it runs locally
  with no network round-trip — it remains the right choice for the
  production API.
* **Few-shot narrows the gap substantially**: seeing 30 labeled examples
  from the training set pushed Jev's precision from 0.75 to 0.85 and
  accuracy from 0.79 to 0.84, at the cost of a few points of recall (0.94
  → 0.86) and a modest latency increase (~310ms → ~340ms) from the larger
  prompt.
* **Zero-shot Jev is the most "sensitive"** (highest recall) — useful if
  missing an at-risk patient is much costlier than a false alarm — but it
  has the weakest precision and accuracy of the three.
* Reproduce these with `python src/compare_jev.py` (zero-shot) and
  `python src/compare_jev_fewshot.py` (few-shot); the raw reports are at
  `src/results/jev_comparison.txt` and `src/results/jev_fewshot_comparison.txt`.
