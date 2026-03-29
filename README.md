# ML API test (group assignment)

Small project: train a scikit-learn model on `data/sample.csv` and expose predictions through a FastAPI app.

This repo is meant for **three students**. Each person works on **their own Git branch**, fixes the incomplete / buggy parts, and **pushes only to that branch**.

## Students and branches

| Student   | Git branch           |
|----------|----------------------|
| Amulu    | `amulu-ml-test`      |
| Bhargavi | `bhargavi-ml-test`   |
| Yeshwanth| `yeshwanth-ml-test`  |

## First-time project setup

1. **Clone the repository** (if you have not already).

   ```bash
   git clone <repository-url>
   cd ml-api-test
   ```

2. **Create and activate a virtual environment** (recommended).

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   # .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**.

   ```bash
   pip install -r requirements.txt
   ```

4. **Check out your branch** (use the branch name for your name in the table above).

   ```bash
   git fetch origin
   git checkout amulu-ml-test        # example: Amulu
   ```

   Replace with `bhargavi-ml-test` or `yeshwanth-ml-test` as appropriate.

5. **Run the API from the repository root** (so imports and `data/sample.csv` resolve correctly).

   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## What is broken (your tasks)

Work on **both** areas below on **your** branch. Keep `prepare_user_sent_data` in `app/model.py` as-is unless your instructor says otherwise.

### 1. `app/model.py` — `train_model()`

Right now `train_model()` does **not** finish training: it loads the CSV and returns the wrong object. You should:

- Use the features in the CSV (all columns except `target`) as `X`, and `target` as `y`.
- Split into train and test sets (for example with `train_test_split`).
- Fit a suitable model (e.g. `RandomForestClassifier`).
- Optionally print or log accuracy on the test set.
- **Return the trained model** (the object that has `.predict(...)`, not the raw `DataFrame`).

Ensure `pd.read_csv("data/sample.csv")` still works when the app is started from the **repo root** as shown above.

### 2. `app/main.py` — `POST /predict`

The handler prepares input with `prepare_user_sent_data(body)` but does **not** complete the flow. You should:

- Call `model.predict(...)` on the DataFrame returned by `prepare_user_sent_data`.
- Return JSON with a clear message for the user, for example:
  - if the predicted class is `1`: something like **"You are passed"**
  - otherwise: **"You are failed"**

Use a sensible JSON shape, e.g. `{"result": "You are passed"}`.

There is a commented sketch in the file; implement it cleanly (fix any syntax or logic issues from the comments).

## Quick manual test

With the server running:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"study_hours":6,"attendance":75,"previous_score":65}'
```

You should get a JSON response with your pass/fail message, not an error or empty body.

## Submitting your work

1. Commit your changes on **your** branch only.

   ```bash
   git status
   git add app/model.py app/main.py   # plus any other files you were asked to change
   git commit -m "Complete train_model and /predict endpoint"
   ```

2. Push **your** branch to `origin`.

   ```bash
   git push -u origin amulu-ml-test       # use your branch name
   ```

Do not force-push shared branches you do not own. If you are unsure, ask your instructor before rewriting history.

## Project layout

- `app/main.py` — FastAPI app and `/predict`
- `app/model.py` — data prep, `train_model`, `prepare_user_sent_data`
- `data/sample.csv` — training data (`study_hours`, `attendance`, `previous_score`, `target`)
- `requirements.txt` — Python dependencies
