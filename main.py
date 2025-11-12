import os
from typing import List, Optional, Literal, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import Borrower, Loan, Payment, RecoveryAction

# ML (lightweight, no sklearn)
import numpy as np
import pickle

MODEL_PATH = "model.pkl"

app = FastAPI(title="Smart Loan Recovery API", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateBorrowerRequest(Borrower):
    pass


class CreateLoanRequest(Loan):
    pass


class CreatePaymentRequest(Payment):
    pass


class TrainResponse(BaseModel):
    trained: bool
    samples_used: int
    auc: Optional[float] = None
    message: str


def _to_str_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(doc)
    if d.get("_id") is not None:
        d["_id"] = str(d["_id"])
    return d


@app.get("/")
def read_root():
    return {"message": "Smart Loan Recovery API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set" if not os.getenv("DATABASE_URL") else "✅ Set",
        "database_name": "❌ Not Set" if not os.getenv("DATABASE_NAME") else "✅ Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["database"] = f"⚠️ Error: {str(e)[:80]}"
    return response


@app.get("/schema")
def get_schema():
    return {
        "borrower": Borrower.model_json_schema(),
        "loan": Loan.model_json_schema(),
        "payment": Payment.model_json_schema(),
        "recoveryaction": RecoveryAction.model_json_schema(),
    }


@app.post("/borrowers")
def create_borrower(payload: CreateBorrowerRequest):
    inserted_id = create_document("borrower", payload)
    return {"id": inserted_id}


@app.get("/borrowers")
def list_borrowers(limit: Optional[int] = 100):
    docs = get_documents("borrower", {}, limit or 100)
    return [_to_str_id(d) for d in docs]


@app.post("/loans")
def create_loan(payload: CreateLoanRequest):
    from bson import ObjectId
    try:
        borrower = db["borrower"].find_one({"_id": ObjectId(payload.borrower_id)})
    except Exception:
        borrower = None
    if not borrower:
        raise HTTPException(status_code=400, detail="Borrower not found")
    inserted_id = create_document("loan", payload)
    return {"id": inserted_id}


@app.get("/loans")
def list_loans(limit: Optional[int] = 100):
    docs = get_documents("loan", {}, limit or 100)
    results = []
    for d in docs:
        item = _to_str_id(d)
        try:
            from bson import ObjectId
            b = db["borrower"].find_one(
                {"_id": ObjectId(d["borrower_id"])},
                {"full_name": 1, "income": 1, "credit_score": 1},
            )
            if b:
                item["borrower"] = {
                    "id": str(b.get("_id")),
                    "full_name": b.get("full_name"),
                    "income": b.get("income"),
                    "credit_score": b.get("credit_score"),
                }
        except Exception:
            pass
        results.append(item)
    return results


@app.post("/payments")
def create_payment(payload: CreatePaymentRequest):
    from bson import ObjectId
    try:
        loan = db["loan"].find_one({"_id": ObjectId(payload.loan_id)})
    except Exception:
        loan = None
    if not loan:
        raise HTTPException(status_code=400, detail="Loan not found")
    inserted_id = create_document("payment", payload)
    return {"id": inserted_id}


# ML utilities (from-scratch logistic regression)
feature_fields = [
    "income",
    "credit_score",
    "employment_status",
    "loan_amount",
    "term_months",
    "interest_rate",
    "past_due_days",
    "num_late_payments",
]


def _encode_employment(status: str) -> int:
    mapping = {
        "employed": 2,
        "self-employed": 1,
        "student": 0,
        "retired": 0,
        "unemployed": -1,
    }
    return mapping.get((status or "").lower(), 0)


def _build_feature_row(borrower: Dict[str, Any], loan: Dict[str, Any]) -> List[float]:
    return [
        float(borrower.get("income", 0) or 0.0),
        float(borrower.get("credit_score", 600) or 600),
        float(_encode_employment(borrower.get("employment_status"))),
        float(loan.get("loan_amount", 0) or 0.0),
        float(loan.get("term_months", 0) or 0.0),
        float(loan.get("interest_rate", 0) or 0.0),
        float(loan.get("past_due_days", 0) or 0.0),
        float(loan.get("num_late_payments", 0) or 0.0),
    ]


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _train_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.05, epochs: int = 1500) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = _sigmoid(z)
        grad_w = (X.T @ (p - y)) / n
        grad_b = float(np.sum(p - y) / n)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def _auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    try:
        # Simple AUC approximation via rank method
        order = np.argsort(y_prob)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(y_prob))
        pos = y_true == 1
        neg = y_true == 0
        n_pos = np.sum(pos)
        n_neg = np.sum(neg)
        if n_pos == 0 or n_neg == 0:
            return None
        auc = (np.sum(ranks[pos]) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
        return float(auc)
    except Exception:
        return None


def _load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def _save_model(model: Dict[str, Any]):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)


class PredictResponse(BaseModel):
    probability_default: float
    label: Literal["default", "non-default"]


@app.post("/train", response_model=TrainResponse)
def train_model():
    loans = list(db["loan"].find({}))
    if not loans:
        X = []
        y = []
        for _ in range(50):
            income = np.random.normal(4000, 500)
            credit = np.random.normal(720, 20)
            X.append([income, credit, 2, 5000, 24, 10, 0, 0])
            y.append(0)
        for _ in range(50):
            income = np.random.normal(1500, 400)
            credit = np.random.normal(580, 30)
            X.append([income, credit, -1, 10000, 36, 24, 30, 3])
            y.append(1)
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
    else:
        X_rows = []
        y_rows = []
        from bson import ObjectId
        for ln in loans:
            try:
                b = db["borrower"].find_one({"_id": ObjectId(ln["borrower_id"])})
                if not b:
                    continue
                X_rows.append(_build_feature_row(b, ln))
                status = ln.get("status", "active")
                label = 1 if status == "defaulted" else 0
                if status != "defaulted" and (ln.get("past_due_days", 0) or 0) >= 60:
                    label = 1
                y_rows.append(label)
            except Exception:
                continue
        if len(X_rows) < 10:
            X = []
            y = []
            for _ in range(50):
                income = np.random.normal(3500, 900)
                credit = np.random.normal(680, 60)
                X.append([income, credit, 2, 8000, 30, 15, 5, 1])
                y.append(0)
            for _ in range(50):
                income = np.random.normal(1800, 600)
                credit = np.random.normal(600, 70)
                X.append([income, credit, -1, 12000, 36, 22, 45, 4])
                y.append(1)
            X = np.array(X, dtype=float)
            y = np.array(y, dtype=float)
        else:
            X = np.array(X_rows, dtype=float)
            y = np.array(y_rows, dtype=float)

    Xs, mu, sigma = _standardize(X)
    w, b = _train_logreg(Xs, y)
    prob = _sigmoid(Xs @ w + b)
    auc = _auc_score(y, prob)

    model = {"w": w, "b": float(b), "mu": mu, "sigma": sigma}
    _save_model(model)

    return TrainResponse(trained=True, samples_used=int(X.shape[0]), auc=auc, message="Model trained and saved")


def _predict_proba_for_loan(loan_id: str) -> float:
    model = _load_model()
    from bson import ObjectId
    ln = db["loan"].find_one({"_id": ObjectId(loan_id)})
    if not ln:
        raise HTTPException(status_code=404, detail="Loan not found")
    b = db["borrower"].find_one({"_id": ObjectId(ln["borrower_id"])})
    if not b:
        raise HTTPException(status_code=404, detail="Borrower not found")

    x = np.array([_build_feature_row(b, ln)], dtype=float)
    if model is None:
        prob = 0.5
        if (ln.get("past_due_days", 0) or 0) > 30:
            prob += 0.2
        if (ln.get("num_late_payments", 0) or 0) >= 2:
            prob += 0.15
        if (b.get("credit_score", 650) or 650) < 620:
            prob += 0.1
        if (b.get("income", 0) or 0) < max(1000, 0.15 * (ln.get("loan_amount", 0) or 0)):
            prob += 0.1
        return float(min(max(prob, 0.01), 0.99))
    else:
        w = np.array(model["w"], dtype=float)
        b0 = float(model["b"])
        mu = np.array(model["mu"], dtype=float)
        sigma = np.array(model["sigma"], dtype=float)
        xs = (x - mu) / sigma
        return float(_sigmoid(xs @ w + b0)[0])


@app.get("/predict/{loan_id}", response_model=PredictResponse)
def predict_loan_default(loan_id: str):
    p = _predict_proba_for_loan(loan_id)
    return PredictResponse(probability_default=p, label="default" if p >= 0.5 else "non-default")


class StrategyResponse(BaseModel):
    probability_default: float
    risk_level: Literal["low","medium","high","severe"]
    recommended_strategy: Literal[
        "gentle-reminder",
        "payment-plan",
        "hardship-assistance",
        "discounted-settlement",
        "legal-escalation"
    ]
    actions: List[str]


def _strategy_from_probability(p: float) -> StrategyResponse:
    if p < 0.3:
        return StrategyResponse(
            probability_default=p,
            risk_level="low",
            recommended_strategy="gentle-reminder",
            actions=[
                "Send friendly reminder via SMS/email",
                "Offer autopay setup",
                "Share due date and outstanding balance",
            ],
        )
    if p < 0.6:
        return StrategyResponse(
            probability_default=p,
            risk_level="medium",
            recommended_strategy="payment-plan",
            actions=[
                "Call to confirm situation",
                "Offer short-term payment plan",
                "Waive late fees upon adherence",
            ],
        )
    if p < 0.8:
        return StrategyResponse(
            probability_default=p,
            risk_level="high",
            recommended_strategy="hardship-assistance",
            actions=[
                "Assess hardship documentation",
                "Temporarily reduce payments",
                "Extend term to lower installment",
            ],
        )
    if p < 0.92:
        return StrategyResponse(
            probability_default=p,
            risk_level="severe",
            recommended_strategy="discounted-settlement",
            actions=[
                "Offer settlement discount for lump sum",
                "Set short deadline for offer",
                "Escalate if no response",
            ],
        )
    return StrategyResponse(
        probability_default=p,
        risk_level="severe",
        recommended_strategy="legal-escalation",
        actions=[
            "Send formal demand notice",
            "Initiate legal review",
            "Proceed with collections compliant with laws",
        ],
    )


@app.get("/strategy/{loan_id}", response_model=StrategyResponse)
def recommend_strategy(loan_id: str):
    p = _predict_proba_for_loan(loan_id)
    return _strategy_from_probability(p)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
