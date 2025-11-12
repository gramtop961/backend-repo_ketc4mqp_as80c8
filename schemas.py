"""
Database Schemas for Smart Loan Recovery System

Each Pydantic model represents a collection in MongoDB. The collection name is the lowercase class name.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal

class Borrower(BaseModel):
    full_name: str = Field(..., description="Borrower's full name")
    email: str = Field(..., description="Contact email")
    phone: Optional[str] = Field(None, description="Phone number")
    income: float = Field(..., ge=0, description="Monthly income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    employment_status: Literal["employed","self-employed","unemployed","student","retired"] = Field("employed")

class Loan(BaseModel):
    borrower_id: str = Field(..., description="Reference to borrower _id as string")
    loan_amount: float = Field(..., ge=0)
    term_months: int = Field(..., ge=1)
    interest_rate: float = Field(..., ge=0, description="Annual interest rate as percent, e.g., 12.5")
    purpose: Optional[str] = Field(None)
    past_due_days: int = Field(0, ge=0)
    num_late_payments: int = Field(0, ge=0)
    status: Literal["active","closed","defaulted"] = Field("active")

class Payment(BaseModel):
    loan_id: str = Field(..., description="Reference to loan _id as string")
    amount: float = Field(..., ge=0)
    paid_on: Optional[str] = Field(None, description="ISO date string; if None, server time will be used")
    method: Optional[str] = Field(None)

class RecoveryAction(BaseModel):
    loan_id: str = Field(...)
    strategy: Literal["gentle-reminder","payment-plan","hardship-assistance","discounted-settlement","legal-escalation"]
    notes: Optional[str] = None
