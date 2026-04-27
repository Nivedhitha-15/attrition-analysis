import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_with_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_calculates_rates_and_sorts():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "department": ["Sales", "Sales", "HR", "HR", "HR"],
            "attrition": ["Yes", "Yes", "Yes", "No", "No"],
        }
    )
    result = attrition_by_department(df)
    # Sales: 2/2 = 100%, HR: 1/3 = 33.33% — highest rate first
    assert result.iloc[0]["department"] == "Sales"
    assert result.iloc[0]["attrition_rate"] == 100.0
    assert result.iloc[1]["department"] == "HR"
    assert result.iloc[1]["attrition_rate"] == 33.33


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "No", "No"],
            "attrition": ["Yes", "No", "No", "No"],
        }
    )
    result = attrition_by_overtime(df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_calculates_rates():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "overtime": ["Yes", "Yes", "No", "No"],
            "attrition": ["Yes", "Yes", "Yes", "No"],
        }
    )
    result = attrition_by_overtime(df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["attrition_rate"] == 50.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns():
    df = pd.DataFrame({"attrition": ["Yes", "No"], "monthly_income": [3000, 6000]})
    result = average_income_by_attrition(df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_calculates_correctly():
    df = pd.DataFrame(
        {
            "attrition": ["Yes", "Yes", "No", "No"],
            "monthly_income": [3000, 5000, 6000, 8000],
        }
    )
    result = average_income_by_attrition(df)
    yes_avg = result[result["attrition"] == "Yes"].iloc[0]["avg_monthly_income"]
    no_avg = result[result["attrition"] == "No"].iloc[0]["avg_monthly_income"]
    assert yes_avg == 4000.0
    assert no_avg == 7000.0


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [1, 1, 2, 2],
            "attrition": ["Yes", "No", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rate_is_per_group_not_share_of_leavers():
    # This test guards against the bug where the denominator was total leavers
    # across the whole dataset instead of each group's own headcount.
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [1, 1, 2, 2],
            "attrition": ["Yes", "No", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    sat1 = result[result["job_satisfaction"] == 1].iloc[0]
    sat2 = result[result["job_satisfaction"] == 2].iloc[0]
    # Satisfaction 1: 1 leaver out of 2 in group = 50%, not 100% of all leavers
    assert sat1["attrition_rate"] == 50.0
    # Satisfaction 2: 0 leavers out of 2 in group = 0%
    assert sat2["attrition_rate"] == 0.0


def test_satisfaction_summary_sorted_by_satisfaction():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "job_satisfaction": [3, 1, 4, 2],
            "attrition": ["No", "Yes", "No", "No"],
        }
    )
    result = satisfaction_summary(df)
    assert list(result["job_satisfaction"]) == [1, 2, 3, 4]
