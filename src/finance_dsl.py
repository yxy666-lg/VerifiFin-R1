#!/usr/bin/env python3
"""Deterministic financial formula DSL for Fin-R1.

The engine validates a small, explicit set of financial formulas.  It does
not execute Python or arbitrary user expressions.  Callers may provide either
a JSON-like mapping or a compact DSL expression such as::

    compound_interest(principal=10000, rate=5%, periods=2) = 11025

Every result is JSON serialisable and records the normalized inputs, expected
value, claimed value, tolerance and pass/fail status.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
import json
import re
from typing import Callable, Mapping


DSL_VERSION = "fin-r1-finance-dsl-v1"
PASS = "pass"
FAIL = "fail"
INVALID = "invalid"


class FinanceDSLValidationError(ValueError):
    """Raised when a request violates the DSL schema or formula contract."""


def _decimal(value: object, *, field: str, percent: bool = False) -> Decimal:
    """Parse a finite decimal; percent strings are normalized to ratios."""
    if isinstance(value, bool) or value is None:
        raise FinanceDSLValidationError(f"{field} must be numeric")

    raw = str(value).strip().replace(",", "")
    is_percent = raw.endswith("%")
    if is_percent:
        raw = raw[:-1].strip()
    try:
        parsed = Decimal(raw)
    except (InvalidOperation, ValueError) as exc:
        raise FinanceDSLValidationError(f"{field} is not a valid number") from exc
    if not parsed.is_finite():
        raise FinanceDSLValidationError(f"{field} must be finite")
    if is_percent:
        parsed /= Decimal("100")
    elif percent and abs(parsed) > Decimal("1"):
        raise FinanceDSLValidationError(
            f"{field}={value!r} is ambiguous; use a ratio such as 0.05 or '5%'"
        )
    return parsed


def _non_negative(value: object, *, field: str) -> Decimal:
    parsed = _decimal(value, field=field)
    if parsed < 0:
        raise FinanceDSLValidationError(f"{field} must be non-negative")
    return parsed


def _rate(value: object, *, field: str = "rate") -> Decimal:
    return _decimal(value, field=field, percent=True)


def _periods(value: object) -> int:
    parsed = _non_negative(value, field="periods")
    integral = parsed.to_integral_value()
    if parsed != integral:
        raise FinanceDSLValidationError("periods must be a non-negative integer")
    return int(integral)


def _require(inputs: Mapping[str, object], *names: str) -> None:
    missing = [name for name in names if name not in inputs]
    if missing:
        raise FinanceDSLValidationError(
            "missing required input(s): " + ", ".join(missing)
        )


def _simple_interest(inputs: Mapping[str, object]) -> tuple[Decimal, dict]:
    _require(inputs, "principal", "rate", "periods")
    principal = _non_negative(inputs["principal"], field="principal")
    rate = _rate(inputs["rate"])
    periods = _non_negative(inputs["periods"], field="periods")
    expected = principal * rate * periods
    return expected, {
        "principal": principal,
        "rate": rate,
        "periods": periods,
    }


def _simple_interest_future_value(
    inputs: Mapping[str, object],
) -> tuple[Decimal, dict]:
    interest, normalized = _simple_interest(inputs)
    return normalized["principal"] + interest, normalized


def _compound_interest(inputs: Mapping[str, object]) -> tuple[Decimal, dict]:
    _require(inputs, "principal", "rate", "periods")
    principal = _non_negative(inputs["principal"], field="principal")
    rate = _rate(inputs["rate"])
    periods = _periods(inputs["periods"])
    if Decimal("1") + rate < 0:
        raise FinanceDSLValidationError("1 + rate must be non-negative")
    with localcontext() as context:
        context.prec = 34
        expected = principal * (Decimal("1") + rate) ** periods
    return expected, {
        "principal": principal,
        "rate": rate,
        "periods": Decimal(periods),
    }


def _holding_period_return(inputs: Mapping[str, object]) -> tuple[Decimal, dict]:
    _require(inputs, "beginning_value", "ending_value")
    beginning = _decimal(inputs["beginning_value"], field="beginning_value")
    ending = _decimal(inputs["ending_value"], field="ending_value")
    income = _decimal(inputs.get("income", 0), field="income")
    if beginning == 0:
        raise FinanceDSLValidationError("beginning_value must not be zero")
    expected = (ending - beginning + income) / beginning
    return expected, {
        "beginning_value": beginning,
        "ending_value": ending,
        "income": income,
    }


def _accounting_equation(inputs: Mapping[str, object]) -> tuple[Decimal, dict]:
    _require(inputs, "assets", "liabilities", "equity")
    assets = _decimal(inputs["assets"], field="assets")
    liabilities = _decimal(inputs["liabilities"], field="liabilities")
    equity = _decimal(inputs["equity"], field="equity")
    # Treat assets as the claim and liabilities + equity as the expected side.
    return liabilities + equity, {
        "assets": assets,
        "liabilities": liabilities,
        "equity": equity,
    }


Formula = Callable[[Mapping[str, object]], tuple[Decimal, dict]]
FORMULAS: dict[str, Formula] = {
    "simple_interest": _simple_interest,
    "simple_interest_future_value": _simple_interest_future_value,
    "compound_interest": _compound_interest,
    "holding_period_return": _holding_period_return,
    "accounting_equation": _accounting_equation,
}


@dataclass(frozen=True)
class ValidationResult:
    version: str
    formula: str
    status: str
    normalized_inputs: dict[str, str]
    expected: str | None
    claimed: str | None
    absolute_error: str | None
    allowed_error: str | None
    unit: str | None
    evidence: dict
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _strings(values: Mapping[str, Decimal]) -> dict[str, str]:
    return {key: format(value, "f") for key, value in values.items()}


def validate(request: Mapping[str, object]) -> ValidationResult:
    """Validate one structured DSL request.

    Required schema::

        {"formula": str, "inputs": object, "claimed": number | "5%"}

    ``accounting_equation`` may omit ``claimed`` because ``assets`` is treated
    as the claimed left-hand side.  ``rel_tolerance`` and ``abs_tolerance``
    default to 1e-6 and 0.01 respectively.
    """
    formula_name = str(request.get("formula", "")).strip()
    unit = str(request["unit"]) if request.get("unit") is not None else None
    try:
        if formula_name not in FORMULAS:
            raise FinanceDSLValidationError(
                f"unknown formula {formula_name!r}; allowed: {', '.join(FORMULAS)}"
            )
        inputs = request.get("inputs")
        if not isinstance(inputs, Mapping):
            raise FinanceDSLValidationError("inputs must be an object")

        expected, normalized = FORMULAS[formula_name](inputs)
        if formula_name == "accounting_equation" and "claimed" not in request:
            claimed = normalized["assets"]
        else:
            if "claimed" not in request:
                raise FinanceDSLValidationError("claimed is required")
            claimed = _decimal(request["claimed"], field="claimed")

        relative = _non_negative(
            request.get("rel_tolerance", "0.000001"), field="rel_tolerance"
        )
        absolute = _non_negative(
            request.get("abs_tolerance", "0.01"), field="abs_tolerance"
        )
        error = abs(expected - claimed)
        allowed = max(absolute, relative * max(Decimal("1"), abs(expected)))
        status = PASS if error <= allowed else FAIL
        evidence = {
            "formula_expression": {
                "simple_interest": "principal * rate * periods",
                "simple_interest_future_value": "principal * (1 + rate * periods)",
                "compound_interest": "principal * (1 + rate) ** periods",
                "holding_period_return": "(ending_value - beginning_value + income) / beginning_value",
                "accounting_equation": "assets = liabilities + equity",
            }[formula_name],
            "comparison": "absolute_error <= max(abs_tolerance, rel_tolerance * max(1, abs(expected)))",
        }
        return ValidationResult(
            version=DSL_VERSION,
            formula=formula_name,
            status=status,
            normalized_inputs=_strings(normalized),
            expected=format(expected, "f"),
            claimed=format(claimed, "f"),
            absolute_error=format(error, "f"),
            allowed_error=format(allowed, "f"),
            unit=unit,
            evidence=evidence,
        )
    except FinanceDSLValidationError as exc:
        return ValidationResult(
            version=DSL_VERSION,
            formula=formula_name,
            status=INVALID,
            normalized_inputs={},
            expected=None,
            claimed=None,
            absolute_error=None,
            allowed_error=None,
            unit=unit,
            evidence={},
            error=str(exc),
        )


DSL_RE = re.compile(
    r"(?P<formula>[a-z_][a-z0-9_]*)\s*\((?P<arguments>[^()]*)\)"
    r"\s*=\s*(?P<claimed>[-+]?\d+(?:\.\d+)?%?)",
    re.IGNORECASE,
)


def parse_expression(expression: str) -> dict:
    """Parse the compact DSL syntax without evaluating arbitrary code."""
    match = DSL_RE.fullmatch(expression.strip())
    if not match:
        raise FinanceDSLValidationError(
            "expected: formula(key=value, ...) = claimed"
        )
    inputs: dict[str, str] = {}
    raw_arguments = match.group("arguments").strip()
    if raw_arguments:
        for item in raw_arguments.split(","):
            if "=" not in item:
                raise FinanceDSLValidationError(f"invalid argument {item!r}")
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", key, re.IGNORECASE):
                raise FinanceDSLValidationError(f"invalid field name {key!r}")
            if key in inputs:
                raise FinanceDSLValidationError(f"duplicate field {key!r}")
            if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", value):
                raise FinanceDSLValidationError(f"invalid numeric value {value!r}")
            inputs[key] = value
    return {
        "formula": match.group("formula").lower(),
        "inputs": inputs,
        "claimed": match.group("claimed"),
    }


def validate_expression(expression: str) -> ValidationResult:
    try:
        return validate(parse_expression(expression))
    except FinanceDSLValidationError as exc:
        return ValidationResult(
            version=DSL_VERSION,
            formula="",
            status=INVALID,
            normalized_inputs={},
            expected=None,
            claimed=None,
            absolute_error=None,
            allowed_error=None,
            unit=None,
            evidence={},
            error=str(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fin-R1 financial formula DSL")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expression", help="compact DSL expression")
    group.add_argument("--json", help="JSON request object")
    args = parser.parse_args()

    if args.expression is not None:
        result = validate_expression(args.expression)
    else:
        try:
            request = json.loads(args.json)
        except json.JSONDecodeError as exc:
            result = ValidationResult(
                version=DSL_VERSION,
                formula="",
                status=INVALID,
                normalized_inputs={},
                expected=None,
                claimed=None,
                absolute_error=None,
                allowed_error=None,
                unit=None,
                evidence={},
                error=f"invalid JSON: {exc.msg}",
            )
        else:
            result = validate(request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
