"""Multi-currency cash accounting book."""

from decimal import Decimal


class InsufficientFundsError(ValueError):
    """Raised when an operation would exceed available cash in a fully funded cash account."""


class CashBook:
    """Multi-currency cash ledger maintaining isolated Decimal balances.

    Invariants:
    1. Each currency has an isolated, non-blended balance.
    2. Currencies are normalized consistently (uppercase, 3-5 alpha characters).
    3. Signed balances are permitted to represent explicit debt/liabilities.
    4. Non-finite values (NaN, infinity) and malformed currency codes are rejected.
    5. Automatic currency conversion is strictly prohibited within the CashBook.
    6. Read-only balance snapshots prevent external state mutation.
    """

    def __init__(self) -> None:
        self._balances: dict[str, Decimal] = {}

    @staticmethod
    def normalize_currency(currency: str) -> str:
        """Validate and normalize a currency code."""
        if not isinstance(currency, str):
            raise TypeError(f"Currency must be a string, got {type(currency).__name__}")
        norm = currency.strip().upper()
        if len(norm) < 3 or len(norm) > 5 or not norm.isalpha():
            raise ValueError(f"Invalid currency code: '{currency}'")
        return norm

    @staticmethod
    def validate_amount(amount: Decimal | float | str) -> Decimal:
        """Validate and convert an amount into a finite Decimal."""
        if isinstance(amount, Decimal):
            dec = amount
        else:
            try:
                dec = Decimal(str(amount))
            except Exception as e:
                raise ValueError(f"Invalid monetary value: {amount}") from e

        if not dec.is_finite():
            raise ValueError(f"Monetary amount must be finite, got {dec}")
        return dec

    def get_balance(self, currency: str) -> Decimal:
        """Retrieve the balance for a specific currency (defaults to 0.0)."""
        norm = self.normalize_currency(currency)
        return self._balances.get(norm, Decimal("0.0"))

    def get_all_balances(self) -> dict[str, Decimal]:
        """Return a copy of all non-zero and registered currency balances."""
        return dict(self._balances)

    def apply_movement(self, currency: str, amount: Decimal | float | str) -> Decimal:
        """Apply a signed cash flow movement to the specified currency balance.

        Args:
            currency: Target currency code.
            amount: Signed Decimal amount (positive = credit/deposit, negative = debit/withdrawal).

        Returns:
            The updated balance for the currency.
        """
        norm = self.normalize_currency(currency)
        dec_amount = self.validate_amount(amount)

        current = self._balances.get(norm, Decimal("0.0"))
        new_balance = current + dec_amount
        self._balances[norm] = new_balance
        return new_balance

    def deposit(self, currency: str, amount: Decimal | float | str) -> Decimal:
        """Credit a positive amount to the specified currency."""
        dec = self.validate_amount(amount)
        if dec < Decimal("0.0"):
            raise ValueError(f"Deposit amount must be positive, got {dec}")
        return self.apply_movement(currency, dec)

    def withdraw(self, currency: str, amount: Decimal | float | str) -> Decimal:
        """Debit a positive amount from the specified currency."""
        dec = self.validate_amount(amount)
        if dec < Decimal("0.0"):
            raise ValueError(f"Withdrawal amount must be positive, got {dec}")
        return self.apply_movement(currency, -dec)

    def has_sufficient_cash(self, currency: str, required_amount: Decimal | float | str) -> bool:
        """Check if the currency balance is greater than or equal to the required amount."""
        norm = self.normalize_currency(currency)
        req = self.validate_amount(required_amount)
        return self.get_balance(norm) >= req

    def check_sufficient_cash(self, currency: str, required_amount: Decimal | float | str) -> None:
        """Enforce that sufficient cash is available, raising InsufficientFundsError otherwise.

        Used to guarantee that fully funded cash equity purchases do not create unfunded positions.
        """
        norm = self.normalize_currency(currency)
        req = self.validate_amount(required_amount)
        current = self.get_balance(norm)
        if current < req:
            raise InsufficientFundsError(
                f"Insufficient funds in {norm}: available balance {current} is less than required {req}"
            )
