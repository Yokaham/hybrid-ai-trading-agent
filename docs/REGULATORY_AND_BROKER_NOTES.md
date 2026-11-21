# Regulatory Compliance & Broker Isolation Architecture

## 1. Jurisdictional & Regulatory Context (India / FEMA)

### 1.1 Foreign Exchange Management Act (FEMA) Overview
- Under Indian law (Foreign Exchange Management Act, 1999), resident individuals are prohibited from undertaking unauthorized forex and derivative transactions on offshore electronic trading platforms.
- The **Reserve Bank of India (RBI)** maintains a published **Alert List** containing unauthorized entities and electronic trading platforms. Notable platforms on the Alert List include **XM, MetaTrader 4, MetaTrader 5, OctaFX, Olymp Trade**, among others.
- Residents remitting funds abroad under the Liberalised Remittance Scheme (LRS) cannot use remittances for margin trading, forex speculation, or overseas derivative contracts.

### 1.2 Mandatory Architectural Safeguards
In strict adherence to regulatory compliance:
1. **No Offshore Live Connections**: Under no circumstances will live execution adapters for XM, MT4/MT5 brokers, or unauthorized offshore forex dealers be developed or permitted in this repository.
2. **Explicit Broker Isolation**: The core quantitative intelligence, feature modeling, and risk engines are 100% decoupled from any specific execution gateway.
3. **Independent Compliance Review**: Any future execution adapter targeting compliant domestic brokerages (e.g., SEBI-registered brokers via regulated Open APIs) must undergo an independent regulatory and compliance audit prior to integration.

---

## 2. Broker Interface Abstraction

The system enforces an interface abstraction pattern (`BaseBrokerAdapter`):

```python
class BaseBrokerAdapter(ABC):
    @abstractmethod
    def get_account_balance(self) -> AccountBalance: ...
    
    @abstractmethod
    def submit_order(self, order: Order) -> ExecutionReport: ...
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...
```

The sole active production adapter within this repository is `PaperSimulatedBroker`. Any unauthorized live broker module will fail validation during initialization and trigger system shutdown.
