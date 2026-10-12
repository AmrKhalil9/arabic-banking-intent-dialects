"""77 ArBanking77 intents -> 12 contact-centre categories (rule list from the earlier routing experiment)."""
CATS = {
    "card_management":   "getting, ordering, activating, linking or replacing a card; card delivery, expiry, card types, contactless, Apple/Google Pay",
    "card_problem":      "a card that is lost, stolen, compromised, swallowed, declined or not working",
    "payment_dispute":   "a card or direct-debit payment that is wrong, unrecognised, pending, reverted, charged twice or with an extra fee or wrong exchange rate",
    "refund":            "asking for a refund or a refund that has not arrived",
    "transfer":          "sending or receiving a bank transfer: failed, pending, declined, cancelled, fees, timing, beneficiary, balance after transfer",
    "cash_atm":          "ATM use, cash withdrawal problems, charges, wrong amount, cash or cheque deposit",
    "top_up":            "topping up the account: by card, bank transfer, cash or cheque; top-up failed, pending, reverted, limits, charges, automatic top-up",
    "exchange_currency": "exchange rates, exchange charges, exchanging currency in the app, supported currencies and countries",
    "pin_security":      "PIN, passcode, blocked PIN, lost or stolen phone",
    "identity":          "verifying identity, source of funds, why verification is needed, age limit",
    "account":           "editing personal details or closing the account",
    "other":             "none of the above",
}
RULES = [  # first match wins, on the English intent name
    ("refund", "refund"),
    ("lost or stolen phone", "pin_security"), ("pin", "pin_security"), ("passcode", "pin_security"),
    ("verify", "identity"), ("age limit", "identity"),
    ("edit personal", "account"), ("terminate", "account"),
    ("top up", "top_up"), ("topping up", "top_up"),
    ("exchange", "exchange_currency"), ("currency", "exchange_currency"), ("country support", "exchange_currency"),
    ("atm", "cash_atm"), ("cash", "cash_atm"), ("cheque", "cash_atm"),
    ("transfer", "transfer"), ("receiving money", "transfer"), ("beneficiary", "transfer"),
    ("lost or stolen card", "card_problem"), ("compromised", "card_problem"), ("swallowed", "card_problem"),
    ("declined card", "card_problem"), ("card not working", "card_problem"), ("virtual card not working", "card_problem"), ("contactless not working", "card_problem"),
    ("payment", "payment_dispute"), ("charged twice", "payment_dispute"), ("extra charge", "payment_dispute"), ("direct debit", "payment_dispute"),
    ("card", "card_management"), ("apple pay", "card_management"), ("visa", "card_management"),
]

def cat_of(intent_en):
    n = intent_en.lower()
    for k, c in RULES:
        if k in n: return c
    raise KeyError(intent_en)
