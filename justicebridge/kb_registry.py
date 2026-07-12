"""
KB store registry — the catalog of ALL legal-topic knowledge bases the system
can search.

This is the backbone of the "planner routes to knowledge" architecture:

    Planner agent  ->  looks at this catalog, decides WHICH KB store(s) are
                       relevant to the citizen's problem (e.g. ["wages",
                       "free_aid"]) and writes them to state["kb_stores"].
    Retrieval agent -> searches ONLY those stores (each is its own Chroma
                       vector collection built from that topic's Acts), so a
                       wage complaint is never matched against family law.

Each KB store is a self-contained vector DB over one legal topic's Act(s).
Adding a new legal vertical = add ONE entry here (+ its Act PDF is fetched
from indiacode automatically by build_corpus.py). Nothing else in the
pipeline needs to change: the planner reads `planner_keywords`, retrieval
reads `acts`/`collection`, risk reads `deadline_key`, output reads
`output_template`.

`always_include=True` stores (free_aid) are appended to EVERY supported
query's search set, because free-legal-aid eligibility is cross-cutting — it
applies regardless of which substantive topic the citizen is asking about.
"""

# store_id -> metadata
KB_STORES = {
    "wages": {
        "topic": "Unpaid wages & labour",
        "description": (
            "unpaid wages, delayed salary, minimum wage, overtime, bonus, "
            "illegal deductions, fines, contractor/employer non-payment, "
            "labour dues"
        ),
        "acts": ["The Code on Wages, 2019"],
        "collection": "kb_wages",
        "supported": True,
        "output_template": "wage_dispute",
        "deadline_key": "wages",
        "planner_keywords": [
            "wage", "wages", "salary", "unpaid", "not paid", "paid", "pay",
            "employer", "employee", "employ", "contractor", "maistry",
            "labour", "labor", "worker", "workplace", "coolie", "mazdoor",
            "overtime", "bonus", "minimum wage", "site", "same work",
            "deduct", "deduction", "fine", "wage slip",
        ],
    },
    "consumer": {
        "topic": "Consumer protection",
        "description": (
            "defective product, faulty goods, deficient service, refund, "
            "replacement, warranty, e-commerce complaint, unfair trade "
            "practice, overcharging, consumer complaint"
        ),
        "acts": ["The Consumer Protection Act, 2019"],
        "collection": "kb_consumer",
        "supported": True,
        "output_template": "consumer_dispute",
        "deadline_key": "consumer",
        "planner_keywords": [
            "defective", "faulty", "refund", "replace", "replacement",
            "warranty", "guarantee", "consumer", "product", "seller", "shop",
            "e-commerce", "online order", "overcharged", "deficient service",
            "unfair trade", "not working", "broken",
        ],
    },
    "family": {
        "topic": "Family & domestic protection",
        "description": (
            "domestic violence, dowry harassment, maintenance, divorce, "
            "judicial separation, cruelty, protection order, marriage rights, "
            "abused wife"
        ),
        "acts": [
            "The Protection of Women from Domestic Violence Act, 2005",
            "The Hindu Marriage Act, 1955",
        ],
        "collection": "kb_family",
        "supported": True,
        "output_template": "family_dispute",
        "deadline_key": "general",
        "planner_keywords": [
            "domestic violence", "husband", "wife", "marriage", "married",
            "dowry", "divorce", "maintenance", "cruelty", "abuse", "abused",
            "beat", "beaten", "harass", "harassment", "protection order",
            "in-laws", "separation",
        ],
    },
    # ---- cross-cutting store: always searched for supported queries ----
    "free_aid": {
        "topic": "Free legal aid eligibility",
        "description": (
            "who qualifies for free government legal aid, Section 12 Legal "
            "Services Authorities Act, DLSA, NALSA, Lok Adalat, Tele-Law"
        ),
        "acts": ["The Legal Services Authorities Act, 1987"],
        "collection": "kb_free_aid",
        "supported": True,
        "output_template": None,
        "deadline_key": "general",
        "always_include": True,
        "planner_keywords": [
            "free legal aid", "legal aid", "cannot afford lawyer", "dlsa",
            "nalsa", "lok adalat", "tele-law", "free lawyer",
        ],
    },
}

# Verticals recognised by the router but NOT yet backed by a KB store — the
# planner routes these straight to the human-handoff branch ("coming soon").
STUB_VERTICALS = {
    "tenancy": {
        "topic": "Tenancy / eviction",
        "planner_keywords": ["evict", "eviction", "landlord", "rent", "tenant",
                             "lease", "vacate", "rented house"],
    },
    "fir": {
        "topic": "Police / FIR / criminal",
        "planner_keywords": ["fir", "police", "arrest", "theft", "assault",
                             "complaint against", "bns", "criminal"],
    },
}


def all_store_ids():
    return list(KB_STORES.keys())


def always_include_ids():
    return [sid for sid, cfg in KB_STORES.items() if cfg.get("always_include")]


def acts_for_store(store_id):
    return KB_STORES.get(store_id, {}).get("acts", [])


def acts_by_store():
    """store_id -> [act titles] — consumed by build_corpus.py."""
    return {sid: cfg["acts"] for sid, cfg in KB_STORES.items()}
