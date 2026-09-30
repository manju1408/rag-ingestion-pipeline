"""Write synthetic policy / FAQ documents for local runs. No real company data."""
from __future__ import annotations

import random
import sys
from pathlib import Path

TOPICS = {
    "billing-policy": [
        ("Late payments", "A late fee of {fee} dollars applies when a bill is unpaid {days} days after the due date."),
        ("Refunds", "Refunds for service outages are credited automatically when an outage lasts longer than {hours} hours."),
        ("Autopay", "Customers enrolled in autopay receive a {pct} percent discount on the monthly plan price."),
    ],
    "equipment-returns": [
        ("Return window", "Rented modems and routers must be returned within {days} days of cancellation."),
        ("Unreturned equipment", "Unreturned equipment is billed at a flat fee of {fee} dollars per device."),
    ],
    "transaction-disputes": [
        ("Filing a dispute", "Disputes must be filed within {days} days of the statement date through the account portal."),
        ("Investigation", "Disputed charges are held during investigation, which completes within {hours} hours for most cases."),
    ],
}


def main(out: str = "data/docs", seed: int = 7) -> None:
    rnd = random.Random(seed)
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    for name, sections in TOPICS.items():
        lines = [f"# {name.replace('-', ' ').title()}", "", "Synthetic policy document for demos.", ""]
        for heading, template in sections:
            lines += [f"## {heading}", "", template.format(fee=rnd.choice([10, 15, 25]),
                      days=rnd.choice([14, 30, 60]), hours=rnd.choice([4, 24, 72]),
                      pct=rnd.choice([5, 10])), ""]
        (root / f"{name}.md").write_text("\n".join(lines))
    print(f"wrote {len(TOPICS)} documents to {root}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
