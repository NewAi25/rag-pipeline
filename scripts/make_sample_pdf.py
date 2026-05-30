"""One-off script: generate `data/sample.pdf`, a tiny 'company handbook'.

Run this once if you want to regenerate the bundled sample PDF:

    pip install fpdf2
    python scripts/make_sample_pdf.py

The output PDF is intentionally short (a few pages) but contains a
handful of distinct policy sections (refunds, PTO, expenses, security)
so you can demo the retriever surfacing different chunks for different
questions.

fpdf2 is NOT a runtime dependency — it's only used here, so it isn't
listed in requirements.txt.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF


SECTIONS: list[tuple[str, str]] = [
    (
        "1. Welcome",
        "Welcome to Acme Corp! This handbook describes our company "
        "policies. It is provided as a sample document for the RAG demo. "
        "If you have questions that are not covered here, please contact "
        "the People team at people@acme.example.",
    ),
    (
        "2. Working Hours",
        "Standard working hours at Acme Corp are Monday through Friday, "
        "9:00 AM to 5:00 PM in your local time zone. We support flexible "
        "schedules: as long as you cover at least four hours of overlap "
        "with your team's core hours and meet your commitments, you may "
        "shift your start and end times by up to two hours.",
    ),
    (
        "3. Paid Time Off (PTO)",
        "All full-time employees accrue 20 days of paid time off per "
        "calendar year, plus 10 company holidays. PTO accrues at a rate "
        "of 1.67 days per month. Unused PTO may be carried over to the "
        "following year up to a maximum of 5 days. Requests should be "
        "submitted in the HR portal at least two weeks in advance for "
        "absences longer than three days.",
    ),
    (
        "4. Refund Policy",
        "Customers may request a full refund within 30 days of purchase "
        "for any reason. Refund requests submitted between 31 and 60 days "
        "after purchase are eligible for a 50% refund. After 60 days, "
        "purchases are non-refundable except where required by local "
        "law. To request a refund, customers should email "
        "refunds@acme.example with their order number. Refunds are "
        "processed within 5 business days to the original payment method.",
    ),
    (
        "5. Expense Reimbursement",
        "Employees may be reimbursed for business expenses including "
        "travel, client meals, conference fees, and approved equipment "
        "purchases. All reimbursable expenses must be pre-approved by a "
        "manager when they exceed $200. Receipts must be submitted via "
        "the Expensify integration within 30 days of the expense. "
        "Reimbursements are paid out with the next payroll cycle.",
    ),
    (
        "6. Remote Work",
        "Acme Corp is a remote-first company. Employees may work from "
        "anywhere in a country where Acme has an established legal entity "
        "(currently: USA, Canada, UK, Germany, India). Employees who wish "
        "to work from a country where Acme does not have an entity for "
        "more than 30 days per year must obtain prior written approval "
        "from the People team and the Legal team.",
    ),
    (
        "7. Information Security",
        "All laptops issued by Acme must have full-disk encryption "
        "enabled and an automatic screen lock set to no more than five "
        "minutes of inactivity. Multi-factor authentication is mandatory "
        "for all company accounts. Confidential customer data must never "
        "be copied to personal devices or personal cloud storage. Report "
        "suspected security incidents immediately to security@acme.example.",
    ),
    (
        "8. Code of Conduct",
        "Acme Corp is committed to providing a respectful, inclusive "
        "workplace. Harassment, discrimination, and retaliation are "
        "strictly prohibited. Concerns may be reported confidentially "
        "to the People team or anonymously via our third-party ethics "
        "hotline. All reports are investigated promptly and in good "
        "faith.",
    ),
    (
        "9. Acknowledgement",
        "Receipt of this handbook does not create an employment contract. "
        "Acme Corp may revise these policies at any time. The most "
        "current version is always available on the internal wiki at "
        "wiki.acme.example/handbook.",
    ),
]


def main() -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=18)
    pdf.cell(0, 12, "Acme Corp Employee Handbook", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, "Sample document for the RAG demo", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for heading, body in SECTIONS:
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, body)
        pdf.ln(3)

    out = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
