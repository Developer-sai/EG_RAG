"""Create the synthetic staff handbook described in Section 3.2.

It belongs to a fictional company, Northbridge Analytics Ltd, and contains what the public
document cannot offer under control: two clauses on annual leave that conflict (25 vs 23 days).
Run:  python data/make_synthetic.py
"""
from pathlib import Path

import pymupdf

OUT = Path(__file__).parent / "synthetic"
ORG = "Northbridge Analytics Ltd (fictional organisation, synthetic test document)"

DOCS = {
    "SYN1_staff_handbook": ("Staff Handbook 2026", [
        ["1. Purpose",
         "This handbook sets out the main terms of employment for all staff of Northbridge Analytics Ltd. It applies to permanent and fixed-term employees from 1 January 2026.",
         "2. Working hours",
         "The standard working week is 37.5 hours, worked Monday to Friday between 8:00 and 18:00. Core hours, when all staff must be available, are 10:00 to 16:00.",
         "Employees may request flexible start and finish times from their line manager. Requests are answered in writing within ten working days."],
        ["3. Annual leave",
         "Full-time employees are entitled to 25 working days of annual leave per leave year, in addition to public holidays. The leave year runs from 1 January to 31 December.",
         "Part-time employees receive annual leave in proportion to the hours they work. Up to five unused days may be carried over into the first three months of the next leave year.",
         "Annual leave requests must be submitted through the HR portal at least two weeks before the first day of leave."],
        ["4. Sick leave",
         "Employees who are absent through illness must inform their line manager before 10:00 on the first day of absence. A medical certificate is required for absences longer than three consecutive days.",
         "Northbridge pays full salary for up to 10 days of certified sick leave in any rolling 12-month period.",
         "Appendix B. Summary of leave entitlements",
         "Full-time employees receive 23 working days of annual leave each leave year. Parental leave and other statutory leave are granted in line with Irish legislation."]]),
}


def build(name: str, title: str, pages: list[list[str]]) -> None:
    doc = pymupdf.open()
    for n, paras in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)                   # A4
        page.insert_text((50, 40), ORG, fontsize=8, color=(0.4, 0.4, 0.4))          # header (furniture)
        page.insert_text((280, 815), f"Page {n} of {len(pages)}", fontsize=8)        # footer (furniture)
        y = 80
        if n == 1:
            page.insert_textbox(pymupdf.Rect(50, y, 545, y + 30), title, fontsize=16, fontname="hebo")
            y += 45
        for text in paras:
            heading = text[:2].strip(". ").isdigit() or text.startswith(("Table", "Appendix"))
            size = 12 if heading else 11
            height = 18
            while page.insert_textbox(pymupdf.Rect(50, y, 545, y + height), text, fontsize=size,
                                      fontname="hebo" if heading else "helv") < 0:
                height += 14                                         # box too small: nothing drawn, retry
            y += height + 8
    OUT.mkdir(exist_ok=True)
    doc.save(OUT / f"{name}.pdf")


if __name__ == "__main__":
    for name, (title, pages) in DOCS.items():
        build(name, title, pages)
    print(f"wrote {len(DOCS)} files to {OUT}")
