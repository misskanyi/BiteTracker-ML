"""Downloadable templates for the Data upload page.

- `data_template_xlsx`: an Excel workbook with the exact column headers the
  validator expects, an instructions sheet and a short worked example.
- `letter_template_docx`: a Word cover letter for the school to sign and attach.

Both are built in memory on request from `src.data_validation`'s column lists, so
they can never drift from what the validator checks. The Word file is written as
plain Office Open XML, so no extra package is needed.
"""

from __future__ import annotations

import io
import zipfile
from xml.sax.saxutils import escape

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from src.data_validation import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

XLSX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
DATA_TEMPLATE_FILENAME = "bitetracker_data_template.xlsx"
LETTER_TEMPLATE_FILENAME = "bitetracker_supporting_letter_template.docx"

# column: (format, description)
COLUMN_GUIDE = {
    "date": ("YYYY-MM-DD", "School day the meals were served. One row per school per day."),
    "school_id": ("Text", "Stable school code used by the programme (e.g. SCH001). Same code on every row for a school."),
    "school_name": ("Text", "Official school name."),
    "meals_served": ("Whole number ≥ 0", "Meals actually served to learners that day."),
    "county": ("Text", "County the school is in."),
    "sub_county": ("Text", "Sub-county the school is in."),
    "enrolled_students": ("Whole number ≥ 0", "Learners enrolled in the feeding programme."),
    "students_present": ("Whole number ≥ 0", "Learners present that day."),
    "students_absent": ("Whole number ≥ 0", "Learners absent that day."),
    "meals_prepared": ("Whole number ≥ 0", "Meals cooked that day."),
    "leftover_meals": ("Whole number ≥ 0", "Meals prepared but not served."),
    "average_cost_per_meal": ("Number ≥ 0 (KES)", "Average cost of one meal."),
    "rainfall_mm": ("Number ≥ 0", "Rainfall that day in millimetres."),
    "temperature_c": ("Number", "Average temperature that day in °C."),
    "school_term": ("Term 1 / Term 2 / Term 3", "School term the date falls in."),
    "is_holiday": ("0 or 1", "1 if the day was a public or school holiday."),
    "is_exam_period": ("0 or 1", "1 if exams were running."),
    "is_school_event": ("0 or 1", "1 if a school event (sports day, prize giving…) affected attendance."),
}

# In `REQUIRED_COLUMNS + OPTIONAL_COLUMNS` order.
EXAMPLE_ROWS = [
    ["2025-01-13", "SCH001", "Example Primary School", 812, "Nairobi", "Embakasi", 900, 830, 70, 840, 28, 45, 0, 24.5, "Term 1", 0, 0, 0],
    ["2025-01-14", "SCH001", "Example Primary School", 798, "Nairobi", "Embakasi", 900, 815, 85, 830, 32, 45, 3.2, 22.1, "Term 1", 0, 0, 0],
    ["2025-01-15", "SCH001", "Example Primary School", 845, "Nairobi", "Embakasi", 900, 860, 40, 850, 5, 45, 0, 25.0, "Term 1", 0, 0, 1],
]

_HEADER_FILL = PatternFill("solid", fgColor="1F6F50")
_REQUIRED_FILL = PatternFill("solid", fgColor="B5462F")
_WHITE_BOLD = Font(bold=True, color="FFFFFF")


def _columns() -> list[str]:
    return REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def _write_header(ws, columns: list[str]) -> None:
    ws.append(columns)
    for idx, name in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=idx)
        cell.font = _WHITE_BOLD
        cell.fill = _REQUIRED_FILL if name in REQUIRED_COLUMNS else _HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[get_column_letter(idx)].width = max(14, len(name) + 4)
    ws.freeze_panes = "A2"


def data_template_xlsx() -> bytes:
    """Excel template: `Data` (headers to fill), `Instructions` and `Example` sheets.

    BiteTracker reads the first sheet, so `Data` is first and must keep its header row.
    """
    columns = _columns()
    wb = Workbook()

    data = wb.active
    data.title = "Data"
    _write_header(data, columns)
    last_row = 5000
    flag_check = DataValidation(type="whole", operator="between", formula1="0", formula2="1", allow_blank=True,
                                error="Enter 0 or 1.", errorTitle="Invalid value")
    term_check = DataValidation(type="list", formula1='"Term 1,Term 2,Term 3"', allow_blank=True)
    count_check = DataValidation(type="decimal", operator="greaterThanOrEqual", formula1="0", allow_blank=True,
                                 error="Enter a number of 0 or more.", errorTitle="Invalid value")
    for check in (flag_check, term_check, count_check):
        data.add_data_validation(check)
    for idx, name in enumerate(columns, start=1):
        col_range = f"{get_column_letter(idx)}2:{get_column_letter(idx)}{last_row}"
        if name.startswith("is_"):
            flag_check.add(col_range)
        elif name == "school_term":
            term_check.add(col_range)
        elif "≥ 0" in COLUMN_GUIDE.get(name, ("",))[0]:
            count_check.add(col_range)

    guide = wb.create_sheet("Instructions")
    guide.append(["BiteTracker data upload template"])
    guide["A1"].font = Font(bold=True, size=14)
    for line in [
        "1. Enter one row per school per school day on the 'Data' sheet. Keep the header row exactly as it is.",
        "2. Columns with red headers are required. Green headers are optional but improve the data quality score.",
        "3. Use school-level daily totals only. Do NOT include learners' names, admission numbers or contact details.",
        "4. Save as .xlsx (or export the Data sheet as .csv) and upload it together with the signed supporting letter.",
        "5. An administrator reviews every valid upload and its letter before the data is considered for use.",
    ]:
        guide.append([line])
    guide.append([])
    guide.append(["Column", "Required?", "Format", "Description"])
    for cell in guide[guide.max_row]:
        cell.font = _WHITE_BOLD
        cell.fill = _HEADER_FILL
    for name in columns:
        fmt, description = COLUMN_GUIDE.get(name, ("", ""))
        guide.append([name, "Required" if name in REQUIRED_COLUMNS else "Optional", fmt, description])
    for letter, width in zip("ABCD", (24, 12, 26, 90)):
        guide.column_dimensions[letter].width = width

    example = wb.create_sheet("Example")
    _write_header(example, columns)
    for row in EXAMPLE_ROWS:
        example.append(row)

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


# --- Supporting letter -------------------------------------------------------------------------

_LETTER_LINES = [
    ("", False),
    ("[SCHOOL LETTERHEAD / NAME]", True),
    ("[Postal address]   [Phone]   [Email]", False),
    ("", False),
    ("Date: ____________________", False),
    ("", False),
    ("To: The BiteTracker Programme Administrator", False),
    ("", False),
    ("RE: SUBMISSION OF SCHOOL MEAL SERVICE DATA", True),
    ("", False),
    ("Dear Sir/Madam,", False),
    ("", False),
    ("I confirm that the attached data file contains the daily school meal records for the school named below, "
     "and that, to the best of my knowledge, the figures are accurate and complete.", False),
    ("", False),
    ("School name: ________________________________     School ID: ______________", False),
    ("County: _____________________     Sub-county: _____________________", False),
    ("Period covered: from ____________ to ____________", False),
    ("Data file name: ________________________________     Number of rows: ________", False),
    ("", False),
    ("The file contains school-level totals only and no personal information about learners. "
     "I authorise the programme to use these records for meal planning and demand forecasting.", False),
    ("", False),
    ("Data contact person: ____________________     Phone: ______________     Email: ____________________", False),
    ("", False),
    ("Yours faithfully,", False),
    ("", False),
    ("", False),
    ("Signature: ______________________          School stamp:", False),
    ("Name: __________________________", False),
    ("Designation: Head Teacher / Deputy Head Teacher", False),
]


def _paragraph(text: str, bold: bool) -> str:
    if not text:
        return "<w:p/>"
    props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    return f'<w:p><w:r>{props}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def letter_template_docx() -> bytes:
    """A minimal, valid Word document with the cover letter to fill, sign and stamp."""
    body = "".join(_paragraph(text, bold) for text, bold in _LETTER_LINES)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        f"{body}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()
