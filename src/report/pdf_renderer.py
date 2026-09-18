"""Renders a Report to a .pdf file.

Independent from docx_renderer.py: both only walk the Report/ReportSection/Block model
(models.py), so neither duplicates the cahier des charges section logic that lives in builder.py.
Chosen over converting the .docx (which would need Word or LibreOffice installed -- a
machine-specific dependency this project avoids everywhere else): reportlab is pure Python.
"""

from pathlib import Path
from typing import List, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable, ListFlowable, ListItem, Paragraph as PdfParagraph, SimpleDocTemplate, Spacer,
    Table as PdfTable, TableStyle,
)

from src.report.models import BulletList, Paragraph, Report, Table

_STYLES = getSampleStyleSheet()
_FINDING_HEADER_STYLE = ParagraphStyle(
    "FindingHeader", parent=_STYLES["BodyText"], fontName="Helvetica-Bold", spaceBefore=8,
)
_TABLE_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5496")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
])


def render_pdf(report: Report, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )

    story: List[Flowable] = [PdfParagraph(_escape(report.title), _STYLES["Title"])]
    if report.subtitle:
        story.append(PdfParagraph(_escape(report.subtitle), _STYLES["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    for section in report.sections:
        story.append(PdfParagraph(_escape(section.title), _STYLES["Heading1"]))
        for block in section.blocks:
            story.extend(_render_block(block))
        story.append(Spacer(1, 0.3 * cm))

    document.build(story)


def _render_block(block: Union[Paragraph, BulletList, Table]) -> List[Flowable]:
    if isinstance(block, Paragraph):
        style = _FINDING_HEADER_STYLE if block.bold else _STYLES["BodyText"]
        return [PdfParagraph(_escape(block.text), style)]
    if isinstance(block, BulletList):
        items = [ListItem(PdfParagraph(_escape(item), _STYLES["BodyText"])) for item in block.items]
        return [ListFlowable(items, bulletType="bullet")]
    if isinstance(block, Table):
        data = [block.headers] + block.rows
        table = PdfTable(data, hAlign="LEFT")
        table.setStyle(_TABLE_STYLE)
        return [table, Spacer(1, 0.2 * cm)]
    raise TypeError(f"Unknown block type: {type(block)}")


def _escape(text: str) -> str:
    """reportlab's Paragraph interprets a subset of HTML/XML markup; escape generated text so
    stray '<' or '&' characters can never break layout or be interpreted as markup."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
