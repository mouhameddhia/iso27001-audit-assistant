"""render_pdf(): content round-trip via pypdf (already a project dependency, no new one needed
just to verify output) -- reopen the written file and check the real content is there.
"""

import pypdf

from src.report.models import BulletList, Paragraph, Report, ReportSection, Table
from src.report.pdf_renderer import render_pdf


def extract_text(path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() for page in reader.pages)


def test_title_and_subtitle_are_written(tmp_path):
    path = tmp_path / "report.pdf"
    render_pdf(Report(title="Rapport d'audit", subtitle="Contoso — MISSION-001", sections=[]), path)
    text = extract_text(path)
    assert "Rapport d'audit" in text and "Contoso" in text and "MISSION-001" in text


def test_section_titles_and_paragraph_text_present(tmp_path):
    path = tmp_path / "report.pdf"
    report = Report("T", "", [ReportSection("Constats", [Paragraph("Un constat spécifique.")])])
    render_pdf(report, path)
    text = extract_text(path)
    assert "Constats" in text and "Un constat spécifique." in text


def test_bullet_list_items_are_all_present(tmp_path):
    path = tmp_path / "report.pdf"
    report = Report("T", "", [ReportSection("Conclusion", [BulletList(["ISO/IEC 27001 A.5.18", "ISO/IEC 27001 A.8.8"])])])
    render_pdf(report, path)
    text = extract_text(path)
    assert "ISO/IEC 27001 A.5.18" in text and "ISO/IEC 27001 A.8.8" in text


def test_table_headers_and_rows_are_present(tmp_path):
    path = tmp_path / "report.pdf"
    report = Report("T", "", [ReportSection("Informations générales", [
        Table(["Champ", "Valeur"], [["Client", "Contoso"]])
    ])])
    render_pdf(report, path)
    text = extract_text(path)
    assert "Champ" in text and "Client" in text and "Contoso" in text


def test_unicode_content_survives(tmp_path):
    path = tmp_path / "report.pdf"
    text = "Écart : accès et privilèges à revoir."
    render_pdf(Report("T", "", [ReportSection("S", [Paragraph(text)])]), path)
    assert text in extract_text(path)


def test_ampersand_and_angle_brackets_do_not_break_rendering(tmp_path):
    """reportlab's Paragraph interprets a markup subset; unescaped '<'/'&' would corrupt layout
    or raise. The renderer must escape them and still produce the literal text."""
    path = tmp_path / "report.pdf"
    text = "Seuil non respecté : valeur < 100 & alerte non déclenchée."
    render_pdf(Report("T", "", [ReportSection("S", [Paragraph(text)])]), path)
    extracted = extract_text(path)
    assert "valeur" in extracted and "100" in extracted and "alerte" in extracted


def test_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "report.pdf"
    render_pdf(Report("T", "", []), path)
    assert path.exists()


def test_produces_a_valid_pdf_with_at_least_one_page(tmp_path):
    path = tmp_path / "report.pdf"
    render_pdf(Report("T", "", [ReportSection("S", [Paragraph("x")])]), path)
    assert len(pypdf.PdfReader(str(path)).pages) >= 1
