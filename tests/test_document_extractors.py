"""Extraction of plain text from client documents (PDF/DOCX/XLSX/CSV), before any anonymization.

Fixtures are generated in-memory with the same libraries the extractor uses to read them, so these
tests exercise the real file formats without needing binary fixture files committed to the repo.
"""

import csv

import pytest

from src.documents.extractors import ExtractionError, extract_csv, extract_docx, extract_pdf, extract_text, extract_xlsx


class TestSourceFileCanBeDeletedRightAfterExtraction:
    """Regression tests: openpyxl's read-only workbook kept its zip file handle open after
    extraction, so deleting the source file immediately afterwards failed with a PermissionError
    on Windows -- found live via the API's upload flow (extract from a temp file, then let
    tempfile.TemporaryDirectory clean it up). Every extractor must release its file handle."""

    def test_csv(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("Nom,Valeur\nA,B\n", encoding="utf-8")
        extract_text(path)
        path.unlink()

    def test_xlsx(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "test.xlsx"
        wb = openpyxl.Workbook()
        wb.active.append(["Nom", "Valeur"])
        wb.active.append(["A", "B"])
        wb.save(str(path))
        extract_text(path)
        path.unlink()

    def test_docx(self, tmp_path):
        docx = pytest.importorskip("docx")
        path = tmp_path / "test.docx"
        document = docx.Document()
        document.add_paragraph("Un paragraphe de test.")
        document.save(str(path))
        extract_text(path)
        path.unlink()

    def test_pdf(self, tmp_path):
        pytest.importorskip("pypdf")
        reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")
        path = tmp_path / "test.pdf"
        c = reportlab_canvas.Canvas(str(path))
        c.drawString(72, 720, "Test PDF.")
        c.save()
        extract_text(path)
        path.unlink()


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ExtractionError, match="Unsupported"):
        extract_text(path)


class TestCsv:
    def test_identity_column_is_wrapped_with_a_trigger_keyword(self, tmp_path):
        path = tmp_path / "comptes.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Nom", "Responsable", "Serveur"])
            writer.writerow(["Compte admin", "Marie Dupont", "SRV-PROD-01"])
        text = extract_csv(path)
        assert "Responsable : Marie Dupont" in text
        assert "Serveur : SRV-PROD-01" in text

    def test_non_identity_column_keeps_its_own_header(self, tmp_path):
        path = tmp_path / "notes.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Statut", "Commentaire"])
            writer.writerow(["Ouvert", "A revoir"])
        text = extract_csv(path)
        assert "Statut: Ouvert" in text

    def test_empty_cells_are_skipped(self, tmp_path):
        path = tmp_path / "sparse.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Nom", "Responsable"])
            writer.writerow(["Compte X", ""])
        text = extract_csv(path)
        assert "Responsable" not in text

    def test_extract_text_dispatches_by_extension(self, tmp_path):
        path = tmp_path / "comptes.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Nom"])
            writer.writerow(["Compte A"])
        assert extract_text(path) == extract_csv(path)


class TestXlsx:
    def test_identity_column_is_wrapped_and_sheet_name_kept(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "comptes.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Comptes"
        ws.append(["Nom", "Responsable", "Email"])
        ws.append(["Compte admin", "Alice Bernard", "alice.bernard@client.example"])
        wb.save(str(path))

        text = extract_xlsx(path)

        assert "[Comptes]" in text
        assert "Responsable : Alice Bernard" in text
        assert "Email : alice.bernard@client.example" in text

    def test_empty_sheet_produces_no_section(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        path = tmp_path / "empty.xlsx"
        wb = openpyxl.Workbook()
        wb.active.title = "Vide"
        wb.save(str(path))
        assert extract_xlsx(path) == ""


class TestDocx:
    def test_paragraphs_and_table_cells_are_extracted(self, tmp_path):
        docx = pytest.importorskip("docx")
        path = tmp_path / "rapport.docx"
        document = docx.Document()
        document.add_paragraph("Rapport d'audit precedent.")
        document.add_paragraph("Le responsable IT est Marie Dupont.")
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Serveur"
        table.rows[0].cells[1].text = "SRV-PROD-01"
        document.save(str(path))

        text = extract_docx(path)

        assert "Rapport d'audit precedent." in text
        assert "Marie Dupont" in text
        assert "Serveur | SRV-PROD-01" in text

    def test_empty_document_yields_empty_text(self, tmp_path):
        docx = pytest.importorskip("docx")
        path = tmp_path / "empty.docx"
        docx.Document().save(str(path))
        assert extract_docx(path) == ""


class TestPdf:
    def test_extracts_text_from_a_generated_pdf(self, tmp_path):
        pytest.importorskip("pypdf")
        reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")
        path = tmp_path / "rapport.pdf"
        c = reportlab_canvas.Canvas(str(path))
        c.drawString(72, 720, "Rapport d'audit precedent.")
        c.drawString(72, 700, "Constat: acces non revus.")
        c.save()

        text = extract_pdf(path)

        assert "Rapport d'audit precedent." in text
        assert "Constat: acces non revus." in text
