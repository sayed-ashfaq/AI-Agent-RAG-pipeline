import os
from typing import Optional

import fitz

from exception_handler.agent_exceptions import FileProcessingError
from custom_logger import GLOBAL_LOGGER as log

class DocHandler:
    """
    PDF save + read (page-wise) for analysis.
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or "Somerandomsession"
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    # -------- Generic multi-format save --------
    def save_file(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            ext = os.path.splitext(filename)[1].lower()
            allowed = {".pdf", ".docx", ".pptx", ".md", ".txt", ".xlsx", ".xls", ".csv", ".db", ".sqlite", ".sqlite3"}
            if ext not in allowed:
                raise ValueError(f"Unsupported file type: {ext}. Allowed: {sorted(allowed)}")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("File saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save file", error=str(e), session_id=self.session_id)
            raise FileProcessingError(f"Failed to save file: {str(e)}", e) from e

    # -------- Generic multi-format read --------
    def read_text(self, path: str) -> str:
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                return self.read_pdf(path)
            if ext == ".docx":
                return self._read_docx(path)
            if ext == ".pptx":
                return self._read_pptx(path)
            if ext == ".md":
                return self._read_md(path)
            if ext == ".txt":
                return self._read_txt(path)
            if ext == ".csv":
                return self._read_csv(path)
            if ext == ".xlsx":
                return self._read_xlsx(path)
            if ext == ".xls":
                return self._read_xls(path)
            if ext in {".db", ".sqlite", ".sqlite3"}:
                return self._read_sqlite(path)
            raise ValueError(f"Unsupported extension for reading: {ext}")
        except Exception as e:
            log.error("Failed to read file", error=str(e), file_path=path, session_id=self.session_id)
            raise FileProcessingError(f"Could not process file: {path}", e) from e

    # Back-compat alias for helpers that try `read_`
    def read_(self, path: str) -> str:
        return self.read_text(path)

    def save_pdf(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save PDF", error=str(e), session_id=self.session_id)
            raise FileProcessingError(f"Failed to save PDF: {str(e)}", e) from e

    def read_pdf(self, pdf_path: str) -> str:
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")  # type: ignore
            text = "\n".join(text_chunks)
            log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read PDF", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise FileProcessingError(f"Could not process PDF: {pdf_path}", e) from e

    # -------- Per-type readers --------
    def _read_docx(self, path: str) -> str:
        import docx2txt
        try:
            text = docx2txt.process(path) or ""
            log.info("DOCX read successfully", file_path=path)
            return text
        except Exception as e:
            log.error("Failed to read DOCX", error=str(e), file_path=path)
            raise

    def _read_pptx(self, path: str) -> str:
        try:
            from pptx import Presentation
            prs = Presentation(path)
            parts = []
            for slide_idx, slide in enumerate(prs.slides, start=1):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        slide_text.append(shape.text)
                if slide_text:
                    parts.append(f"\n--- Slide {slide_idx} ---\n" + "\n".join(slide_text))
            text = "\n".join(parts)
            log.info("PPTX read successfully", file_path=path, slides=len(prs.slides))
            return text
        except Exception as e:
            log.error("Failed to read PPTX", error=str(e), file_path=path)
            raise

    def _read_md(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            log.info("MD read successfully", file_path=path)
            return content
        except Exception as e:
            log.error("Failed to read MD", error=str(e), file_path=path)
            raise

    def _read_txt(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            log.info("TXT read successfully", file_path=path)
            return content
        except Exception as e:
            log.error("Failed to read TXT", error=str(e), file_path=path)
            raise

    def _read_csv(self, path: str) -> str:
        import csv
        try:
            lines = []
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    lines.append(", ".join("" if c is None else str(c) for c in row))
            text = "\n".join(lines)
            log.info("CSV read successfully", file_path=path, rows=len(lines))
            return text
        except Exception as e:
            log.error("Failed to read CSV", error=str(e), file_path=path)
            raise

    def _read_xlsx(self, path: str) -> str:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            parts = []
            for ws in wb.worksheets:
                parts.append(f"\n--- Sheet: {ws.title} ---")
                for row in ws.iter_rows(values_only=True):
                    parts.append("\t".join("" if c is None else str(c) for c in row))
            text = "\n".join(parts)
            log.info("XLSX read successfully", file_path=path, sheets=len(wb.worksheets))
            return text
        except Exception as e:
            log.error("Failed to read XLSX", error=str(e), file_path=path)
            raise

    def _read_xls(self, path: str) -> str:
        try:
            import xlrd
            wb = xlrd.open_workbook(path)
            parts = []
            for sheet in wb.sheets():
                parts.append(f"\n--- Sheet: {sheet.name} ---")
                for rx in range(sheet.nrows):
                    row = [sheet.cell_value(rx, cx) for cx in range(sheet.ncols)]
                    parts.append("\t".join("" if c is None else str(c) for c in row))
            text = "\n".join(parts)
            log.info("XLS read successfully", file_path=path, sheets=wb.nsheets)
            return text
        except Exception as e:
            log.error("Failed to read XLS", error=str(e), file_path=path)
            raise

    def _read_sqlite(self, path: str) -> str:
        try:
            import sqlite3
            # Open database read-only
            uri = f"file:{path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # List tables
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
            tables = [r[0] for r in cur.fetchall()]
            if not tables:
                log.info("SQLite DB has no user tables", file_path=path)
                return ""

            parts: list[str] = []
            for t in tables:
                parts.append(f"\n--- Table: {t} ---")
                # Get columns
                try:
                    cur.execute(f"PRAGMA table_info('{t}')")
                    cols = [row[1] for row in cur.fetchall()]
                except Exception:
                    cols = []
                if cols:
                    parts.append("# Columns: " + ", ".join(cols))
                # Dump limited rows
                try:
                    cur.execute(f"SELECT * FROM '{t}' LIMIT 1000")
                    rows = cur.fetchall()
                    for r in rows:
                        if isinstance(r, sqlite3.Row):
                            vals = [r[k] for k in r.keys()]
                        else:
                            vals = list(r)
                        parts.append("\t".join("" if v is None else str(v) for v in vals))
                except Exception as e:  # pragma: no cover - best-effort
                    parts.append(f"# Error reading table '{t}': {e}")

            conn.close()
            text = "\n".join(parts)
            log.info("SQLite read successfully", file_path=path, tables=len(tables))
            return text
        except Exception as e:
            log.error("Failed to read SQLite DB", error=str(e), file_path=path)
            raise