"""Generación del informe técnico en PDF.

Usamos ReportLab (`platypus`) para componer el documento a partir de
elementos (párrafos, tablas, espaciadores) en lugar de dibujar línea a
línea. El contenido combina:

* Datos deterministas (tecnologías, estructura, hallazgos de Semgrep).
* El análisis narrativo redactado por el LLM.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------- #
# Fuentes: intentamos registrar una fuente Unicode; si no, usamos la    #
# fuente por defecto y saneamos el texto.                               #
# --------------------------------------------------------------------- #
_UNICODE_FONT = None
_FONT_CANDIDATES = [
    ("Arial", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("Segoe UI", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("Arial", "/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]


def _register_fonts() -> None:
    """Registra la primera fuente Unicode disponible en el sistema."""
    global _UNICODE_FONT
    for name, regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists():
            try:
                pdfmetrics.registerFont(TTFont(name, regular))
                if Path(bold).exists():
                    pdfmetrics.registerFont(TTFont(f"{name}-Bold", bold))
                _UNICODE_FONT = name
                return
            except Exception:
                continue


def _safe(text: str) -> str:
    """Garantiza que el texto es representable con la fuente activa."""
    if _UNICODE_FONT:
        return text.replace("\x00", "")
    # Sin fuente Unicode: limitamos a latin-1 (ReportLab por defecto).
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _styles():
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontName=_UNICODE_FONT or "Helvetica",
        fontSize=10, leading=14, alignment=TA_LEFT, spaceAfter=6,
    )
    heading = ParagraphStyle(
        "Heading", parent=styles["Heading1"], fontName=(f"{_UNICODE_FONT}-Bold" if _UNICODE_FONT else "Helvetica-Bold"),
        fontSize=16, leading=20, spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#1a1a2e"),
    )
    subheading = ParagraphStyle(
        "SubHeading", parent=styles["Heading2"], fontName=(f"{_UNICODE_FONT}-Bold" if _UNICODE_FONT else "Helvetica-Bold"),
        fontSize=12, leading=16, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#16213e"),
    )
    title = ParagraphStyle(
        "Title", parent=styles["Title"], fontName=(f"{_UNICODE_FONT}-Bold" if _UNICODE_FONT else "Helvetica-Bold"),
        fontSize=22, leading=26, textColor=colors.HexColor("#0f3460"), spaceAfter=4,
    )
    meta = ParagraphStyle(
        "Meta", parent=body, fontSize=9, leading=12, textColor=colors.HexColor("#555555"),
    )
    return body, heading, subheading, title, meta


def _table(data: list[list[str]], header: bool = True):
    table = Table(data, hAlign="LEFT", colWidths=None)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), (f"{_UNICODE_FONT}-Bold" if _UNICODE_FONT else "Helvetica-Bold")),
        ]
    table.setStyle(TableStyle(style))
    return table


def generate_report(data: dict, output_path: Path) -> str:
    """Genera el PDF y devuelve la ruta del archivo creado."""
    _register_fonts()
    body, heading, subheading, title_style, meta = _styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Análisis técnico del repositorio",
    )

    story = []
    story.append(Paragraph(_safe(data.get("project_name", "Repositorio")), title_style))
    story.append(Paragraph(_safe(f"URL: {data.get('repo_url', '-')}"), meta))
    story.append(Paragraph(_safe(f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}"), meta))
    story.append(Spacer(1, 0.6 * cm))

    tech = data.get("tech_summary") or {}
    languages = tech.get("languages") or {}
    technologies = tech.get("technologies") or []

    story.append(Paragraph("Resumen", heading))
    story.append(Paragraph(
        _safe(
            f"El repositorio contiene {tech.get('total_files', 0)} archivos analizados "
            f"({tech.get('total_lines', 0)} líneas). Lenguajes principales: "
            + (", ".join(list(languages)[:8]) or "no detectados") + "."
        ),
        body,
    ))

    story.append(Paragraph("Lenguajes detectados", subheading))
    if languages:
        story.append(_table([["Lenguaje", "Archivos"]] + [[_safe(k), str(v)] for k, v in languages.items()]))
    else:
        story.append(Paragraph("No se detectaron lenguajes.", body))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Frameworks y tecnologías", subheading))
    if technologies:
        story.append(Paragraph(_safe(" · ".join(technologies)), body))
    else:
        story.append(Paragraph("No se detectaron frameworks conocidos.", body))

    story.append(Paragraph("Estructura del proyecto", subheading))
    structure = data.get("structure") or []
    if structure:
        for item in structure:
            story.append(Paragraph(_safe(f"• {item}"), body))
    else:
        story.append(Paragraph("Estructura no disponible.", body))

    story.append(PageBreak())

    story.append(Paragraph("Análisis del código (IA)", heading))
    story.append(Paragraph(_safe(data.get("analysis_summary", "(análisis no disponible)")), body))

    story.append(Paragraph("Hallazgos de Semgrep", heading))
    findings = data.get("semgrep_findings") or []
    if findings:
        rows = [["Severidad", "Archivo", "Línea", "Regla / descripción"]]
        for f in findings:
            rows.append([
                _safe(f.get("severity", "?")),
                _safe(f.get("path", "?")),
                str(f.get("line") or "-"),
                _safe((f.get("rule", "") + " — " + f.get("message", "")).strip(" —")),
            ])
        story.append(_table(rows))
    else:
        story.append(Paragraph("Sin hallazgos (o Semgrep no disponible).", body))

    story.append(Paragraph("Nota sobre seguridad", subheading))
    story.append(Paragraph(
        _safe(
            "Los hallazgos son *posibles* problemas detectados por análisis estático. "
            "No constituyen una garantía de seguridad: siempre debe realizarse una "
            "revisión manual y pruebas adicionales."
        ),
        body,
    ))

    doc.build(story)
    return str(output_path)
