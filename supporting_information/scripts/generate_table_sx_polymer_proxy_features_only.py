from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import zipfile

import pandas as pd


import sys
from pathlib import Path

_SI_ROOT = Path(__file__).resolve().parents[1]
if str(_SI_ROOT) not in sys.path:
    sys.path.insert(0, str(_SI_ROOT))
from _paths import (  # noqa: E402
    BENCHMARK,
    CONFIG_YAML,
    DATASET_META_JSON,
    FINAL,
    INTERVALS_CSV,
    OUT,
    PROCESSED_CSV,
    RESULTS_CSV,
    ROOT,
    RUN,
    S8_MANIFEST,
    SIMILARITY_CSV,
    SUMMARY_CSV,
    XGB_SHAP_CSV,
    XGB_SHAP_FIG,
)

FEATURES_PY = ROOT / "tg-uq-generalization-benchmark" / "src" / "cms_tg" / "features.py"
CONFIG_YAML = ROOT / "tg-uq-generalization-benchmark" / "configs" / "tg.yaml"

TABLE_TITLE = "Table Sx. Polymer proxy features used in the predictive representation"


def proxy_feature_table() -> pd.DataFrame:
    rows = [
        {
            "feature name": "proxy_heavy_atoms",
            "definition": "Number of heavy atoms, excluding hydrogens, in the repeat-unit molecule.",
            "calculation source": "RDKit Mol.GetNumHeavyAtoms()",
            "unit/type": "count; integer",
            "included in model?": "yes",
        },
        {
            "feature name": "proxy_rings",
            "definition": "Number of rings in the repeat-unit molecular graph.",
            "calculation source": "RDKit Mol.GetRingInfo().NumRings()",
            "unit/type": "count; integer",
            "included in model?": "yes",
        },
        {
            "feature name": "proxy_aromatic_frac",
            "definition": "Fraction of heavy atoms that are aromatic atoms.",
            "calculation source": "Custom RDKit atom loop: sum(atom.GetIsAromatic()) / heavy_atoms",
            "unit/type": "fraction; continuous 0-1",
            "included in model?": "yes",
        },
        {
            "feature name": "proxy_hetero_frac",
            "definition": "Fraction of heavy atoms that are heteroatoms, defined as atoms other than carbon or hydrogen.",
            "calculation source": "Custom RDKit atom loop: sum(atomic_num not in [1, 6]) / heavy_atoms",
            "unit/type": "fraction; continuous 0-1",
            "included in model?": "yes",
        },
        {
            "feature name": "proxy_halogen_frac",
            "definition": "Fraction of heavy atoms that are halogens: F, Cl, Br, or I.",
            "calculation source": "Custom RDKit atom loop: sum(atomic_num in [9, 17, 35, 53]) / heavy_atoms",
            "unit/type": "fraction; continuous 0-1",
            "included in model?": "yes",
        },
    ]
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{latex_escape(TABLE_TITLE)}}}",
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\hline",
        " & ".join(latex_escape(col) for col in cols) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[col]) for col in cols) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def excel_col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def worksheet_xml(df: pd.DataFrame) -> str:
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    xml_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{excel_col_name(c_idx)}{r_idx}"
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        xml_rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        "</worksheet>"
    )


def write_xlsx(path: Path, df: pd.DataFrame) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Proxy features" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml(df))


def write_docx(path: Path, df: pd.DataFrame) -> None:
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    table_rows = []
    for row in rows:
        cells = "".join(
            "<w:tc><w:p><w:r><w:t>"
            + escape(str(value))
            + "</w:t></w:r></w:p></w:tc>"
            for value in row
        )
        table_rows.append(f"<w:tr>{cells}</w:tr>")
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        f"<w:p><w:r><w:t>{escape(TABLE_TITLE)}</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Only custom polymer proxy features actually included in the model are listed.</w:t></w:r></w:p>"
        f"<w:tbl>{''.join(table_rows)}</w:tbl>"
        + '<w:sectPr><w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/></w:sectPr>'
        + "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            "</Relationships>",
        )
        zf.writestr("word/document.xml", document_xml)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = proxy_feature_table()

    base = OUT / "TableSx_Polymer_Proxy_Features_Used"
    table.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".md").write_text(dataframe_to_markdown(table), encoding="utf-8")
    base.with_suffix(".tex").write_text(dataframe_to_latex(table), encoding="utf-8")
    write_xlsx(base.with_suffix(".xlsx"), table)
    write_docx(base.with_suffix(".docx"), table)

    notes = f"""{TABLE_TITLE}
Generated: {datetime.now().isoformat()}

Purpose:
- Focused SI reproducibility table for Reviewer 2 Comment 5 and Reviewer 1 Comment 3.
- Lists only polymer proxy features that are actually used by the predictive representation.

Inputs checked:
- Feature implementation: {FEATURES_PY}
- Configuration: {CONFIG_YAML}
- Config setting: features.add_polymer_proxy_features = true

Important scope note:
- This table intentionally lists only the five custom polymer proxy features.
- It does not list the full RDKit descriptor set.
- No unused proxy features are included.

Proxy features included:
{chr(10).join('- ' + name for name in table['feature name'])}
"""
    (OUT / "TableSx_Polymer_Proxy_Features_Used_notes.txt").write_text(notes, encoding="utf-8")

    print(TABLE_TITLE)
    print(table.to_string(index=False))
    print("Wrote:")
    for path in [
        base.with_suffix(".csv"),
        base.with_suffix(".md"),
        base.with_suffix(".tex"),
        base.with_suffix(".xlsx"),
        base.with_suffix(".docx"),
        OUT / "TableSx_Polymer_Proxy_Features_Used_notes.txt",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
