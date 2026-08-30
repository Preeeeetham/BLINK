"""
Converts everything_about_blikn.md into a high-quality PDF with embedded annotated images,
clean typography, tables, and page layout using headless Microsoft Edge.
"""

import base64
import os
from pathlib import Path
import re
import subprocess


def convert_markdown_to_pdf():
    workspace = Path(__file__).resolve().parent.parent
    md_path = workspace / "everything_about_blikn.md"
    pdf_path = workspace / "everything_about_blikn.pdf"
    html_tmp = workspace / "everything_about_blikn_temp.html"

    md_text = md_path.read_text(encoding="utf-8")

    # Embed images as Base64 to ensure 100% reliable local loading
    def embed_img(match):
        alt = match.group(1)
        rel_path = match.group(2).strip()
        img_p = (workspace / rel_path).resolve()
        if img_p.exists():
            with open(img_p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = img_p.suffix.lower()
            mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            return (
                f'<div class="img-container">'
                f'<img src="data:{mime};base64,{b64}" alt="{alt}">'
                f'<div class="img-caption">{alt}</div>'
                f'</div>'
            )
        return match.group(0)

    html_body = re.sub(r'!\[(.*?)\]\((.*?)\)', embed_img, md_text)

    # Process markdown lines
    lines = html_body.split("\n")
    out = []
    in_code = False
    code_buf = []
    in_table = False
    table_buf = []
    in_list = False

    for line in lines:
        if line.startswith("```"):
            if in_list:
                out.append("</ul>")
                in_list = False
            if in_code:
                code_text = "\n".join(code_buf)
                out.append(f"<pre><code>{code_text}</code></pre>")
                code_buf = []
                in_code = False
            else:
                in_code = True
                code_buf = []
            continue
        if in_code:
            code_buf.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Table handling
        if "|" in line and not line.startswith("#"):
            if in_list:
                out.append("</ul>")
                in_list = False
            if not in_table:
                in_table = True
                table_buf = []
            table_buf.append(line)
            continue
        else:
            if in_table:
                t_html = ["<table>"]
                is_header = True
                for r in table_buf:
                    if re.match(r"^\s*\|?\s*[-:]+[-| :]*\|?\s*$", r):
                        is_header = False
                        continue
                    cells = [c.strip() for c in r.strip("|").split("|")]
                    tag = "th" if is_header else "td"
                    row_str = "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
                    t_html.append(row_str)
                t_html.append("</table>")
                out.append("\n".join(t_html))
                table_buf = []
                in_table = False

        # Lists
        if line.startswith("- ") or line.startswith("* "):
            if not in_list:
                in_list = True
                out.append("<ul>")
            out.append(f"<li>{line[2:]}</li>")
            continue
        elif re.match(r"^\d+\.\s", line):
            if not in_list:
                in_list = True
                out.append("<ol>")
            num_match = re.match(r"^\d+\.\s(.*)", line)
            out.append(f"<li>{num_match.group(1)}</li>")
            continue
        else:
            if in_list:
                out.append("</ul>" if "<ul>" in out[-2] else "</ol>")
                in_list = False

        # Headings & Paragraphs
        if line.startswith("# "):
            out.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("#### "):
            out.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith("---"):
            out.append("<hr>")
        elif line.strip():
            if line.startswith("<div") or line.startswith("</div>") or line.startswith("<img"):
                out.append(line)
            else:
                out.append(f"<p>{line}</p>")

    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")

    parsed_content = "\n".join(out)

    # Format inline elements
    parsed_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', parsed_content)
    parsed_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', parsed_content)
    parsed_content = re.sub(r'`(.*?)`', r'<code>\1</code>', parsed_content)

    # Math cleanup for clean HTML rendering
    math_replacements = {
        "$T_0$": "T<sub>0</sub>",
        "$T_1$": "T<sub>1</sub>",
        "$T_t$": "T<sub>t</sub>",
        "$T_B$": "T<sub>B</sub>",
        "$V_{\\max}$": "V<sub>max</sub>",
        "$P_c$": "P<sub>c</sub>",
        "$\\Delta T$": "ΔT",
        "$\\mu\\text{m}$": "µm",
        "$\\text{K}$": " K",
        "$\\text{km/h}$": " km/h",
        "$\\text{m/s}$": " m/s",
        "$\\text{hPa}$": " hPa",
        "$\\text{mm/hr}$": " mm/hr",
        "$\\text{dB}$": " dB",
        "$\\mathbf{f}_{0 \\to 1}$": "f<sub>0→1</sub>",
        "$\\mathbf{f}_{1 \\to 0}$": "f<sub>1→0</sub>",
        "$\\vec{u}$": "u",
        "$\\vec{v}$": "v",
    }
    for k, v in math_replacements.items():
        parsed_content = parsed_content.replace(k, v)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Project BLINK Master Technical Guide</title>
<style>
    @page {{
        size: A4;
        margin: 14mm 14mm 14mm 14mm;
        @bottom-right {{
            content: "Page " counter(page);
            font-size: 8pt;
            color: #64748b;
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.5;
        color: #1e293b;
        background: #ffffff;
        margin: 0;
        padding: 0;
    }}
    h1 {{
        font-size: 20pt;
        color: #0f172a;
        margin-top: 0;
        margin-bottom: 4px;
        border-bottom: 2.5px solid #2563eb;
        padding-bottom: 6px;
    }}
    h2 {{
        font-size: 13pt;
        color: #1e3a8a;
        margin-top: 20px;
        margin-bottom: 6px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 3px;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 11pt;
        color: #0369a1;
        margin-top: 14px;
        margin-bottom: 4px;
        page-break-after: avoid;
    }}
    h4 {{
        font-size: 10pt;
        color: #334155;
        margin-top: 10px;
        margin-bottom: 3px;
        page-break-after: avoid;
    }}
    p {{
        margin: 5px 0;
        text-align: justify;
    }}
    ul, ol {{
        margin: 4px 0 8px 18px;
        padding: 0;
    }}
    li {{
        margin: 2.5px 0;
    }}
    hr {{
        border: none;
        border-top: 1px solid #cbd5e1;
        margin: 16px 0;
    }}
    code {{
        font-family: "Consolas", "Courier New", monospace;
        font-size: 8.5pt;
        background: #f1f5f9;
        color: #0f172a;
        padding: 1px 3px;
        border-radius: 3px;
        border: 1px solid #e2e8f0;
    }}
    pre {{
        background: #0f172a;
        color: #e2e8f0;
        padding: 8px 12px;
        border-radius: 5px;
        font-family: "Consolas", "Courier New", monospace;
        font-size: 8pt;
        overflow-x: auto;
        margin: 8px 0;
        page-break-inside: avoid;
    }}
    pre code {{
        background: transparent;
        color: #38bdf8;
        padding: 0;
        border: none;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 5px 7px;
        text-align: left;
    }}
    th {{
        background: #f1f5f9;
        color: #0f172a;
        font-weight: 700;
    }}
    tr:nth-child(even) {{
        background: #f8fafc;
    }}
    .img-container {{
        text-align: center;
        margin: 14px 0;
        page-break-inside: avoid;
    }}
    .img-container img {{
        max-width: 96%;
        height: auto;
        border-radius: 5px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .img-caption {{
        font-size: 8pt;
        color: #64748b;
        margin-top: 3px;
        font-style: italic;
    }}
</style>
</head>
<body>
{parsed_content}
</body>
</html>"""

    html_tmp.write_text(html_template, encoding="utf-8")
    print(f"Generated HTML template: {html_tmp}")

    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_bin = next((p for p in edge_paths if os.path.exists(p)), None)

    if not edge_bin:
        raise RuntimeError("Microsoft Edge binary not found for PDF generation.")

    cmd = [
        edge_bin,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path.resolve()}",
        html_tmp.resolve().as_uri(),
    ]

    print("Running Edge PDF print-to-pdf...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"SUCCESS: Created {pdf_path} ({size_kb:.1f} KB)")
        if html_tmp.exists():
            html_tmp.unlink()
        return True
    else:
        print(f"FAILED: Edge returned {res.returncode}. Stderr: {res.stderr}")
        return False


if __name__ == "__main__":
    convert_markdown_to_pdf()
