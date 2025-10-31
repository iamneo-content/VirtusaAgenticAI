# diagram_publisher.py
# One public function: publish_diagrams(...)
from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path
import streamlit.components.v1 as components

def publish_diagrams(
    mermaid_map: Dict[str, str],
    out_dir: str,
    title: str = "HLD Diagrams",
    theme: str = "default",            # mermaid: default | neutral | dark
    preview: bool = True,              # show inline in Streamlit
    save_fullpage_html: bool = True,   # writes <out_dir>/full_diagrams.html
    hld_markdown: Optional[str] = None,# if provided, also build a printable HLD.html from markdown
    hld_html_out_path: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Build a full diagrams HTML page (client-side Mermaid) and optionally a printable HLD HTML from markdown.
    Returns:
      { "full_html": "<path>|None", "hld_html": "<path>|None" }
    """

    # --- helpers kept INSIDE this function (no extra public APIs) ---
    theme = (theme or "default").lower()
    if theme not in {"default", "neutral", "dark"}:
        theme = "default"

    HTML_SHELL = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    :root { --pad: 24px; }
    @media print { a[href]:after { content: ""; } }
    body{margin:var(--pad);font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu;line-height:1.4}
    h1{margin-top:0}
    .mermaid{max-width:100%; margin:16px 0;}
    .toolbar{position:sticky;top:0;background:#fff;padding-bottom:8px;margin-bottom:16px;border-bottom:1px solid #eee}
    .btn{padding:8px 12px;border:1px solid #ddd;border-radius:8px;background:#f7f7f7;cursor:pointer}
    .meta{color:#555;margin-top:12px}
  </style>
</head>
<body>
  <div class="toolbar">
    <button class="btn" onclick="window.print()">Print / Save as PDF</button>
  </div>
  <h1>__TITLE__</h1>
  __BODY__
  <div class="meta">Generated</div>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
  mermaid.initialize({ startOnLoad: true, securityLevel: "loose", theme: "__THEME__" });
  await mermaid.run();
</script>
</body>
</html>"""

    def _blocks_from_map(_map: Dict[str, str]) -> str:
        parts = []
        for name, code in _map.items():
            parts.append(f"<h3 style='margin:8px 0'>{name}</h3>\n<div class='mermaid'>\n{code}\n</div>")
        return "\n".join(parts)

    def _md_to_html_with_mermaid(md_text: str) -> str:
        import re, html as _html
        try:
            import markdown as _markdown
        except Exception:
            _markdown = None
        def repl(m):
            code = m.group(1).strip()
            return f'\n<div class="mermaid">\n{code}\n</div>\n'
        md_pre = re.sub(r"```mermaid\s*(.*?)```", repl, md_text, flags=re.S | re.I)
        body = _markdown.markdown(md_pre, extensions=["fenced_code", "tables"]) if _markdown else f"<pre>{_html.escape(md_pre)}</pre>"
        return (HTML_SHELL
                .replace("__BODY__", body)
                .replace("__THEME__", theme)
                .replace("__TITLE__", title))

    # --- main flow ---
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    # 1) Full diagrams page (from mermaid_map)
    full_html_path = None
    diagrams_body = _blocks_from_map(mermaid_map)
    # __TITLE__ is already rendered by HTML_SHELL; don't inject an extra <h1>
    full_html_text = (HTML_SHELL
                      .replace("__BODY__", diagrams_body)
                      .replace("__THEME__", theme)
                      .replace("__TITLE__", title))
    if save_fullpage_html:
        full_html_path = str(base / "full_diagrams.html")
        Path(full_html_path).write_text(full_html_text, encoding="utf-8")
    if preview:
        est_height = 280 + max(1, len(mermaid_map)) * 360
        components.html(full_html_text, height=min(est_height, 2800), scrolling=True)

    # 2) Printable HLD HTML (optional, from Markdown with ```mermaid)
    hld_html_path = None
    if hld_markdown is not None:
        if hld_html_out_path:
            hld_html_path = hld_html_out_path
        else:
            # default: sibling /hld/HLD.html
            hld_html_path = str(base.parent / "hld" / "HLD.html")
        Path(hld_html_path).parent.mkdir(parents=True, exist_ok=True)
        Path(hld_html_path).write_text(_md_to_html_with_mermaid(hld_markdown), encoding="utf-8")

    return {"full_html": full_html_path, "hld_html": hld_html_path}


def render_mermaid_inline(mermaid_code: str, *, key: str, height: int = 520, theme: str = "default") -> None:
    """
    Render a SINGLE Mermaid diagram inline in Streamlit using the Mermaid CDN.
    Public API (no extra helpers exposed). All helpers/constants live inside.
    """
    # --- everything is scoped inside this function ---
    theme = (theme or "default").lower()
    if theme not in {"default", "neutral", "dark"}:
        theme = "default"

    def _build_html(code: str, container_id: str) -> str:
        return f"""
        <div id="{container_id}" class="mermaid">
{code}
        </div>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose', theme: '{theme}' }});
          mermaid.init(undefined, document.getElementById("{container_id}"));
        </script>
        """

    container_id = f"mermaid-{key}"
    html = _build_html(mermaid_code, container_id)
    components.html(html, height=height, scrolling=True)