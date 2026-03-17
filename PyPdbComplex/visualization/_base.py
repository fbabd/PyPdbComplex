"""
Shared HTML page template for all visualization modules.

Usage example::

    from ._base import html_page

    extra_css = \"\"\"
        .my-chart { width: 100%; }
    \"\"\"

    body = f\"\"\"
    <div class="container">
        <h1>{title}</h1>
        ...
    </div>
    \"\"\"

    return html_page(title, body, extra_css=extra_css,
                     cdn_scripts=["https://cdn.plot.ly/plotly-2.27.0.min.js"])
"""

from __future__ import annotations
from typing import List, Optional


def common_css(max_width: int = 1400) -> str:
    """
    Returns the shared CSS used across all visualization pages.

    Includes: box-sizing reset, body font/background, .container layout,
    h1 heading style, and stat-card component.

    Args:
        max_width: Max width of .container in pixels (default 1400).
    """
    return f"""
        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }}

        .container {{
            max-width: {max_width}px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        }}

        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 28px;
        }}

        /* Stat cards — gradient accent component */
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 8px;
            color: white;
        }}

        .stat-label {{
            font-size: 11px;
            opacity: 0.9;
            text-transform: uppercase;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}

        /* Info box — blue left-border callout */
        .info-box {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #2196f3;
        }}

        .info-box p {{
            margin: 5px 0;
            color: #1565c0;
            font-size: 14px;
        }}
    """


def html_page(
    title: str,
    body: str,
    *,
    extra_css: str = '',
    cdn_scripts: Optional[List[str]] = None,
    cdn_styles: Optional[List[str]] = None,
    max_width: int = 1400,
) -> str:
    """
    Build a complete, self-contained HTML page.

    Args:
        title:       ``<title>`` text for the page.
        body:        Full HTML content to place inside ``<body>``.
        extra_css:   Additional CSS rules appended after ``common_css()``.
        cdn_scripts: CDN ``<script src>`` URLs to include in ``<head>``.
        cdn_styles:  CDN ``<link rel="stylesheet">`` URLs for ``<head>``.
        max_width:   Max width of the ``.container`` div (default 1400 px).

    Returns:
        A complete HTML string ready to write to a file or serve directly.
    """
    cdn_styles = cdn_styles or []
    cdn_scripts = cdn_scripts or []

    style_links = '\n    '.join(
        f'<link rel="stylesheet" href="{url}">' for url in cdn_styles
    )
    script_tags = '\n    '.join(
        f'<script src="{url}"></script>' for url in cdn_scripts
    )

    # Only emit the tags block when there is content to emit
    head_links = f'\n    {style_links}' if style_links else ''
    head_scripts = f'\n    {script_tags}' if script_tags else ''

    return (
        f'<!DOCTYPE html>\n'
        f'<html lang="en">\n'
        f'<head>\n'
        f'    <meta charset="utf-8">\n'
        f'    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'    <title>{title}</title>'
        f'{head_links}'
        f'{head_scripts}\n'
        f'    <style>\n'
        f'{common_css(max_width)}\n'
        f'{extra_css}\n'
        f'    </style>\n'
        f'</head>\n'
        f'<body>\n'
        f'{body}\n'
        f'</body>\n'
        f'</html>'
    )
