"""Parse the official EUR-Lex XHTML of Regulation (EU) 2024/1689 into structured units.

EUR-Lex publishes the Official Journal text with stable ELI identifiers (``rct_N``, ``art_N``,
``anx_X``, ``cpt_X.sct_N``) and numbered paragraph containers (``005.001`` = Article 5(1)). Enumerated
points are laid out as two-column tables (``(a)`` | text); we flatten those into ``(a) text`` lines so
the structure survives as plain text.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from .models import Paragraph, Unit

_ENUM = re.compile(r"^(\(?[0-9]{1,3}[a-z]?\)|[0-9]{1,3}\.|\([a-z]{1,2}\)|\([ivxlc]+\)|—|-)$")
_PARA_DIV = re.compile(r"^(\d{3})\.(\d{3})$")
_WS = re.compile(r"[ \t   ]+")
_TITLE_CLASSES = {"oj-ti-art", "oj-sti-art", "oj-doc-ti", "oj-ti-section-1", "oj-ti-section-2"}


_SPACE_BEFORE = re.compile(r"\s+([.,;:)\]^’])")
_SPACE_AFTER = re.compile(r"([(\[‘])\s+")


def _clean(text: str) -> str:
    """Collapse whitespace and undo the spaces get_text(" ") inserts around inline markup."""
    text = _WS.sub(" ", text).strip()
    return _SPACE_AFTER.sub(r"\1", _SPACE_BEFORE.sub(r"\1", text))


def _normalise_markup(root: Tag) -> None:
    # Footnote references are <a><span class="oj-super">N</span></a>: drop them.
    for sup in root.select("a > span.oj-super"):
        sup.parent.decompose()
    # Remaining superscripts are exponents (e.g. 10^25 FLOPs): keep them readable.
    for sup in root.select("span.oj-super"):
        sup.replace_with(f"^{sup.get_text(strip=True)}")


def _is_text_block(el: Tag) -> bool:
    """Leaf text containers: <p> paragraphs, and bare <span>s placed directly in table cells (Annex I)."""
    if el.name == "p":
        return not (set(el.get("class") or []) & _TITLE_CLASSES)
    return el.name == "span" and el.parent is not None and el.parent.name == "td"


def _lines(el: Tag) -> list[str]:
    """Flatten an element into text lines, merging enumeration labels with the text that follows."""
    raw = [_clean(b.get_text(" ")) for b in el.find_all(_is_text_block)]
    out: list[str] = []
    pending: str | None = None
    for line in raw:
        if not line:
            continue
        if _ENUM.match(line):
            pending = f"{pending} {line}" if pending else line
            continue
        out.append(f"{pending} {line}" if pending else line)
        pending = None
    if pending:
        out.append(pending)
    return out


def _title_text(div: Tag | None) -> str | None:
    if div is None:
        return None
    t = div.find("div", class_="eli-title", recursive=False)
    return _clean(t.get_text(" ")) if t else None


def _container_info(art: Tag) -> dict[str, str | None]:
    info: dict[str, str | None] = {
        "chapter": None,
        "chapter_title": None,
        "section": None,
        "section_title": None,
    }
    for parent in art.parents:
        pid = parent.get("id") if isinstance(parent, Tag) else None
        if not pid:
            continue
        if m := re.fullmatch(r"cpt_([IVXL]+)\.sct_(\d+)", pid):
            info["section"], info["section_title"] = m.group(2), _title_text(parent)
        elif m := re.fullmatch(r"cpt_([IVXL]+)", pid):
            info["chapter"], info["chapter_title"] = m.group(1), _title_text(parent)
    if info["chapter_title"]:
        title = info["chapter_title"].capitalize()  # source is ALL CAPS
        info["chapter_title"] = re.sub(r"\b(ai|eu|gpai)\b", lambda m: m.group(1).upper(), title)
    return info


def _parse_recital(div: Tag) -> Unit:
    num = div["id"].split("_", 1)[1]
    text = " ".join(_lines(div))
    text = re.sub(rf"^\({num}\)\s*", "", text)
    return Unit(
        id=div["id"],
        kind="recital",
        number=num,
        paragraphs=[Paragraph(id=f"{div['id']}.0", label=f"Recital ({num})", text=text)],
    )


def _parse_article(div: Tag) -> Unit:
    num = div["id"].split("_", 1)[1]
    sti = div.find("p", class_="oj-sti-art")
    title = _clean(sti.get_text(" ")) if sti else ""
    paragraphs: list[Paragraph] = []
    para_divs = [c for c in div.find_all("div", recursive=False) if _PARA_DIV.match(c.get("id", ""))]
    if para_divs:
        for pd in para_divs:
            pno = str(int(_PARA_DIV.match(pd["id"]).group(2)))
            paragraphs.append(
                Paragraph(id=f"{div['id']}.{pno}", label=f"Article {num}({pno})", text="\n".join(_lines(pd)))
            )
    else:
        paragraphs.append(Paragraph(id=f"{div['id']}.0", label=f"Article {num}", text="\n".join(_lines(div))))
    return Unit(
        id=div["id"], kind="article", number=num, title=title, paragraphs=paragraphs, **_container_info(div)
    )


def _parse_annex(div: Tag) -> Unit:
    num = div["id"].split("_", 1)[1]
    titles = [_clean(p.get_text(" ")) for p in div.find_all("p", class_="oj-doc-ti", recursive=False)]
    title = titles[1] if len(titles) > 1 else ""
    # Top-level children: intro paragraphs, then one table (or sub-container) per numbered point.
    paragraphs: list[Paragraph] = []
    intro: list[str] = []
    point = 0
    for child in div.children:
        if not isinstance(child, Tag):
            continue
        classes = set(child.get("class") or [])
        if child.name == "p" and classes & _TITLE_CLASSES:
            continue
        lines = _lines(child) if child.name != "p" else [_clean(child.get_text(" "))]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        if child.name == "p" and not paragraphs and child.find_next_sibling(["table", "div"]) is not None:
            intro.extend(lines)
            continue
        point += 1
        m = re.match(r"^(\d{1,2}[a-z]?)\.\s", lines[0]) or re.match(
            r"^(?:Section|Part)\s+([A-Z0-9]+)", lines[0]
        )
        label = f"Annex {num}, point {m.group(1)}" if m else f"Annex {num} ({point})"
        body = "\n".join(intro + lines)
        intro = []
        paragraphs.append(Paragraph(id=f"{div['id']}.{point}", label=label, text=body))
    if intro:
        paragraphs.append(
            Paragraph(id=f"{div['id']}.{point + 1}", label=f"Annex {num}", text="\n".join(intro))
        )
    return Unit(id=div["id"], kind="annex", number=num, title=title, paragraphs=paragraphs)


def parse_html(html: str, include_recitals: bool = True) -> list[Unit]:
    with warnings.catch_warnings():  # EUR-Lex serves XHTML; the lenient HTML parser is what we want
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    _normalise_markup(soup)
    units: list[Unit] = []

    def by_prefix(prefix: str) -> Iterable[Tag]:
        return soup.find_all("div", id=re.compile(rf"^{prefix}_[0-9IVXL]+$"))

    if include_recitals:
        units += [_parse_recital(d) for d in by_prefix("rct")]
    units += [_parse_article(d) for d in by_prefix("art")]
    units += [_parse_annex(d) for d in by_prefix("anx")]
    if not any(u.kind == "article" for u in units):
        raise ValueError("No articles found: the EUR-Lex HTML layout may have changed.")
    return units


def load_units(path: Path) -> list[Unit]:
    with path.open(encoding="utf-8") as f:
        return [Unit.model_validate_json(line) for line in f if line.strip()]


def save_units(units: list[Unit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for u in units:
            f.write(u.model_dump_json() + "\n")
