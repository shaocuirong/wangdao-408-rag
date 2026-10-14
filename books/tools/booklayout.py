# -*- coding: utf-8 -*-
"""Layout reconstruction for the OCR'd textbooks.

Turns raw OCR boxes into clean reading-ordered text, then hangs that text on the
bookmark hierarchy.

Coordinate facts derived from the scans (1353x1885 render):
  * y < 9%   -> running head: book page number + chapter name.  Dropped, but the page
                number is captured so citations can name the printed page.
  * y > 93%  -> publisher promo watermark ("配套课程公众号【研大】"/"联系微信..."). Dropped.
  * Headings such as "1.1" and "操作系统的基本概念" arrive as two boxes sharing a row,
    so same-row boxes must be re-joined before anything else.
"""
import re

PROMO_RE = re.compile(r"公众号|研大|微信|扫一扫|视频讲解|配套课程|酷酷刷题|包邮|刷题小程序")
# OCR sometimes clips the watermark into a bare phone-number fragment.
PHONE_RE = re.compile(r"[0-9]{9,}")
NUM_RE   = re.compile(r"^[0-9０-９]{1,4}$")
ROMAN_RE = re.compile(r"^[IVXLCDM]{1,6}$")

_WS_RE = re.compile(r"[\s\u3000]+")
_PUNCT = "，。、；：？！“”‘’（）《》〈〉【】—…·．,.;:?!'\"()<>[]{}~_-—/|+=*&^%$#@"

TOP_FRAC = 0.09
BOT_FRAC = 0.93


def norm_key(t):
    """Whitespace/punctuation-insensitive key used to match heading text.

    Digits are deliberately preserved: they are what separates "1.1" from "1.2"
    and "1．数据" from "2．数据元素", so folding them away causes false anchors.
    """
    t = _WS_RE.sub("", t)
    t = t.replace("．", ".").replace("。", ".")
    return t.strip(_PUNCT)


def norm_plain(t):
    """Lighter normalisation, used for duplicate detection."""
    return _WS_RE.sub("", t).strip(_PUNCT)


def is_promo(t):
    return bool(PROMO_RE.search(t)) or bool(PHONE_RE.search(t))


def split_margins(lines, w, h):
    """Return (body_lines, header_lines, footer_lines)."""
    body, head, foot = [], [], []
    for L in lines:
        if L["y1"] < TOP_FRAC * h:
            head.append(L)
        elif L["y0"] > BOT_FRAC * h:
            foot.append(L)
        else:
            body.append(L)
    return body, head, foot


def book_page_label(header_lines, w):
    """The printed page number sits at the outer edge of the running head."""
    cands = []
    for L in header_lines:
        t = L["t"].strip()
        if NUM_RE.match(t) or ROMAN_RE.match(t):
            cands.append((L["x0"] if L["x0"] < w / 2 else w - L["x1"], t))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands[0][1]


def merge_rows(lines):
    """Group boxes that share a visual row, then order them left to right."""
    if not lines:
        return []
    ordered = sorted(lines, key=lambda L: (L["y0"], L["x0"]))
    rows, cur = [], [ordered[0]]
    for L in ordered[1:]:
        cur_top = min(x["y0"] for x in cur)
        cur_bot = max(x["y1"] for x in cur)
        ov = min(cur_bot, L["y1"]) - max(cur_top, L["y0"])
        ref = min(cur_bot - cur_top, L["y1"] - L["y0"]) or 1
        if ov > 0.45 * ref:
            cur.append(L)
        else:
            rows.append(cur)
            cur = [L]
    rows.append(cur)

    out = []
    for row in rows:
        row.sort(key=lambda L: L["x0"])
        parts, prev_x1, prev_h = [], None, 0
        for L in row:
            t = L["t"].strip()
            if not t:
                continue
            if prev_x1 is not None:
                hgt = max(prev_h, L["y1"] - L["y0"]) or 1
                if L["x0"] - prev_x1 > 0.45 * hgt:
                    parts.append(" ")
            parts.append(t)
            prev_x1 = L["x1"]
            prev_h = L["y1"] - L["y0"]
        text = "".join(parts).strip()
        if text:
            out.append({
                "t": text,
                "y0": min(L["y0"] for L in row),
                "y1": max(L["y1"] for L in row),
                "x0": min(L["x0"] for L in row),
                "x1": max(L["x1"] for L in row),
            })
    out.sort(key=lambda r: r["y0"])
    return out


def _ascii_word_end(c):
    return c.isascii() and (c.isalnum() or c in ")]}")


def _ascii_word_start(c):
    return c.isascii() and (c.isalnum() or c in "([{")


def join_lines(texts):
    """Concatenate OCR line fragments the way Chinese typesetting expects."""
    s = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if not s:
            s = t
            continue
        if s.endswith("-") and re.match(r"^[A-Za-z]", t):
            s = s[:-1] + t
        elif _ascii_word_end(s[-1]) and _ascii_word_start(t[0]):
            s += " " + t
        else:
            s += t
    return s


def build_blocks(rows):
    """Split rows into paragraph blocks using vertical gaps and indentation."""
    if not rows:
        return []
    heights = [r["y1"] - r["y0"] for r in rows]
    med_h = sorted(heights)[len(heights) // 2] or 20
    lefts = sorted(r["x0"] for r in rows)
    med_left = lefts[len(lefts) // 2]

    blocks, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        gap = r["y0"] - prev["y1"]
        new_para = gap > 1.25 * med_h
        if not new_para and (r["x0"] - med_left) > 0.8 * med_h and len(cur[-1]["t"]) < 30:
            new_para = True
        if new_para:
            blocks.append(cur)
            cur = [r]
        else:
            cur.append(r)
    blocks.append(cur)
    return [join_lines([r["t"] for r in b]) for b in blocks]
