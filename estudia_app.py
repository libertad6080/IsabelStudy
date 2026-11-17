# -*- coding: utf-8 -*-
"""Study Buddy v2: sube un tema (ZIP de imágenes) y Gemini genera 3 niveles que, juntos, cubren TODO el tema.
Se desbloquea el siguiente nivel aprobando el test. Ejecutar:  streamlit run estudia_app.py
(usa la misma GOOGLE_API_KEY de .streamlit/secrets.toml que IAG.py)"""
import base64, difflib, io, json, math, random, re, shutil, time, unicodedata, zipfile, os
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import streamlit as st
from openai import OpenAI
from PIL import Image

st.set_page_config(page_title="Study Buddy", page_icon="🎓", layout="wide")

BASE = Path(__file__).parent / "temas"
BASE.mkdir(exist_ok=True)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODELO_DEFECTO = "gemini-3.5-flash-lite"  # mismo que IAG.py; cámbialo con MODELO_IA en secrets.toml
PASA = 0.8       # % de aciertos para aprobar un nivel
INTENTO = 15     # preguntas por intento del test (se sacan de un banco más grande, repartidas por apartados)
VERSION = 2

LEVELS = {
    1: dict(nombre="Explorador", est="⭐", rows=5, style="VERY simple English (CEFR A1-A2), sentences of max 10 words, easy distractors.",
            game=dict(rounds=6, grid=2, pairs=4, groups=2)),
    2: dict(nombre="Aventurero", est="⭐⭐", rows=5, style="Simple English (A2), sentences of max 15 words, mix of facts and easy why/how.",
            game=dict(rounds=8, grid=3, pairs=6, groups=3)),
    3: dict(nombre="Experto", est="⭐⭐⭐", rows=6, style="Textbook-level English (A2-B1) with technical terms, cause/effect and comparison, plausible distractors.",
            game=dict(rounds=10, grid=4, pairs=8, groups=4)),
}

# ───────────────────────── IA (Gemini vía endpoint compatible OpenAI, como en IAG.py) ─────────────────────────
def api_key():
    try: k = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception: k = ""
    return k or os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")


def modelo():
    try: return st.secrets.get("MODELO_IA", MODELO_DEFECTO)
    except Exception: return MODELO_DEFECTO


def llm(messages, max_tokens=4000, as_json=False):
    cli = OpenAI(base_url=GEMINI_BASE_URL, api_key=api_key())
    kw = dict(model=modelo(), messages=messages, max_tokens=max_tokens)
    if as_json: kw["response_format"] = {"type": "json_object"}
    reasoning = True
    for intento in range(6):
        try:
            extra = {"reasoning_effort": "minimal"} if reasoning else {}
            return cli.chat.completions.create(**kw, **extra).choices[0].message.content or ""
        except Exception as e:
            s = str(e)
            if "429" in s or "RESOURCE_EXHAUSTED" in s: time.sleep(15 * (intento + 1))
            elif any(c in s for c in ("500", "502", "503", "504", "UNAVAILABLE", "overloaded", "timed out", "Timeout", "Connection")): time.sleep(4 * (intento + 1))  # Gemini saturado: espera y reintenta
            elif reasoning and ("reasoning" in s.lower() or "400" in s): reasoning = False
            else: raise
    raise RuntimeError("Gemini no responde (saturado o límite de peticiones). Espera un minuto y reintenta.")


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    try: return json.loads(t)
    except Exception: return json.loads(re.search(r"\{.*\}", t, re.S).group(0))


def ask_json(prompt, max_tokens, ok):
    err = "respuesta incompleta"
    for _ in range(2):
        try:
            d = parse_json(llm([{"role": "user", "content": prompt}], max_tokens=max_tokens, as_json=True))
            if ok(d): return d
        except Exception as e:
            err = e
    raise RuntimeError(f"Gemini no dio una respuesta válida ({err}). Pulsa de nuevo para reintentar.")


def fill(t, **kw):
    for k, v in kw.items(): t = t.replace(f"@@{k}@@", str(v))
    return t


P_BASE = """You are reading scanned pages of a Spanish bilingual primary-school textbook (5º Primaria, in English).
Ignore the child's handwriting and teacher marks. Pages come labelled "PAGE k:".
Return ONLY JSON: {"source_text": "...", "figures": [{"page": k, "box": [ymin, xmin, ymax, xmax], "label": "...", "caption": "..."}]}
- source_text: COMPLETE English transcription of ALL teaching content, in textbook order, organised in sections that start with a line "## Section title"
  (one section per sub-topic of the book). Keep EVERY definition, list item, example, number, name and fact; skip only exercises' blank lines. Max 2500 words.
- figures: real pictures/photos/cartoons/maps/diagrams that illustrate ONE concept. box = tight box around the picture, coordinates 0-1000 relative to that page image.
  label = 1-3 English words naming what it shows (e.g. "Canyon", "Mitochondria"), unique and distinct. caption = one short English sentence.
  NOT text blocks, NOT handwriting, NOT whole pages. Max 12 figures."""

STUDENT = "Student: Spanish girl, 10 years old, 5º Primaria, bilingual school, BASIC English for her age. Level @@N@@ English style: @@STYLE@@\n"
PART = "\nThis level covers ONLY this PART of the unit (other levels cover the rest). Sections: @@SECTIONS@@\n\nMATERIAL (PART):\n@@TEXT@@"

P_SUMMARY = STUDENT + """Write a COMPLETE study summary of the PART. It must cover EVERY section and EVERY definition, list item, example, number and name in it; skip nothing.
About @@WORDS@@ words in ENGLISH, one "## " heading per section (exact section titles), short paragraphs and bullets, **bold** key terms.
After each section add a paragraph starting with "💡 " written in SPANISH that explains the hardest idea of that section simply, with an analogy for a 10-year-old.
Also give @@VOCAB@@ vocabulary items (English term + Spanish translation) spread across all sections.
Return ONLY JSON: {"summary_md": "...", "vocab": [{"en": "term", "es": "traducción"}]}""" + PART

P_SCHEMES = STUDENT + """Create @@K@@ fill-in-the-blank schemes that TOGETHER cover all the key ideas of the PART: one scheme per section (merge neighbouring sections if there are more than @@K@@); use the section title as the scheme title.
Each scheme has @@ROWS@@ rows. Text in ENGLISH; instructions and hints in SPANISH (max 2 sentences, hints guide without giving the answer).
Return ONLY JSON: {"schemes": [{"title": "...", "instructions_es": "...", "layout": "flow or list", "bank": ["Word", "..."],
 "rows": [{"text": "English sentence with exactly one ___ blank", "answer": "Word from bank", "hint_es": "..."}]}]}
Every row in a scheme has a different answer (1-3 words); add 1-2 distractor words to each bank; layout "flow" for sequences or processes.""" + PART

P_TEST = STUDENT + """Write @@Q@@ multiple-choice questions in ENGLISH that cover ALL sections of the PART evenly (every definition and key fact should be asked somewhere).
Each question: exactly 4 options, one correct, plausible distractors, "section" = exact section title, "explain_es" = explanation in SPANISH (max 2 sentences) of why the right answer is right.
Return ONLY JSON: {"test": [{"section": "...", "q": "...", "options": ["A", "B", "C", "D"], "answer_idx": 0, "explain_es": "..."}]}""" + PART

TREE_ROWS = {2: 9, 3: 15}   # nodos por esquema en árbol (niveles 2 y 3)
TREE_DEPTH = {2: "Use depths 0 to 2 only (title -> main branches -> details).",
              3: "Use depths 0 to 3 and DEVELOP the whole process or topic of the section stage by stage, in order, down to the details: every main stage (d=1) has its own sub-steps (d=2) and the details (d=3) that explain how or why."}
P_SCHEMES_TREE = STUDENT + """Create @@K@@ MULTI-LEVEL fill-in-the-blank schemes (concept trees, like an indented textbook scheme) that TOGETHER cover all the key ideas of the PART: one scheme per section (merge neighbouring sections if there are more than @@K@@); use the section title as the scheme title.
Each scheme is a list of @@ROWS@@ nodes in READING ORDER (a parent always comes before its children). Each node has a depth "d": d=0 is the single title idea of the scheme, d=1 the main branches, d=2 the sub-branches, d=3 the details or steps. @@DEPTH@@
Text in ENGLISH (short phrases or one sentence, max 12 words); instructions and hints in SPANISH (max 2 sentences, hints guide without giving the answer).
About 70% of the nodes contain exactly one ___ blank and give its "answer" (1-3 words, taken from the bank); the other nodes (the d=0 title and some branch headings) have no blank and "answer": "". Every answer is different; add 1-2 distractor words to each bank.
Return ONLY JSON: {"schemes": [{"title": "...", "instructions_es": "...", "layout": "tree", "bank": ["Word", "..."], "rows": [{"d": 0, "text": "Title of the scheme", "answer": "", "hint_es": ""}, {"d": 1, "text": "Main branch with a ___ blank", "answer": "Word", "hint_es": "..."}]}]}""" + PART

MAX_STOPS = 6   # paradas máximas del viaje por nivel (una por apartado; si hay más apartados se agrupan)
JV = 3          # versión de las paradas guardadas (si cambia, se vuelven a dibujar)
KIT_RANGE = {1: "3-5", 2: "4-7", 3: "5-9"}
LEVEL_RULE = {
    1: "LEVEL 1 (youngest): every English text is ONE very short sentence (max 10 words). says_en is playful and tiny. body_en: ONE short sentence. Generic drawing: 4-5 parts.",
    2: "LEVEL 2: English sentences max 15 words. body_en: 1-2 sentences saying what the part does for the whole and why that matters. Generic drawing: 5-6 parts.",
    3: "LEVEL 3 (expert): use the technical terms from the material. body_en: 2 sentences with cause and effect or comparison. Generic drawing: 6-7 parts.",
}
P_JOURNEY = STUDENT + """Create ONE STOP of a virtual guided tour in the spirit of the TV series "Once Upon a Time... Life" (Érase una vez la vida):
friendly little characters, where every piece of the picture has a personality and tells what it does and why it is useful for the WHOLE (the body, the cell, the planet, the machine...).
Cover ONLY the sections of this stop and use only facts from the material. Stops already made: @@PREV@@.
@@RULE@@
STEP 1 - CHOOSE THE PICTURE.
READY-MADE REALISTIC PICTURES (cutaway cells):
@@KITS@@
If this stop is mainly about what a cell is made of or its parts, you MUST use the matching ready-made picture: set "kit" to its name, set "svg" to "" and give "parts" ONLY with ids of that picture, using @@KR@@ parts that the material talks about
(if it names fewer than 3, add the most basic ones). Each part's "name" is the English name listed above (or the material's own word for the same thing). Everything you write for a part must be true and about THAT part only.
Otherwise set "kit":"none" and draw the SVG yourself: viewBox="0 0 800 520", no scripts, no external images or fonts, NO <text> elements anywhere (labels are added by the app), recognisable realistic shapes (not plain circles), flat bright colours with soft gradients.
Draw <g id="bg"> (frame of the whole, always visible), then separate big groups <g id="p1">, <g id="p2">... (>=90px, well separated, no transform attributes). The FIRST child of every group must be <title>exact part name</title>.
STEP 2 - TEACH. Each part: short, true, simple text. "clue_en" describes the part's function WITHOUT saying its name. "story" has 3-5 beats in the right order showing how the parts WORK TOGETHER like a mini-adventure
(e.g. for a cell: a message reaches the membrane -> the nucleus sends orders -> the mitochondria give energy...; for a vaccine: a harmless germ piece arrives -> white cells notice it -> they build antibodies -> memory cells remember it). Each beat uses an id from "parts".
Return ONLY JSON: {"kit": "animal_cell | plant_cell | none", "scene": "English title of the picture", "emoji": "one emoji for the stop", "story_es": "2-3 sentences in Spanish in storyteller voice, starting like 'Érase una vez…'",
 "traveler": "one emoji that travels through the story", "svg": "", "parts": [{"id": "nuc", "name": "English name", "emoji": "one emoji", "says_en": "what the part says about itself, first person, 1 short sentence",
   "text_en": "what it is", "text_es": "explicación en español con una analogía", "body_en": "what it does for the whole and why it is useful", "body_es": "lo mismo en español, simple", "clue_en": "clue without the name",
   "clue_es": "la misma pista, traducida al español sin decir el nombre", "why_es": "1-2 frases en español muy sencillas que explican POR QUÉ esa pista describe a esta pieza, con una analogía"}],
 "story": [{"part": "nuc", "en": "what happens in this beat", "es": "lo mismo en español"}], "outro_es": "one Spanish sentence to close the stop"}""" + PART


# ───────────────────────── almacenamiento ─────────────────────────
def tdir(s): return BASE / s
def load(s): return json.loads((tdir(s) / "tema.json").read_text("utf-8"))
def save(s, m): (tdir(s) / "tema.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), "utf-8")
def topics(): return sorted([p.name for p in BASE.iterdir() if (p / "tema.json").exists()], key=lambda s: load(s)["created"])
def nkey(s): return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def b64(im, q=75):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return base64.b64encode(b.getvalue()).decode()


def import_zip(name, f):
    slug = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()).strip("-") or "tema"
    if tdir(slug).exists(): slug += f"-{int(time.time())}"
    (tdir(slug) / "pages").mkdir(parents=True)
    try: _extract(f, slug, name)
    except Exception:
        shutil.rmtree(tdir(slug), ignore_errors=True); raise
    return slug


def _extract(f, slug, name):
    with zipfile.ZipFile(f) as z:
        names = sorted([n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and "__MACOSX" not in n], key=nkey)
        if not names: raise ValueError("El ZIP no contiene imágenes.")
        for i, n in enumerate(names, 1):
            im = Image.open(io.BytesIO(z.read(n))).convert("RGB"); im.thumbnail((1600, 1600))
            im.save(tdir(slug) / "pages" / f"{i:03d}.jpg", quality=80)
    save(slug, {"name": name, "npages": len(names), "created": time.time(), "base": None, "levels": {}, "progress": {}})


def crop_figs(d, figs):
    (d / "figs").mkdir(exist_ok=True); out, seen = [], set()
    for i, f in enumerate(figs):
        try:
            y0, x0, y1, x1 = [max(0, min(1000, float(v))) for v in f["box"]]
            lab = str(f["label"]).strip()
            if y1 - y0 < 60 or x1 - x0 < 60 or not lab or lab.lower() in seen: continue
            im = Image.open(d / "pages" / f"{int(f['page']):03d}.jpg"); W, H = im.size
            c = im.crop((int(x0 * W / 1000), int(y0 * H / 1000), int(x1 * W / 1000), int(y1 * H / 1000)))
            c.thumbnail((520, 520)); c.save(d / "figs" / f"{i:03d}.jpg", quality=80)
            seen.add(lab.lower()); out.append({"file": f"{i:03d}.jpg", "label": lab, "caption": str(f.get("caption", ""))})
        except Exception:
            continue
    return out


# ───────────────────────── plan: repartir TODO el tema en 3 niveles ─────────────────────────
def split_sections(text):
    parts = re.split(r"(?m)^##\s+(.+?)\s*$", text); secs = []
    for i in range(1, len(parts), 2):
        t, b = parts[i].strip(), parts[i + 1].strip()
        if secs and secs[-1]["title"].lower() == t.lower(): secs[-1]["text"] += "\n" + b
        else: secs.append({"title": t, "text": b})
    pre = parts[0].strip()
    if pre:
        if secs: secs[0]["text"] = pre + "\n" + secs[0]["text"]
        else: secs = [{"title": "Part 1", "text": pre}]
    return secs


def make_plan(text):
    secs = split_sections(text)
    if len(secs) < 3:  # el modelo no puso apartados: troceamos por párrafos
        paras = [p for p in re.split(r"\n\s*\n|\n", text) if p.strip()] or [text]
        k = max(3, min(6, len(paras))); secs = [{"title": f"Part {i + 1}", "text": "\n".join(paras[i * len(paras) // k:(i + 1) * len(paras) // k])} for i in range(k)]
    tot = sum(len(s["text"].split()) for s in secs) or 1; groups, acc = [[], [], []], 0
    for s in secs:
        w = len(s["text"].split()); groups[min(2, int((acc + w / 2) / (tot / 3)))].append(s); acc += w
    if not all(groups): n = len(secs); groups = [secs[i * n // 3:(i + 1) * n // 3] for i in range(3)]
    return {str(i + 1): {"sections": [s["title"] for s in g], "text": "\n\n".join(f"## {s['title']}\n{s['text']}" for s in g)} for i, g in enumerate(groups)}


def build_base(slug):
    m, d = load(slug), tdir(slug)
    pages = sorted((d / "pages").glob("*.jpg")); texts, figs = [], []
    nb = (len(pages) + 9) // 10; bar = st.progress(0.0, text="Gemini está leyendo las páginas…")
    for b in range(nb):
        content = [{"type": "text", "text": P_BASE}]
        for k, p in enumerate(pages[b * 10:(b + 1) * 10], 1):
            im = Image.open(p); im.thumbnail((1280, 1280))
            content += [{"type": "text", "text": f"PAGE {k}:"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(im, 70)}}]
        data = parse_json(llm([{"role": "user", "content": content}], max_tokens=10000, as_json=True))
        texts.append(str(data.get("source_text", "")))
        for f in data.get("figures", []):
            try: f["page"] = int(f["page"]) + b * 10; figs.append(f)
            except Exception: pass
        bar.progress((b + 1) / nb)
    text = "\n\n".join(texts)
    m["base"] = {"v": VERSION, "source_text": text, "figures": crop_figs(d, figs)}
    m["plan"] = make_plan(text); m["base_ts"] = time.time(); m["levels"] = {}; m["progress"] = {}
    save(slug, m)


# ───────────────────────── generación de cada nivel (3 llamadas, se puede reanudar) ─────────────────────────
def clean_scheme(s):
    tree = s.get("layout") == "tree"; rows, used, prev = [], set(), 0
    for r in s.get("rows", []):
        a = str(r.get("answer", "")).strip(); t = str(r.get("text", "")).strip()
        if tree:
            if not t: continue
            if a and a.lower() in used: t = t.replace("___", a); a = ""      # respuesta repetida: el nodo se queda como texto
            if not a: t = t.replace("___", "").strip()
            elif "___" not in t: t += " ___"
            try: d = int(r.get("d", 1))
            except Exception: d = 1
            d = 0 if not rows else max(1, min(3, d, prev + 1)); prev = d
            if a: used.add(a.lower()); p_ = t.split("___"); t = p_[0] + "___" + "".join(p_[1:])
            rows.append({"d": d, "text": t, "answer": a, "hint_es": str(r.get("hint_es", ""))}); continue
        if not a or a.lower() in used: continue
        if "___" not in t: t += " ___"
        used.add(a.lower()); parts = t.split("___")
        rows.append({"text": parts[0] + "___" + "".join(parts[1:]), "answer": a, "hint_es": str(r.get("hint_es", ""))})
    ans = [r["answer"] for r in rows if r["answer"]]
    bank = list(dict.fromkeys([str(w) for w in s.get("bank", [])] + ans)); random.shuffle(bank)
    return {"title": str(s.get("title", "Scheme")), "instructions_es": str(s.get("instructions_es", "Arrastra cada palabra a su hueco.")),
            "layout": "tree" if tree else ("flow" if s.get("layout") == "flow" else "list"), "bank": bank, "rows": rows}


def clean_tests(items, titles):
    out = []
    for q in items:
        try:
            o = [str(x) for x in q["options"]][:4]; a = o[int(q["answer_idx"])]
            if len(o) != 4 or len(set(o)) != 4: continue
            random.shuffle(o); sec = str(q.get("section", ""))
            if sec not in titles: sec = (difflib.get_close_matches(sec, titles, 1, 0.3) or [titles[0]])[0]
            out.append({"section": sec, "q": str(q["q"]), "options": o, "answer_idx": o.index(a), "explain_es": str(q.get("explain_es", ""))})
        except Exception:
            continue
    return out


def build_level(slug, n):
    m = load(slug); L, part = LEVELS[n], m["plan"][str(n)]
    titles, text = part["sections"], part["text"]; words = len(text.split())
    pack = m["levels"].setdefault(str(n), {}); bar = st.progress(0.0, text="Creando el resumen…")
    common = dict(N=n, STYLE=L["style"], SECTIONS=" | ".join(titles), TEXT=text)
    if "summary_md" not in pack:
        d = ask_json(fill(P_SUMMARY, WORDS=max(450, min(1100, int(words * 0.7))), VOCAB=max(10, min(24, 3 * len(titles))), **common), 9000,
                     lambda d: len(str(d.get("summary_md", ""))) > 200)
        pack["summary_md"] = str(d["summary_md"]); pack["vocab"] = [v for v in d.get("vocab", []) if isinstance(v, dict) and v.get("en")]
        save(slug, m)
    bar.progress(0.33, text="Creando los esquemas…")
    if "schemes" not in pack:
        tree = n >= 2   # niveles 2 y 3: esquemas en árbol (por niveles); el 3 desarrolla todo el proceso hasta el detalle
        d = ask_json(fill(P_SCHEMES_TREE if tree else P_SCHEMES, K=max(1, min(6, len(titles))), ROWS=TREE_ROWS[n] if tree else L["rows"], DEPTH=TREE_DEPTH.get(n, ""), **common),
                     14000 if tree else 9000, lambda d: d.get("schemes"))
        sch = [s for s in (clean_scheme(x) for x in d["schemes"]) if sum(1 for r in s["rows"] if r["answer"]) >= 2]
        if not sch: raise RuntimeError("Esquemas vacíos, reintenta.")
        pack["schemes"] = sch; save(slug, m)
    bar.progress(0.66, text="Creando las preguntas del test…")
    if "test" not in pack:
        d = ask_json(fill(P_TEST, Q=max(18, min(36, round(words / 20))), **common), 12000, lambda d: len(clean_tests(d.get("test", []), titles or ["-"])) >= 8)
        pack["test"] = clean_tests(d["test"], titles); save(slug, m)
    bar.progress(0.8, text="Dibujando el viaje…")
    try: build_journey(slug, n, bar)
    except Exception: pass  # si falla no bloquea el nivel; se reintenta desde la pestaña Explora
    bar.progress(1.0)


_SVG_NS = "http://www.w3.org/2000/svg"
_CASE_TAGS = {n.lower(): n for n in ("linearGradient", "radialGradient", "clipPath", "textPath", "feGaussianBlur", "feOffset", "feMerge", "feMergeNode",
                                     "feBlend", "feColorMatrix", "feComposite", "feFlood", "feDropShadow", "foreignObject")}
_CASE_ATTRS = {n.lower(): n for n in ("viewBox", "gradientUnits", "gradientTransform", "preserveAspectRatio", "stdDeviation", "patternUnits", "patternTransform",
                                      "spreadMethod", "clipPathUnits", "pathLength", "markerWidth", "markerHeight", "refX", "refY", "startOffset", "textLength",
                                      "baseFrequency", "numOctaves", "stitchTiles", "filterUnits", "primitiveUnits", "maskUnits", "maskContentUnits")}


class _LenientSVG(HTMLParser):
    """Reconstruye un SVG aunque el modelo se haya dejado etiquetas sin cerrar o mal cruzadas (cierra solo lo que falta)."""
    def __init__(self):
        super().__init__(convert_charrefs=True); self.root = None; self.stack = []

    def _mk(self, tag, attrs):
        el = ET.Element(_CASE_TAGS.get(tag, tag), {_CASE_ATTRS.get(k, k): (v or "") for k, v in attrs if k and re.fullmatch(r"[A-Za-z_][\w.\-]*(:[\w.\-]+)?", k) and k != "xmlns"})
        if self.stack: self.stack[-1].append(el)
        elif self.root is None: self.root = el
        else: return None
        return el

    def handle_starttag(self, tag, attrs):
        el = self._mk(tag, attrs)
        if el is not None: self.stack.append(el)

    def handle_startendtag(self, tag, attrs): self._mk(tag, attrs)

    def handle_endtag(self, tag):
        tag = _CASE_TAGS.get(tag, tag)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i].tag == tag: del self.stack[i:]; return

    def handle_data(self, data):
        if self.stack and data.strip():
            el = self.stack[-1]
            if len(el): el[-1].tail = (el[-1].tail or "") + data
            else: el.text = (el.text or "") + data


def parse_svg(svg):
    """Lee el SVG del modelo: primero estricto; si el XML está roto, lo repara de forma tolerante."""
    svg = re.sub(r"^```(?:svg|xml)?|```$", "", svg.strip(), flags=re.M).strip()
    a, b = svg.find("<svg"), svg.rfind("</svg>")
    if a < 0: raise ValueError("no hay <svg>")
    svg = svg[a:b + 6] if b > a else svg[a:]
    if "xmlns=" not in svg[:300]: svg = svg.replace("<svg", f'<svg xmlns="{_SVG_NS}"', 1)
    try: return ET.fromstring(svg)
    except ET.ParseError: pass
    svg = re.sub(r"&(?!#?\w+;)", "&amp;", svg)  # & sueltos
    try: return ET.fromstring(svg)
    except ET.ParseError: pass
    p = _LenientSVG(); p.feed(svg); p.close()
    if p.root is None: raise ValueError("SVG vacío")
    p.root.set("xmlns", _SVG_NS)
    return ET.fromstring(ET.tostring(p.root, encoding="unicode"))


def fallback_svg(parts):
    """Dibujo de reserva: fichas con relieve (3D) y el emoji de cada pieza, bien separadas y fáciles de pulsar."""
    from xml.sax.saxutils import escape
    n = len(parts); cols = 2 if n <= 4 else 3; rows = math.ceil(n / cols); ch = 440 / rows
    pal = [("#bfe3ff", "#6cb4f2", "#2f7bc4"), ("#ffd6d6", "#ff8c8c", "#d14a4a"), ("#d3f5d0", "#7fd67a", "#3d9c3a"), ("#fff1b8", "#ffd24d", "#d49b00"), ("#e4d9ff", "#a98cf2", "#6a4cc2"),
           ("#ffd9ee", "#f58ac2", "#c4488f"), ("#ffe0c2", "#ffa45c", "#d4691a"), ("#c9f5ee", "#5fd8c4", "#1f9c88"), ("#d8dcff", "#8e97f0", "#4a52c2")]
    defs = "".join(f'<linearGradient id="gT{k}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/></linearGradient>' for k, (a, b, c) in enumerate(pal))
    out = [f'<svg xmlns="{_SVG_NS}" viewBox="0 0 800 520"><defs>{defs}</defs><g id="bg"><rect x="10" y="10" width="780" height="500" rx="40" fill="#eef2f7" stroke="#cbd5e1" stroke-width="5"/></g>']
    for i, p in enumerate(parts):
        row, col = divmod(i, cols); n_row = min(cols, n - row * cols)
        cx = 40 + (720 / n_row) * (col + .5); cy = 40 + ch * (row + .5) - 8
        w = min(720 / n_row * .78, 210); h = min(ch * .70, 150); d = 16; k = i % len(pal); c = pal[k][2]
        emo = escape(p.get("emoji") or "✨")
        out.append(f'<g id="p{i + 1}"><ellipse cx="{cx:.0f}" cy="{cy + h / 2 + d + 8:.0f}" rx="{w * .5:.0f}" ry="10" fill="#000" fill-opacity=".18"/>'
                   f'<rect x="{cx - w / 2:.0f}" y="{cy - h / 2 + d:.0f}" width="{w:.0f}" height="{h:.0f}" rx="26" fill="{c}"/>'
                   f'<rect x="{cx - w / 2:.0f}" y="{cy - h / 2:.0f}" width="{w:.0f}" height="{h:.0f}" rx="26" fill="url(#gT{k})" stroke="#fff" stroke-opacity=".7" stroke-width="3"/>'
                   f'<text x="{cx:.0f}" y="{cy:.0f}" text-anchor="middle" dominant-baseline="central" font-size="{h * .55:.0f}">{emo}</text></g>')
    return "".join(out) + "</svg>"


def clean_svg(svg, ids):
    root = parse_svg(svg); tag = lambda e: e.tag.split("}")[-1].lower()
    if tag(root) != "svg": raise ValueError("no es svg")
    bad = {"script", "foreignobject", "image", "iframe", "use", "a", "style", "text", "tspan"}
    for c, p in [(c, p) for p in root.iter() for c in p]:
        if tag(c) in bad: p.remove(c)
    for el in root.iter():
        for a in list(el.attrib):
            if a.lower().startswith("on") or "href" in a.lower(): del el.attrib[a]
    if "viewBox" not in root.attrib: root.set("viewBox", "0 0 800 520")
    found = [i for i in ids if root.find(f".//*[@id='{i}']") is not None]
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    return ET.tostring(root, encoding="unicode"), found


# ───────────────────────── dibujos realistas listos (células en corte, vista 2D en perspectiva caballera) ─────────────────────────
import random as _rnd

def _arc(cx, cy, rx, ry, a0, a1):
    p = lambda a: (cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
    x0, y0 = p(a0); x1, y1 = p(a1)
    return f"M{x0:.1f},{y0:.1f} A{rx},{ry} 0 {1 if abs(a1 - a0) > 180 else 0} 1 {x1:.1f},{y1:.1f}"

def _at(inner, x, y, s=1.0, r=0):
    return f'<g transform="translate({x} {y}) rotate({r}) scale({s})">{inner}</g>'

def _hit(w, h, x=0, y=0):  # zona invisible que se puede pulsar
    return f'<rect x="{x - w / 2}" y="{y - h / 2}" width="{w}" height="{h}" rx="14" fill="none" pointer-events="all"/>'

DEFS = """<defs>
<linearGradient id="gShell" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#46d3bf"/><stop offset="1" stop-color="#157a76"/></linearGradient>
<radialGradient id="gCut" cx=".42" cy=".38" r=".75"><stop offset="0" stop-color="#6f56a8"/><stop offset="1" stop-color="#3a2b63"/></radialGradient>
<radialGradient id="gNuc" cx=".35" cy=".3" r=".8"><stop offset="0" stop-color="#fff6a8"/><stop offset=".6" stop-color="#f5cd4a"/><stop offset="1" stop-color="#d9a21f"/></radialGradient>
<radialGradient id="gNcl" cx=".35" cy=".3" r=".8"><stop offset="0" stop-color="#c9f78d"/><stop offset="1" stop-color="#37b24a"/></radialGradient>
<linearGradient id="gMito" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff9a9a"/><stop offset="1" stop-color="#c64a5c"/></linearGradient>
<linearGradient id="gGol" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffc88a"/><stop offset="1" stop-color="#ee8a3e"/></linearGradient>
<radialGradient id="gVes" cx=".35" cy=".3" r=".8"><stop offset="0" stop-color="#d9f3fb"/><stop offset="1" stop-color="#63b9dc"/></radialGradient>
<linearGradient id="gCen" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff8a8a"/><stop offset="1" stop-color="#bf3b3b"/></linearGradient>
<linearGradient id="gWall" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#a6e08f"/><stop offset="1" stop-color="#5fb36a"/></linearGradient>
<radialGradient id="gPlantCut" cx=".4" cy=".35" r=".8"><stop offset="0" stop-color="#f3fbe6"/><stop offset="1" stop-color="#cfe9c4"/></radialGradient>
<linearGradient id="gChl" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7ee07a"/><stop offset="1" stop-color="#2d9a43"/></linearGradient>
<radialGradient id="gCvac" cx=".4" cy=".3" r=".9"><stop offset="0" stop-color="#e6f7fd"/><stop offset="1" stop-color="#86cde8"/></radialGradient>
</defs>"""

# ---------- orgánulos dibujados en el origen (0,0) ----------
def s_nucleus():
    pores = "".join(f'<circle cx="{112 * math.cos(math.radians(a)):.1f}" cy="{90 * math.sin(math.radians(a)):.1f}" r="3.2" fill="#b8820f"/>' for a in range(0, 360, 28))
    chrom = "".join(f'<path d="M{x},{y} q7,-10 14,0 t14,0 t14,0" fill="none" stroke="#e0b030" stroke-width="2.6" stroke-linecap="round"/>' for x, y in [(-68, -22), (-60, 34), (20, -42), (24, 40)])
    return (f'<ellipse rx="112" ry="90" fill="url(#gNuc)" stroke="#c28d12" stroke-width="5"/>{pores}'
            f'<ellipse cy="8" rx="88" ry="68" fill="#fff6c2" stroke="#e3b53a" stroke-width="3"/>{chrom}'
            f'<circle cx="6" cy="14" r="36" fill="url(#gNcl)" stroke="#2f9a3a" stroke-width="3"/>'
            f'<ellipse cx="-4" cy="2" rx="13" ry="7" fill="#fff" opacity=".6" transform="rotate(-30 -4 2)"/>')

def s_er():
    out = ""
    for k, (rx, ry) in enumerate([(130, 106), (146, 122), (162, 138)]):
        out += f'<path d="{_arc(0, 0, rx, ry, -35 + k * 4, 118 - k * 3)}" fill="none" stroke="#2c9fc0" stroke-width="13" stroke-linecap="round"/>'
        out += f'<path d="{_arc(0, 0, rx, ry, -35 + k * 4, 118 - k * 3)}" fill="none" stroke="#8fe3f4" stroke-width="6" stroke-linecap="round"/>'
    for a in range(-20, 112, 17):
        x, y = 166 * math.cos(math.radians(a)), 142 * math.sin(math.radians(a))
        out += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#f6c487" stroke="#a9702f" stroke-width="1.3"/>'
    return out

def s_golgi():
    out = ""
    for i in range(5):
        y = i * 15; w = 160 - abs(i - 2) * 8
        d = f"M{-w / 2},{y} q20,-18 40,0 t40,0 t40,0 t{w - 120},0"
        out += f'<path d="{d}" fill="none" stroke="#bd561b" stroke-width="16" stroke-linecap="round"/><path d="{d}" fill="none" stroke="url(#gGol)" stroke-width="10" stroke-linecap="round"/>'
    out += '<circle cx="100" cy="6" r="8" fill="#ffb877" stroke="#bd561b" stroke-width="2.5"/><circle cx="104" cy="52" r="6" fill="#ffb877" stroke="#bd561b" stroke-width="2.5"/><circle cx="-100" cy="36" r="7" fill="#ffb877" stroke="#bd561b" stroke-width="2.5"/>'
    return out

def _bean():
    cr = "".join(f'<path d="M{x},-19 q8,9.5 0,19 q-8,9.5 0,19" fill="none" stroke="#a83347" stroke-width="3.4" stroke-linecap="round"/>' for x in range(-42, 46, 17))
    return ('<ellipse rx="64" ry="31" fill="url(#gMito)" stroke="#9e2f42" stroke-width="4"/><ellipse rx="55" ry="23" fill="#ffc2c2" opacity=".5"/>' + cr +
            '<ellipse cx="-22" cy="-17" rx="24" ry="5" fill="#fff" opacity=".4"/>')

def s_mito():
    return _at(_bean(), 0, 0, 1, -18) + _at(_bean(), 62, -66, .85, 38)

def s_lyso():
    def one(r):
        dots = "".join(f'<circle cx="{r * .45 * math.cos(a):.1f}" cy="{r * .45 * math.sin(a):.1f}" r="{max(2, r / 8):.1f}" fill="#7cc23a"/>' for a in (0.3, 2.3, 4.3))
        return f'<circle r="{r}" fill="#f4faa8" stroke="#86cc45" stroke-width="{r / 3.5:.1f}"/>{dots}<ellipse cx="{-r * .35:.1f}" cy="{-r * .4:.1f}" rx="{r * .3:.1f}" ry="{r * .15:.1f}" fill="#fff" opacity=".6"/>'
    return one(24) + _at(one(17), 48, 38)

BLOB = "M-40,-5 C-42,-30 -10,-40 14,-31 C42,-24 50,2 36,22 C22,42 -18,44 -35,27 C-45,17 -41,5 -40,-5 Z"
def s_vac():
    b = f'<path d="{BLOB}" fill="url(#gVes)" stroke="#4aa6c8" stroke-width="4"/><ellipse cx="-14" cy="-16" rx="14" ry="6" fill="#fff" opacity=".7" transform="rotate(-25 -14 -16)"/>'
    return b + _at(b, 62, 58, .5)

def s_cent():
    cyl = ('<rect x="-36" y="-13" width="72" height="26" rx="9" fill="url(#gCen)" stroke="#8c2323" stroke-width="3"/>' +
           "".join(f'<line x1="{x}" y1="-12" x2="{x}" y2="12" stroke="#8c2323" stroke-width="2.4"/>' for x in (-22, -8, 6, 22)))
    return _at(cyl, 0, 30) + _at(cyl, -6, -24, 1, 90)

def s_ribo():
    _rnd.seed(4); pts = [(_rnd.uniform(-52, 52), _rnd.uniform(-24, 24)) for _ in range(13)]
    return _hit(120, 64) + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#f6c487" stroke="#a9702f" stroke-width="1.6"/>' for x, y in pts)

def s_chlo():
    def one():
        grana = "".join(f'<rect x="{-26 + i * 18}" y="-9" width="14" height="5" rx="2" fill="#17692b"/><rect x="{-26 + i * 18}" y="-2" width="14" height="5" rx="2" fill="#17692b"/><rect x="{-26 + i * 18}" y="5" width="14" height="5" rx="2" fill="#17692b"/>' for i in range(3))
        return f'<ellipse rx="56" ry="31" fill="url(#gChl)" stroke="#1d7a33" stroke-width="4"/>{grana}<ellipse cx="-20" cy="-19" rx="20" ry="4.5" fill="#fff" opacity=".45"/>'
    return _at(one(), 0, 0, 1, -12) + _at(one(), 56, 52, .9, 14)

# ---------- células completas ----------
def kit_animal():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 520">' + DEFS +
           '<g id="bg"><ellipse cx="400" cy="506" rx="290" ry="12" fill="#000" opacity=".12"/></g>'
           '<g id="memb"><ellipse cx="400" cy="283" rx="338" ry="214" fill="url(#gShell)" stroke="#0f6663" stroke-width="4"/>'
           '<path d="M120,380 C95,300 130,190 215,125" fill="none" stroke="#fff" stroke-opacity=".35" stroke-width="12" stroke-linecap="round"/>'
           '<ellipse cx="400" cy="262" rx="322" ry="196" fill="none" stroke="#7df0dd" stroke-width="12"/></g>'
           '<g id="cyto"><ellipse cx="400" cy="262" rx="312" ry="186" fill="url(#gCut)"/></g>'
           f'<g id="er">{_at(s_er(), 410, 222, .92)}</g>'
           f'<g id="nuc">{_at(s_nucleus(), 410, 222, .92)}</g>'
           f'<g id="golgi">{_at(s_golgi(), 322, 352, .95)}</g>'
           f'<g id="mito">{_at(s_mito(), 150, 215)}</g>'
           f'<g id="lyso">{_at(s_lyso(), 590, 138)}</g>'
           f'<g id="vac">{_at(s_vac(), 165, 325)}</g>'
           f'<g id="cent">{_at(s_cent(), 640, 275, .95)}</g>'
           f'<g id="ribo">{_at(s_ribo(), 520, 395)}</g></svg>')
    return dict(svg=svg, fixed=["memb", "cyto"], lbl={"memb": [400, 476], "cyto": [255, 98]},
                parts={"memb": "Cell membrane", "cyto": "Cytoplasm", "nuc": "Nucleus", "er": "Endoplasmic reticulum", "golgi": "Golgi apparatus",
                       "mito": "Mitochondria", "ribo": "Ribosomes", "lyso": "Lysosomes", "vac": "Vacuole", "cent": "Centrioles"})

def _block(x, y, w, h, rx, depth, fill):
    return "".join(f'<rect x="{x + k}" y="{y + k}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>' for k in range(depth, 0, -2))

def kit_plant():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 520">' + DEFS +
           '<g id="bg"><ellipse cx="410" cy="500" rx="300" ry="12" fill="#000" opacity=".12"/></g>'
           f'<g id="wall">{_block(60, 40, 620, 390, 44, 30, "#2f6f46")}<rect x="60" y="40" width="620" height="390" rx="44" fill="url(#gWall)" stroke="#2c7a46" stroke-width="5"/></g>'
           '<g id="memb"><rect x="80" y="60" width="580" height="350" rx="32" fill="#f6c453" stroke="#c9902a" stroke-width="3"/></g>'
           '<g id="cyto"><rect x="96" y="76" width="548" height="318" rx="24" fill="url(#gPlantCut)"/></g>'
           f'<g id="cvac">{_at(f"""<ellipse rx="150" ry="84" fill="url(#gCvac)" stroke="#4ea9cf" stroke-width="5"/><ellipse cx="-50" cy="-44" rx="48" ry="12" fill="#fff" opacity=".6" transform="rotate(-12 -50 -44)"/>""", 450, 292)}</g>'
           f'<g id="er">{_at(s_er(), 235, 185, .66)}</g>'
           f'<g id="nuc">{_at(s_nucleus(), 235, 185, .66)}</g>'
           f'<g id="golgi">{_at(s_golgi(), 205, 340, .7)}</g>'
           f'<g id="chlo">{_at(s_chlo(), 475, 130, .95)}</g>'
           f'<g id="mito">{_at(s_mito(), 122, 298, .62)}</g>'
           f'<g id="ribo">{_at(s_ribo(), 365, 120, .85)}</g></svg>')
    return dict(svg=svg, fixed=["wall", "memb", "cyto"], lbl={"wall": [370, 420], "memb": [370, 68], "cyto": [570, 385]},
                parts={"wall": "Cell wall", "memb": "Cell membrane", "cyto": "Cytoplasm", "nuc": "Nucleus", "cvac": "Central vacuole", "chlo": "Chloroplasts",
                       "mito": "Mitochondria", "golgi": "Golgi apparatus", "er": "Endoplasmic reticulum", "ribo": "Ribosomes"})


KITS = {"animal_cell": kit_animal, "plant_cell": kit_plant}


def kit_svg(K, keep):
    """Devuelve el dibujo del kit quitando los orgánulos que no se enseñan en esta parada (el marco —membrana, citoplasma, pared— se queda de decorado)."""
    ET.register_namespace("", "http://www.w3.org/2000/svg"); root = ET.fromstring(K["svg"])
    for g in list(root):
        gid = g.attrib.get("id")
        if gid and gid != "bg" and gid not in keep and gid not in K["fixed"]: root.remove(g)
    return ET.tostring(root, encoding="unicode")


def journey_groups(part):
    """Reparte los apartados del nivel en paradas (máx. MAX_STOPS): así el viaje recorre TODOS los puntos del nivel."""
    secs = split_sections(part["text"]) or [{"title": "Part 1", "text": part["text"]}]
    k = min(MAX_STOPS, len(secs)); out = []
    for i in range(k):
        g = secs[i * len(secs) // k:(i + 1) * len(secs) // k]
        title = " & ".join(s["title"] for s in g[:2]) + ("…" if len(g) > 2 else "")
        out.append({"title": title, "sections": [s["title"] for s in g], "text": "\n\n".join(f"## {s['title']}\n{s['text']}" for s in g)})
    return out


def rekey_by_titles(svg, parts):
    """Las formas del dibujo se emparejan con sus datos por el <title> de cada grupo (así nombre, texto y forma siempre coinciden). Quita los <title> al terminar."""
    root = ET.fromstring(svg); tag = lambda e: e.tag.split("}")[-1].lower(); titles = {}
    for g in root.iter():
        gid = g.attrib.get("id", "")
        if re.fullmatch(r"p\d+", gid):
            t = next((c.text for c in g if tag(c) == "title" and c.text), None)
            if t: titles[gid] = t.strip().lower()
    for p_ in root.iter():
        for c in [c for c in p_ if tag(c) == "title"]: p_.remove(c)
    out = svg if not titles else ET.tostring(root, encoding="unicode")
    if len(titles) < len(parts) / 2: return out, parts  # sin títulos: nos fiamos de los ids
    res, used = [], set()
    for p in parts:
        sc = [(difflib.SequenceMatcher(None, t, p["name"].lower()).ratio(), gid) for gid, t in titles.items() if gid not in used]
        if sc and max(sc)[0] >= .6: gid = max(sc)[1]; used.add(gid); res.append({**p, "id": gid})
    return out, res


def svg_pobre(svg, parts):
    """True si las formas del dibujo del modelo son demasiado simples (p. ej. círculos lisos): mejor usar las fichas."""
    try:
        root = ET.fromstring(svg); cnt = []
        for p in parts:
            el = root.find(f".//*[@id='{p['id']}']"); cnt.append(sum(1 for _ in el.iter()) if el is not None else 0)
        return not cnt or sum(cnt) / len(cnt) < 4
    except Exception: return True


class DibujoPobre(Exception): pass


def build_stop(slug, n, idx):
    m = load(slug); pack = m["levels"][str(n)]; g = journey_groups(m["plan"][str(n)])[idx]
    prev = ", ".join(s.get("scene", "") for s in pack.get("journey", {}).values() if s.get("jv") == JV) or "none"
    kits = "\n".join(f'- "{k}": parts ' + ", ".join(f'{i}={nm}' for i, nm in f()["parts"].items()) for k, f in KITS.items())
    err = "respuesta incompleta"; extra = ""; TRIES = 3
    for attempt in range(TRIES):
        try:
            raw_txt = llm([{"role": "user", "content": fill(P_JOURNEY, N=n, STYLE=LEVELS[n]["style"], RULE=LEVEL_RULE[n], PREV=prev, KITS=kits, KR=KIT_RANGE[n],
                                                                  SECTIONS=" | ".join(g["sections"]), TEXT=g["text"]) + extra}], max_tokens=9000, as_json=True)
            d = parse_json(raw_txt)
            if isinstance(d, list): d = {"parts": d}
            pr = d.get("parts") or d.get("pieces") or d.get("piezas") or d.get("items") or []
            if isinstance(pr, dict): pr = [{"id": k, **v} for k, v in pr.items() if isinstance(v, dict)]  # partes como {id: {...}}
            d["parts"] = pr
            keys = ("id", "name", "emoji", "says_en", "text_en", "text_es", "body_en", "body_es", "clue_en", "clue_es", "why_es")
            allp = [{k: str(p.get(k, "")) for k in keys} for p in d.get("parts", []) if isinstance(p, dict)]
            if len(allp) < 3:
                try: (BASE / "_ultima_respuesta_parada.txt").write_text(str(raw_txt), encoding="utf-8")  # para diagnosticar
                except Exception: pass
                raise ValueError(f"el modelo dio {len(allp)} partes (claves: {', '.join(list(d)[:8])}; {len(str(raw_txt))} caracteres) - devuelve entre 4 y 5 partes")
            maxp = int(KIT_RANGE[n].split("-")[1]); remap = None; fixed, lbl = [], {}
            kit = KITS.get(str(d.get("kit", "none")))
            try:
                if kit:
                    K = kit(); seen = set(); keep = []
                    for p in allp:
                        if p["id"] in K["parts"] and p["id"] not in seen: seen.add(p["id"]); keep.append(p)
                        p["name"] = p["name"] or K["parts"].get(p["id"], "")
                    parts = keep[:maxp]
                    if len(parts) < 3: raise ValueError("faltan partes: usa SOLO los ids de la lista del dibujo listo (kit)")
                    ids = {p["id"] for p in parts}; svg = kit_svg(K, ids); fixed = [i for i in K["fixed"] if i in ids]; lbl = {i: v for i, v in K["lbl"].items() if i in ids}
                else:
                    svg, found = clean_svg(d.get("svg", ""), [p["id"] for p in allp]); parts = [p for p in allp if p["id"] in found]
                    svg, parts = rekey_by_titles(svg, parts)
                    if len(parts) < 3:  # los ids/títulos no cuadran: emparejamos por orden los grupos del dibujo con las partes
                        root_ = ET.fromstring(svg)
                        gs = [c.attrib["id"] for c in root_ if c.tag.split("}")[-1].lower() == "g" and re.fullmatch(r"\w+", c.attrib.get("id", "")) and c.attrib["id"] != "bg"]
                        if len(gs) < 3: raise ValueError("faltan formas en el dibujo")
                        k_ = min(len(gs), len(allp), maxp); remap = {allp[i]["id"]: gs[i] for i in range(k_)}
                        parts = [{**allp[i], "id": gs[i]} for i in range(k_)]
                    if svg_pobre(svg, parts): raise DibujoPobre("dibujo demasiado simple")
            except Exception as e_dib:
                if attempt < TRIES - 1 and not isinstance(e_dib, DibujoPobre): raise
                parts = allp[:maxp]; remap = {p["id"]: f"p{i + 1}" for i, p in enumerate(parts)}  # último intento: dibujo de reserva
                for i, p in enumerate(parts): p["id"] = f"p{i + 1}"
                svg = fallback_svg(parts); fixed, lbl = [], {}
            if remap: d["story"] = [{**b_, "part": remap[str(b_.get("part"))]} for b_ in d.get("story", []) if isinstance(b_, dict) and str(b_.get("part")) in remap]
            for p in parts:
                p["emoji"] = p["emoji"] or "✨"; p["says_en"] = p["says_en"] or p["text_en"]; p["clue_en"] = p["clue_en"] or p["text_en"]
                p["body_en"] = p["body_en"] or p["text_en"]; p["body_es"] = p["body_es"] or p["text_es"]
                p["clue_es"] = p["clue_es"] or p["body_es"]; p["why_es"] = p["why_es"] or p["text_es"]
            ids = {p["id"] for p in parts}
            story = [{"part": str(b.get("part")), "en": str(b.get("en", "")), "es": str(b.get("es", ""))} for b in d.get("story", []) if isinstance(b, dict) and str(b.get("part")) in ids]
            if len(story) < 2: story = [{"part": p["id"], "en": p["text_en"], "es": p["text_es"]} for p in parts]
            if len(parts) >= 3:
                pack.setdefault("journey", {})[str(idx)] = {"jv": JV, "scene": str(d.get("scene", "Explore")), "emoji": str(d.get("emoji", "🔬")) or "🔬", "story_es": str(d.get("story_es", "")),
                                                            "traveler": str(d.get("traveler", "✨")) or "✨", "svg": svg, "parts": parts, "fixed": fixed, "lbl": lbl, "story": story, "outro_es": str(d.get("outro_es", ""))}
                save(slug, m); return
        except Exception as e:
            err = e
            extra = (f"\n\nIMPORTANT: your previous answer failed ({e}). Draw a SIMPLER svg: fewer shapes, every tag closed (self-close <path .../>, <circle .../>), "
                     "no &, no comments, <title> as first child of each group. Return the JSON again.")
    raise RuntimeError(f"No pude dibujar la parada ({err}). Inténtalo otra vez.")


def journey_ready(pack): return {k: v for k, v in pack.get("journey", {}).items() if v.get("jv") == JV}


def build_journey(slug, n, bar=None):
    """Dibuja todas las paradas que faltan del viaje del nivel (una llamada a Gemini por parada)."""
    groups = journey_groups(load(slug)["plan"][str(n)])
    for i in range(len(groups)):
        if str(i) in journey_ready(load(slug)["levels"][str(n)]): continue
        if bar: bar.progress(i / len(groups), text=f"Dibujando el viaje: parada {i + 1} de {len(groups)}…")
        build_stop(slug, n, i)
    if bar: bar.progress(1.0)


# ───────────────────────── viaje con TU dibujo (lo subes en SVG; la IA lo mira, reconoce las piezas y monta la parada en inglés) ─────────────────────────
import copy
PHOTO_MAX = {1: 5, 2: 8, 3: 12}   # piezas máximas por nivel
_XLINK = "http://www.w3.org/1999/xlink"
_INK = {"#000", "#000000", "black", "#fff", "#ffffff", "white", "none", "currentcolor", "rgb(0,0,0)", "rgb(255,255,255)"}
_BLACK = {"#000", "#000000", "black", "rgb(0,0,0)"}; _WHITE = {"#fff", "#ffffff", "white", "rgb(255,255,255)"}
_SHAPES = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon"}

P_PHOTO = STUDENT + """You are given ONE educational picture (e.g. a cell cut open). It may contain numbered callouts (circles with numbers, arrows) or labels in Spanish or any language: use them only to recognise the parts.
Build ONE STOP of a virtual guided tour from it, in the spirit of the TV series "Once Upon a Time... Life": every part has a personality and tells what it does and why it is useful for the WHOLE.
Use the MATERIAL for facts when it talks about the part; otherwise use simple, correct school-level knowledge. Stops already made: @@PREV@@.
@@RULE@@
STEP 1 - READ THE PICTURE. Choose the @@MAXP@@ most important parts that are visible and identifiable (all of them if there are fewer). For EACH part give:
- "name": its name in ENGLISH.
- "box": [ymin, xmin, ymax, xmax] tight box around the STRUCTURE itself (where a callout arrow points), NOT around numbers, arrows or text. Coordinates 0-1000 relative to the whole picture.
- "point": [y, x] one point INSIDE the structure (0-1000), on the structure's own colour, not on a gap or a line.
STEP 2 - TEACH. Each part: short, true, simple text. "clue_en" describes the part's function WITHOUT saying its name. "story" has 3-5 beats in the right order showing how the parts WORK TOGETHER like a mini-adventure. Each beat uses an id from "parts".
Return ONLY JSON: {"scene": "English title of the picture", "emoji": "one emoji", "story_es": "2-3 sentences in Spanish in storyteller voice, starting like 'Érase una vez…'", "traveler": "one emoji that travels through the story",
 "parts": [{"id": "p1", "name": "English name", "box": [0, 0, 0, 0], "point": [0, 0], "emoji": "one emoji", "says_en": "what the part says about itself, first person, 1 short sentence",
   "text_en": "what it is", "text_es": "explicación en español con una analogía", "body_en": "what it does for the whole and why it is useful", "body_es": "lo mismo en español, simple", "clue_en": "clue without the name",
   "clue_es": "la misma pista, traducida al español sin decir el nombre", "why_es": "1-2 frases en español muy sencillas que explican POR QUÉ esa pista describe a esta pieza, con una analogía"}],
 "story": [{"part": "p1", "en": "what happens in this beat", "es": "lo mismo en español"}], "outro_es": "one Spanish sentence to close the stop"}

MATERIAL (context):
@@TEXT@@"""


def _tg(e): return e.tag.split("}")[-1].lower() if isinstance(e.tag, str) else ""


def _paint(el):
    """(relleno, trazo) de un elemento mirando style y atributos. Sin 'fill' = negro (valor por defecto del SVG)."""
    st_ = {k.strip(): v.strip().lower() for k, v in (x.split(":", 1) for x in (el.attrib.get("style") or "").split(";") if ":" in x)}
    return (st_.get("fill", el.attrib.get("fill", "#000")).lower().replace(" ", ""), st_.get("stroke", el.attrib.get("stroke", "none")).lower().replace(" ", ""))


def _callout_shape(e):
    """Forma pequeña típica de una llamada: círculo blanco con borde negro, número o flecha negros sin borde, o línea negra."""
    if _tg(e) not in _SHAPES or len(e.attrib.get("d", "")) > 3000: return False
    f, s = _paint(e)
    return (f in _WHITE and s in _BLACK) or (f in _BLACK and s == "none") or (f == "none" and s in _BLACK)


def strip_annotations(root):
    """Quita las llamadas del dibujo (textos, círculos con número, flechas, líneas negras). Es exacto: borra los elementos del SVG, no los tapa."""
    n = 0
    def walk(p):
        nonlocal n
        for c in list(p):
            t = _tg(c)
            if t in ("text", "tspan") or _callout_shape(c): p.remove(c); n += 1; continue
            if t == "g":
                sh = [e for e in c.iter() if _tg(e) in _SHAPES]
                if sh and len(sh) <= 6 and not any(_tg(e) == "g" for e in c.iter() if e is not c) and all(set(_paint(e)) <= _INK for e in sh): p.remove(c); n += 1; continue
                walk(c)
    walk(root); return n


def svg_str(root):
    ET.register_namespace("", _SVG_NS); ET.register_namespace("xlink", _XLINK)
    for e in root.iter():
        for a in list(e.attrib):
            if a.startswith("{") and not a.startswith("{%s}" % _XLINK): del e.attrib[a]
    for p in list(root.iter()):
        for c in list(p):
            if isinstance(c.tag, str) and (c.tag.split("}")[0].lstrip("{") not in (_SVG_NS, "") or _tg(c) in ("metadata", "namedview")): p.remove(c)
    return ET.tostring(root, encoding="unicode")


def svg_prepare(raw, quitar=True):
    """SVG subido -> (svg limpio sin llamadas, svg original con tamaño explícito para que la IA lo vea, ancho, alto, nº de elementos quitados)."""
    s = raw.decode("utf-8", "replace")
    s = re.sub(r"<\?xml.*?\?>|<!DOCTYPE[^>\[]*(\[.*?\])?>|<!--.*?-->", "", s, flags=re.S)
    root = parse_svg(s)
    for p in list(root.iter()):
        for c in list(p):
            if _tg(c) in ("script", "foreignobject", "iframe", "embed", "object", "audio", "video"): p.remove(c)
            elif _tg(c) == "style": c.text = re.sub(r"@import[^;]*;|url\(\s*['\"]?https?:[^)]*\)", "", c.text or "")
    for el in root.iter():
        for a in list(el.attrib):
            if a.lower().startswith("on"): del el.attrib[a]
            elif "href" in a.lower() and not el.attrib[a].strip().startswith(("#", "data:image/")): del el.attrib[a]
    num = lambda v: float(m_.group(1)) if (m_ := re.match(r"\s*(-?[\d.]+)", v or "")) else None
    if root.attrib.get("viewBox"):
        x0, y0, W, H = [float(t) for t in re.split(r"[ ,]+", root.attrib["viewBox"].strip())][:4]
    else:
        W, H = num(root.attrib.get("width")), num(root.attrib.get("height"))
        if not W or not H: raise ValueError("el SVG no indica su tamaño (width/height o viewBox)")
        root.set("viewBox", f"0 0 {W} {H}")
    for k in ("width", "height", "x", "y"): root.attrib.pop(k, None)
    orig = copy.deepcopy(root); orig.set("width", str(W)); orig.set("height", str(H))
    n = strip_annotations(root) if quitar else 0
    return svg_str(root), svg_str(orig), W, H, n


def svg_png(svg, width=1400):
    """Convierte el SVG en imagen para que Gemini lo vea. Usa PyMuPDF (pip install pymupdf) o, si no, CairoSVG."""
    err = []
    try:
        import fitz
        pg = fitz.open(stream=svg.encode("utf-8"), filetype="svg")[0]; z = width / pg.rect.width
        return Image.open(io.BytesIO(pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False).tobytes("png"))).convert("RGB")
    except ImportError: pass
    except Exception as e: err.append(f"PyMuPDF: {e}")
    try:
        import cairosvg
        return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width, background_color="white"))).convert("RGB")
    except ImportError: pass
    except Exception as e: err.append(f"CairoSVG: {e}")
    raise RuntimeError("No puedo convertir el SVG en imagen para que la IA lo vea. Instala PyMuPDF con «pip install pymupdf» y reinicia la app"
                       + (f" ({'; '.join(err)})" if err else "") + ", o sube también una captura PNG del mismo dibujo en el recuadro opcional.")


def _box(b):
    try:
        y0, x0, y1, x1 = [max(0.0, min(1000.0, float(v))) for v in b]
        return [min(y0, y1), min(x0, x1), max(y0, y1), max(x0, x1)] if (abs(y1 - y0) > 8 and abs(x1 - x0) > 8) else None
    except Exception: return None


def photo_layout(W, H):
    s = min(800 / W, 520 / H); w, h = W * s, H * s
    return (800 - w) / 2, (520 - h) / 2, w, h


def photo_svg(W, H, parts):
    """SVG de la parada: el dibujo (se inserta al mostrar: <!--FIG-->) y, encima, una zona pulsable invisible por cada pieza + el punto donde irá su número."""
    ox, oy, w, h = photo_layout(W, H); P = lambda nx, ny: (ox + nx / 1000 * w, oy + ny / 1000 * h)
    out = [f'<svg xmlns="{_SVG_NS}" viewBox="0 0 800 520"><g id="bg"><!--FIG--></g>']; lbl = {}
    for p in sorted(parts, key=lambda p: -(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1])):  # las pequeñas encima, para poder pulsarlas
        ymin, xmin, ymax, xmax = p["box"]; x0, y0 = P(xmin, ymin); x1, y1 = P(xmax, ymax)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; bw, bh = max(x1 - x0, 46), max(y1 - y0, 46)
        out.append(f'<g id="{p["id"]}"><rect class="hl" x="{cx - bw / 2:.1f}" y="{cy - bh / 2:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="{min(bw, bh) * .3:.1f}" fill="#fff" fill-opacity=".01" pointer-events="all"/></g>')
        ay, ax = p.get("point") or [(ymin + ymax) / 2, (xmin + xmax) / 2]; lx, ly = P(ax, ay); lbl[p["id"]] = [round(lx), round(ly)]
    return "".join(out) + "</svg>", lbl


def build_photo_stop(slug, n, raw, quitar=True, ref=None):
    """Tu SVG -> parada del viaje: la IA localiza las piezas y escribe todo en inglés (con apoyo en español). El dibujo se muestra tal cual, sin tapar nada."""
    clean, orig, W, H, _ = svg_prepare(raw, quitar)
    im = Image.open(io.BytesIO(ref)).convert("RGB") if ref else svg_png(orig)
    im.thumbnail((1280, 1280)); m = load(slug)
    prev = ", ".join(s.get("scene", "") for s in m.get("photos", {}).get(str(n), [])) or "none"
    prompt = fill(P_PHOTO, N=n, STYLE=LEVELS[n]["style"], RULE=LEVEL_RULE[n].split(" Generic drawing")[0], PREV=prev, MAXP=PHOTO_MAX[n], TEXT=m["plan"][str(n)]["text"][:6000])
    content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(im, 85)}}]
    err = "respuesta incompleta"
    for attempt in range(3):
        try:
            d = parse_json(llm([{"role": "user", "content": content}], max_tokens=9000, as_json=True))
            photo_assemble(slug, n, d, clean, W, H); return
        except Exception as e:
            err = e
    raise RuntimeError(f"No pude montar la parada ({err}). Inténtalo otra vez.")


def photo_assemble(slug, n, d, clean, W, H):
    keys = ("id", "name", "emoji", "says_en", "text_en", "text_es", "body_en", "body_es", "clue_en", "clue_es", "why_es")
    parts, idmap = [], {}
    for p in d.get("parts", []):
        if not isinstance(p, dict): continue
        bx = _box(p.get("box"))
        if not bx or not str(p.get("name", "")).strip(): continue
        q = {k: str(p.get(k, "")) for k in keys}; old = q["id"]; q["id"] = f"hz{len(parts) + 1}"; idmap[old] = q["id"]
        q["box"] = bx
        try:
            py, px = [max(0.0, min(1000.0, float(v))) for v in p["point"]]; q["point"] = [py, px] if bx[0] - 30 <= py <= bx[2] + 30 and bx[1] - 30 <= px <= bx[3] + 30 else None
        except Exception: q["point"] = None
        q["emoji"] = q["emoji"] or "✨"; q["says_en"] = q["says_en"] or q["text_en"]; q["clue_en"] = q["clue_en"] or q["text_en"]
        q["body_en"] = q["body_en"] or q["text_en"]; q["body_es"] = q["body_es"] or q["text_es"]
        q["clue_es"] = q["clue_es"] or q["body_es"]; q["why_es"] = q["why_es"] or q["text_es"]
        parts.append(q)
    parts = parts[:PHOTO_MAX[n]]
    if len(parts) < 3: raise ValueError(f"solo encontré {len(parts)} piezas con su recuadro; necesito al menos 3")
    ids = {p["id"] for p in parts}
    story = [{"part": idmap.get(str(b.get("part")), ""), "en": str(b.get("en", "")), "es": str(b.get("es", ""))} for b in d.get("story", []) if isinstance(b, dict)]
    story = [b for b in story if b["part"] in ids]
    if len(story) < 2: story = [{"part": p["id"], "en": p["text_en"], "es": p["text_es"]} for p in parts]
    svg, lbl = photo_svg(W, H, parts)
    for p in parts: p.pop("box"); p.pop("point")
    (tdir(slug) / "photos").mkdir(exist_ok=True)
    fn = f"{n}_{int(time.time() * 1000)}.svg"; (tdir(slug) / "photos" / fn).write_text(clean, "utf-8")
    m = load(slug)
    m.setdefault("photos", {}).setdefault(str(n), []).append({"jv": JV, "scene": str(d.get("scene", "My picture")), "emoji": str(d.get("emoji", "🔬")) or "🔬", "story_es": str(d.get("story_es", "")),
                                                              "traveler": str(d.get("traveler", "✨")) or "✨", "svg": svg, "parts": parts, "fixed": [], "lbl": lbl, "badge": True, "story": story,
                                                              "outro_es": str(d.get("outro_es", "")), "photo": fn, "fw": W, "fh": H})
    save(slug, m)


def photo_stops(slug, m, lvl):
    """Paradas guardadas, con tu dibujo ya colocado dentro del SVG de la parada."""
    out = []
    for s in m.get("photos", {}).get(str(lvl), []):
        try: fig = (tdir(slug) / "photos" / s["photo"]).read_text("utf-8")
        except Exception: continue
        ox, oy, w, h = photo_layout(s["fw"], s["fh"])
        fig = fig.replace("<svg", f'<svg x="{ox:.1f}" y="{oy:.1f}" width="{w:.1f}" height="{h:.1f}"', 1)
        out.append({**s, "svg": s["svg"].replace("<!--FIG-->", fig), "title": s["scene"], "ready": True})
    return out



def tutor(slug, lvl, question, ctx=""):
    m = load(slug); mat = (m["plan"][str(lvl)]["text"] if lvl and m.get("plan") else m["base"]["source_text"])[:14000]
    sysm = ("Eres 'Profe', un tutor amable para una niña española de 10 años (5º Primaria, colegio bilingüe, inglés básico). "
            "Responde SIEMPRE en español sencillo, máximo 6 frases, con los términos clave en inglés entre paréntesis. "
            "Usa solo el material del tema; si no aparece, dilo. Termina con ánimo breve.\n\nMATERIAL:\n" + mat)
    return llm([{"role": "system", "content": sysm}, {"role": "user", "content": (ctx + "\n" + question).strip()}], max_tokens=700)


# ───────────────────────── progreso ─────────────────────────
def set_progress(slug, lvl, **kw):
    m = load(slug); m["progress"].setdefault(str(lvl), {}).update(kw); save(slug, m)


def passed(m, n): return bool(m["progress"].get(str(n), {}).get("passed"))
def cur_level(m): return next((n for n in (1, 2, 3) if not passed(m, n)), 3)


def draw(qs, n):
    by = {}
    for i, q in enumerate(qs): by.setdefault(q.get("section", ""), []).append(i)
    for v in by.values(): random.shuffle(v)
    order = []
    while len(order) < min(n, len(qs)):
        for v in by.values():
            if v and len(order) < n: order.append(v.pop())
    random.shuffle(order); return order


def record_attempt(slug, lvl, qs, S, mode):
    m = load(slug); pr = m["progress"].setdefault(str(lvl), {}); pend = pr.setdefault("pending", {}); sec = {}
    for qid, ok in S["res"].items():
        q = qs[qid]; a = sec.setdefault(q["section"], [0, 0]); a[1] += 1; a[0] += int(ok)
        if ok: pend.pop(q["q"], None)
        else: pend[q["q"]] = {"q": q["q"], "a": q["options"][q["answer_idx"]], "e": q["explain_es"], "s": q["section"]}
    n = len(S["res"]); sc = sum(S["res"].values()); ok_level = mode == "level" and n > 0 and sc / n >= PASA
    newly = ok_level and not pr.get("passed")
    pr.setdefault("attempts", []).append({"ts": time.strftime("%d/%m %H:%M"), "score": sc, "total": n, "mode": mode, "passed": bool(ok_level), "sections": sec})
    pr["attempts"] = pr["attempts"][-30:]
    if ok_level: pr["passed"] = True
    save(slug, m); return ok_level, newly


@st.cache_data(show_spinner=False)
def load_figs(slug, ts):
    out = []
    for f in load(slug)["base"]["figures"]:
        im = Image.open(tdir(slug) / "figs" / f["file"]); im.thumbnail((420, 420))
        out.append({"src": "data:image/jpeg;base64," + b64(im, 78), "label": f["label"], "caption": f["caption"]})
    return out


# ───────────────────────── componente HTML interactivo (juegos, esquemas, escena) ─────────────────────────
HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;font-family:'Segoe UI',system-ui,sans-serif}
body{margin:0;padding:12px;background:#fff;color:#1f2937;border-radius:14px}
.bar{display:flex;justify-content:space-between;font-weight:700;margin-bottom:8px}
button{cursor:pointer;border:0;border-radius:10px;padding:10px 14px;font-size:16px;background:#e0e7ff;color:#1e1b4b;font-weight:600}
button:hover{filter:brightness(.95)} button:disabled{opacity:.7;cursor:default}
.opts{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:10px}
.ok{background:#bbf7d0!important}.bad{background:#fecaca!important}
.imgwrap{position:relative;width:min(100%,420px);margin:auto;border-radius:12px;overflow:hidden}
.imgwrap img{width:100%;display:block}.tiles{position:absolute;inset:0;display:grid}
.tile{background:linear-gradient(135deg,#6366f1,#8b5cf6);border:1px solid #fff;cursor:pointer;transition:opacity .5s}.tile.off{opacity:0;pointer-events:none}
.mem{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.card{height:118px;border-radius:12px;background:#6366f1;display:flex;align-items:center;justify-content:center;cursor:pointer;color:#fff;font-size:30px;overflow:hidden}
.card .f{display:none;width:100%;height:100%;align-items:center;justify-content:center;text-align:center;padding:4px;font-size:15px;color:#111;background:#f1f5f9}
.card .f img{width:100%;height:100%;object-fit:cover}
.card.up .f,.card.ok .f{display:flex}.card.up .b,.card.ok .b{display:none}.card.ok{outline:4px solid #22c55e}
.g4{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.dz{text-align:center}
.dz img{width:100%;height:110px;object-fit:cover;border-radius:10px}
.chip{display:inline-block;padding:8px 14px;border-radius:999px;margin:4px;cursor:grab;font-weight:700;user-select:none;border:2px solid transparent}
.chip.sel{border-color:#111;transform:scale(1.06)}
.zone{display:inline-block;min-width:140px;min-height:42px;border:2px dashed #94a3b8;border-radius:12px;vertical-align:middle;padding:2px;text-align:center;margin:2px}
.zone.over{background:#e0f2fe}.zone.ok{border-color:#22c55e;background:#f0fdf4}.zone.bad{border-color:#ef4444;background:#fef2f2}
.bankbox{background:#f8fafc;border-radius:12px;padding:8px;margin-bottom:10px;min-height:52px}
.row{margin:6px 0;font-size:18px;line-height:2.1}.arrow{text-align:center;font-size:22px;color:#6366f1;line-height:1}
.hb{font-size:14px;color:#334155;background:#fef9c3;border-radius:8px;padding:6px 10px;margin:0 0 6px}
.mini{padding:2px 8px;font-size:14px;background:#fef9c3}.msg{text-align:center;margin:8px;font-size:17px}
.stage{background:#f0f9ff;border-radius:14px;padding:6px;perspective:900px}
.part{transition:opacity .7s,transform .9s,filter .4s;transform-box:fill-box;transform-origin:center}.part.hid{opacity:0;pointer-events:none}
.part.next{opacity:.3;pointer-events:auto;cursor:pointer;animation:pu 1.1s infinite;filter:drop-shadow(0 0 8px #f59e0b)}
.part.done{opacity:1;pointer-events:auto;cursor:pointer}.part.sel{filter:drop-shadow(0 0 7px #f59e0b)}
@keyframes pu{0%,100%{opacity:.25}50%{opacity:.9}}
.part.dim{opacity:.25;filter:grayscale(.7)}.part.glow{opacity:1;animation:gw .9s ease-in-out infinite}@keyframes gw{0%,100%{filter:drop-shadow(0 0 3px #fff) drop-shadow(0 0 10px #f59e0b)}50%{filter:drop-shadow(0 0 5px #fff) drop-shadow(0 0 16px #f59e0b) drop-shadow(0 0 30px #fbbf24)}}.ring{fill:none;stroke:#f97316;stroke-width:6;stroke-dasharray:16 10;pointer-events:none;transform-box:fill-box;transform-origin:center;animation:rg 1s ease-in-out infinite;filter:drop-shadow(0 0 4px #fff)}@keyframes rg{0%,100%{opacity:.55;transform:scale(1)}50%{opacity:1;transform:scale(1.1)}}.part.good{opacity:1;filter:drop-shadow(0 0 14px #22c55e)}
.part.glow rect.hl{fill:#f59e0b;fill-opacity:.05;stroke:#f59e0b;stroke-width:3}.part.good rect.hl{fill:#22c55e;fill-opacity:.08;stroke:#22c55e;stroke-width:3}
.legend{display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;margin-top:10px;font-size:.85em}.legend b{display:inline-block;min-width:22px;text-align:center;background:#fff;border:2px solid #6366f1;border-radius:999px}
.trow{display:flex;align-items:center;flex-wrap:wrap;gap:4px;padding:6px 10px;margin:7px 0;border-left:6px solid var(--tc,#6366f1);background:#fff;border-radius:10px;box-shadow:0 1px 4px #0002;position:relative}
.trow.d0{--tc:#6366f1;background:#eef2ff;font-size:1.12em;font-weight:800}.trow.d1{--tc:#06b6d4}.trow.d2{--tc:#22c55e}.trow.d3{--tc:#f59e0b}
.trow:not(.d0)::before{content:'↳';position:absolute;left:-24px;top:7px;color:#94a3b8;font-weight:800}
.part.shake{animation:shk .4s}@keyframes shk{0%,100%{transform:translateX(0)}25%{transform:translateX(-8px)}75%{transform:translateX(8px)}}
.jmap{display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:8px;margin:14px 0}.jar{color:#94a3b8;font-size:22px;margin:0 2px}
.stopc{width:150px;min-height:150px;border-radius:18px;background:linear-gradient(160deg,#e0e7ff,#fce7f3);padding:10px;text-align:center;cursor:pointer;position:relative;transition:transform .2s;box-shadow:0 2px 6px #0002}
.stopc:hover{transform:translateY(-3px) scale(1.03)}.stopc .sn{position:absolute;top:6px;left:12px;font-weight:800;color:#6366f1}
.stopc .se{font-size:46px;margin:8px 0 2px}.stopc small{display:block;margin-top:4px;color:#475569}
.stopc.lock{background:#f1f5f9;opacity:.7}.stopc.done{background:linear-gradient(160deg,#bbf7d0,#fef9c3)}
.stopc.now{outline:3px solid #f59e0b;animation:bnc 1.3s infinite}@keyframes bnc{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
.steps{display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:6px 0}
.st{padding:6px 10px;font-size:14px;background:#f1f5f9}.st.dn{background:#bbf7d0}.st.on{background:#6366f1;color:#fff}.st:disabled{opacity:.45}
.mini2{padding:5px 10px;font-size:14px}
.chr{display:flex;gap:12px;align-items:flex-start}.av{font-size:40px;line-height:1}
.bub{display:inline-block;background:#fff;border-radius:14px;padding:2px 10px;margin-left:4px;font-style:italic}
.story{font-size:17px;line-height:1.5}.cnt{font-weight:700;margin-left:8px}
.cols{display:grid;grid-template-columns:minmax(270px,5fr) 7fr;gap:14px;align-items:start}.side{min-width:0}
.cols .stage{position:sticky;top:0}@media(max-width:720px){.cols{grid-template-columns:1fr}.cols .stage{order:-1;position:static}}
.job{background:#f8fafc;border:2px solid #cbd5e1;border-radius:12px;padding:8px 10px;margin:6px 0;cursor:pointer;font-size:15px;line-height:1.35}
.job small{display:block;color:#475569;margin-top:2px}.job.sel{border-color:#6366f1;background:#eef2ff}.job.ok{background:#dcfce7;border-color:#22c55e;cursor:default}
.opt{display:block;width:100%;text-align:left;margin:6px 0;font-size:16px;background:#eef2ff;line-height:1.35}
.qn{font-weight:800;color:#6366f1}.qt{font-size:19px;font-weight:700;margin:4px 0}.qes{color:#475569;font-size:15px}
.cast span{display:inline-block;background:#fff;border-radius:999px;padding:3px 10px;margin:3px 2px;font-size:14px;font-weight:600}
.dots{letter-spacing:4px;color:#6366f1;font-size:18px;margin:4px 0}
.lbl{opacity:0;pointer-events:none;transition:opacity .4s,transform .9s}.lbl.on{opacity:1}
.part.ask{animation:gl 1.3s infinite;cursor:pointer}@keyframes gl{0%,100%{filter:drop-shadow(0 0 0 #f59e0b)}50%{filter:drop-shadow(0 0 11px #f59e0b)}}
.info{background:#fef9c3;border-radius:12px;padding:10px 14px;margin-top:8px;min-height:80px;font-size:17px;line-height:1.45}
</style></head><body><div id="root"></div><script>
const P=__PAYLOAD__, MODE="__MODE__", R=document.getElementById('root'), $=id=>document.getElementById(id);
const PAL=['#fde68a','#bbf7d0','#bfdbfe','#fbcfe8','#ddd6fe','#fed7aa','#a7f3d0','#fecdd3'];
const shuffle=a=>{a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a};
const rand=a=>a[Math.floor(Math.random()*a.length)];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const FIGS=P.figs, C=P.cfg;
const endScreen=(txt,again)=>{R.innerHTML=`<h2>🎉 ¡Terminado!</h2><p class=msg>${txt}</p><button id=again>🔁 Jugar otra vez</button>`;$('again').onclick=again};

function gReveal(){
  const items=shuffle(FIGS).slice(0,C.rounds);let i=0,score=0;
  function round(){
    if(i>=items.length)return endScreen(`Has conseguido ⭐ ${score} puntos`,gReveal);
    const f=items[i],n=C.grid;let hidden=n*n,done=false;
    const opts=shuffle([f,...shuffle(FIGS.filter(x=>x.label!==f.label)).slice(0,3)]);
    R.innerHTML=`<div class=bar><span>Round ${i+1}/${items.length}</span><span>⭐ ${score}</span></div>
    <div class=imgwrap><img src="${f.src}"><div class=tiles style="grid-template-columns:repeat(${n},1fr)">${'<div class=tile></div>'.repeat(n*n)}</div></div>
    <p class=msg id=msg>👆 Toca los cuadros para destapar la imagen. ¡Cuantos menos destapes, más puntos!</p>
    <div class=opts>${opts.map((o,k)=>`<button data-k=${k}>${esc(o.label)}</button>`).join('')}</div>`;
    const tiles=[...R.querySelectorAll('.tile')];
    const open=t=>{if(!t.classList.contains('off')){t.classList.add('off');hidden--}};
    tiles.forEach(t=>t.onclick=()=>{if(!done)open(t)});open(rand(tiles));
    R.querySelectorAll('.opts button').forEach(b=>b.onclick=()=>{
      if(done)return;const o=opts[+b.dataset.k];
      if(o.label===f.label){done=true;const pts=hidden+1;score+=pts;tiles.forEach(t=>t.classList.add('off'));b.classList.add('ok');
        $('msg').innerHTML=`✅ ¡Correcto! <b>${esc(f.label)}</b> +${pts}⭐<br><small>${esc(f.caption)}</small>`;
        const nx=document.createElement('button');nx.textContent='Siguiente ➜';nx.style.cssText='display:block;margin:10px auto';nx.onclick=()=>{i++;round()};R.appendChild(nx);
      }else{b.classList.add('bad');b.disabled=true;$('msg').textContent='❌ Casi… ¡prueba otra vez! Te destapo un cuadro 💡';
        const hid=tiles.filter(t=>!t.classList.contains('off'));if(hid.length>1)open(rand(hid))}
    });
  }
  round();
}

function gMem(){
  const items=shuffle(FIGS).slice(0,C.pairs);
  const cards=shuffle(items.flatMap((f,id)=>[{id,t:'i',f},{id,t:'t',f}]));
  let first=null,lock=false,moves=0,found=0;
  R.innerHTML=`<div class=bar><span>🃏 Encuentra cada imagen con su palabra</span><span id=mv>Movimientos: 0</span></div>
  <div class=mem>${cards.map((c,k)=>`<div class=card data-k=${k}><span class=b>❓</span><div class=f>${c.t==='i'?`<img src="${c.f.src}">`:`<b>${esc(c.f.label)}</b>`}</div></div>`).join('')}</div>`;
  R.querySelectorAll('.card').forEach(el=>el.onclick=()=>{
    if(lock||el.classList.contains('up')||el.classList.contains('ok'))return;
    el.classList.add('up');const c=cards[+el.dataset.k];
    if(!first){first={el,c};return}
    moves++;$('mv').textContent='Movimientos: '+moves;
    if(first.c.id===c.id&&first.c.t!==c.t){[first.el,el].forEach(x=>{x.classList.remove('up');x.classList.add('ok')});first=null;
      if(++found===items.length)setTimeout(()=>endScreen(`Lo has hecho en ${moves} movimientos`,gMem),600)}
    else{lock=true;const a=first.el;first=null;setTimeout(()=>{a.classList.remove('up');el.classList.remove('up');lock=false},900)}
  });
}

function gDrag(){
  const pool=shuffle(FIGS),groups=[];for(let g=0;g<C.groups;g++){const s=pool.slice(g*4,g*4+4);if(s.length===4)groups.push(s)}
  let gi=0;
  function round(){
    if(gi>=groups.length)return endScreen('¡Has puesto todas las palabras en su sitio!',gDrag);
    const g=groups[gi];let solved=0,sel=null;
    R.innerHTML=`<div class=bar><span>🎯 Arrastra cada palabra a su imagen (o toca palabra y luego imagen)</span><span>${gi+1}/${groups.length}</span></div>
    <div id=bank class=bankbox>${shuffle(g).map((f,k)=>`<span class=chip draggable=true data-l="${esc(f.label)}" style="background:${PAL[k%8]}">${esc(f.label)}</span>`).join('')}</div>
    <div class=g4>${g.map((f,k)=>`<div class=dz><img src="${f.src}"><div class=zone data-k=${k}></div></div>`).join('')}</div><p class=msg id=msg></p>`;
    const chips=[...R.querySelectorAll('.chip')];
    chips.forEach(c=>{c.ondragstart=e=>e.dataTransfer.setData('text/plain',c.dataset.l);c.onclick=()=>{chips.forEach(x=>x.classList.remove('sel'));if(sel===c){sel=null}else{sel=c;c.classList.add('sel')}}});
    const put=(z,label)=>{
      if(z.classList.contains('ok')||!label)return;const k=+z.dataset.k;
      if(g[k].label===label){z.classList.add('ok');z.innerHTML=`<span class=chip style="background:#bbf7d0;cursor:default">${esc(label)}</span>`;
        const c=chips.find(x=>x.dataset.l===label);if(c)c.remove();sel=null;$('msg').textContent='✅ ¡Bien!';
        if(++solved===4)setTimeout(()=>{gi++;round()},900)}
      else{z.classList.add('bad');setTimeout(()=>z.classList.remove('bad'),700);$('msg').textContent='❌ Ese no es… ¡mira bien la imagen!'}
    };
    R.querySelectorAll('.zone').forEach(z=>{z.ondragover=e=>{e.preventDefault();z.classList.add('over')};z.ondragleave=()=>z.classList.remove('over');
      z.ondrop=e=>{e.preventDefault();z.classList.remove('over');put(z,e.dataTransfer.getData('text/plain'))};z.onclick=()=>{if(sel)put(z,sel.dataset.l)}});
  }
  if(!groups.length)R.innerHTML='<p>Necesito al menos 4 imágenes.</p>';else round();
}

function scheme(){
  const S=P.data,rows=S.rows,bank=S.bank;let placed=rows.map(()=>null),sel=null,checked=false,msg='',hints=new Set();
  const col=w=>PAL[bank.indexOf(w)%PAL.length];
  const chip=(w,extra='')=>`<span class="chip ${sel===w?'sel':''}" data-w="${esc(w)}" ${extra} style="background:${col(w)}">${esc(w)}</span>`;
  function draw(){
    const free=bank.filter(w=>!placed.includes(w));
    let h=`<h3 style="margin:0 0 4px">${esc(S.title)}</h3><p style="margin:0 0 8px;color:#475569">${esc(S.instructions_es)}</p>
    <div class=bankbox id=bank>${free.map(w=>chip(w)).join('')||'<i>✔ Ya has colocado todas las palabras</i>'}</div>`;
    h+=rows.map((r,i)=>{const [a,b]=r.text.split('___'),w=placed[i];
      const cls=checked&&w!==null?(w===r.answer?'ok':'bad'):'';
      if(S.layout==='tree'){const d=r.d||0,m=d*28;return `<div class="trow d${d}" style="margin-left:${m}px">${r.answer?`${esc(a)}<span class="zone ${cls}" data-z=${i}>${w!==null?`<span class=chip data-p=${i} style="background:${col(w)}">${esc(w)}</span>`:''}</span>${esc(b||'')} <button class="mini" data-h=${i}>💡</button>`:`<b>${esc(r.text)}</b>`}</div>${hints.has(i)&&r.answer?`<div class=hb style="margin-left:${m}px">💡 ${esc(r.hint_es)}</div>`:''}`}
      return `${S.layout==='flow'&&i>0?'<div class=arrow>⬇</div>':''}<div class=row>${esc(a)}<span class="zone ${cls}" data-z=${i}>${w!==null?`<span class=chip data-p=${i} style="background:${col(w)}">${esc(w)}</span>`:''}</span>${esc(b||'')}
      <button class="mini" data-h=${i}>💡</button></div>${hints.has(i)?`<div class=hb>💡 ${esc(r.hint_es)}</div>`:''}`}).join('');
    h+=`<div style="margin-top:10px"><button id=chk>✅ Comprobar</button> <button id=rst>↺ Empezar de nuevo</button></div><p class=msg>${msg}</p>`;
    R.innerHTML=h;
    R.querySelectorAll('#bank .chip').forEach(c=>{c.draggable=true;c.ondragstart=e=>e.dataTransfer.setData('text/plain',c.dataset.w);c.onclick=()=>{sel=sel===c.dataset.w?null:c.dataset.w;draw()}});
    R.querySelectorAll('.zone').forEach(z=>{const i=+z.dataset.z;z.ondragover=e=>{e.preventDefault();z.classList.add('over')};z.ondragleave=()=>z.classList.remove('over');
      z.ondrop=e=>{e.preventDefault();put(i,e.dataTransfer.getData('text/plain'))};z.onclick=()=>{if(sel)put(i,sel)}});
    R.querySelectorAll('[data-p]').forEach(c=>c.onclick=ev=>{ev.stopPropagation();placed[+c.dataset.p]=null;checked=false;msg='';draw()});
    R.querySelectorAll('[data-h]').forEach(b=>b.onclick=()=>{const i=+b.dataset.h;hints.has(i)?hints.delete(i):hints.add(i);draw()});
    $('rst').onclick=()=>{placed=rows.map(()=>null);sel=null;checked=false;msg='';draw()};
    $('chk').onclick=()=>{checked=true;const need=rows.filter(r=>r.answer).length,ok=rows.filter((r,i)=>r.answer&&placed[i]===r.answer).length;
      msg=ok===need?'🎉 ¡Esquema perfecto! ¡Muy bien!':(rows.some((r,i)=>r.answer&&placed[i]===null)?`Llevas ${ok} bien. Te faltan huecos por rellenar.`:`${ok} de ${need} bien. Pulsa 💡 si necesitas ayuda y cambia las rojas.`);draw()};
  }
  function put(i,w){if(!w||!bank.includes(w))return;placed=placed.map(p=>p===w?null:p);placed[i]=w;sel=null;checked=false;msg='';draw()}
  draw();
}

function journey(){
  const D=P.data,STOPS=D.stops,KEY='sb_j2_'+D.key,LV=D.level,NS='http://www.w3.org/2000/svg';
  let prog={};try{prog=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
  const saveP=()=>{try{localStorage.setItem(KEY,JSON.stringify(prog))}catch(e){}};
  const PH=[['name','🔤','Nombra'],['quiz','🎯','Reto']];if(LV>=2)PH.push(['fnc','🧠','Función']);if(LV>=3)PH.push(['seq','📖','Historia']);
  const stars=n=>n?'⭐'.repeat(n):'☆☆☆';

  function map(){
    const done=STOPS.filter((s,i)=>prog[i]).length,nxt=STOPS.findIndex((s,i)=>s.ready&&!prog[i]);
    R.innerHTML=`<div class=bar><span>🗺️ Tu viaje por el tema</span><span>${done}/${STOPS.length} paradas</span></div>
    <p class=msg id=mm>${done===STOPS.length?'🏆 ¡Has recorrido TODO el tema! Ahora prueba los 🎮 Juegos y los 🧩 Esquemas.':'Cada parada es un apartado del tema. Pincha en la que se mueve para seguir el viaje.'}</p>
    <div class=jmap>${STOPS.map((s,i)=>`${i?'<span class=jar>➜</span>':''}<div class="stopc ${s.ready?'':'lock'} ${prog[i]?'done':''} ${i===nxt?'now':''}" data-i=${i}>
      <div class=sn>${i+1}</div><div class=se>${s.ready?esc(s.emoji):'🎨'}</div><b>${esc(s.title)}</b><small>${s.ready?stars(prog[i]||0):'No dibujada'}</small></div>`).join('')}</div>`;
    R.querySelectorAll('.stopc').forEach(c=>c.onclick=()=>{const i=+c.dataset.i;
      if(STOPS[i].ready)stop(i);else $('mm').textContent='🎨 Esta parada no se pudo dibujar. Pulsa «Reintentar» debajo del mapa.'});
  }

  function stop(i){
    const S=STOPS[i],parts=S.parts,fixed=new Set(S.fixed||[]);
    let ph=0,reach=0;
    R.innerHTML=`<div class=bar><button id=bk class=mini2>🗺️ Mapa</button><span>${esc(S.emoji)} ${i+1}. ${esc(S.title)}</span><span>${stars(prog[i]||0)}</span></div>
      <div class=steps id=steps></div><div class=cols><div class=side><div class=info id=info></div><div id=ctl style="margin-top:8px"></div></div><div class=stage id=stage>${S.svg}</div></div>`;
    const svg=R.querySelector('#stage svg');svg.style.cssText='width:100%;height:auto;display:block;transition:transform .15s ease-out;transform-origin:50% 55%';
    const stg=$('stage');stg.onpointermove=e=>{const r=stg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;svg.style.transform=`rotateY(${(x*16).toFixed(1)}deg) rotateX(${(-y*12).toFixed(1)}deg)`};stg.onpointerleave=()=>{svg.style.transform=''};
    const bg=svg.querySelector('#bg'),W={},E={};
    const LY=document.createElementNS(NS,'g'),Lb={};
    parts.forEach(p=>{const el=svg.querySelector('#'+p.id);const w=document.createElementNS(NS,'g');el.parentNode.insertBefore(w,el);w.appendChild(el);w.setAttribute('class','part');W[p.id]=w;E[p.id]=el;
      let x,y;if(S.lbl&&S.lbl[p.id]){x=S.lbl[p.id][0];y=S.lbl[p.id][1]}else{const bb=el.getBBox();x=bb.x+bb.width/2;y=(bb.y+bb.height/2>300)?bb.y-16:bb.y+bb.height+18}
      x=S.badge?x:Math.max(80,Math.min(720,x));y=Math.max(22,Math.min(500,y));
      const g=document.createElementNS(NS,'g');g.setAttribute('class','lbl');const t=document.createElementNS(NS,'text');
      [['x',x],['y',y+7],['text-anchor','middle'],['font-size',S.badge?17:21],['font-weight',700],['fill','#1e1b4b']].forEach(([k,v])=>t.setAttribute(k,v));t.textContent=S.badge?String(parts.indexOf(p)+1):p.name;g.appendChild(t);LY.appendChild(g);Lb[p.id]=g;g._t=t;g._xy=[x,y]});
    svg.appendChild(LY);
    parts.forEach(p=>{const g=Lb[p.id],t=g._t,[x,y]=g._xy,L=t.getComputedTextLength()||90,r=document.createElementNS(NS,'rect');
      [['x',x-L/2-10],['y',y-16],['width',L+20],['height',30],['rx',15],['fill','#fff'],['fill-opacity',.95],['stroke','#6366f1'],['stroke-width',2.5]].forEach(([k,v])=>r.setAttribute(k,v));g.insertBefore(r,t)});
    /* aro naranja que parpadea alrededor de la pieza que hay que mirar */
    const RG=document.createElementNS(NS,'ellipse');RG.setAttribute('class','ring');RG.style.display='none';svg.appendChild(RG);
    const aim=id=>{if(!id){RG.style.display='none';return}const bb=W[id].getBBox();
      [['cx',bb.x+bb.width/2],['cy',bb.y+bb.height/2],['rx',Math.max(30,bb.width/2+16)],['ry',Math.max(30,bb.height/2+16)]].forEach(([k,v])=>RG.setAttribute(k,v));RG.style.display=''};
    const all=()=>parts.map(p=>W[p.id]),lall=()=>parts.map(p=>Lb[p.id]);
    const legend=()=>S.badge?'<div class=legend>'+parts.map((p,k)=>`<span><b>${k+1}</b> ${esc(p.name)}</span>`).join('')+'</div>':'';
    const clear=()=>{all().forEach(w=>{w.onclick=null;w.setAttribute('class','part done')});lall().forEach(g=>g.classList.remove('on'));aim(null);if(bg)bg.style.opacity=1};
    function steps(){$('steps').innerHTML=PH.map((p,k)=>`<button class="st ${k===ph?'on':''} ${k<reach?'dn':''}" data-k=${k} ${k>reach?'disabled':''}>${p[1]} ${p[2]}</button>`).join('<span class=jar>›</span>');
      $('steps').querySelectorAll('.st').forEach(b=>b.onclick=()=>go(+b.dataset.k))}
    const unlock=k=>{reach=Math.max(reach,k);steps()};
    function go(k){ph=k;reach=Math.max(reach,k);clear();steps();({name,quiz,fnc,seq})[PH[k][0]]()}

    const SC={};
    function more(a,b){clear();lall().forEach(g=>g.classList.add('on'));unlock(ph+1);
      $('info').innerHTML=`<div class=story><h3 style="margin:0">✅ ¡Reto superado!</h3>${a} de ${b} a la primera.<br>Hay otro reto más difícil: <b>${PH[ph+1][1]} ${PH[ph+1][2]}</b>.</div>${legend()}`;
      $('ctl').innerHTML=`<button id=n2>${PH[ph+1][1]} ${PH[ph+1][2]} ➜</button> <button id=r2>🔁 Repetir este reto</button>`;
      $('n2').onclick=()=>go(ph+1);$('r2').onclick=()=>go(ph)}
    function finish(a,b){
      const ks=PH.map(p=>p[0]).filter(k=>k!=='name'),r=ks.reduce((s,k)=>s+(SC[k]||0),0)/ks.length,st=r>=.8?3:r>=.5?2:1;
      prog[i]=Math.max(prog[i]||0,st);saveP();clear();lall().forEach(g=>g.classList.add('on'));
      const hasNext=i+1<STOPS.length,rdy=hasNext&&STOPS[i+1].ready;
      $('info').innerHTML=`<div class=story><h3 style="margin:0">🎉 ¡Parada completada! ${stars(st)}</h3>${a} de ${b} a la primera en el último reto.${hasNext&&!rdy?'<br>🎨 La siguiente parada no está dibujada: pulsa «Reintentar» debajo del mapa.':''}${!hasNext?'<br>🏆 ¡Última parada del viaje!':''}</div>${legend()}`;
      $('ctl').innerHTML=`${rdy?'<button id=ns>Siguiente parada ➜</button> ':''}<button id=mp>🗺️ Mapa</button> <button id=rp>🔁 Repetir el reto</button>`;
      if(rdy)$('ns').onclick=()=>stop(i+1);$('mp').onclick=map;$('rp').onclick=()=>go(ph)}

    /* RETO · FUNCIÓN (niveles 2 y 3): se ilumina una pieza y se elige, en inglés, qué hace; al acertar, la explicación va en español */
    function fnc(){
      const ord=shuffle(parts),N=ord.length;let q=0,first=0;
      function ask(){
        if(q>=N){SC.fnc=first/N;return ph<PH.length-1?more(first,N):finish(first,N)}
        const p=ord[q],opts=shuffle([p,...shuffle(parts.filter(x=>x.id!==p.id)).slice(0,3)]);let wrong=0;
        all().forEach(w=>{w.classList.remove('good','shake');w.classList.toggle('glow',w===W[p.id]);w.classList.toggle('dim',w!==W[p.id])});
        lall().forEach(g=>g.classList.remove('on'));aim(p.id);
        const head=`<div class=qn>Pregunta ${q+1} de ${N}</div><div class=qt>🧠 What does the <b>${esc(p.name)}</b> do?</div>`;
        $('info').innerHTML=head+'<div class=qes>👉 Elige la respuesta correcta.</div>';
        $('ctl').innerHTML=opts.map((o,k)=>`<button class=opt data-k=${k}>${esc(o.body_en)}</button>`).join('')+`<div class=cnt>⭐ ${first} a la primera</div>`;
        $('ctl').querySelectorAll('.opt').forEach(b=>b.onclick=()=>{const o=opts[+b.dataset.k];
          if(o.id===p.id){if(!wrong)first++;$('ctl').querySelectorAll('.opt').forEach(x=>x.disabled=true);b.classList.add('ok');
            W[p.id].classList.remove('glow','dim');W[p.id].classList.add('good');Lb[p.id].classList.add('on');aim(null);
            $('info').innerHTML=head+`<div class=story style="margin-top:8px">✅ <b>¡Muy bien!</b></div><div class=qes style="margin-top:6px">🇪🇸 <b>${esc(p.body_es)}</b></div><div class=story style="margin-top:6px;background:#ecfeff;padding:6px 10px;border-radius:10px">💡 ${esc(p.why_es||p.text_es)}</div>`;
            $('ctl').insertAdjacentHTML('afterbegin',`<button id=nq style="margin-bottom:8px">${q+1>=N?'Terminar ➜':'Siguiente ➜'}</button>`);
            $('nq').onclick=()=>{q++;ask()}}
          else{wrong++;b.classList.add('bad');b.disabled=true;$('info').insertAdjacentHTML('beforeend','<div class=story style="margin-top:6px">❌ Casi… ¡prueba otra vez!</div>')}})}
      ask()}

    /* RETO · HISTORIA (nivel 3): se ordenan los pasos del proceso; cada acierto ilumina la pieza que actúa y enseña el paso en español */
    function seq(){
      const B=S.story,N=B.length,sh=shuffle(B.map((b,k)=>k)),used=new Set();let nx=0,wrong=0;
      function draw(){
        all().forEach(w=>{w.classList.remove('glow','dim');w.classList.add('done')});aim(null);
        const dn=B.slice(0,nx).map((b,k)=>`<div class=story style="margin-top:6px">✅ <b>${k+1}.</b> ${esc(b.en)}<br><small>${esc(b.es)}</small></div>`).join('');
        $('info').innerHTML=`<div class=qn>Historia · paso ${Math.min(nx+1,N)} de ${N}</div><div class=qt>📖 Put the story in order: what happens ${nx?'next':'first'}?</div>${nx===0&&S.story_es?`<div class=qes>${esc(S.story_es)}</div>`:'<div class=qes>👉 Pulsa el paso que va ahora.</div>'}${dn}`;
        $('ctl').innerHTML=sh.filter(k=>!used.has(k)).map(k=>`<button class=opt data-k=${k}>${esc(B[k].en)}</button>`).join('');
        $('ctl').querySelectorAll('.opt').forEach(b=>b.onclick=()=>{const k=+b.dataset.k;
          if(k===nx){used.add(k);const id=B[k].part;nx++;if(W[id]){W[id].classList.add('good');Lb[id].classList.add('on')}
            if(nx>=N){const ok=N-Math.min(wrong,N);SC.seq=ok/N;lall().forEach(g=>g.classList.add('on'));return ph<PH.length-1?more(ok,N):finish(ok,N)}
            draw()}
          else{wrong++;b.classList.add('bad');b.disabled=true}})}
      draw()}

    $('bk').onclick=map;

    /* FASE 1 · NOMBRA: una pieza brilla y se elige su nombre en la lista de la izquierda; al acertar se explica en español para qué sirve */
    function name(){
      const order=shuffle(parts),N=order.length;let q=0;
      function ask(){
        if(q>=N)return fin();
        const p=order[q],opts=shuffle([p,...shuffle(parts.filter(x=>x.id!==p.id)).slice(0,3)]);
        all().forEach(w=>{w.classList.remove('good','shake');w.classList.toggle('glow',w===W[p.id]);w.classList.toggle('dim',w!==W[p.id])});
        lall().forEach(g=>g.classList.remove('on'));order.slice(0,q).forEach(x=>Lb[x.id].classList.add('on'));aim(p.id);
        $('info').innerHTML=`<div class=qn>Pieza ${q+1} de ${N}</div><div class=qt>👀 ¿Cómo se llama la pieza que brilla?</div>`;
        $('ctl').innerHTML=opts.map((o,k)=>`<button class=opt data-k=${k}>${esc(o.name)}</button>`).join('');
        $('ctl').querySelectorAll('.opt').forEach(b=>b.onclick=()=>{const o=opts[+b.dataset.k];
          if(o.id===p.id){$('ctl').querySelectorAll('.opt').forEach(x=>x.disabled=true);b.classList.add('ok');
            W[p.id].classList.remove('glow','dim');W[p.id].classList.add('good');Lb[p.id].classList.add('on');aim(null);
            $('info').innerHTML=`<div class=qn>Pieza ${q+1} de ${N}</div><div class=qt>👀 ¿Cómo se llama la pieza que brilla?</div><div class=story style="margin:8px 0 6px">✅ <b>¡Muy bien!</b> Es <b>${esc(p.name)}</b>${S.badge?' (nº '+(parts.indexOf(p)+1)+')':''}.</div><div class=chr><div class=av>${esc(p.emoji)}</div><div><b>${esc(p.name)}</b><br>📘 ${esc(p.body_es)}<br><small>${esc(p.text_es)}</small></div></div>`;
            $('ctl').insertAdjacentHTML('afterbegin',`<button id=nq style="margin-bottom:8px">${q+1>=N?'🎯 ¡Al reto!':'Siguiente ➜'}</button>`);
            $('nq').onclick=()=>{q++;ask()};if(q+1>=N)unlock(1)}
          else{b.classList.add('bad');b.disabled=true;$('info').innerHTML+=`<div class=story style="margin-top:6px">❌ Casi… ¡prueba otra vez!</div>`}})}
      function fin(){clear();lall().forEach(g=>g.classList.add('on'));
        $('info').innerHTML=`<div class=story><h3 style="margin:0">🎉 ¡Ya conoces todas las piezas!</h3>Ahora el reto: te daré una descripción en inglés y tú eliges la pieza.</div>${legend()}`;
        $('ctl').innerHTML='<button id=nx>🎯 Ir al reto ➜</button>';$('nx').onclick=()=>go(1);unlock(1)}
      ask();
    }

    /* FASE 2 · RETO: descripción SOLO en inglés a la izquierda; se toca la pieza en el dibujo de la derecha */
    function quiz(){
      const order=shuffle(parts),N=order.length;let q=0,first=0;
      function ask(){
        if(q>=N)return end();
        const p=order[q];let wrong=0,locked=false;clear();
        $('info').innerHTML=`<div class=qn>Pregunta ${q+1} de ${N}</div><div class=qt>${esc(p.clue_en||p.body_en)}</div><div class=qes>👉 Toca la pieza en el dibujo.</div>`;
        $('ctl').innerHTML=`<div class=cnt id=sc>⭐ ${first} a la primera</div>`;
        parts.forEach(pp=>W[pp.id].onclick=ev=>{ev.stopPropagation();if(locked)return;
          if(pp.id===p.id){locked=true;if(!wrong)first++;W[p.id].classList.remove('glow');W[p.id].classList.add('good');Lb[p.id].classList.add('on');aim(null);
            $('info').innerHTML=`<div class=qn>Pregunta ${q+1} de ${N}</div><div class=qt>${esc(p.clue_en||p.body_en)}</div><div class=story style="margin-top:8px">✅ <b>¡Muy bien!</b> Es <b>${esc(p.name)}</b>${S.badge?' (nº '+(parts.indexOf(p)+1)+')':''}.</div><div class=qes style="margin-top:8px">🇪🇸 <b>${esc(p.clue_es||p.body_es)}</b></div><div class=story style="margin-top:6px;background:#ecfeff;padding:6px 10px;border-radius:10px">💡 ${esc(p.why_es||p.text_es)}</div>`;
            $('ctl').innerHTML=`<button id=nq>${q+1>=N?'Ver resultado ➜':'Siguiente ➜'}</button>`;$('nq').onclick=()=>{q++;ask()}}
          else{wrong++;W[pp.id].classList.add('shake');setTimeout(()=>W[pp.id].classList.remove('shake'),450);
            if(wrong===1)$('info').insertAdjacentHTML('beforeend','<div class=story style="margin-top:6px">❌ Casi… ¡prueba otra vez!</div>');
            if(wrong>=2){W[p.id].classList.add('glow');aim(p.id)}}})}
      function end(){SC.quiz=first/N;if(ph<PH.length-1)more(first,N);else finish(first,N)}
      ask();
    }
    go(0);
  }
  map();
}
({reveal:gReveal,memory:gMem,drag:gDrag,scheme:scheme,journey:journey})[MODE]();
</script></body></html>"""


def render(mode, data, cfg, figs, height):
    payload = json.dumps({"data": data, "cfg": cfg, "figs": figs}, ensure_ascii=False).replace("</", "<\\/")
    page = HTML.replace("__PAYLOAD__", payload).replace("__MODE__", mode)
    if hasattr(st, "iframe"): st.iframe(page, height=height)  # Streamlit >= 1.6x (sustituye a components.html)
    else:
        import streamlit.components.v1 as components; components.html(page, height=height, scrolling=True)


# ───────────────────────── pestañas ─────────────────────────
def tab_resumen(slug, m, pack, lvl):
    part = m["plan"][str(lvl)]
    st.caption("En este nivel estudias: " + " · ".join(part["sections"]))
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown(pack["summary_md"])
        was = m["progress"].get(str(lvl), {}).get("read", False)
        if st.checkbox("✅ He leído el resumen", value=was, key=f"read_{slug}_{lvl}") != was: set_progress(slug, lvl, read=not was)
    with c2:
        if pack.get("vocab"):
            st.markdown("**📚 Vocabulary**")
            st.markdown("| English | Español |\n|---|---|\n" + "\n".join(f"| **{v['en']}** | {v.get('es', '')} |" for v in pack["vocab"]))
    figs = m["base"]["figures"]
    if figs:
        with st.expander("🖼️ Imágenes del tema"):
            cols = st.columns(4)
            for i, f in enumerate(figs): cols[i % 4].image(str(tdir(slug) / "figs" / f["file"]), caption=f["label"])
    st.divider(); st.markdown("#### 🤖 ¿Algo no se entiende? Pregúntale a Profe (en español)")
    q = st.text_input("Tu pregunta", key=f"q_{slug}_{lvl}", placeholder="Ej: ¿Qué diferencia hay entre evaporación y condensación?")
    if st.button("Preguntar", key=f"ask_{slug}_{lvl}") and q.strip():
        with st.spinner("Profe está pensando…"):
            try: st.chat_message("assistant").write(tutor(slug, lvl, q))
            except Exception as e: st.error(f"Gemini no respondió: {e}")


def tab_explora_foto(slug, m, lvl):
    st.caption("📐 Sube tu dibujo en **SVG** (por ejemplo la célula). La IA lo mira, reconoce las piezas y monta el viaje **en inglés**. El dibujo se muestra tal cual: sin texto encima y sin taparlo.")
    ek, nk = f"err_foto_{slug}_{lvl}", f"upn_{slug}_{lvl}"; u = st.session_state.get(nk, 0)
    up = st.file_uploader("Tu dibujo (SVG)", type=["svg"], key=f"upsvg_{slug}_{lvl}_{u}")
    quitar = st.checkbox("🧹 Quitar del dibujo los números, textos y flechas", value=True, key=f"quitar_{slug}_{lvl}")
    with st.expander("Opcional: captura PNG/JPG del mismo dibujo (solo si la app no puede leer el SVG)"):
        ref = st.file_uploader("Captura del dibujo completo", type=["png", "jpg", "jpeg"], key=f"upref_{slug}_{lvl}_{u}")
    if up:
        try:
            clean, _o, W, H, nq = svg_prepare(up.getvalue(), quitar)
            c1, c2 = st.columns([1, 2])
            c1.image(clean, width=300); c1.caption(f"Así se verá ({nq} elementos quitados)" if quitar else "Así se verá")
            if c2.button("🪄 Crear la parada con este dibujo", type="primary", key=f"mkfoto_{slug}_{lvl}"):
                with st.spinner("Gemini está mirando el dibujo y preparando la parada… (tarda un poco)"):
                    try: build_photo_stop(slug, lvl, up.getvalue(), quitar, ref.getvalue() if ref else None); st.session_state.pop(ek, None); st.session_state[nk] = u + 1
                    except Exception as e: st.session_state[ek] = str(e)
                st.rerun()
        except Exception as e: st.error(f"No puedo leer ese SVG: {e}")
    if st.session_state.get(ek): st.error(st.session_state[ek])
    stops = photo_stops(slug, load(slug), lvl)
    if not stops: st.info("Todavía no hay ninguna parada. Sube tu dibujo para empezar el viaje."); return
    render("journey", {"stops": stops, "key": f"{slug}_{lvl}_foto", "level": lvl}, None, [], 680)
    with st.expander("🗑️ Quitar una parada"):
        k = st.selectbox("Parada", range(len(stops)), format_func=lambda i: f"{i + 1}. {stops[i]['title']}", key=f"delfoto_{slug}_{lvl}")
        if st.button("Quitar", key=f"delfotob_{slug}_{lvl}"):
            mm = load(slug); gone = mm["photos"][str(lvl)].pop(k); save(slug, mm)
            try: (tdir(slug) / "photos" / gone["photo"]).unlink()
            except Exception: pass
            st.rerun()


def tab_explora(slug, m, pack, lvl):
    modo = st.radio("¿Con qué dibujo quieres hacer el viaje?", ["📐 Con mi propio dibujo (SVG)", "🎨 Con dibujos automáticos"], horizontal=True, key=f"modo_{slug}_{lvl}")
    if modo.startswith("📐"): return tab_explora_foto(slug, m, lvl)
    groups = journey_groups(m["plan"][str(lvl)]); J = journey_ready(pack); ek = f"err_ex_{slug}_{lvl}"
    if len(J) < len(groups) and not st.session_state.get(ek):
        bar = st.progress(0.0, text="🎨 Dibujando el viaje… (una parada por apartado; tarda un poco, déjalo trabajar)")
        try: build_journey(slug, lvl, bar)
        except Exception as e: st.session_state[ek] = str(e)
        st.rerun()
    if st.session_state.get(ek):
        st.error(st.session_state[ek])
        if st.button("🔁 Reintentar dibujar las paradas que faltan", key=ek + "_b"): st.session_state.pop(ek); st.rerun()
        if not J: return
    st.caption("🗺️ Una parada por cada apartado del nivel. En cada una: 📖 historia → 🔍 descubres las piezas → 🧩 las desmontas y las unes con su trabajo → 🎬 las ves en acción → 🎯 reto.")
    stops = [{**(J.get(str(i)) or {}), "title": g["title"], "ready": str(i) in J} for i, g in enumerate(groups)]
    render("journey", {"stops": stops, "key": f"{slug}_{lvl}", "level": lvl}, None, [], 680)
    with st.expander("🔧 Redibujar una parada"):
        ready = sorted(int(k) for k in J)
        i = st.selectbox("Parada", ready, format_func=lambda i: f"{i + 1}. {groups[i]['title']}", key=f"rd_{slug}_{lvl}")
        if st.button("🔁 Redibujar esta parada", key=f"rdb_{slug}_{lvl}"):
            with st.spinner("Gemini está dibujando…"):
                try: build_stop(slug, lvl, i); st.rerun()
                except Exception as e: st.error(str(e))


def tab_juegos(slug, m, lvl):
    figs = load_figs(slug, m["base_ts"])
    if len(figs) < 4:
        st.warning("Gemini encontró menos de 4 imágenes útiles en este tema. Prueba con «Preparar de nuevo» en el menú lateral."); return
    g = st.radio("Elige juego", ["🔍 Adivina la imagen", "🃏 Memory", "🎯 Arrastra la palabra"], horizontal=True, key=f"g_{slug}")
    mode, h = {"🔍": ("reveal", 800), "🃏": ("memory", 600), "🎯": ("drag", 640)}[g.split()[0]]
    render(mode, None, LEVELS[lvl]["game"], figs, h)


def tab_esquemas(slug, m, pack, lvl):
    sch = pack["schemes"]
    st.caption(f"Hay {len(sch)} esquemas y entre todos cubren lo que estudias en este nivel. Arrastra las palabras de colores a su hueco.")
    i = st.selectbox("Esquema", range(len(sch)), format_func=lambda i: f"{i + 1}. {sch[i]['title']}", key=f"sc_{slug}_{lvl}") if len(sch) > 1 else 0
    render("scheme", sch[i], None, [], (240 + 62 * len(sch[i]["rows"])) if sch[i].get("layout") == "tree" else 260 + 80 * len(sch[i]["rows"]))
    if st.button("✨ Crear otro esquema nuevo con IA", key=f"newsc_{slug}_{lvl}"):
        with st.spinner("Gemini está diseñando otro esquema…"):
            try:
                part = m["plan"][str(lvl)]
                if lvl >= 2:
                    fmt = (f"English text, Spanish hints. It is a MULTI-LEVEL tree of {TREE_ROWS[lvl]} nodes in reading order (parent before its children), each with depth d (0 = title, 1 = branches, 2 = sub-branches{', 3 = details of the process' if lvl == 3 else ''}); "
                           f"about 70% of the nodes contain one ___ blank. Return ONLY JSON {{\"title\":\"\",\"instructions_es\":\"\",\"layout\":\"tree\",\"bank\":[],\"rows\":[{{\"d\":0,\"text\":\"\",\"answer\":\"\",\"hint_es\":\"\"}}]}}")
                else:
                    fmt = (f"English text, Spanish hints. Return ONLY JSON {{\"title\":\"\",\"instructions_es\":\"\",\"layout\":\"flow or list\",\"bank\":[],\"rows\":[{{\"text\":\"sentence with one ___\",\"answer\":\"\",\"hint_es\":\"\"}}]}} "
                           f"with {LEVELS[lvl]['rows']} rows.")
                pr = (f"Create ONE new fill-in-the-blanks scheme (different from: {', '.join(s['title'] for s in sch)}) for a 10-year-old Spanish girl with basic English, level {lvl}. "
                      + fmt + "\n\nMATERIAL:\n" + part["text"][:9000])
                new = clean_scheme(ask_json(pr, 4000, lambda d: d.get("rows")))
                m["levels"][str(lvl)]["schemes"].append(new); save(slug, m); st.rerun()
            except Exception as e:
                st.error(f"No pude crear el esquema: {e}")


def new_state(order, run=0): return {"order": order, "i": 0, "score": 0, "ans": None, "res": {}, "help": "", "run": run, "saved": False}
def nxt(ks): S = st.session_state[ks]; S["i"] += 1; S["ans"] = None; S["help"] = ""
def restart(ks, order): st.session_state[ks] = new_state(order, st.session_state[ks]["run"] + 1)


def pick(ks, qid, j):
    S = st.session_state[ks]; ok = j == st.session_state[ks + "_q"][qid]["answer_idx"]
    S["ans"] = j; S["help"] = ""; S["res"][qid] = ok; S["score"] += int(ok)


def tab_test(slug, m, pack, lvl):
    qs = pack["test"]; pend = m["progress"].get(str(lvl), {}).get("pending", {}); mode = "level"
    st.caption(f"Banco de {len(qs)} preguntas. Cada intento tiene {min(INTENTO, len(qs))} preguntas de todos los apartados. "
               f"Necesitas un {int(PASA * 100)} % de aciertos para aprobar el nivel y pasar al siguiente.")
    if pend:
        ch = st.radio("¿Qué quieres hacer?", ["📝 Test del nivel", f"🎯 Practicar las {len(pend)} falladas"], horizontal=True, key=f"modo_{slug}_{lvl}")
        mode = "level" if ch.startswith("📝") else "practice"
    ks = f"t_{slug}_{lvl}_{mode}"; st.session_state[ks + "_q"] = qs
    mk = (lambda: draw(qs, INTENTO)) if mode == "level" else (lambda: [i for i, q in enumerate(qs) if q["q"] in pend])
    if ks not in st.session_state: st.session_state[ks] = new_state(mk())
    S = st.session_state[ks]; n = len(S["order"])
    if n == 0:
        st.info("No quedan preguntas pendientes. ¡Bien!"); return
    if S["i"] >= n:
        if not S["saved"]:
            S["saved"] = True; ok_level, newly = record_attempt(slug, lvl, qs, S, mode)
            if newly: st.session_state["just_passed"] = (slug, lvl, S["score"], n); st.rerun()
        pct = S["score"] / n
        (st.success if pct >= PASA else st.warning)(f"### Has acertado {S['score']} de {n}  {'⭐' * round(5 * pct)}")
        if mode == "level" and pct < PASA: st.write(f"Necesitabas {math.ceil(PASA * n)} aciertos. Repasa el resumen y los esquemas, mira tus errores en **📊 Resultados** y vuelve a intentarlo.")
        if mode == "level" and pct >= PASA: st.write("Ya tenías este nivel aprobado. ¡Sigue así!")
        c1, c2 = st.columns(2)
        c1.button("🔁 Otro intento" if mode == "level" else "🔁 Volver a practicar", on_click=restart, args=(ks, mk()), use_container_width=True)
        if S["saved"] and any(not v for v in S["res"].values()) and mode == "level":
            c2.button("🎯 Practicar solo las falladas", on_click=restart, args=(ks, [q for q, v in S["res"].items() if not v]), use_container_width=True)
        return
    qid = S["order"][S["i"]]; q = qs[qid]
    st.progress(S["i"] / n, text=f"Pregunta {S['i'] + 1} de {n}")
    st.caption(f"Apartado: {q['section']}"); st.subheader(q["q"])
    for j, o in enumerate(q["options"]):
        st.button(f"{'ABCD'[j]})  {o}", key=f"{ks}_{S['run']}_{S['i']}_{j}", on_click=pick, args=(ks, qid, j), disabled=S["ans"] is not None, use_container_width=True)
    if S["ans"] is not None:
        if S["ans"] == q["answer_idx"]: st.success("🎉 ¡Muy bien!")
        else:
            st.error(f"❌ La respuesta correcta era: **{q['options'][q['answer_idx']]}**"); st.info("💡 " + q["explain_es"])
            if st.button("🤖 Explícamelo con otras palabras", key=f"hlp_{ks}_{S['run']}_{S['i']}"):
                with st.spinner("Profe lo explica…"):
                    try: S["help"] = tutor(slug, lvl, f"No entiendo por qué la respuesta correcta es «{q['options'][q['answer_idx']]}». Explícamelo más fácil, con un ejemplo.", f"Pregunta: {q['q']}\nOpciones: {q['options']}")
                    except Exception as e: S["help"] = f"Gemini no respondió: {e}"
            if S["help"]: st.chat_message("assistant").write(S["help"])
        st.button("Siguiente ➜" if S["i"] + 1 < n else "Ver mi resultado ➜", on_click=nxt, args=(ks,), type="primary")


def tab_resultados(slug, m, lvl_now):
    bad_all, hay = {}, False
    for n in (1, 2, 3):
        pr = m["progress"].get(str(n), {}); att = pr.get("attempts", [])
        estado = "✅ aprobado" if passed(m, n) else ("▶️ en curso" if n == lvl_now else "🔒 bloqueado")
        st.markdown(f"### {LEVELS[n]['est']} Nivel {n} · {estado}")
        if not att: st.caption("Todavía no has hecho el test de este nivel."); continue
        hay = True
        st.markdown("\n".join(f"- {a['ts']} · {'test' if a['mode'] == 'level' else 'práctica'}: **{a['score']}/{a['total']}** ({round(100 * a['score'] / max(a['total'], 1))} %) {'✅' if a['passed'] else ''}" for a in att[-5:]))
        agg = {}
        for a in att[-6:]:
            for s, (ok, t) in a["sections"].items(): x = agg.setdefault(s, [0, 0]); x[0] += ok; x[1] += t
        good = {s: v for s, v in agg.items() if v[1] and v[0] / v[1] >= PASA}; bad = {s: v for s, v in agg.items() if v[1] and v[0] / v[1] < PASA}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Lo que ya sabes**")
            for s, (ok, t) in good.items(): st.write(f"• {s} — {round(100 * ok / t)} %")
            if not good: st.caption("Aún nada por encima del 80 %. ¡Ánimo!")
        with c2:
            st.markdown("**🔁 Tienes que repasar**")
            for s, (ok, t) in bad.items(): st.write(f"• {s} — {round(100 * ok / t)} %"); bad_all[f"Nivel {n}: {s}"] = round(100 * ok / t)
            if not bad: st.caption("¡Nada pendiente!")
        pend = pr.get("pending", {})
        if pend:
            with st.expander(f"❌ Preguntas que sigues fallando ({len(pend)})"):
                for p in pend.values(): st.markdown(f"**{p['q']}**  \n✔️ {p['a']}  \n💡 {p['e']}  \n*({p['s']})*")
    if hay and bad_all:
        if st.button("🤖 Pídele a Profe un plan de repaso", key=f"plan_{slug}"):
            with st.spinner("Profe está preparando tu plan…"):
                try: st.session_state[f"plan_{slug}"] = tutor(slug, None, "Estos son los apartados que me cuestan (porcentaje de aciertos): " + "; ".join(f"{k} ({v} %)" for k, v in bad_all.items()) + ". Dame un plan de repaso corto y animado, en 4 pasos.")
                except Exception as e: st.error(str(e))
        if st.session_state.get(f"plan_{slug}"): st.chat_message("assistant").write(st.session_state[f"plan_{slug}"])
    elif hay: st.success("🌟 No tienes nada pendiente de repasar. ¡Enhorabuena!")


# ───────────────────────── app ─────────────────────────
def level_ready(m, n): return all(k in m["levels"].get(str(n), {}) for k in ("summary_md", "schemes", "test"))


def run_step(key, aviso, fn, *args):
    """Ejecuta la parte de IA automáticamente al entrar. Si falla, muestra el error y un botón de reintento (sin bucles)."""
    if st.session_state.get(key):
        st.error(st.session_state[key])
        if st.button("🔁 Reintentar", key=key + "_b"): st.session_state.pop(key); st.rerun()
        st.stop()
    st.info(aviso)
    with st.spinner("Trabajando con la IA…"):
        try: fn(*args)
        except Exception as e: st.session_state[key] = str(e)
    st.rerun()


st.markdown("<style>.block-container{padding-top:1.5rem} div.stButton>button{border-radius:12px}</style>", unsafe_allow_html=True)
if not api_key():
    st.error("Falta la clave de Gemini: crea `.streamlit/secrets.toml` junto a este archivo con `GOOGLE_API_KEY = \"tu_clave\"` (la misma que usa IAG.py).")

with st.sidebar:
    st.title("🎓 Study Buddy")
    slugs = topics()
    cur = st.radio("📚 Tus temas", slugs, format_func=lambda s: load(s)["name"], key="cur") if slugs else None
    with st.expander("➕ Subir un tema nuevo", expanded=not slugs):
        nm = st.text_input("Nombre del tema", placeholder="Natural – Cells")
        zf = st.file_uploader("ZIP con las imágenes (ordenadas por nombre: 001.jpg, 002.jpg…)", type=["zip"])
        if st.button("Crear tema", disabled=not (nm.strip() and zf)):
            try: import_zip(nm.strip(), zf); st.session_state.pop("cur", None); st.rerun()
            except Exception as e: st.error(str(e))

if not cur:
    st.header("¡Bienvenida! 👋")
    st.write("Sube un tema desde el menú de la izquierda (un ZIP con las fotos de las páginas). Gemini creará 3 niveles que juntos cubren todo el tema; se pasa al siguiente aprobando el test.")
    st.stop()

m = load(cur)
if m["base"] and m["base"].get("v") != VERSION:  # temas creados con la versión anterior: hay que releerlos con el método nuevo
    m["base"] = None; m["levels"] = {}; m["progress"] = {}; save(cur, m); st.toast("Este tema se creó con la versión anterior: hay que prepararlo de nuevo.")
lvl_now = cur_level(m); lvl = st.session_state.get(f"lvl_over_{cur}") or lvl_now

with st.sidebar:
    if m["base"]:
        st.markdown("**🏆 Tu camino**")
        for n in (1, 2, 3):
            st.write(("✅" if passed(m, n) else "▶️" if n == lvl_now else "🔒") + f" Nivel {n} · {LEVELS[n]['nombre']}")
    with st.expander("⚙️ Gestionar este tema"):
        if st.button("🔄 Preparar de nuevo (relee las imágenes)"):
            m["base"] = None; m["levels"] = {}; m["progress"] = {}; save(cur, m); st.rerun()
        if m["base"] and st.button(f"🔄 Regenerar nivel {lvl}"):
            m["levels"].pop(str(lvl), None); save(cur, m); st.rerun()
        if st.button("🗑️ Borrar tema"):
            shutil.rmtree(tdir(cur)); st.session_state.pop("cur", None); st.rerun()
    if m["base"]:
        with st.expander("👨‍👩‍👧 Modo padres"):
            ov = st.selectbox("Ver el nivel…", ["Automático (según progreso)", 1, 2, 3], key=f"ovsel_{cur}")
            st.session_state[f"lvl_over_{cur}"] = None if ov == "Automático (según progreso)" else ov
            if st.button("♻️ Borrar el progreso de este tema"):
                m["progress"] = {}; save(cur, m); st.session_state.pop(f"lvl_over_{cur}", None); st.rerun()

st.title(m["name"])
if not m["base"]:
    run_step(f"err_base_{cur}", f"🪄 Preparando el tema con IA ({m['npages']} páginas). Tarda unos minutos: Gemini lee todas las páginas, reparte el contenido en 3 niveles y recorta las imágenes. "
             "Déjalo trabajar y no pulses nada mientras tanto.", build_base, cur)

jp = st.session_state.get("just_passed")
if jp and jp[0] == cur:
    _, l, sc, n = jp
    if st.session_state.get("jp_shown") != jp: st.balloons(); st.session_state["jp_shown"] = jp
    st.success(f"### 🎉 ¡Has aprobado el nivel {l}!  {sc}/{n}")
    if l < 3 and not level_ready(m, l + 1):
        run_step(f"err_lvl_{cur}_{l + 1}", f"🪄 Estoy preparando el nivel {l + 1}. Tarda unos minutos; no pulses nada.", build_level, cur, l + 1)
    st.write("🏆 ¡Has completado todo el tema! Puedes seguir repasando cuando quieras." if l >= 3 else f"Ya puedes pasar al **nivel {l + 1}**.")
    st.button(f"➡️ Empezar el nivel {l + 1}" if l < 3 else "Ver mis resultados", type="primary", on_click=lambda: st.session_state.pop("just_passed", None))
    st.stop()

st.caption(f"{LEVELS[lvl]['est']} Nivel {lvl} · {LEVELS[lvl]['nombre']}" + (" · tema completado 🏆" if all(passed(m, k) for k in (1, 2, 3)) else ""))
pack = m["levels"].get(str(lvl), {})
if not level_ready(m, lvl):
    run_step(f"err_lvl_{cur}_{lvl}", f"🪄 Creando el nivel {lvl} (resumen, esquemas, escenas y preguntas). Tarda unos minutos; no pulses nada mientras tanto. "
             f"Este nivel cubre: {' · '.join(m['plan'][str(lvl)]['sections'])}", build_level, cur, lvl)

t1, t2, t3, t4, t5, t6 = st.tabs(["📖 Resumen", "🔬 Explora", "🎮 Juegos", "🧩 Esquemas", "✅ Repasa", "📊 Resultados"])
with t1: tab_resumen(cur, m, pack, lvl)
with t2: tab_explora(cur, m, pack, lvl)
with t3: tab_juegos(cur, m, lvl)
with t4: tab_esquemas(cur, m, pack, lvl)
with t5: tab_test(cur, m, pack, lvl)
with t6: tab_resultados(cur, m, lvl_now)
