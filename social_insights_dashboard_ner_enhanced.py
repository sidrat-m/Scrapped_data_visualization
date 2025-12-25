# Social Insights Dashboard (NLP Enhanced) — patched to avoid KeyError and be defensive
# - Ensures fig_hashtag (and other figs) are always defined in make_figures_from_post
# - Makes callback defensive when building grid_children (uses figs.get with safe defaults)
# - Clears load_json_file cache when Reload is clicked and also before loading a file
#
# Save/overwrite this file and run:
#   python social_insights_dashboard.py
#
# Optional deps (for wordcloud/NER): wordcloud, matplotlib, spacy, xx_ent_wiki_sm or en_core_web_sm

import os
import glob
import json
import re
import unicodedata
import traceback
from functools import lru_cache
from collections import Counter, OrderedDict
from datetime import datetime
import io
import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Optional libraries
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

try:
    import spacy
    # prefer multilingual small model if installed, fallback to en model
    try:
        NER_NLP = spacy.load("xx_ent_wiki_sm")
    except Exception:
        try:
            NER_NLP = spacy.load("en_core_web_sm")
        except Exception:
            NER_NLP = None
    SPACY_AVAILABLE = NER_NLP is not None
except Exception:
    SPACY_AVAILABLE = False
    NER_NLP = None

from dash import Dash, dcc, html, Input, Output

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRAPED_ROOT = os.path.join(BASE_DIR, "ScrappedData")  # folder containing candidate subfolders
PORT = 8050

# regex patterns
_RE_LAUGH = re.compile(r"\b(haha|hehe|lol|rofl|lmao|হা|হাহা|হাহাহা|হা-হা)\b", re.IGNORECASE)
_RE_ANGER = re.compile(r"😠|😡|👿")
RE_BANGLA = re.compile(r'[\u0980-\u09FF]+')
RE_WORD = re.compile(r"[\u0980-\u09FF]+|[A-Za-z0-9']+")

# -------------------------
# Helpers: filesystem
# -------------------------
def list_candidates(root=SCRAPED_ROOT):
    if not os.path.isdir(root):
        return []
    entries = sorted(os.listdir(root))
    folders = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    return folders

def list_json_files(candidate_folder):
    if not candidate_folder:
        return []
    folder = os.path.join(SCRAPED_ROOT, candidate_folder)
    if not os.path.isdir(folder):
        return []
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    return [os.path.basename(p) for p in files]

# -------------------------
# Helpers: JSON parsing & normalization
# -------------------------
def normalize_text(s):
    return unicodedata.normalize("NFC", str(s)).strip()

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur

@lru_cache(maxsize=256)
def load_json_file(candidate_folder, filename):
    path = os.path.join(SCRAPED_ROOT, candidate_folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def parse_post_json(obj):
    post_section = obj.get("post", {}) if isinstance(obj, dict) else {}
    raw = post_section.get("raw_data", {}) or {}
    content = (post_section.get("content") or safe_get(raw, "content") or "").strip()
    timestamp = safe_get(raw, "date_posted") or safe_get(post_section, "timestamp") or safe_get(post_section, "date")
    dt = None
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except Exception:
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = None

    likes = safe_get(post_section, "likes") or safe_get(raw, "likes") or safe_get(raw, "num_likes") or 0
    shares = safe_get(post_section, "shares") or safe_get(raw, "num_shares") or 0
    comments_count = safe_get(post_section, "comments_count") or safe_get(raw, "num_comments") or 0

    reaction_list = safe_get(raw, "num_likes_type") or safe_get(raw, "reactions") or []
    reaction_counter = Counter()
    if isinstance(reaction_list, list):
        for r in reaction_list:
            try:
                reaction_counter[r.get("type", "Other")] += int(r.get("num", 0))
            except Exception:
                pass

    attachments = safe_get(raw, "attachments") or safe_get(post_section, "attachments") or []
    hashtags = safe_get(raw, "hashtags") or []

    def to_int_safe(x):
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return 0

    post_record = {
        "content": normalize_text(content),
        "datetime": dt,
        "likes": to_int_safe(likes),
        "shares": to_int_safe(shares),
        "comments_count": to_int_safe(comments_count),
        "reaction_breakdown": dict(reaction_counter),
        "attachments": attachments,
        "hashtags": hashtags if isinstance(hashtags, list) else [],
        "raw": raw,
    }

    comments = obj.get("comments") or []
    comments_records = []
    for c in comments:
        text = normalize_text(c.get("comment_text") or "")
        sentiment = c.get("sentiment") or c.get("tone") or "Unknown"
        emotion = c.get("emotion") or "Unknown"
        tone = c.get("tone") or "Unknown"
        intent = c.get("intent") or "Unknown"
        likes_c = c.get("likes_count") or c.get("likes") or 0
        try:
            likes_c = int(likes_c)
        except Exception:
            likes_c = 0
        dtc = c.get("date_created") or c.get("created_at") or None
        dtc_parsed = None
        if dtc:
            try:
                dtc_parsed = datetime.fromisoformat(dtc.replace("Z", "+00:00"))
            except Exception:
                try:
                    dtc_parsed = datetime.strptime(dtc, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dtc_parsed = None
        comments_records.append({
            "text": text,
            "sentiment": sentiment,
            "emotion": emotion,
            "tone": tone,
            "intent": intent,
            "likes": likes_c,
            "date": dtc_parsed,
            "user_name": c.get("user_name") or c.get("user") or "",
            "raw": c,
        })

    return post_record, comments_records

# -------------------------
# Text analysis helpers
# -------------------------
def tokenize(text):
    if text is None:
        return []
    text = normalize_text(text)
    return [w for w in RE_WORD.findall(text) if len(w) > 0]

def compute_text_stats(content, comments_texts):
    tokens = []
    if content:
        tokens.extend(tokenize(content))
    for t in comments_texts:
        tokens.extend(tokenize(t))
    tokens = [t for t in tokens if len(t.strip()) > 0]
    total_words = len(tokens)
    unique_words = len(set(tokens))
    avg_word_length = (sum(len(t) for t in tokens) / total_words) if total_words else 0.0
    word_counts = Counter(tokens)
    top_words = word_counts.most_common(50)
    bigrams = Counter()
    for i in range(len(tokens)-1):
        bigrams[(tokens[i], tokens[i+1])] += 1
    top_bigrams = [(" ".join(k), v) for k, v in bigrams.most_common(30)]
    return total_words, unique_words, avg_word_length, top_words, top_bigrams

def generate_wordcloud_image(word_counts, width=600, height=300, background_color="white"):
    if not WORDCLOUD_AVAILABLE or not word_counts:
        return None
    try:
        wc = WordCloud(width=width, height=height, background_color=background_color, collocations=False)
        wc.generate_from_frequencies(word_counts)
        buf = io.BytesIO()
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        return "data:image/png;base64," + img_b64
    except Exception:
        return None

def extract_named_entities(texts):
    entities = Counter()
    if not texts or all(not t for t in texts):
        return entities

    joined = "\n".join([t for t in texts if t])
    if SPACY_AVAILABLE and NER_NLP is not None:
        try:
            doc = NER_NLP(joined)
            for ent in doc.ents:
                if ent.label_.lower() in ("person", "per", "person_name", "org", "place", "gpe", "loc"):
                    entities[ent.text.strip()] += 1
            if not entities:
                for ent in doc.ents:
                    if ent.label_.lower() in ("person",):
                        entities[ent.text.strip()] += 1
            return entities
        except Exception:
            pass

    cap_seq = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', joined)
    for name in cap_seq:
        entities[name.strip()] += 1
    tokens = [t for t in RE_WORD.findall(joined) if RE_BANGLA.search(t)]
    bigrams = Counter()
    for i in range(len(tokens)-1):
        bigrams[tokens[i] + " " + tokens[i+1]] += 1
    for k, v in bigrams.most_common(30):
        if v > 1:
            entities[k] += v
    return entities

# -------------------------
# Figures builder (enhanced + safe fallbacks)
# -------------------------
def make_figures_from_post(post_record, comments_records):
    content = post_record.get("content", "") or ""
    likes = int(post_record.get("likes", 0) or 0)
    shares = int(post_record.get("shares", 0) or 0)
    comments_count = int(post_record.get("comments_count", 0) or 0)
    hashtags_field = post_record.get("hashtags", []) or []

    df_post = pd.DataFrame([{"content": content, "likes": likes, "shares": shares, "comments_count": comments_count, "hashtags": hashtags_field}])
    if comments_records and isinstance(comments_records, (list, tuple)) and len(comments_records) > 0:
        df_comments = pd.DataFrame(comments_records)
        for c in ["text", "sentiment", "emotion", "tone", "intent", "likes", "date", "user_name"]:
            if c not in df_comments.columns:
                df_comments[c] = None
    else:
        df_comments = pd.DataFrame(columns=["text", "sentiment", "emotion", "tone", "intent", "likes", "date", "user_name"])

    reaction_counter = Counter(post_record.get("reaction_breakdown") or {})
    if not reaction_counter:
        reaction_counter = Counter({"Like": likes})
    rx_names = list(reaction_counter.keys())
    rx_vals = list(reaction_counter.values())
    fig_rx_pie = px.pie(values=rx_vals, names=rx_names, title="Reaction Breakdown (post)") if rx_vals else px.scatter(title="Reaction Breakdown (no data)")
    fig_rx_bar = px.bar(x=rx_names, y=rx_vals, labels={"x":"Reaction","y":"Count"}, title="Reaction Types") if rx_vals else px.scatter(title="Reaction Types")

    # sentiment/emotion/tone/intent pies
    if not df_comments.empty and "sentiment" in df_comments.columns:
        sent_counts = df_comments["sentiment"].fillna("Unknown").astype(str).value_counts()
        fig_sentiment_pie = px.pie(values=sent_counts.values, names=sent_counts.index, title="Comment Sentiment (raw)", hole=0.3)
        fig_sentiment_pie.update_traces(textinfo='percent+label')
    else:
        fig_sentiment_pie = px.pie(values=[1], names=["No comments"], title="Comment Sentiment (none)")

    if not df_comments.empty and "emotion" in df_comments.columns:
        emo_counts = df_comments["emotion"].fillna("Unknown").astype(str).value_counts()
        fig_emotion_pie = px.pie(values=emo_counts.values, names=emo_counts.index, title="Comment Emotion (raw)", hole=0.3)
        fig_emotion_pie.update_traces(textinfo='percent+label')
    else:
        fig_emotion_pie = px.pie(values=[1], names=["No comments"], title="Comment Emotion (none)")

    if not df_comments.empty and "tone" in df_comments.columns:
        tone_counts = df_comments["tone"].fillna("Unknown").astype(str).value_counts()
        fig_tone_pie = px.pie(values=tone_counts.values, names=tone_counts.index, title="Comment Tone (raw)", hole=0.3)
        fig_tone_pie.update_traces(textinfo='percent+label')
    else:
        fig_tone_pie = px.pie(values=[1], names=["No comments"], title="Comment Tone (none)")

    if not df_comments.empty and "intent" in df_comments.columns:
        intent_counts = df_comments["intent"].fillna("Unknown").astype(str).value_counts()
        fig_intent_pie = px.pie(values=intent_counts.values, names=intent_counts.index, title="Comment Intent (raw)", hole=0.3)
        fig_intent_pie.update_traces(textinfo='percent+label')
    else:
        fig_intent_pie = px.pie(values=[1], names=["No comments"], title="Comment Intent (none)")

    # text stats
    comments_texts = df_comments["text"].fillna("").tolist() if not df_comments.empty else []
    total_words, unique_words, avg_word_len, top_words, top_bigrams = compute_text_stats(content, comments_texts)

    # top words bar
    if top_words:
        words, counts = zip(*top_words[:30])
        fig_top_words = px.bar(x=list(words), y=list(counts), labels={"x":"Word","y":"Count"}, title="Top Words (post + comments)")
    else:
        fig_top_words = px.bar(title="Top Words (none)")

    # wordcloud image
    wc_img = generate_wordcloud_image(dict(top_words))
    fig_wordcloud = wc_img

    # NER entities
    ner_entities = extract_named_entities([content] + comments_texts)
    if ner_entities:
        ents, ent_counts = zip(*ner_entities.most_common(30))
        fig_ner_bar = px.bar(x=list(ents), y=list(ent_counts), labels={"x":"Entity","y":"Count"}, title="Named Entities (people/places/orgs - heuristics/spaCy)")
    else:
        fig_ner_bar = px.bar(title="Named Entities (none)")

    # engagement
    eng_labels = ["Likes", "Shares", "Comments"]
    eng_values = [likes, shares, comments_count]
    fig_eng_bar = px.bar(x=eng_labels, y=eng_values, labels={"x":"Metric","y":"Count"}, title="Engagement Metrics (post)")
    fig_eng_bar.update_traces(marker_color=["#1f77b4","#ff7f0e","#2ca02c"], text=eng_values, textposition="auto")
    maxv = max(eng_values) if eng_values else 1
    fig_eng_bar.update_yaxes(range=[0, maxv * 1.15])

    df_scatter = pd.DataFrame([{"post_len": len(content), "engagement": likes + shares + comments_count}])
    fig_postlen_engage = px.scatter(df_scatter, x="post_len", y="engagement", size="engagement", title="Post Length vs Engagement")

    fig_rx_time = px.bar(x=rx_names, y=rx_vals, title="Reactions by Type (single post)") if rx_names else px.scatter(title="Reactions by Type (none)")
    try:
        fig_polar = go.Figure(go.Scatterpolar(r=rx_vals + [rx_vals[0]] if rx_vals else [0], theta=rx_names + [rx_names[0]] if rx_names else ["None"], fill='toself'))
        fig_polar.update_layout(title="Reaction Mix (polar)")
    except Exception:
        fig_polar = px.line(title="Reaction Mix")
    if rx_names and rx_vals:
        df_rx = pd.DataFrame({"reaction": rx_names, "count": rx_vals})
        fig_rx_violin = px.violin(df_rx, y="count", x="reaction", box=True, title="Reaction Distribution")
    else:
        fig_rx_violin = px.violin(title="Reaction Distribution")

    # heatmap emotion vs sentiment
    if not df_comments.empty and "sentiment" in df_comments.columns and "emotion" in df_comments.columns:
        try:
            pivot = pd.crosstab(df_comments["emotion"].fillna("Unknown"), df_comments["sentiment"].fillna("Unknown"))
            fig_heat = ff.create_annotated_heatmap(z=pivot.values.tolist(), x=pivot.columns.tolist(), y=pivot.index.tolist(), colorscale='Viridis', showscale=True)
            fig_heat.update_layout(title="Emotion vs Sentiment (comments)")
        except Exception:
            fig_heat = px.bar(title="Emotion vs Sentiment (no data)")
    else:
        fig_heat = px.bar(title="Emotion vs Sentiment (no data)")

    # treemap
    if top_words:
        labels_tt = [w for w,_ in top_words[:30]]
        values_tt = [c for _,c in top_words[:30]]
        fig_treemap = px.treemap(names=labels_tt, values=values_tt, title="Word Frequency Treemap")
    else:
        fig_treemap = px.treemap(names=["none"], values=[1], title="Word Frequency Treemap")

    if not df_comments.empty:
        df_preview = df_comments.sort_values("likes", ascending=False).head(15)[["user_name","text","likes","sentiment","emotion","tone","intent"]].fillna("")
        comments_preview = df_preview.to_dict(orient="records")
    else:
        comments_preview = [{"user_name":"","text":"No comments","likes":0,"sentiment":"","emotion":"","tone":"","intent":""}]

    # Ensure commonly referenced figures exist to avoid KeyError in caller
    try:
        _ = fig_hashtag
    except NameError:
        # build fig_hashtag from hcounts if available, else empty
        try:
            hcounts = Counter([h.lower() for h in (hashtags_field or []) if h])
            if hcounts:
                hs, hv = zip(*hcounts.most_common(20))
                fig_hashtag = px.bar(x=list(hs), y=list(hv), labels={"x":"Hashtag","y":"Count"}, title="Top Hashtags")
            else:
                fig_hashtag = px.bar(title="Top Hashtags (none)")
        except Exception:
            fig_hashtag = px.bar(title="Top Hashtags (none)")

    try:
        _ = fig_word_freq
    except NameError:
        # alias top words bar as word_freq for backward compatibility
        fig_word_freq = fig_top_words

    # KPI numbers extended
    kpi = {
        "likes": likes,
        "shares": shares,
        "comments": comments_count,
        "total": likes + shares + comments_count,
        "total_words": total_words,
        "unique_words": unique_words,
        "avg_word_len": round(avg_word_len, 2)
    }

    return {
        "fig_rx_pie": fig_rx_pie,
        "fig_rx_bar": fig_rx_bar,
        "fig_sentiment_pie": fig_sentiment_pie,
        "fig_emotion_pie": fig_emotion_pie,
        "fig_tone_pie": fig_tone_pie,
        "fig_intent_pie": fig_intent_pie,
        "fig_top_words": fig_top_words,
        "fig_wordcloud_img": fig_wordcloud,
        "fig_ner_bar": fig_ner_bar,
        "fig_word_freq": fig_word_freq,
        "fig_hashtag": fig_hashtag,
        "fig_postlen_engage": fig_postlen_engage,
        "fig_rx_time": fig_rx_time,
        "fig_polar": fig_polar,
        "fig_rx_violin": fig_rx_violin,
        "fig_heat": fig_heat,
        "fig_treemap": fig_treemap,
        "fig_eng_bar": fig_eng_bar,
        "fig_kpi": kpi,
        "comments_preview": comments_preview,
    }

# -------------------------
# Dash app layout & callbacks
# -------------------------
app = Dash(__name__)
app.title = "Social Insights Dashboard (NLP Enhanced)"

candidates = list_candidates()
candidate_options = [{"label": c, "value": c} for c in candidates]
default_candidate = candidates[0] if candidates else None
file_options = [{"label": f, "value": f} for f in list_json_files(default_candidate)] if default_candidate else []

app.layout = html.Div([
    html.H1("Social Insights Dashboard", style={"textAlign":"center"}),

    html.Div([
        html.Div([
            html.Label("Select candidate folder:"),
            dcc.Dropdown(id="candidate-dropdown", options=candidate_options, value=default_candidate, clearable=False, style={"width":"100%"}),
            html.Br(),
            html.Label("Select JSON post file:"),
            dcc.Dropdown(id="file-dropdown", options=file_options, value=file_options[0]["value"] if file_options else None, clearable=False, style={"width":"100%"}),
            html.Br(),
            html.Button("Reload file list", id="reload-files", n_clicks=0),
            html.Div(id="file-meta", style={"marginTop":"10px", "fontSize":"12px", "color":"#444"}),
            html.Div(style={"marginTop":"8px", "fontSize":"12px", "color":"#666"}, children=[
                html.Div(f"WordCloud available: {'Yes' if WORDCLOUD_AVAILABLE else 'No'}"),
                html.Div(f"spaCy NER available: {'Yes' if SPACY_AVAILABLE else 'No'}"),
            ])
        ], style={"width":"26%", "display":"inline-block", "verticalAlign":"top", "padding":"10px", "boxSizing":"border-box", "borderRight":"1px solid #ddd"}),

        # Right column: post content on top, then KPI row, then graphs
        html.Div([
            html.Div(id="post-content", style={
                "whiteSpace":"pre-wrap", "padding":"12px", "border":"1px solid #eee",
                "backgroundColor":"#fff8f0", "marginBottom":"12px", "maxHeight":"260px", "overflow":"auto",
                "fontSize":"15px", "lineHeight":"1.4"
            }),
            html.Div(id="kpi-row", style={"display":"flex", "gap":"16px", "flexWrap":"wrap", "marginBottom":"12px"}),
            html.Div(id="graphs-grid", children=[], style={"display":"grid", "gridTemplateColumns":"repeat(2, 1fr)", "gap":"20px", "padding":"10px"})
        ], style={"width":"73%", "display":"inline-block", "verticalAlign":"top", "padding":"10px", "boxSizing":"border-box"})
    ], style={"width":"98%", "margin":"0 auto", "display":"flex", "gap":"10px"})
], style={"fontFamily":"Noto Sans Bengali, Kalpurush, 'Noto Serif Bengali', sans-serif"})

# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output("candidate-dropdown", "options"),
    Output("candidate-dropdown", "value"),
    Input("reload-files", "n_clicks"),
)
def _reload_files(n):
    # Clear cached JSON loads so newly-saved files are re-read when clicking Reload
    try:
        load_json_file.cache_clear()
    except Exception:
        pass
    candidates = list_candidates()
    candidate_options = [{"label": c, "value": c} for c in candidates]
    default_candidate = candidates[0] if candidates else None
    return candidate_options, default_candidate

@app.callback(
    Output("file-dropdown", "options"),
    Output("file-dropdown", "value"),
    Input("candidate-dropdown", "value")
)
def on_candidate_change(candidate_value):
    files = list_json_files(candidate_value)
    options = [{"label": f, "value": f} for f in files]
    value = options[0]["value"] if options else None
    return options, value

# Main callback now returns post-content, file-meta, kpi-row, graphs-grid
@app.callback(
    Output("post-content", "children"),
    Output("file-meta", "children"),
    Output("kpi-row", "children"),
    Output("graphs-grid", "children"),
    Input("candidate-dropdown", "value"),
    Input("file-dropdown", "value"),
)
def on_file_selected(candidate_folder, filename):
    if not candidate_folder or not filename:
        return "(No post selected)", "No file selected", [], []

    path = os.path.join(SCRAPED_ROOT, candidate_folder, filename)
    if not os.path.exists(path):
        return "", html.Div(f"Selected file not found: {path}", style={"color":"red"}), [], []

    try:
        # Clear cached JSON loads immediately before reading to ensure fresh data
        try:
            load_json_file.cache_clear()
        except Exception:
            pass

        raw = load_json_file(candidate_folder, filename)
        post_record, comments_records = parse_post_json(raw)
        figs = make_figures_from_post(post_record, comments_records)

        # Post content displayed on top (preserve newlines)
        post_content = post_record.get("content") or "(No post content)"

        # file-meta small lines
        dt = post_record.get("datetime")
        meta_lines = [
            html.Div(f"File: {filename}"),
            html.Div(f"Post datetime: {dt.isoformat() if dt else 'Unknown'}"),
            html.Div(f"Likes: {post_record.get('likes',0)}, Shares: {post_record.get('shares',0)}, Comments scraped: {post_record.get('comments_count',0)}")
        ]

        # KPI children: include additional text stats
        k = figs.get("fig_kpi", {"likes":0,"shares":0,"comments":0,"total":0})
        likes = k.get("likes", post_record.get("likes",0))
        shares = k.get("shares", post_record.get("shares",0))
        comments_n = k.get("comments", post_record.get("comments_count",0))
        total = k.get("total", likes + shares + comments_n)
        total_words = k.get("total_words", 0)
        unique_words = k.get("unique_words", 0)
        avg_word_len = k.get("avg_word_len", 0.0)

        def kpi_card(title, value, color="#1f77b4"):
            return html.Div([
                html.Div(title, style={"fontSize":"13px", "color":"#333", "marginBottom":"6px", "textAlign":"center"}),
                html.Div(str(value), style={"fontSize":"20px", "fontWeight":"700", "color": color, "textAlign":"center"})
            ], style={"padding":"10px", "border":"1px solid #ddd", "borderRadius":"6px", "backgroundColor":"#fff", "minWidth":"160px"})

        kpi_children = [
            kpi_card("Likes", likes, "#d62728"),
            kpi_card("Shares", shares, "#ff7f0e"),
            kpi_card("Comments", comments_n, "#2ca02c"),
            kpi_card("Total", total, "#1f77b4"),
            kpi_card("Total words", total_words, "#666"),
            kpi_card("Unique words", unique_words, "#666"),
            kpi_card("Avg word len", avg_word_len, "#666"),
        ]

        # defensive figure helper
        def _safe_fig(key, default_title="No data"):
            return figs.get(key) if figs.get(key) is not None else px.scatter(title=default_title)

        # Wordcloud handled specially (may be base64 string)
        wordcloud_component = html.Div(
            html.Img(src=figs.get("fig_wordcloud_img"), style={"maxWidth":"100%", "height":"auto"})
            if figs.get("fig_wordcloud_img") else html.Div("WordCloud not available", style={"padding":"10px"}),
            style={"padding":"10px", "border":"1px solid #eee", "backgroundColor":"#fff"}
        )

        grid_children = [
            dcc.Graph(figure=_safe_fig("fig_rx_pie", "Reaction Breakdown (no data)")),
            dcc.Graph(figure=_safe_fig("fig_rx_bar", "Reaction Types (no data)")),
            dcc.Graph(figure=_safe_fig("fig_sentiment_pie", "Comment Sentiment (none)")),
            dcc.Graph(figure=_safe_fig("fig_emotion_pie", "Comment Emotion (none)")),
            dcc.Graph(figure=_safe_fig("fig_tone_pie", "Comment Tone (none)")),
            dcc.Graph(figure=_safe_fig("fig_intent_pie", "Comment Intent (none)")),
            dcc.Graph(figure=_safe_fig("fig_top_words", "Top Words (none)")),
            wordcloud_component,
            dcc.Graph(figure=_safe_fig("fig_ner_bar", "Named Entities (none)")),
            dcc.Graph(figure=_safe_fig("fig_emotion_bar", "Comment Emotions (none)")),
            dcc.Graph(figure=_safe_fig("fig_word_freq", "Top Words (none)")),
            dcc.Graph(figure=_safe_fig("fig_hashtag", "Top Hashtags (none)")),
            dcc.Graph(figure=_safe_fig("fig_postlen_engage", "Post Length vs Engagement")),
            dcc.Graph(figure=_safe_fig("fig_rx_time", "Reactions by Type (none)")),
            dcc.Graph(figure=_safe_fig("fig_polar", "Reaction Mix")),
            dcc.Graph(figure=_safe_fig("fig_rx_violin", "Reaction Distribution")),
            dcc.Graph(figure=_safe_fig("fig_heat", "Emotion vs Sentiment (none)")),
            dcc.Graph(figure=_safe_fig("fig_treemap", "Word Treemap")),
            dcc.Graph(figure=_safe_fig("fig_eng_bar", "Engagement Metrics (none)")),
            html.Div([
                html.H4("Top comments (preview)"),
                html.Ul([html.Li(f"{c['user_name']} ({c['likes']} likes): {c['text'][:160]} — Sent:{c.get('sentiment','')}, Emo:{c.get('emotion','')}, Tone:{c.get('tone','')}, Intent:{c.get('intent','')}") for c in figs.get("comments_preview", [])])
            ], style={"padding":"10px", "border":"1px solid #eee", "backgroundColor":"#fafafa"})
        ]

        return post_content, meta_lines, kpi_children, grid_children

    except Exception as e:
        tb = traceback.format_exc()
        print("Exception in on_file_selected callback:\n", tb)
        user_msg = f"Error processing file '{filename}': {type(e).__name__}: {str(e)}"
        return "(error)", html.Div([html.Div(user_msg, style={"color":"red","fontWeight":"bold"}), html.Pre(str(e), style={"whiteSpace":"pre-wrap","color":"#444"})]), [], []

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    print("Starting Social Insights Dashboard (NLP Enhanced)")
    print("Scraped root:", SCRAPED_ROOT)
    app.run(debug=True, port=PORT, use_reloader=True)