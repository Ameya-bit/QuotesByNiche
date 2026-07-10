"""Publication figures for the B5H0 induction-head post.

Recomputes everything from ``niche_model.pt`` using the *same* analysis functions
that live in ``niche_attention_analysis.ipynb`` (lifted verbatim below, marked
"from the notebook" -- the logic is not reinvented here). Each figure is written
to ``figures/`` as both PNG (for the post) and SVG.

Run from the repo root:  ./venv/bin/python interp/figures/make_figures.py
"""
from __future__ import annotations

import os
import sys

# Make interp/ importable so ``niche_classes`` resolves regardless of the
# working directory: running ``python interp/figures/make_figures.py`` puts
# figures/, not interp/, on sys.path. This makes the documented run command work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from niche_classes import load_model

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Style: light background, colorblind-safe (Okabe-Ito), minimal chartjunk.
# ---------------------------------------------------------------------------
OKABE = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#111111",
}
INK = "#1a1a1a"
MUTED = "#b9c0c8"        # de-emphasised bars / dots
GRID = "#e6e8eb"
PANEL_BG = "#ffffff"
FIG_BG = "#ffffff"

# Sequential single-hue colormap for attention weights (colorblind-safe: one hue).
ATTN_CMAP = LinearSegmentedColormap.from_list(
    "attn", ["#ffffff", "#d6e6f2", "#87bddf", "#3d8fc4", "#0072B2", "#004f7d"]
)

# Verdict -> colour for the summary table.
VERDICT_COLOR = {
    "INDUCTION": OKABE["green"],
    "SINK (~pos 0)": OKABE["orange"],
    # Distinct dark-slate vs light-grey: the verdict is shown as a bare colour dot,
    # so these two "neither induction nor sink" cases must be told apart by shade.
    "previous-token": "#5f6f7c",
    "other": "#aab2b9",
}

mpl.rcParams.update({
    "figure.facecolor": FIG_BG,
    "axes.facecolor": PANEL_BG,
    "savefig.facecolor": FIG_BG,
    # DejaVu Sans is matplotlib-bundled (reproducible on any machine) and has full
    # glyph coverage for the arrows / ␣ / curly quotes these figures use.
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Arial"],
    "font.size": 11,
    "axes.edgecolor": "#c7ccd1",
    "axes.linewidth": 0.8,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def glyph(ch: str) -> str:
    """Printable label for a single character (spaces are invisible otherwise)."""
    return "␣" if ch == " " else ch


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=200)
    print(f"  wrote figures/{name}.png + .svg")


# ---------------------------------------------------------------------------
# Analysis functions -- lifted verbatim from niche_attention_analysis.ipynb.
# ---------------------------------------------------------------------------
m = load_model(os.path.join(HERE, "..", "..", "niche_model.pt"), map_location="cpu")


def independent_weights(attention_block, attention_head):
    w_e = m.model.token_embedding.weight.transpose(-1, -2).detach()
    w_u = m.model.last_layer.weight.detach()
    qkv = m.model.blocks[attention_block].comp_full_attend.qkv.weight.detach()
    q, k, v = tuple(t.view(m.config.n_head, m.config.n_embd // m.config.n_head, -1).detach()
                    for t in qkv.split(m.config.n_embd))
    w_o = torch.stack(m.model.blocks[attention_block].comp_full_attend.lin_proj.weight.split(
        m.config.n_embd // m.config.n_head, dim=-1)).detach()
    return w_e, w_u, q, k, v, w_o


def identify_copying_ov(w_u, w_o, v, w_e, attention_block, attention_head):
    w_ov_h = w_u @ w_o[attention_head] @ v[attention_head] @ w_e
    w_ov_h_eigen = torch.linalg.eig(w_ov_h)
    w_ov_h_rank = m.config.n_embd // m.config.n_head
    ov_evalues = w_ov_h_eigen.eigenvalues
    ov_evalues_mag = torch.abs(ov_evalues)
    ov_evalues_indsort = torch.argsort(ov_evalues_mag, descending=True)
    ov_evalues_sorted = ov_evalues[ov_evalues_indsort][:w_ov_h_rank]
    copying_score = sum(ov_evalues_sorted.real) / sum(torch.abs(ov_evalues_sorted))
    return float(copying_score.real)


def identify_copying_qk(attention_block, attention_head, sentence, token, iterate):
    chars = [i for i, s in enumerate(sentence) if s == token]
    idx = torch.tensor([[m.stoi[c] for c in sentence]])
    with torch.no_grad():
        m.model(idx, store_attn=True)
    A = torch.squeeze(m.model.blocks[attention_block].comp_full_attend.attn_weights[:, attention_head])
    q = chars[iterate]
    topk = torch.topk(A[q], 5)
    ranked = [(p, sentence[p], round(w, 4))
              for p, w in zip(topk.indices.tolist(), topk.values.tolist())]
    return q, chars, ranked, A


def case(label, sentence, token, iterate=-1, watch=None):
    """run_case from the notebook, returning a record instead of printing."""
    q, occs, ranked, A = identify_copying_qk(5, 0, sentence, token, iterate)
    targets = {p + 1 for p in occs if p < q}
    amax_pos, amax_ch, amax_w = ranked[0]
    verdict = ("INDUCTION" if amax_pos in targets else
               "previous-token" if amax_pos == q - 1 else
               "SINK (~pos 0)" if amax_pos <= 1 else "other")
    watched = []
    if watch:
        for wc in watch:
            for p, c in enumerate(sentence):
                if c == wc and p <= q:
                    watched.append((wc, p, float(A[q][p].item())))
    return {
        "label": label, "sentence": sentence, "token": token, "q": q,
        "query_char": sentence[q], "occs": occs, "targets": sorted(targets),
        "verdict": verdict, "top_pos": amax_pos, "top_char": amax_ch,
        "top_w": amax_w, "ranked": ranked, "watch": watched,
    }


# ===========================================================================
# FIGURE 1 -- MAKE: raw vs norm-normalized OV diagonal
# ===========================================================================
def figure_make(write: bool = True):
    w_e, w_u, q, k, v, w_o = independent_weights(5, 0)
    w_ov_h = w_u @ w_o[0] @ v[0] @ w_e
    norms = torch.linalg.norm(w_u, dim=1) * torch.linalg.norm(w_e, dim=0)
    raw = torch.diagonal(w_ov_h)
    normed = raw / norms

    itos = m.itos
    raw_by = {itos[i]: float(raw[i]) for i in range(m.config.vocab_size)}
    nor_by = {itos[i]: float(normed[i]) for i in range(m.config.vocab_size)}

    TOP_N = 13
    order = [c for c, _ in sorted(nor_by.items(), key=lambda x: x[1], reverse=True)[:TOP_N]]
    order = order[::-1]                       # so the largest ends up at the top of the bars

    # Parentheses sit far outside the top-13 -- '(' is rank 39/177 and ')' is dead
    # last (177/177, the *most* anti-copied token in the vocab). Show them as a
    # separate reference group below the top block so the bracket-grammar
    # hypothesis (fig5) can be read straight off the OV diagonal: the head does
    # not self-copy either paren, and actively pushes ')' down.
    paren_chars = ['(', ')']
    main_y = list(range(len(order)))
    paren_y = [-1.8 - i for i in range(len(paren_chars))]      # sit below the group

    all_chars = order + paren_chars
    all_y = main_y + paren_y
    raw_vals = [raw_by[c] for c in all_chars]
    nor_vals = [nor_by[c] for c in all_chars]

    def bar_colors(chars):
        out = []
        for c in chars:
            if c in ("(", ")"):
                out.append(OKABE["vermillion"])       # bracket reference group
            elif c == '"':
                out.append(OKABE["blue"])              # the copying head's signature
            else:
                out.append(MUTED)
        return out

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 6.6), sharey=True)

    axL.barh(all_y, raw_vals, color=bar_colors(all_chars), height=0.68,
             edgecolor="white", linewidth=0.6)
    axR.barh(all_y, nor_vals, color=bar_colors(all_chars), height=0.68,
             edgecolor="white", linewidth=0.6)

    axL.set_yticks(all_y)
    axL.set_yticklabels([glyph(c) for c in all_chars], fontfamily="monospace", fontsize=11)
    axL.set_title("Raw OV diagonal", pad=10)
    axR.set_title("Norm-normalized OV diagonal", pad=10)
    axL.set_xlabel("self-copy logit  (attend $X$ → boost $X$)")
    axR.set_xlabel("logit ÷ (‖embed‖·‖unembed‖) per token")

    # Value labels; parens carry the vermillion accent so bar and number read as
    # one reference group. Negative bars (e.g. ')') get their label on the left.
    label_col = [OKABE["vermillion"] if c in ("(", ")") else "#556" for c in all_chars]
    sep_y = (min(main_y) + max(paren_y)) / 2       # divider between block and parens

    for ax, vals, fmt in ((axL, raw_vals, "{:.1f}"), (axR, nor_vals, "{:+.2f}")):
        ax.axvline(0, color="#cfd4d9", lw=0.8, zorder=0)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(sep_y, color="#d3d7db", lw=0.8, ls=(0, (4, 3)), zorder=1)
        pad = 0.02 * max(abs(v) for v in vals)
        for yi, val, lc in zip(all_y, vals, label_col):
            if val >= 0:
                ax.text(val + pad, yi, fmt.format(val), va="center", ha="left",
                        fontsize=8.5, color=lc)
            else:
                ax.text(val - pad, yi, fmt.format(val), va="center", ha="right",
                        fontsize=8.5, color=lc)
        ax.margins(x=0.14)
        ax.set_ylim(min(paren_y) - 0.8, max(main_y) + 0.7)

    # Flag the bracket reference group and where the parens actually rank.
    axR.text(0.315, (paren_y[0] + paren_y[-1]) / 2,
             "brackets are not self-copied\n('(' #39,  ')' last of 177)",
             fontsize=8, style="italic", color=OKABE["vermillion"],
             va="center", ha="left")

    # Call out the quote -- the copying head's signature -- surviving to the top.
    qi = order.index('"')
    axR.annotate('the  "  logit survives\nnormalization —\nstill #1',
                 xy=(nor_vals[qi], qi), xytext=(0.085, qi - 4.6),
                 fontsize=9, color=OKABE["blue"], ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=OKABE["blue"], lw=1.4,
                                 connectionstyle="arc3,rad=-0.28"))

    # Headline/subtitle intentionally omitted -- the LaTeX \postfigure caption
    # describes the figure; the panel keeps only visual decoding aids.
    fig.subplots_adjust(top=0.92, wspace=0.06)
    if write:
        save(fig, "fig1_make_ov_diagonal")
    return fig


# ===========================================================================
# FIGURE 2 -- THE SPIKE: copying score across all 24 heads
# ===========================================================================
def figure_spike(write: bool = True):
    scores = {}
    for b in range(m.config.n_layers):
        for h in range(m.config.n_head):
            w_e, w_u, q, k, v, w_o = independent_weights(b, h)
            scores[(b, h)] = identify_copying_ov(w_u, w_o, v, w_e, b, h)

    heads = [(b, h) for b in range(m.config.n_layers) for h in range(m.config.n_head)]
    labels = [f"B{b}H{h}" for (b, h) in heads]
    vals = [scores[(b, h)] for (b, h) in heads]
    is_star = [(b, h) == (5, 0) for (b, h) in heads]
    colors = [OKABE["vermillion"] if s else MUTED for s in is_star]

    x = range(len(heads))
    fig, ax = plt.subplots(figsize=(11, 4.9))
    ax.bar(list(x), vals, color=colors, width=0.72, edgecolor="white", linewidth=0.6,
           zorder=3)

    star_i = is_star.index(True)
    runner = sorted(vals, reverse=True)[1]
    ax.axhline(runner, color="#b0b6bc", lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(0.4, runner + 0.014, f"next best = {runner:.2f}",
            va="bottom", ha="left", fontsize=8.5, color="#7a828a")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=90, fontsize=7.5, fontfamily="monospace")
    # Emphasise the B5H0 tick label.
    ax.get_xticklabels()[star_i].set_color(OKABE["vermillion"])
    ax.get_xticklabels()[star_i].set_fontweight("bold")

    ax.set_ylabel("copying score\n(Σ Re λ / Σ |λ|, top-64 eigenvalues)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.annotate(f"B5H0 = {vals[star_i]:.3f}\nthe only contender",
                xy=(star_i, vals[star_i]),
                xytext=(star_i + 2.6, vals[star_i] + 0.02),
                fontsize=10.5, fontweight="bold", color=OKABE["vermillion"],
                ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", color=OKABE["vermillion"], lw=1.6,
                                connectionstyle="arc3,rad=0.15"))

    # Description lives in the LaTeX caption; keep only the in-plot callouts.
    fig.subplots_adjust(top=0.96, bottom=0.15)
    if write:
        save(fig, "fig2_spike_copying_score")
    return fig


# ===========================================================================
# FIGURE 3 -- DIVERT: induction stripe tracks the swapped content (cat / dog)
# ===========================================================================
def _attn(sentence, token, iterate=1):
    q, occs, ranked, A = identify_copying_qk(5, 0, sentence, token, iterate)
    return A.numpy(), list(sentence), q, occs


def figure_divert(write: bool = True):
    panels = [
        ("the cat sat on the mat, the cat ran", "c", "a"),
        ("the dog sat on the mat, the dog ran", "d", "o"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    ims = []
    for ax, (sent, tok, succ) in zip(axes, panels):
        A, chars, q, occs = _attn(sent, tok, iterate=1)
        T = len(chars)
        target = occs[0] + 1                         # char right after the first occurrence
        im = ax.imshow(A, cmap=ATTN_CMAP, vmin=0, vmax=1, aspect="equal")
        ims.append(im)

        ticks = range(T)
        ax.set_xticks(list(ticks))
        ax.set_yticks(list(ticks))
        ax.set_xticklabels([glyph(c) for c in chars], fontfamily="monospace", fontsize=7)
        ax.set_yticklabels([glyph(c) for c in chars], fontfamily="monospace", fontsize=7)
        ax.set_xlabel("key position  (attended-to token)")
        ax.set_ylabel("query position  (token doing the attending)")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

        # Highlight the whole query row and the induction cell.
        ax.add_patch(Rectangle((-0.5, q - 0.5), T, 1, fill=False,
                               edgecolor=OKABE["orange"], lw=1.4, alpha=0.55))
        ax.add_patch(Rectangle((target - 0.5, q - 0.5), 1, 1, fill=False,
                               edgecolor=OKABE["vermillion"], lw=2.4))

        w = A[q, target]
        # Emphasise the two axis labels that define the induction match.
        ax.get_yticklabels()[q].set_color(OKABE["vermillion"])
        ax.get_yticklabels()[q].set_fontweight("bold")
        ax.get_xticklabels()[target].set_color(OKABE["vermillion"])
        ax.get_xticklabels()[target].set_fontweight("bold")

        ax.annotate(f"query {glyph(tok)!s}  →  {glyph(succ)!s}\nweight {w:.2f}",
                    xy=(target, q), xytext=(target + 4.5, q - 6.5),
                    fontsize=10, fontweight="bold", color=OKABE["vermillion"],
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=OKABE["vermillion"],
                                    lw=1.7, connectionstyle="arc3,rad=0.2"))
        ax.set_title(f"“{sent}”", fontsize=10.5, loc="left", pad=8, fontfamily="monospace")

    cbar = fig.colorbar(ims[0], ax=axes, fraction=0.025, pad=0.02, shrink=0.72)
    cbar.set_label("attention weight", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Headline/subtitle omitted -- described by the LaTeX caption.
    if write:
        save(fig, "fig3_divert_induction_stripe")
    return fig


# ===========================================================================
# FIGURE 4 / 5 -- BREAK: paired-prompt verdict tables
#   The quote/induction prompts (P1-P5) and the parenthesis prompts (P6-C8) get
#   their own table so each control reads on its own terms. The honest-wrinkle
#   caveats now live in the LaTeX captions, not baked into the image.
# ===========================================================================
def _verdict_table(cases, name, write, fig_w=11.0):
    n = len(cases)
    row_h = 1.0
    fig_h = 0.52 * n + 3.2
    # Column anchors are fixed in 0..100 data coords while text is a fixed point
    # size, so a physically wider figure spreads the columns apart (each gap spans
    # more inches and the text fills proportionally less of it) without any
    # re-layout. fig_w lets one table run wider than another (fig4 does).
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0.3, n + 2.6)
    ax.axis("off")

    # Column x-anchors (data coords 0..100). PROMPT + a ~39-char TEST STRING fill
    # the left half, so QUERY and the three TOP-k cells are packed into the right
    # half at a ~9-unit pitch. Each TOP-k cell is "char @pos weight"; VERDICT is a
    # compact colour cue (dot + label), not a filled chip, to free the space.
    X_LABEL, X_SENT, X_QUERY = 1.5, 20, 51
    X_T1, X_T2, X_T3 = 60, 72, 84       # wider ~12-unit pitch -> roomier columns
    X_VERDICT = 95        # verdict is now just a centred colour dot (see legend)
    POS_DX = 2.0     # small "@pos" annotation sits just right of its glyph
    TW_DX = 5.6      # attention weight sits clear to the right of the char + @pos
    top_y = n + 1.05

    # Header.
    for x, txt in ((X_LABEL, "PROMPT"), (X_SENT, "TEST STRING"), (X_QUERY, "QUERY"),
                   (X_T1, "TOP 1"), (X_T2, "TOP 2"), (X_T3, "TOP 3")):
        ax.text(x, top_y, txt, fontsize=9.5, fontweight="bold", color="#5b6570",
                ha="left", va="center")
    ax.text(X_VERDICT, top_y, "VERDICT", fontsize=9.5, fontweight="bold",
            color="#5b6570", ha="center", va="center")   # centred over the dots
    ax.plot([X_LABEL, 99], [top_y - 0.55, top_y - 0.55], color="#c7ccd1", lw=1.0)

    # Zebra banding by prompt pair (P1, P2, ... in order of appearance).
    pairs = []
    for c in cases:
        pr = c["label"].split("-")[0].split(" ")[0]
        if pr not in pairs:
            pairs.append(pr)
    for r, c in enumerate(cases):
        y = n - r
        pair = c["label"].split("-")[0].split(" ")[0]
        if pairs.index(pair) % 2 == 0:
            ax.add_patch(Rectangle((X_LABEL - 0.8, y - 0.5), 98.3, row_h,
                                   color="#f4f6f8", zorder=0))

        ax.text(X_LABEL, y, c["label"], fontsize=10.5, ha="left", va="center",
                fontweight="bold", color=INK)
        sent = c["sentence"]
        shown = sent if len(sent) <= 40 else sent[:38] + "…"
        ax.text(X_SENT, y, shown, fontsize=7.5, ha="left", va="center",
                fontfamily="monospace", color="#3a4149")
        ax.text(X_QUERY, y, glyph(c["query_char"]), fontsize=11, ha="left",
                va="center", fontfamily="monospace", color="#3a4149")
        ax.text(X_QUERY + POS_DX, y, f"@{c['q']}", fontsize=7.5, ha="left",
                va="center", color="#9aa4ad")

        # TOP 1 / 2 / 3 -- the three most-attended tokens, each as "char @pos weight".
        # The winner (top-1) is emphasised and runners-up are lighter, so the
        # ranking reads at a glance while the exact weights expose near-ties (the
        # induction target only just beating the sink in the P4 rows).
        for xk, (p, ch, w) in zip((X_T1, X_T2, X_T3), c["ranked"][:3]):
            lead = xk == X_T1
            ax.text(xk, y, glyph(ch), fontsize=10.5 if lead else 9.5, ha="left",
                    va="center", fontfamily="monospace",
                    fontweight="bold" if lead else "normal",
                    color=INK if lead else "#5a636b")
            ax.text(xk + POS_DX, y, f"@{p}", fontsize=7.5, ha="left", va="center",
                    color="#9aa4ad")
            ax.text(xk + TW_DX, y, f"{w:.2f}", fontsize=8, ha="left", va="center",
                    color=OKABE["blue"], fontweight="bold" if lead else "normal")

        # Verdict -- a single centred colour dot (see the legend swatches for the
        # mapping); the text label is dropped to give the TOP-k columns more room.
        ax.plot([X_VERDICT], [y], marker="o", ms=8, color=VERDICT_COLOR[c["verdict"]],
                alpha=0.9, markeredgecolor="none", zorder=2)

    # Legend for verdict colours.
    lx = X_LABEL
    ly = n + 1.7
    for label, key in (("induction", "INDUCTION"), ("sink (pos ~0)", "SINK (~pos 0)"),
                       ("previous-token", "previous-token"), ("other", "other")):
        ax.add_patch(Rectangle((lx, ly - 0.16), 1.4, 0.32,
                               facecolor=VERDICT_COLOR[key], alpha=0.5, edgecolor="none"))
        ax.text(lx + 2.0, ly, label, fontsize=9.5, va="center", ha="left", color="#5b6570")
        lx += 2.0 + len(label) * 0.72 + 3.0

    # Annotation legend: what the small numbers in the query / top-k cells mean.
    ax.text(50, ly, "@N = token position   ·   weight = attention (softmax)",
            fontsize=8.5, style="italic", va="center", ha="left", color="#8a939c")

    # Height-aware header: reserve a fixed number of inches for title + subtitle
    # so short and tall tables get the same absolute spacing (no overlap).
    # Title/subtitle omitted -- the LaTeX caption describes the table; only the
    # in-figure legend and honest-wrinkle callout remain. A little top air keeps
    # the legend row from being cropped by bbox_inches="tight".
    header_in = 0.3
    fig.subplots_adjust(top=1 - header_in / fig_h, bottom=0.04)
    if write:
        save(fig, name)
    return fig


def figure_break_quotes(write: bool = True):
    cases = [
        case("P1-A xylophone", 'The man said, "xylophone do I write?". He then spoke, "Grashoper', '"'),
        case("P1-B what",      'The man said, "what do I write?". He then spoke, "Grashoper', '"'),
        case("P2-A cat", 'the cat sat on the mat, the cat ran', 'c', iterate=1),
        case("P2-B dog", 'the dog sat on the mat, the dog ran', 'd', iterate=1),
        case("P3-A repeat", 'Zarathustra spoke. Later, Zarathustra', 'Z', iterate=1),
        # Stronger "no repeat" control: the sole Z sits *mid-string* (pos 22) with
        # real prior context, so a SINK verdict is a meaningful null -- the head
        # declines to fire absent an earlier match, rather than the degenerate
        # first-token case where pos 0 can only attend to itself.
        case("P3-B norepeat", 'the crowd below heard Zarathustra', 'Z', iterate=0),
        case("P4-A apple-last",  '"apple" ... "banana" ... "apple', '"'),
        case("P4-B banana-last", '"banana" ... "apple" ... "banana', '"'),
        case("P5-A prior-opens", 'He said "one. She said "two. He said "', '"'),
        case("P5-B prior-closes", 'one" and two" and three" and', '"'),
    ]
    return _verdict_table(
        cases,
        name="fig4_break_quote_grammar",
        write=write,
        fig_w=14.0,        # wider than fig5 -> roomier columns
    )


def figure_break_parens(write: bool = True):
    cases = [
        case("P6-A paren-succ",  'x(ab x(cd', 'x'),
        case("P6-B letter-succ", 'xzab xzcd', 'x'),
        case("P7-A unmatched", 'So (the cat naps',  's'),
        case("P7-B matched",   'So (the cat) naps', 's'),
        case("C8 open-vs-closed", 'A (B (C) D', 'D'),
    ]
    return _verdict_table(
        cases,
        name="fig5_break_paren_grammar",
        write=write,
    )


if __name__ == "__main__":
    print("Rendering figures ->", HERE)
    figure_make()
    figure_spike()
    figure_divert()
    figure_break_quotes()
    figure_break_parens()
    plt.close("all")
    print("done.")
