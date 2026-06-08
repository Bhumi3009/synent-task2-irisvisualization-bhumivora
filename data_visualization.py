import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.datasets import load_iris

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris — Field Study",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Azeret+Mono:wght@300;400;500;600&family=Nunito+Sans:wght@300;400;600&display=swap');

:root {
  --ink:       #0D0E0B;

  --card:      #FAFAF6;
  --line:      #D9D4C7;
  --muted:     #867E6C;
  --accent1:   #3B6E52;
  --accent2:   #7B4F8E;
  --accent3:   #C85C38;
  --highlight: #E8F0EB;
  --mono:      'Azeret Mono', monospace;
  --serif:     'Libre Baskerville', Georgia, serif;
  --sans:      'Nunito Sans', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--sans);
  color: var(--ink);
}

.stApp {
  background-color: var(--paper);
  background-image:
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 27px,
      rgba(0,0,0,0.03) 27px,
      rgba(0,0,0,0.03) 28px
    );
}

[data-testid="stSidebar"] {
  background-color: #0D1A12 !important;
  border-right: 1px solid #1E3020;
}
[data-testid="stSidebar"] * { color: #A8C4AB !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #D6EAD8 !important; }
[data-testid="stSidebar"] label {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: #5A8A60 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
  background: #142018 !important;
  border: 1px solid #2A4030 !important;
  color: #A8C4AB !important;
  border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: #5A8A60 !important;
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: 0.8px;
}

#MainMenu, footer, header { visibility: hidden; }

.pg-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  padding: 20px 0 22px;
  border-bottom: 2px solid var(--ink);
  margin-bottom: 28px;
}
.pg-eyebrow {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--accent1);
  margin: 0 0 6px;
}
.pg-title {
  font-family: var(--serif);
  font-size: 42px;
  font-weight: 700;
  color: var(--ink);
  margin: 0;
  line-height: 1.05;
  letter-spacing: -1px;
}
.pg-title em {
  font-style: italic;
  color: var(--accent1);
}
.pg-meta {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  text-align: right;
  line-height: 1.6;
}
.pg-meta strong { color: var(--ink); font-weight: 600; }

[data-testid="metric-container"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-top: 3px solid var(--ink);
  border-radius: 4px;
  padding: 16px 18px !important;
  box-shadow: 3px 3px 0 var(--line);
}
[data-testid="metric-container"] label {
  font-family: var(--mono) !important;
  font-size: 9px !important;
  letter-spacing: 1.4px;
  text-transform: uppercase;
  color: var(--muted) !important;
  font-weight: 400;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  font-size: 30px !important;
  color: var(--ink) !important;
  font-weight: 600;
  letter-spacing: -1px;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-family: var(--mono);
  font-size: 10px !important;
  color: var(--muted) !important;
}

.sec-head {
  font-family: var(--serif);
  font-size: 20px;
  font-weight: 700;
  color: var(--ink);
  margin: 28px 0 2px;
  letter-spacing: -0.3px;
}
.sec-sub {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 18px;
}

.pill {
  display: inline-block;
  padding: 2px 12px;
  border-radius: 2px;
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 1px;
  text-transform: uppercase;
  font-weight: 600;
  margin-right: 6px;
}
.pill-s { background: #E4EFEA; color: #3B6E52; border: 1px solid #BDD5C5; }
.pill-v { background: #EBE4F0; color: #7B4F8E; border: 1px solid #CBBBDB; }
.pill-g { background: #F5EAE4; color: #C85C38; border: 1px solid #E0C0AF; }

.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 2px solid var(--ink);
  background: transparent;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted);
  padding: 10px 20px;
  border-radius: 0;
  background: transparent;
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom: 3px solid #3B6E52 !important;
  font-weight: 600 !important;
  background: #E8F0EB !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line) !important;
  border-radius: 4px;
  overflow: hidden;
  box-shadow: 4px 4px 0 rgba(0,0,0,0.05);
}

.stTextInput > div > div {
  background: var(--card) !important;
  border: 1px solid var(--line) !important;
  border-radius: 4px !important;
  font-family: var(--mono) !important;
  font-size: 12px !important;
}

.stDownloadButton > button {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  letter-spacing: 1px;
  text-transform: uppercase;
  background: var(--ink) !important;
  color: var(--paper) !important;
  border: none !important;
  border-radius: 3px !important;
  padding: 8px 18px !important;
  box-shadow: 3px 3px 0 #3B6E52;
  transition: box-shadow 0.15s, transform 0.15s;
}
.stDownloadButton > button:hover {
  box-shadow: 1px 1px 0 #3B6E52 !important;
  transform: translate(2px, 2px);
}

hr {
  border: none;
  border-top: 1px solid var(--line);
  margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Palette & plot theme ───────────────────────────────────────────────────────
PAL = {"Setosa": "#3B6E52", "Versicolor": "#7B4F8E", "Virginica": "#C85C38"}
BG, GRID, INK = "#FAFAF6", "#E2DDD4", "#0D0E0B"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    GRID,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.5,
    "grid.linestyle":    ":",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelcolor":   INK,
    "xtick.color":       "#867E6C",
    "ytick.color":       "#867E6C",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "text.color":        INK,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
    "axes.labelpad":     8,
    "axes.labelsize":    10,
    "legend.frameon":    False,
    "legend.fontsize":   9,
})

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["Species"] = iris.target
    df["Species"] = df["Species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})
    return df

df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Field Study")
    st.markdown("*Iris* genus · 1936 Fisher dataset")
    st.markdown("---")
    st.markdown("**Species**")
    selected_species = st.multiselect(
        "Species", options=["Setosa", "Versicolor", "Virginica"],
        default=["Setosa", "Versicolor", "Virginica"], label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Scatter — X axis**")
    x_feature = st.selectbox("X", df.columns[:-1], index=0, label_visibility="collapsed")
    st.markdown("**Scatter — Y axis**")
    y_feature = st.selectbox("Y", df.columns[:-1], index=2, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Histogram feature**")
    hist_feature = st.selectbox("Histogram", df.columns[:-1], index=0, label_visibility="collapsed")
    hist_bins = st.slider("Bins", 5, 40, 18)
    st.markdown("---")
    st.markdown("**Box plot feature**")
    box_feature = st.selectbox("Box", df.columns[:-1], index=2, label_visibility="collapsed", key="box")
    st.markdown("---")
    st.caption("sklearn · 150 samples · 4 features · 3 classes")

# ── Filter ─────────────────────────────────────────────────────────────────────
if not selected_species:
    st.warning("Select at least one species in the sidebar.")
    st.stop()

filtered = df[df["Species"].isin(selected_species)]

# ── Header ─────────────────────────────────────────────────────────────────────
pill_map = {"S": "pill-s", "V": "pill-v", "G": "pill-g"}
pills = "".join(
    f'<span class="pill {pill_map[s[0]]}">{s}</span>'
    for s in selected_species
)
st.markdown(f"""
<div class="pg-header">
  <div>
    <p class="pg-eyebrow">Exploratory Data Analysis · Botany</p>
    <h1 class="pg-title"><em>Iris</em> — Field Study</h1>
    <div style="margin-top:10px;">{pills}</div>
  </div>
  <div class="pg-meta">
    <strong>{len(filtered)}</strong> specimens<br>
    <strong>4</strong> morphological features<br>
    <strong>{filtered["Species"].nunique()}</strong> species selected
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Specimens", len(filtered),
          delta=f"{len(filtered)-150} vs full" if len(filtered) < 150 else "full set")
c2.metric("Features", 4)
c3.metric("Species", filtered["Species"].nunique())
c4.metric("Avg petal length", f"{filtered['petal length (cm)'].mean():.2f} cm")
c5.metric("Avg sepal length", f"{filtered['sepal length (cm)'].mean():.2f} cm")
st.markdown("<hr>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Distribution", "Histogram", "Scatter", "Box Plot", "Correlation", "Raw Data"
])

def sec(title, sub):
    st.markdown(f'<p class="sec-head">{title}</p><p class="sec-sub">{sub}</p>',
                unsafe_allow_html=True)

# Tab 1 ─────────────────────────────────────────────────────────────────────────
with tab1:
    sec("Species Distribution", "Specimen count · filtered selection")
    counts = filtered["Species"].value_counts().reindex(selected_species).dropna()
    colors = [PAL[s] for s in counts.index]
    total  = counts.sum()
    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bars = ax.bar(counts.index, counts.values, color=colors, width=0.38, zorder=2, linewidth=0)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=INK)
        ax.set_ylabel("Specimen count", labelpad=8)
        ax.set_ylim(0, counts.max() * 1.3)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", length=0, pad=8)
        for bar, (sp, val) in zip(bars, counts.items()):
            pct = val / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, -counts.max()*0.05,
                    f"{pct:.0f}%", ha="center", va="top", fontsize=8, color="#867E6C")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col_info:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        for sp in counts.index:
            n = int(counts[sp]); pct = n / total * 100; color = PAL[sp]
            st.markdown(f"""
            <div style="padding:14px 16px;margin-bottom:10px;background:#FAFAF6;
                        border:1px solid #D9D4C7;border-left:4px solid {color};border-radius:3px;">
              <div style="font-family:'Azeret Mono',monospace;font-size:9px;letter-spacing:1.2px;
                          text-transform:uppercase;color:#867E6C;margin-bottom:4px;">{sp}</div>
              <div style="font-family:'Azeret Mono',monospace;font-size:26px;
                          font-weight:600;color:{color};line-height:1;">{n}</div>
              <div style="font-family:'Azeret Mono',monospace;font-size:10px;
                          color:#867E6C;margin-top:2px;">{pct:.1f}% of selection</div>
            </div>""", unsafe_allow_html=True)

# Tab 2 ─────────────────────────────────────────────────────────────────────────
with tab2:
    sec(f"{hist_feature.split(' (')[0].title()} Distribution",
        f"Overlaid histograms per species · {hist_bins} bins · dashed = mean")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for sp in selected_species:
        sub = filtered[filtered["Species"] == sp][hist_feature]
        ax.hist(sub, bins=hist_bins, alpha=0.6, color=PAL[sp],
                edgecolor=BG, linewidth=1, label=sp, zorder=2)
        ax.axvline(sub.mean(), color=PAL[sp], linewidth=1.5, linestyle="--", alpha=0.9, zorder=3)
    patches = [mpatches.Patch(color=PAL[sp], label=sp, alpha=0.7) for sp in selected_species]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel(hist_feature); ax.set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Tab 3 ─────────────────────────────────────────────────────────────────────────
with tab3:
    sec("Scatter Plot",
        f"{x_feature.split(' (')[0].title()} vs {y_feature.split(' (')[0].title()} · colour = species")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for sp in selected_species:
        sub = filtered[filtered["Species"] == sp]
        ax.scatter(sub[x_feature], sub[y_feature],
                   color=PAL[sp], label=sp,
                   alpha=0.78, s=65, edgecolors=BG, linewidths=0.8, zorder=3)
    ax.set_xlabel(x_feature); ax.set_ylabel(y_feature)
    ax.legend(fontsize=9, markerscale=1.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Tab 4 ─────────────────────────────────────────────────────────────────────────
with tab4:
    sec(f"Box Plot — {box_feature.split(' (')[0].title()}",
        "Median · IQR · whiskers · outliers per species")
    data_by_sp = [filtered[filtered["Species"] == sp][box_feature].values
                  for sp in selected_species]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data_by_sp, patch_artist=True, widths=0.38,
        medianprops=dict(color=BG, linewidth=2.5),
        whiskerprops=dict(linewidth=1.2, color="#867E6C", linestyle=":"),
        capprops=dict(linewidth=1.5, color="#867E6C"),
        flierprops=dict(marker="o", markersize=5, alpha=0.55, linestyle="none"),
    )
    for patch, sp in zip(bp["boxes"], selected_species):
        patch.set_facecolor(PAL[sp]); patch.set_alpha(0.82); patch.set_linewidth(0)
    for flier, sp in zip(bp["fliers"], selected_species):
        flier.set_markerfacecolor(PAL[sp]); flier.set_markeredgecolor(PAL[sp])
    ax.set_xticks(range(1, len(selected_species)+1))
    ax.set_xticklabels(selected_species)
    ax.set_ylabel(box_feature); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Tab 5 ─────────────────────────────────────────────────────────────────────────
with tab5:
    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        sec("Correlation Heatmap", "Pearson r · all numeric features")
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("iris", ["#C85C38", "#F2EEE5", "#3B6E52"], N=256)
        corr = filtered.drop(columns="Species").corr().round(2)
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                    center=0, vmin=-1, vmax=1,
                    linewidths=1.5, linecolor=BG,
                    annot_kws={"size": 12, "weight": "bold"},
                    square=True, ax=ax, cbar_kws={"shrink": 0.75, "pad": 0.02})
        ax.tick_params(axis="x", rotation=30, labelsize=8.5)
        ax.tick_params(axis="y", rotation=0, labelsize=8.5)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col_b:
        sec("Pair Plot", "All feature pairs · KDE on diagonal")
        g = sns.pairplot(
            filtered, hue="Species", palette=PAL,
            diag_kind="kde", corner=False,
            plot_kws=dict(alpha=0.6, s=25, edgecolor=BG, linewidth=0.4),
            diag_kws=dict(alpha=0.55), height=1.65
        )
        g.figure.set_facecolor(BG)
        for ax_ in g.axes.flatten():
            if ax_:
                ax_.set_facecolor(BG)
                for spine in ax_.spines.values():
                    spine.set_edgecolor(GRID)
        g.figure.tight_layout()
        st.pyplot(g.figure, use_container_width=True)
        plt.close()

# Tab 6 ─────────────────────────────────────────────────────────────────────────
with tab6:
    sec("Raw Specimen Data", f"{len(filtered)} records · 5 columns · filterable")
    col_search, _ = st.columns([1, 3])
    with col_search:
        search = st.text_input("Filter rows", placeholder="e.g. Setosa",
                               label_visibility="collapsed")
    disp = (filtered[filtered.apply(
                lambda r: search.lower() in str(r.values).lower(), axis=1)]
            if search else filtered)
    st.dataframe(
        disp.reset_index(drop=True).style
        .apply(lambda s: [
            "background-color:#E4EFEA;color:#3B6E52" if v == "Setosa"
            else "background-color:#EBE4F0;color:#7B4F8E" if v == "Versicolor"
            else "background-color:#F5EAE4;color:#C85C38"
            for v in s
        ], subset=["Species"])
        .format(precision=2),
        use_container_width=True, height=430,
    )
    dl1, dl2 = st.columns([1, 5])
    with dl1:
        st.download_button(
            "⬇  Export CSV",
            data=disp.to_csv(index=False).encode(),
            file_name="iris_filtered.csv",
            mime="text/csv",
        )
    with dl2:
        st.markdown(f"""
        <p style="font-family:'Azeret Mono',monospace;font-size:10px;
                  color:#867E6C;margin-top:12px;">{len(disp)} rows · {disp.shape[1]} columns</p>
        """, unsafe_allow_html=True)
        