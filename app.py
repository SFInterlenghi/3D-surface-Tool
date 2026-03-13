import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import copy
import io
import zipfile
import warnings

from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from scipy.interpolate import RBFInterpolator
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="3D Surface Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0;
    }
    .sub-title {
        font-family: 'DM Sans', sans-serif; text-align: center;
        font-size: 1.05rem; opacity: 0.7; margin-top: 0; margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(58,123,213,0.08); border-radius: 10px; padding: 12px 16px;
    }
    .feature-card {
        background: rgba(58,123,213,0.06); border: 1px solid rgba(58,123,213,0.15);
        border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
    }
    .feature-card h4 { margin: 0 0 0.3rem 0; font-family: 'Space Grotesk', sans-serif; }
    .feature-card p { margin: 0; opacity: 0.75; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
mapcolors_list = px.colors.named_colorscales()
themes = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
RBF_KERNELS = ["thin_plate_spline", "multiquadric", "inverse_multiquadric", "gaussian", "linear", "cubic", "quintic"]

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
_defaults = dict(
    page="home", df=None, df_clean=None,
    pls2=None, pls3=None, RMSE=None, C=None, Y_pred=None,
    trans=None, NC=None, Ccalc=None, Ccalc_r=None,
    var1=None, var2=None, computed=False, model_method=None,
    rbf_model=None, rbf_constraint_model=None,
    predicted_point=None, predicted_extrapolated=False,
    extrap_pending=False, extrap_x1=None, extrap_x2=None,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
# MODEL / MATH HELPERS — PLS
# ═════════════════════════════════════════════════════════════════════════════

def get_model(ENT):
    OR = str(ENT)
    if OR == '2':
        NC = 6
        trans = lambda x1, x2: (1, x1, x2, x1*x1, x2*x2, x1*x2)
    elif OR == '3':
        NC = 10
        trans = lambda x1, x2: (1, x1, x2, x1*x1, x2*x2, x1*x2,
                                 x1**3, x2**3, x1*x1*x2, x2*x2*x1)
    elif OR == '4':
        NC = 15
        trans = lambda x1, x2: (1, x1, x2, x1*x1, x2*x2, x1*x2,
                                 x1**3, x2**3, x1*x1*x2, x2*x2*x1,
                                 x1**4, x2**4, x1**3*x2, x2**3*x1, x1*x1*x2*x2)
    elif OR == '5':
        NC = 22
        trans = lambda x1, x2: (1, x1, x2, x1*x1, x2*x2, x1*x2,
                                 x1**3, x2**3, x1*x1*x2, x2*x2*x1,
                                 x1**4, x2**4, x1**3*x2, x2**3*x1, x1*x1*x2*x2,
                                 x1**5, x1**4*x2, x1**3*x2*x2, x1*x2**4,
                                 x1*x1*x2**3, x2**5, x1*x2**4)
    else:
        NC = 6
        trans = lambda x1, x2: (1, x1, x2, x1*x1, x2*x2, x1*x2)
    return trans, NC


def get_best_NC(dados, trans, NC):
    C = np.zeros((dados.shape[0], NC))
    for i in range(dados.shape[0]):
        C[i, :] = trans(dados.iloc[i, 0], dados.iloc[i, 1])
    RMSE = np.zeros((NC - 1, 1))
    for i in np.arange(1, NC - 1):
        pls = PLSRegression(n_components=i, scale=False)
        pls.fit(C, dados.iloc[:, 2])
        RMSE[i - 1] = np.sqrt(mean_squared_error(dados.iloc[:, 2], pls.predict(C)))
    return RMSE, C


def get_model_PLS(RMSE, dados, C):
    NC2 = max(int(np.argmin(RMSE)), 1)
    pls = PLSRegression(n_components=NC2, scale=True)
    pls.fit(C, dados.iloc[:, 2])
    return pls.predict(C), pls


def get_surface_pls(dados, trans, pls2, pls3, LIM, N=200):
    var1 = np.linspace(dados.iloc[:, 0].min(), dados.iloc[:, 0].max(), N)
    var2 = np.linspace(dados.iloc[:, 1].min(), dados.iloc[:, 1].max(), N)
    Ccalc = np.zeros((N, N))
    Ccalc_r = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            inp = np.array(trans(var1[j], var2[i])).reshape(1, -1)
            p1 = float(pls2.predict(inp).ravel()[0])
            p2 = float(pls3.predict(inp).ravel()[0])
            Ccalc[i, j] = p1
            Ccalc_r[i, j] = p1
            if LIM is not None and p2 < LIM:
                Ccalc_r[i, j] = np.nan
    return Ccalc, Ccalc_r, var1, var2


# ═════════════════════════════════════════════════════════════════════════════
# MODEL / MATH HELPERS — RBF
# ═════════════════════════════════════════════════════════════════════════════

def fit_rbf(dados, kernel="thin_plate_spline", smoothing=0.0):
    """Fit an RBF interpolator from the first two columns → third column."""
    X = dados.iloc[:, :2].values
    y = dados.iloc[:, 2].values
    rbf = RBFInterpolator(X, y, kernel=kernel, smoothing=smoothing)
    Y_pred = rbf(X)
    return Y_pred, rbf


def get_surface_rbf(dados, rbf_model, rbf_constraint, LIM, N=200):
    var1 = np.linspace(dados.iloc[:, 0].min(), dados.iloc[:, 0].max(), N)
    var2 = np.linspace(dados.iloc[:, 1].min(), dados.iloc[:, 1].max(), N)
    grid_x, grid_y = np.meshgrid(var1, var2)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    Ccalc = rbf_model(grid_pts).reshape(N, N)
    Ccalc_r = Ccalc.copy()

    if LIM is not None and rbf_constraint is not None:
        constraint_vals = rbf_constraint(grid_pts).reshape(N, N)
        Ccalc_r[constraint_vals < LIM] = np.nan

    return Ccalc, Ccalc_r, var1, var2


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def figure_rmse(RMSE, theme):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.linspace(1, RMSE.shape[0], RMSE.shape[0] - 1),
        y=RMSE.reshape((-1,)),
        name="Componentes", mode='markers', marker_color='rgba(255,182,193,.9)',
    ))
    fig.update_traces(marker_line_width=2, marker_size=10)
    fig.update_layout(xaxis_title="Número de componentes", yaxis_title="RMSE", template=theme)
    return fig


def figure_pred(dados, Y_pred, theme, method_name="PLS"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=dados.iloc[:, 2].values.ravel(), name="Real",
        mode='markers', marker_color='#6951e2', marker_size=12,
    ))
    fig.add_trace(go.Scatter(
        y=np.asarray(Y_pred).ravel(), name=f"Predito ({method_name})",
        mode='markers', marker_symbol="x", marker_color='#eb6ecc', marker_size=8,
    ))
    fig.update_traces(marker_line_width=1)
    fig.update_layout(xaxis_title="Amostra", yaxis_title="Variável", template=theme)
    return fig


def figure_surface(Ccalc, var1, var2, Ccalc_r, dados, theme, colormap, predicted_point=None):
    """3D surface. If predicted_point=(x1, x2, z), show it as a marker."""
    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=Ccalc, x=var1, y=var2, surfacecolor=Ccalc_r,
        colorscale=colormap, opacity=0.5, showscale=False, name="Superfície (ghost)",
    ))
    fig.add_trace(go.Surface(
        z=Ccalc_r, x=var1, y=var2, colorscale=colormap,
        colorbar=dict(orientation="v", x=0.9, xanchor="right"), name="Superfície",
    ))
    fig.update_traces(contours_z=dict(
        show=True, usecolormap=True, highlightcolor="limegreen", project_z=True,
    ))

    if predicted_point is not None:
        px1, px2, pz = predicted_point
        fig.add_trace(go.Scatter3d(
            x=[px1], y=[px2], z=[pz],
            mode='markers+text', name='Predição',
            text=[f'ẑ={pz}'], textposition='top center',
            textfont=dict(size=12, color='white'),
            marker=dict(size=8, color='#ffdd00', symbol='diamond',
                        line=dict(width=2, color='white')),
        ))

    col_names = dados.columns.tolist()
    fig.update_layout(
        autosize=True,
        scene=dict(
            xaxis=dict(title=col_names[0], showbackground=True),
            yaxis=dict(title=col_names[1], showbackground=True),
            zaxis=dict(title=col_names[2], showbackground=True),
        ),
        height=700, margin=dict(l=5, r=5, b=5, t=5), template=theme,
    )
    return fig


def figure_contour(Ccalc, Ccalc_r, var1, var2, dados, theme, colormap, rest_value=None, predicted_point=None):
    """2D contour plot with optional constraint boundary and predicted point."""
    col_names = dados.columns.tolist()
    fig = go.Figure()

    # Filled contour of the surface
    fig.add_trace(go.Contour(
        z=Ccalc_r, x=var1, y=var2,
        colorscale=colormap,
        contours=dict(showlabels=True, labelfont=dict(size=11, color='white')),
        colorbar=dict(title=col_names[2]),
        name=col_names[2],
    ))

    # Constraint boundary line
    if rest_value is not None:
        fig.add_trace(go.Contour(
            z=Ccalc, x=var1, y=var2,
            contours=dict(
                type='constraint', operation='=', value=rest_value,
                showlabels=True,
            ),
            name=f'Restrição = {rest_value}',
            showscale=False,
            line=dict(width=3, color='#ffdd00', dash='dash'),
        ))

    # Predicted point
    if predicted_point is not None:
        px1, px2, pz = predicted_point
        fig.add_trace(go.Scatter(
            x=[px1], y=[px2],
            mode='markers+text', name=f'Predição (ẑ={pz})',
            text=[f'ẑ={pz}'], textposition='top center',
            textfont=dict(size=12),
            marker=dict(size=14, color='#ffdd00', symbol='star',
                        line=dict(width=2, color='white')),
        ))

    fig.update_layout(
        xaxis_title=col_names[0], yaxis_title=col_names[1],
        height=600, template=theme,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def figure_correlation(df_numeric, theme):
    """Correlation heatmap."""
    corr = df_numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale='RdBu_r', zmin=-1, zmax=1, text=np.round(corr.values, 2),
        texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        height=500, template=theme,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def figure_scatter_matrix(df_numeric, theme):
    """Scatter matrix of numeric columns."""
    fig = px.scatter_matrix(
        df_numeric, dimensions=df_numeric.columns.tolist(),
        opacity=0.6, height=600,
    )
    fig.update_traces(diagonal_visible=True, marker=dict(size=3))
    fig.update_layout(template=theme, margin=dict(l=40, r=10, t=10, b=40))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def col_letter(idx):
    result = ""
    while True:
        result = chr(idx % 26 + ord('A')) + result
        idx = idx // 26 - 1
        if idx < 0:
            break
    return result


def load_raw_sheet(uploaded_file, sheet_name):
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, sheet_name=sheet_name,
                         header=None, engine="openpyxl", dtype=str)


def auto_detect_layout(raw):
    """Guess header row, data start row, start column, and end column.
    
    Header: first row where >= 3 cells are non-empty AND mostly non-numeric text.
    Data start: first row after header where at least one cell is numeric.
    Columns: first and last columns that contain at least one numeric value in the
             data region.
    
    Returns 1-based row numbers and 0-based column indices.
    """
    total_rows, total_cols = raw.shape

    # --- Detect header row ---
    header_row_0 = 0  # fallback: first row
    for r in range(total_rows):
        row_vals = raw.iloc[r]
        non_empty = row_vals.dropna()
        non_empty = non_empty[non_empty.astype(str).str.strip() != '']
        if len(non_empty) < 3:
            continue
        # Check that most non-empty cells are NOT purely numeric
        numeric_count = sum(1 for v in non_empty if _is_numeric_str(str(v)))
        text_count = len(non_empty) - numeric_count
        if text_count >= numeric_count:
            header_row_0 = r
            break

    # --- Detect data start row ---
    data_start_0 = header_row_0 + 1
    for r in range(header_row_0 + 1, total_rows):
        row_vals = raw.iloc[r]
        if any(_is_numeric_str(str(v)) for v in row_vals.dropna()):
            data_start_0 = r
            break

    # --- Detect column range (based on data region) ---
    data_region = raw.iloc[data_start_0:]
    first_col, last_col = 0, total_cols - 1

    for c in range(total_cols):
        col_vals = data_region.iloc[:, c].dropna()
        numeric_count = sum(1 for v in col_vals if _is_numeric_str(str(v)))
        if numeric_count >= 1:
            first_col = c
            break

    for c in range(total_cols - 1, -1, -1):
        col_vals = data_region.iloc[:, c].dropna()
        numeric_count = sum(1 for v in col_vals if _is_numeric_str(str(v)))
        if numeric_count >= 1:
            last_col = c
            break

    return header_row_0 + 1, data_start_0 + 1, first_col, last_col


def _is_numeric_str(s):
    """Check if a string represents a number."""
    s = s.strip()
    if not s:
        return False
    try:
        float(s.replace(',', '.'))
        return True
    except ValueError:
        return False


def go_to(page):
    st.session_state["page"] = page


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════════════════

def page_home():

    st.markdown('<p class="main-title">📊 Módulo de Visualização de Superfícies</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Gere superfícies 3D interpolando via PLS ou RBF — faça upload de um arquivo Excel para começar</p>', unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="feature-card"><h4>📂 Upload flexível</h4>
        <p>Dados em qualquer posição — escolha planilha, cabeçalho e colunas.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="feature-card"><h4>📈 PLS + RBF</h4>
        <p>Dois métodos de interpolação com validação automática.</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="feature-card"><h4>🌐 3D + Contorno 2D</h4>
        <p>Superfície rotacionável e mapa de contorno com restrições.</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="feature-card"><h4>🔬 Exploração de dados</h4>
        <p>Estatísticas, correlação e scatter matrix antes de modelar.</p></div>""", unsafe_allow_html=True)

    st.divider()

    # ── Step 1: Upload ──
    st.markdown("### 1 · Carregar arquivo")
    uploaded_file = st.file_uploader(
        "Selecione um arquivo Excel (.xlsx)", type=["xlsx"],
        help="Faça upload de um arquivo .xlsx.", key="file_uploader",
    )
    if uploaded_file is None:
        st.info("Faça upload de um arquivo Excel para continuar.")
        return

    # Sheet selector
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    sheet_names = xls.sheet_names

    st.markdown("### 2 · Configurar localização dos dados")
    col_s1, _ = st.columns([1, 2])
    with col_s1:
        sheet = st.selectbox("📄 Planilha", options=sheet_names, key="sel_sheet") if len(sheet_names) > 1 else sheet_names[0]
        if len(sheet_names) == 1:
            st.markdown(f"📄 Planilha: **{sheet}**")

    raw = load_raw_sheet(uploaded_file, sheet)
    total_rows, total_cols = raw.shape

    with st.expander("👀 Pré-visualização (primeiras 25 linhas)", expanded=True):
        preview = raw.head(25).copy()
        preview.index = range(1, len(preview) + 1)
        preview.columns = [col_letter(i) for i in range(preview.shape[1])]
        st.dataframe(preview, use_container_width=True, height=320)
        st.caption(f"Arquivo: **{total_rows}** linhas × **{total_cols}** colunas")

    # Auto-detect layout
    det_header, det_data, det_sc, det_ec = auto_detect_layout(raw)

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    letters = [col_letter(i) for i in range(total_cols)]
    with col_h1:
        header_row = st.number_input("Linha do cabeçalho", 1, total_rows, det_header, key="sel_header_row",
                                     help="Detectado automaticamente — ajuste se necessário.")
    with col_h2:
        det_data_clamped = max(header_row + 1, det_data)
        data_start = st.number_input("Início dos dados", header_row + 1, total_rows,
                                     min(det_data_clamped, total_rows), key="sel_data_start",
                                     help="Detectado automaticamente — ajuste se necessário.")
    with col_h3:
        start_col = st.selectbox("Coluna inicial", letters, det_sc, key="sel_col_start")
    with col_h4:
        end_col = st.selectbox("Coluna final", letters, det_ec, key="sel_col_end")

    sc, ec = letters.index(start_col), letters.index(end_col)
    if ec < sc:
        st.error("A coluna final deve ser posterior à inicial.")
        return

    # Parse
    hdr = raw.iloc[header_row - 1, sc:ec + 1].values
    col_names = [str(c).strip() if pd.notna(c) else f"Col_{col_letter(sc + i)}" for i, c in enumerate(hdr)]
    data_slice = raw.iloc[data_start - 1:, sc:ec + 1].copy()
    data_slice.columns = col_names
    for c in data_slice.columns:
        data_slice[c] = pd.to_numeric(data_slice[c], errors="coerce")
    data_slice = data_slice.dropna(how="all").reset_index(drop=True)

    # Empty columns
    empty_cols = [c for c in data_slice.columns if data_slice[c].isna().all()]
    if empty_cols:
        st.markdown("#### ⚠️ Colunas vazias detectadas")
        st.warning(f"Colunas vazias: **{', '.join(empty_cols)}**")
        if st.radio("Remover?", ["Sim, remover colunas vazias", "Não, manter"],
                    index=0, key="remove_empty", horizontal=True) == "Sim, remover colunas vazias":
            data_slice = data_slice.drop(columns=empty_cols)
            st.success(f"✅ {len(empty_cols)} coluna(s) removida(s).")

    # Parsed preview
    st.markdown("### 3 · Tabela interpretada")
    st.dataframe(data_slice.head(20), use_container_width=True, height=300)

    numeric_cols = list(data_slice.select_dtypes(include=[np.number]).columns)
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Linhas", data_slice.shape[0])
    with c2: st.metric("Colunas numéricas", len(numeric_cols))
    with c3: st.metric("Colunas não-numéricas", data_slice.shape[1] - len(numeric_cols))

    if len(numeric_cols) < 3:
        st.error("São necessárias pelo menos **3 colunas numéricas**.")
        return

    # ── Data Exploration ──
    st.divider()
    st.markdown("### 4 · Exploração de dados")

    tab_stats, tab_corr, tab_scatter = st.tabs(["📋 Estatísticas", "🔥 Correlação", "🔵 Scatter Matrix"])

    df_num = data_slice[numeric_cols]

    with tab_stats:
        # Condensed table: count, mean, std, min, max (no quartiles — those go in the boxplot)
        stats_df = df_num.describe().T[["count", "mean", "std", "min", "max"]]
        st.dataframe(stats_df.style.format({"count": "{:.0f}", "mean": "{:.4f}",
                                            "std": "{:.4f}", "min": "{:.4f}", "max": "{:.4f}"}),
                     use_container_width=True)

        # Interactive box plots for distribution / quartiles
        st.markdown("**Distribuição por variável**")
        fig_box = go.Figure()
        for col in numeric_cols:
            fig_box.add_trace(go.Box(y=df_num[col].dropna(), name=col, boxmean='sd'))
        fig_box.update_layout(
            height=450, template="plotly_dark",
            yaxis_title="Valor", showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True, key="fig_box")

    with tab_corr:
        fig_corr = figure_correlation(df_num, "plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True, key="fig_corr")

    with tab_scatter:
        # Limit to max 8 columns for readability
        scatter_cols = numeric_cols[:8]
        if len(numeric_cols) > 8:
            st.caption(f"Exibindo as primeiras 8 de {len(numeric_cols)} colunas para legibilidade.")
        fig_sm = figure_scatter_matrix(data_slice[scatter_cols], "plotly_dark")
        st.plotly_chart(fig_sm, use_container_width=True, key="fig_scatter")

    # Store and navigate
    st.session_state["df"] = data_slice
    st.session_state["computed"] = False

    st.divider()
    st.markdown("### ✅ Dados prontos!")
    st.success(f"**{data_slice.shape[0]}** linhas × **{len(numeric_cols)}** colunas numéricas disponíveis.")
    st.button("🚀 Ir para análise de superfície", on_click=go_to, args=("analysis",),
              use_container_width=True, type="primary")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def page_analysis():

    df = st.session_state.get("df")
    if df is None:
        st.warning("Nenhum dado carregado.")
        st.button("← Voltar para Home", on_click=go_to, args=("home",))
        return

    st.markdown('<p class="main-title">📈 Análise de Superfície</p>', unsafe_allow_html=True)
    st.button("← Voltar para Home", on_click=go_to, args=("home",), key="back_top")
    st.divider()

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    # ── Model config ──
    st.markdown("### Configuração do modelo")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        x1 = st.selectbox("Eixo x₁", numeric_cols, 0, key="sel_x1")
    with col_m2:
        x2 = st.selectbox("Eixo x₂", numeric_cols, min(1, len(numeric_cols)-1), key="sel_x2")
    with col_m3:
        x3 = st.selectbox("Eixo z (resposta)", numeric_cols, min(2, len(numeric_cols)-1), key="sel_x3")

    # Method selection
    col_method, col_mp1, col_mp2 = st.columns(3)
    with col_method:
        method = st.selectbox("Método de interpolação", ["PLS (Polinomial)", "RBF (Radial Basis Function)"],
                              key="sel_method")
    with col_mp1:
        if method == "PLS (Polinomial)":
            order = st.selectbox("Ordem do polinômio", [2, 3, 4, 5], 1, key="sel_order")
        else:
            rbf_kernel = st.selectbox("Kernel RBF", RBF_KERNELS, 0, key="sel_rbf_kernel")
    with col_mp2:
        if method == "PLS (Polinomial)":
            st.markdown("")  # spacer
        else:
            rbf_smooth = st.number_input("Smoothing", 0.0, 100.0, 0.0, step=0.1,
                                         help="0 = interpolação exata. Valores maiores suavizam o ajuste.",
                                         key="sel_rbf_smooth")

    # Constraint
    col_r1, col_r2 = st.columns(2)
    rest_options = numeric_cols + ["Problema sem restrições"]
    with col_r1:
        rest_name = st.selectbox("Variável com restrição", rest_options,
                                 len(rest_options)-1, key="sel_rest")
    rest_value = None
    with col_r2:
        if rest_name != "Problema sem restrições":
            rest_value = st.number_input("Valor mínimo", 0.0, key="sel_rest_val")

    # Appearance
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        theme = st.selectbox("Template", themes, 0, key="sel_theme")
    with col_t2:
        colormap = st.selectbox("Mapa de cores", mapcolors_list,
                                mapcolors_list.index("viridis") if "viridis" in mapcolors_list else 0,
                                key="sel_cmap")

    # ── Compute ──
    compute_btn = st.button("🚀 Calcular superfície", use_container_width=True, type="primary")

    if compute_btn:
        tags = [x1, x2, x3]
        df_clean = df[tags].dropna()
        if df_clean.shape[0] < 5:
            st.error(f"Apenas {df_clean.shape[0]} linhas válidas — insuficiente.")
            return

        with st.spinner("Calculando modelo e gerando superfície…"):

            if method == "PLS (Polinomial)":
                trans, NC = get_model(order)
                st.session_state["trans"] = trans
                st.session_state["NC"] = NC

                if rest_name == "Problema sem restrições":
                    RMSE, C = get_best_NC(df_clean, trans, NC)
                    Y_pred, pls2 = get_model_PLS(RMSE, df_clean, C)
                    pls3 = pls2
                else:
                    tags2 = [x1, x2, rest_name]
                    df_clean2 = df[tags2].dropna()
                    RMSE, C = get_best_NC(df_clean, trans, NC)
                    RMSE2, C2 = get_best_NC(df_clean2, trans, NC)
                    Y_pred, pls2 = get_model_PLS(RMSE, df_clean, C)
                    _, pls3 = get_model_PLS(RMSE2, df_clean2, C2)

                Ccalc, Ccalc_r, var1, var2 = get_surface_pls(df_clean, trans, pls2, pls3, rest_value)
                st.session_state.update(dict(
                    pls2=pls2, pls3=pls3, RMSE=RMSE, C=C, Y_pred=Y_pred,
                    Ccalc=Ccalc, Ccalc_r=Ccalc_r, var1=var1, var2=var2,
                    computed=True, df_clean=df_clean, model_method="PLS",
                    rbf_model=None, rbf_constraint_model=None,
                    predicted_point=None, predicted_extrapolated=False,
                    extrap_pending=False,
                ))

            else:  # RBF
                Y_pred, rbf_model = fit_rbf(df_clean, kernel=rbf_kernel, smoothing=rbf_smooth)
                rbf_constraint = None
                if rest_name != "Problema sem restrições":
                    tags2 = [x1, x2, rest_name]
                    df_clean2 = df[tags2].dropna()
                    _, rbf_constraint = fit_rbf(df_clean2, kernel=rbf_kernel, smoothing=rbf_smooth)

                Ccalc, Ccalc_r, var1, var2 = get_surface_rbf(
                    df_clean, rbf_model, rbf_constraint, rest_value)

                # Compute RMSE for display
                rmse_val = np.sqrt(mean_squared_error(df_clean.iloc[:, 2], Y_pred))

                st.session_state.update(dict(
                    pls2=None, pls3=None, RMSE=None, C=None, Y_pred=Y_pred,
                    Ccalc=Ccalc, Ccalc_r=Ccalc_r, var1=var1, var2=var2,
                    computed=True, df_clean=df_clean, model_method="RBF",
                    rbf_model=rbf_model, rbf_constraint_model=rbf_constraint,
                    rbf_rmse=rmse_val, predicted_point=None,
                    predicted_extrapolated=False, extrap_pending=False,
                ))

        st.success("Superfície calculada com sucesso!")
        st.rerun()

    # ── Results ──
    if not st.session_state.get("computed"):
        st.divider()
        st.markdown(
            """<div style="text-align:center; padding:3rem 2rem; opacity:0.5;">
            <p style="font-size:2.5rem;">🔬</p>
            <h3>Configure o modelo acima e clique em "Calcular superfície"</h3>
            </div>""", unsafe_allow_html=True)
        return

    # Load results
    tags = [st.session_state["sel_x1"], st.session_state["sel_x2"], st.session_state["sel_x3"]]
    df_clean = st.session_state["df_clean"]
    theme = st.session_state["sel_theme"]
    colormap = st.session_state["sel_cmap"]
    Y_pred = st.session_state["Y_pred"]
    Ccalc = st.session_state["Ccalc"]
    Ccalc_r = st.session_state["Ccalc_r"]
    var1 = st.session_state["var1"]
    var2 = st.session_state["var2"]
    model_method = st.session_state["model_method"]

    # Read predicted point from session (persists across reruns)
    predicted_point = st.session_state.get("predicted_point")
    predicted_extrapolated = st.session_state.get("predicted_extrapolated", False)

    # Only show point on figures if it's within model bounds
    figure_point = predicted_point if (predicted_point and not predicted_extrapolated) else None

    st.divider()

    # ── Validation ──
    st.markdown(f"### Validação do modelo ({model_method})")

    if model_method == "PLS":
        RMSE = st.session_state["RMSE"]
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig1 = figure_rmse(RMSE, theme)
            st.plotly_chart(fig1, use_container_width=True, key="fig_rmse")
        with col_v2:
            fig2 = figure_pred(df_clean, Y_pred, theme, "PLS")
            st.plotly_chart(fig2, use_container_width=True, key="fig_pred")
    else:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            rbf_rmse = st.session_state.get("rbf_rmse", 0)
            st.metric("RMSE (RBF)", f"{rbf_rmse:.6f}")
            r2 = 1 - (np.sum((df_clean.iloc[:, 2].values - np.asarray(Y_pred).ravel())**2) /
                       np.sum((df_clean.iloc[:, 2].values - df_clean.iloc[:, 2].mean())**2))
            st.metric("R²", f"{r2:.6f}")
        with col_v2:
            fig2 = figure_pred(df_clean, Y_pred, theme, "RBF")
            st.plotly_chart(fig2, use_container_width=True, key="fig_pred")

    st.divider()

    # ── 3D Surface ──
    st.markdown("### Superfície 3D")
    fig3 = figure_surface(Ccalc, var1, var2, Ccalc_r, df_clean, theme, colormap, figure_point)
    st.plotly_chart(fig3, use_container_width=True, key="fig_surf")

    st.divider()

    # ── 2D Contour ──
    st.markdown("### Mapa de contorno 2D")
    fig_cont = figure_contour(Ccalc, Ccalc_r, var1, var2, df_clean, theme, colormap, rest_value, figure_point)
    st.plotly_chart(fig_cont, use_container_width=True, key="fig_contour")

    st.divider()

    # ── Point prediction ──
    st.markdown("### Predição pontual")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        x1_val = st.number_input(f"Valor de {tags[0]}", value=float(df_clean.iloc[:, 0].mean()), key="x1_pred")
    with col_p2:
        x2_val = st.number_input(f"Valor de {tags[1]}", value=float(df_clean.iloc[:, 1].mean()), key="x2_pred")
    with col_p3:
        predict_btn = st.button("Calcular predição", use_container_width=True)

    # Check if point is within data range
    def _is_extrapolated(x1v, x2v, dfc):
        x1_min, x1_max = dfc.iloc[:, 0].min(), dfc.iloc[:, 0].max()
        x2_min, x2_max = dfc.iloc[:, 1].min(), dfc.iloc[:, 1].max()
        return not (x1_min <= x1v <= x1_max and x2_min <= x2v <= x2_max)

    def _compute_prediction(x1v, x2v, m_method, extrapolated):
        if m_method == "PLS":
            trans = st.session_state["trans"]
            NC = st.session_state["NC"]
            pls3 = st.session_state["pls3"]
            Chat = np.zeros((1, NC))
            Chat[0, :] = trans(x1v, x2v)
            return round(float(np.asarray(pls3.predict(Chat)).ravel()[0]), 4)
        else:
            rbf = st.session_state["rbf_model"]
            return round(float(rbf(np.array([[x1v, x2v]])).ravel()[0]), 4)

    if predict_btn:
        is_extrap = _is_extrapolated(x1_val, x2_val, df_clean)

        if is_extrap:
            # Store pending state and rerun to show confirmation dialog
            st.session_state["extrap_pending"] = True
            st.session_state["extrap_x1"] = x1_val
            st.session_state["extrap_x2"] = x2_val
            st.rerun()
        else:
            # Within range — compute and store immediately
            y_hat = _compute_prediction(x1_val, x2_val, model_method, False)
            st.session_state["predicted_point"] = (x1_val, x2_val, y_hat)
            st.session_state["predicted_extrapolated"] = False
            st.session_state["extrap_pending"] = False
            st.rerun()

    # Handle pending extrapolation confirmation
    if st.session_state.get("extrap_pending"):
        ep_x1 = st.session_state["extrap_x1"]
        ep_x2 = st.session_state["extrap_x2"]

        st.warning(
            f"⚠️ **Estimativa extrapolada fora da range do modelo.** "
            f"O ponto ({tags[0]}={ep_x1}, {tags[1]}={ep_x2}) está fora do intervalo dos dados. "
            f"Proceder? Avalie o resultado com cuidado."
        )
        col_yes, col_no, _ = st.columns([1, 1, 3])
        with col_yes:
            if st.button("✅ Sim, calcular mesmo assim", use_container_width=True):
                y_hat = _compute_prediction(ep_x1, ep_x2, model_method, True)
                st.session_state["predicted_point"] = (ep_x1, ep_x2, y_hat)
                st.session_state["predicted_extrapolated"] = True
                st.session_state["extrap_pending"] = False
                st.rerun()
        with col_no:
            if st.button("❌ Não, cancelar", use_container_width=True):
                st.session_state["extrap_pending"] = False
                st.rerun()

    # Show prediction result (if one exists in session)
    if predicted_point is not None:
        px1, px2, pz = predicted_point

        if predicted_extrapolated:
            st.warning(f"⚠️ **ẑ = {pz}** — valor extrapolado fora dos limites do modelo. Interprete com cautela.")
        else:
            st.metric(label="ẑ = f(x₁, x₂)", value=pz)

        if rest_name != "Problema sem restrições" and rest_value is not None:
            if pz >= rest_value:
                st.success(f"**ẑ = {pz}** → estado **viável** (≥ {rest_value})")
            else:
                st.error(f"**ẑ = {pz}** → estado **não viável** (< {rest_value})")

    st.divider()

    # ── Equation + Downloads ──
    st.markdown("### Equação e downloads")
    if model_method == "PLS":
        st.latex(r"\hat{y} = \left(\frac{x - \mu}{\sigma}\right) \cdot \tilde{B} + B_0")
    else:
        st.latex(r"\hat{y} = \sum_{i=1}^{N} w_i \, \phi\!\left(\|x - x_i\|\right)")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        if model_method == "PLS":
            pls2 = st.session_state["pls2"]
            try:
                x_mean = np.asarray(pls2._x_mean).ravel()
                x_std = np.asarray(pls2._x_std).ravel()
                coefs = np.asarray(pls2.coef_).ravel()
                intercept_val = float(np.asarray(pls2.intercept_).ravel()[0])
                n = len(x_mean)
                ic = np.full(n, np.nan); ic[0] = intercept_val
                par_df = pd.DataFrame({"mean": x_mean, "std": x_std, "coef": coefs[:n], "intercept": ic})
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
                    par_df.to_excel(w, sheet_name="parameters-PLS", index=False)
                buf.seek(0)
                st.download_button("📥 Baixar parâmetros PLS (.xlsx)", buf,
                                   "parametros_PLS.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            except Exception as e:
                st.warning(f"Parâmetros indisponíveis: {e}")
        else:
            st.info("Para RBF, os parâmetros são os pesos do interpolador (download não aplicável).")

    with col_d2:
        try:
            all_figs = [("real_vs_predicted", fig2), ("3D_surface", fig3), ("contour_2D", fig_cont)]
            if model_method == "PLS":
                all_figs.insert(0, ("RMSE_components", fig1))

            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zipf:
                for name, fig in all_figs:
                    zipf.writestr(f"{name}.html", fig.to_html(include_plotlyjs="cdn", full_html=True))
            zbuf.seek(0)
            st.download_button("📥 Baixar figuras (.zip)", zbuf,
                               "figuras.zip", "application/zip", use_container_width=True)
        except Exception as e:
            st.warning(f"Erro ao gerar figuras: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state["page"] == "home":
    page_home()
else:
    page_analysis()
