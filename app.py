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
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-title {
        font-family: 'DM Sans', sans-serif;
        text-align: center;
        font-size: 1.05rem;
        opacity: 0.7;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(58, 123, 213, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .feature-card {
        background: rgba(58, 123, 213, 0.06);
        border: 1px solid rgba(58, 123, 213, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .feature-card h4 {
        margin: 0 0 0.3rem 0;
        font-family: 'Space Grotesk', sans-serif;
    }
    .feature-card p {
        margin: 0;
        opacity: 0.75;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
mapcolors_list = px.colors.named_colorscales()
themes = [
    "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn",
    "simple_white", "none",
]

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
_defaults = dict(
    page="home", df=None, df_clean=None,
    pls2=None, pls3=None, RMSE=None, C=None, Y_pred=None,
    trans=None, NC=None, Ccalc=None, Ccalc_r=None,
    var1=None, var2=None, computed=False,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
# MODEL / MATH HELPERS
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
        x1 = dados.iloc[i, 0]
        x2 = dados.iloc[i, 1]
        C[i, :] = trans(x1, x2)
    RMSE = np.zeros((NC - 1, 1))
    for i in np.arange(1, NC - 1):
        pls2 = PLSRegression(n_components=i, scale=False)
        pls2.fit(C, dados.iloc[:, 2])
        Y_pred = pls2.predict(C)
        RMSE[i - 1] = np.sqrt(mean_squared_error(dados.iloc[:, 2], Y_pred))
    return RMSE, C


def get_model_PLS(RMSE, dados, C):
    NC2 = max(int(np.argmin(RMSE)), 1)
    pls2 = PLSRegression(n_components=NC2, scale=True)
    pls2.fit(C, dados.iloc[:, 2])
    Y_pred = pls2.predict(C)
    return Y_pred, pls2


def get_surface(dados, trans, pls2, pls3, LIM):
    A = copy.deepcopy(dados)
    N1, N2 = 200, 200
    var1 = np.linspace(A.iloc[:, 0].min(), A.iloc[:, 0].max(), N1)
    var2 = np.linspace(A.iloc[:, 1].min(), A.iloc[:, 1].max(), N2)
    Ccalc = np.zeros((N2, N1))
    Ccalc_r = np.zeros((N2, N1))
    for i in range(N2):
        for j in range(N1):
            inp = np.array(trans(var1[j], var2[i])).reshape(1, -1)
            p1 = float(pls2.predict(inp).ravel()[0])
            p2 = float(pls3.predict(inp).ravel()[0])
            Ccalc[i, j] = p1
            Ccalc_r[i, j] = p1
            if LIM is not None and p2 < LIM:
                Ccalc_r[i, j] = np.nan
    return Ccalc, Ccalc_r, var1, var2


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def figure1(RMSE, theme):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.linspace(1, RMSE.shape[0], RMSE.shape[0] - 1),
        y=RMSE.reshape((-1,)),
        name="Número de componentes",
        mode='markers', marker_color='rgba(255,182,193,.9)',
    ))
    fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    fig.update_layout(xaxis_title="Número de componentes", yaxis_title="RMSE", template=theme)
    return fig


def figure2(dados, Y_pred, theme):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=dados.iloc[:, 2].values.reshape((-1,)),
        name="Real", mode='markers', marker_color='#6951e2', marker_size=12,
    ))
    fig.add_trace(go.Scatter(
        y=Y_pred.reshape((-1,)),
        name="Predito", mode='markers', marker_symbol="x",
        marker_color='#eb6ecc', marker_size=8,
    ))
    fig.update_traces(mode='markers', marker_line_width=1)
    fig.update_layout(xaxis_title="Amostra", yaxis_title="Variável", template=theme)
    return fig


def figure3(Ccalc, var1, var2, Ccalc_r, dados, theme, colormap):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Surface(
        z=Ccalc, x=var1, y=var2, surfacecolor=Ccalc_r,
        colorscale=colormap, opacity=0.5, showscale=False,
    ))
    fig.add_trace(go.Surface(
        z=Ccalc_r, x=var1, y=var2, colorscale=colormap,
        colorbar=dict(orientation="v", x=0.9, xanchor="right"),
    ))
    fig.update_traces(contours_z=dict(
        show=True, usecolormap=True, highlightcolor="limegreen", project_z=True,
    ))
    fig.update_layout(
        autosize=True,
        scene=dict(
            xaxis=dict(title=dados.columns.tolist()[0], showbackground=True),
            yaxis=dict(title=dados.columns.tolist()[1], showbackground=True),
            zaxis=dict(title=dados.columns.tolist()[2], showbackground=True),
        ),
        height=700, margin=dict(l=5, r=5, b=5, t=5), template=theme,
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def col_letter(idx):
    """0-based index → Excel column letter (0→A, 25→Z, 26→AA)."""
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


def go_to(page):
    st.session_state["page"] = page


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════════════════

def page_home():

    # ── Header ──
    st.markdown('<p class="main-title">📊 Módulo de Visualização de Superfícies</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Gere superfícies 3D interpolando via PLS — faça upload de um arquivo Excel para começar</p>', unsafe_allow_html=True)

    # ── Feature cards ──
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""<div class="feature-card">
            <h4>📂 Upload flexível</h4>
            <p>Aceita arquivos .xlsx com dados em qualquer posição — escolha a planilha, linha de cabeçalho e intervalo de colunas.</p>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="feature-card">
            <h4>📈 Regressão PLS</h4>
            <p>Interpolação polinomial de ordem 2 a 5 com validação automática via RMSE e comparação real × predito.</p>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown("""<div class="feature-card">
            <h4>🌐 Superfície 3D interativa</h4>
            <p>Visualização rotacionável com mapas de cores, restrições de viabilidade e predição pontual.</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Step 1: File upload ──
    st.markdown("### 1 · Carregar arquivo")
    uploaded_file = st.file_uploader(
        "Selecione um arquivo Excel (.xlsx)",
        type=["xlsx"],
        help="Faça upload de um arquivo .xlsx com os dados para gerar a superfície.",
        key="file_uploader",
    )

    if uploaded_file is None:
        st.info("Faça upload de um arquivo Excel para continuar.")
        return

    # ── Sheet selector ──
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    sheet_names = xls.sheet_names

    st.markdown("### 2 · Configurar localização dos dados")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        if len(sheet_names) > 1:
            sheet = st.selectbox("📄 Planilha", options=sheet_names, key="sel_sheet")
        else:
            sheet = sheet_names[0]
            st.markdown(f"📄 Planilha: **{sheet}**")

    raw = load_raw_sheet(uploaded_file, sheet)
    total_rows, total_cols = raw.shape

    # ── Raw preview ──
    with st.expander("👀 Pré-visualização do arquivo (primeiras 25 linhas)", expanded=True):
        preview = raw.head(25).copy()
        preview.index = range(1, len(preview) + 1)
        preview.columns = [col_letter(i) for i in range(preview.shape[1])]
        st.dataframe(preview, use_container_width=True, height=320)
        st.caption(f"Arquivo: **{total_rows}** linhas × **{total_cols}** colunas")

    # ── Location selectors ──
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_letters_all = [col_letter(i) for i in range(total_cols)]

    with col_h1:
        header_row = st.number_input(
            "Linha do cabeçalho", min_value=1, max_value=total_rows,
            value=1, step=1,
            help="Linha (como no Excel) onde estão os nomes das variáveis.",
            key="sel_header_row",
        )
    with col_h2:
        data_start = st.number_input(
            "Início dos dados",
            min_value=header_row + 1, max_value=total_rows,
            value=min(header_row + 1, total_rows), step=1,
            help="Primeira linha com dados numéricos.",
            key="sel_data_start",
        )
    with col_h3:
        start_col = st.selectbox("Coluna inicial", options=col_letters_all,
                                 index=0, key="sel_col_start")
    with col_h4:
        end_col = st.selectbox("Coluna final", options=col_letters_all,
                               index=total_cols - 1, key="sel_col_end")

    start_col_idx = col_letters_all.index(start_col)
    end_col_idx = col_letters_all.index(end_col)

    if end_col_idx < start_col_idx:
        st.error("A coluna final deve ser igual ou posterior à coluna inicial.")
        return

    # ── Parse data ──
    header_idx = header_row - 1
    data_start_idx = data_start - 1

    header_values = raw.iloc[header_idx, start_col_idx:end_col_idx + 1].values
    col_names = [str(c).strip() if pd.notna(c) else f"Col_{col_letter(start_col_idx + i)}"
                 for i, c in enumerate(header_values)]

    data_slice = raw.iloc[data_start_idx:, start_col_idx:end_col_idx + 1].copy()
    data_slice.columns = col_names

    for c in data_slice.columns:
        data_slice[c] = pd.to_numeric(data_slice[c], errors="coerce")
    data_slice = data_slice.dropna(how="all").reset_index(drop=True)

    # ── Detect and handle empty columns ──
    empty_cols = [c for c in data_slice.columns if data_slice[c].isna().all()]

    if empty_cols:
        st.markdown("#### ⚠️ Colunas vazias detectadas")
        st.warning(
            f"As seguintes **{len(empty_cols)}** coluna(s) estão completamente vazias: "
            f"**{', '.join(empty_cols)}**"
        )
        remove_choice = st.radio(
            "Deseja remover essas colunas?",
            options=["Sim, remover colunas vazias", "Não, manter todas as colunas"],
            index=0, key="remove_empty_cols", horizontal=True,
        )
        if remove_choice == "Sim, remover colunas vazias":
            data_slice = data_slice.drop(columns=empty_cols)
            st.success(f"✅ {len(empty_cols)} coluna(s) vazia(s) removida(s).")

    # ── Step 3: Parsed preview ──
    st.markdown("### 3 · Tabela interpretada")
    st.dataframe(data_slice.head(20), use_container_width=True, height=300)

    numeric_cols = list(data_slice.select_dtypes(include=[np.number]).columns)
    non_numeric = [c for c in data_slice.columns if c not in numeric_cols]

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.metric("Linhas", data_slice.shape[0])
    with col_i2:
        st.metric("Colunas numéricas", len(numeric_cols))
    with col_i3:
        st.metric("Colunas não-numéricas", len(non_numeric))

    if len(numeric_cols) < 3:
        st.error("São necessárias pelo menos **3 colunas numéricas** para gerar a superfície.")
        return

    # ── Store and navigate ──
    st.session_state["df"] = data_slice
    st.session_state["computed"] = False

    st.divider()
    st.markdown("### ✅ Dados prontos!")
    st.success(
        f"**{data_slice.shape[0]}** linhas × **{len(numeric_cols)}** colunas numéricas "
        f"disponíveis para análise."
    )
    st.button(
        "🚀 Ir para análise de superfície",
        on_click=go_to, args=("analysis",),
        use_container_width=True, type="primary",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def page_analysis():

    df = st.session_state.get("df")

    if df is None:
        st.warning("Nenhum dado carregado. Volte para a página inicial para fazer upload.")
        st.button("← Voltar para Home", on_click=go_to, args=("home",))
        return

    # ── Header ──
    st.markdown('<p class="main-title">📈 Análise de Superfície</p>', unsafe_allow_html=True)
    st.button("← Voltar para Home", on_click=go_to, args=("home",), key="back_top")

    st.divider()

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

    # ── Model configuration ──
    st.markdown("### Configuração do modelo")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        x1 = st.selectbox("Eixo x₁", options=numeric_cols, index=0, key="sel_x1")
    with col_m2:
        x2 = st.selectbox("Eixo x₂", options=numeric_cols,
                          index=min(1, len(numeric_cols) - 1), key="sel_x2")
    with col_m3:
        x3 = st.selectbox("Eixo z (resposta)", options=numeric_cols,
                          index=min(2, len(numeric_cols) - 1), key="sel_x3")
    with col_m4:
        order = st.selectbox("Ordem do polinômio", options=[2, 3, 4, 5], index=1, key="sel_order")

    # ── Constraint ──
    col_r1, col_r2 = st.columns(2)
    rest_options = numeric_cols + ["Problema sem restrições"]
    with col_r1:
        rest_name = st.selectbox("Variável com restrição", options=rest_options,
                                 index=len(rest_options) - 1, key="sel_rest")
    rest_value = None
    with col_r2:
        if rest_name != "Problema sem restrições":
            rest_value = st.number_input("Valor mínimo da restrição", value=0.0, key="sel_rest_val")

    # ── Appearance ──
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        theme = st.selectbox("Template da figura", options=themes, index=0, key="sel_theme")
    with col_t2:
        colormap = st.selectbox("Mapa de cores", options=mapcolors_list,
                                index=mapcolors_list.index("viridis") if "viridis" in mapcolors_list else 0,
                                key="sel_cmap")

    # ── Compute ──
    compute_btn = st.button("🚀 Calcular superfície", use_container_width=True, type="primary")

    if compute_btn:
        tags = [x1, x2, x3]
        df_clean = df[tags].dropna()

        if df_clean.shape[0] < 5:
            st.error(f"Dados insuficientes: apenas {df_clean.shape[0]} linhas válidas.")
            return

        with st.spinner("Calculando modelo PLS e gerando superfície…"):
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

            Ccalc, Ccalc_r, var1, var2 = get_surface(df_clean, trans, pls2, pls3, rest_value)

            st.session_state.update(dict(
                pls2=pls2, pls3=pls3, RMSE=RMSE, C=C,
                Y_pred=Y_pred, Ccalc=Ccalc, Ccalc_r=Ccalc_r,
                var1=var1, var2=var2, computed=True, df_clean=df_clean,
            ))

        st.success("Superfície calculada com sucesso!")
        st.rerun()

    # ── Results ──
    if not st.session_state.get("computed"):
        st.divider()
        st.markdown(
            """<div style="text-align:center; padding: 3rem 2rem; opacity: 0.5;">
                <p style="font-size: 2.5rem;">🔬</p>
                <h3>Configure o modelo acima e clique em "Calcular superfície"</h3>
            </div>""", unsafe_allow_html=True,
        )
        return

    # ── Load computed results ──
    tags = [st.session_state["sel_x1"], st.session_state["sel_x2"], st.session_state["sel_x3"]]
    df_clean = st.session_state["df_clean"]
    theme = st.session_state["sel_theme"]
    colormap = st.session_state["sel_cmap"]
    RMSE = st.session_state["RMSE"]
    Y_pred = st.session_state["Y_pred"]
    Ccalc = st.session_state["Ccalc"]
    Ccalc_r = st.session_state["Ccalc_r"]
    var1 = st.session_state["var1"]
    var2 = st.session_state["var2"]

    st.divider()

    # ── Validation plots ──
    st.markdown("### Validação do modelo")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig1 = figure1(RMSE, theme)
        st.plotly_chart(fig1, use_container_width=True, key="fig_rmse")
    with col_v2:
        fig2 = figure2(df_clean, Y_pred, theme)
        st.plotly_chart(fig2, use_container_width=True, key="fig_pred")

    st.divider()

    # ── 3D Surface ──
    st.markdown("### Superfície gerada")
    fig3 = figure3(Ccalc, var1, var2, Ccalc_r, df_clean, theme, colormap)
    st.plotly_chart(fig3, use_container_width=True, key="fig_surf")

    st.divider()

    # ── Point prediction ──
    st.markdown("### Predição pontual")
    col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
    with col_p1:
        x1_val = st.number_input(f"Valor de {tags[0]}",
                                 value=float(df_clean.iloc[:, 0].mean()), key="x1_pred")
    with col_p2:
        x2_val = st.number_input(f"Valor de {tags[1]}",
                                 value=float(df_clean.iloc[:, 1].mean()), key="x2_pred")
    with col_p3:
        predict_btn = st.button("Calcular predição", use_container_width=True)

    if predict_btn:
        trans = st.session_state["trans"]
        NC = st.session_state["NC"]
        pls3 = st.session_state["pls3"]

        Chat = np.zeros((1, NC))
        Chat[0, :] = trans(x1_val, x2_val)
        y_hat = round(float(np.asarray(pls3.predict(Chat)).ravel()[0]), 4)

        st.metric(label="ẑ = f(x₁, x₂)", value=y_hat)

        if rest_name != "Problema sem restrições" and rest_value is not None:
            if y_hat >= rest_value:
                st.success(f"O estado estimado **ẑ = {y_hat}** corresponde a um estado **viável** (≥ {rest_value}).")
            else:
                st.error(f"O estado estimado **ẑ = {y_hat}** corresponde a um estado **não viável** (< {rest_value}).")

    st.divider()

    # ── Equation + Downloads ──
    st.markdown("### Equação e downloads")
    st.latex(r"\hat{y} = \left(\frac{x - \mu}{\sigma}\right) \cdot \tilde{B} + B_0")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        pls2 = st.session_state["pls2"]
        try:
            x_mean = np.asarray(pls2._x_mean).ravel()
            x_std = np.asarray(pls2._x_std).ravel()
            coefs = np.asarray(pls2.coef_).ravel()
            intercept_val = float(np.asarray(pls2.intercept_).ravel()[0])
            n = len(x_mean)
            intercept_col = np.full(n, np.nan)
            intercept_col[0] = intercept_val

            parametros_xls = pd.DataFrame({
                "mean": x_mean, "std": x_std,
                "coef": coefs[:n], "intercept": intercept_col,
            })
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                parametros_xls.to_excel(writer, sheet_name="parameters-PLS", index=False)
            buf.seek(0)
            st.download_button(
                "📥 Baixar parâmetros (.xlsx)", data=buf,
                file_name="parametros_PLS.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Não foi possível gerar parâmetros: {e}")

    with col_d2:
        try:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zipf:
                for fig, name in [
                    (fig1, "RMSE_components"),
                    (fig2, "real_vs_predicted"),
                    (fig3, "3D_surface"),
                ]:
                    zipf.writestr(f"{name}.html",
                                  fig.to_html(include_plotlyjs="cdn", full_html=True))
            zbuf.seek(0)
            st.download_button(
                "📥 Baixar figuras (.zip)", data=zbuf,
                file_name="figuras.zip", mime="application/zip",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Não foi possível gerar as figuras: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════════════════════

if st.session_state["page"] == "home":
    page_home()
else:
    page_analysis()
