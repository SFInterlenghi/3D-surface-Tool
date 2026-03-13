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
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');

    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
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
    .stSidebar [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(58, 123, 213, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colorscale options
# ─────────────────────────────────────────────────────────────────────────────
mapcolors_list = px.colors.named_colorscales()

# Plotly theme list
themes = [
    "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn",
    "simple_white", "none",
]

# ─────────────────────────────────────────────────────────────────────────────
# Model / math helpers (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

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
        # Default to order 2
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
    NC2 = np.argmin(RMSE)
    if NC2 == 0:
        NC2 = 1
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
            Pdavez = var2[i]
            Tdavez = var1[j]
            inp = np.array(trans(Tdavez, Pdavez)).reshape(1, -1)
            predict_ = float(pls2.predict(inp).ravel()[0])
            predict_2 = float(pls3.predict(inp).ravel()[0])
            Ccalc[i, j] = predict_
            Ccalc_r[i, j] = predict_

            if LIM is not None and predict_2 < LIM:
                Ccalc_r[i, j] = np.nan

    return Ccalc, Ccalc_r, var1, var2


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────────

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_layout(margin=dict(l=20, r=20, b=35, t=25))
    return fig


def figure1(RMSE, theme):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.linspace(1, RMSE.shape[0], RMSE.shape[0] - 1),
        y=RMSE.reshape((-1,)),
        name="Número de componentes",
        mode='markers',
        marker_color='rgba(255,182,193,.9)',
    ))
    fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    fig.update_layout(
        xaxis_title="Número de componentes",
        yaxis_title="RMSE",
        template=theme,
    )
    return fig


def figure2(dados, Y_pred, theme):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=dados.iloc[:, 2].values.reshape((-1,)),
        name="Real",
        mode='markers',
        marker_color='#6951e2',
        marker_size=12,
    ))
    fig.add_trace(go.Scatter(
        y=Y_pred.reshape((-1,)),
        name="Predito",
        mode='markers',
        marker_symbol="x",
        marker_color='#eb6ecc',
        marker_size=8,
    ))
    fig.update_traces(mode='markers', marker_line_width=1)
    fig.update_layout(
        xaxis_title="Amostra",
        yaxis_title="Variável",
        template=theme,
    )
    return fig


def figure3(Ccalc, var1, var2, Ccalc_r, dados, theme, colormap):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Surface(
        z=Ccalc, x=var1, y=var2,
        surfacecolor=Ccalc_r,
        colorscale=colormap,
        opacity=0.5,
        showscale=False,
    ))
    fig.add_trace(go.Surface(
        z=Ccalc_r, x=var1, y=var2,
        colorscale=colormap,
        colorbar=dict(orientation="v", x=0.9, xanchor="right"),
    ))
    fig.update_traces(contours_z=dict(
        show=True, usecolormap=True,
        highlightcolor="limegreen", project_z=True,
    ))
    fig.update_layout(
        autosize=True,
        scene=dict(
            xaxis=dict(title=dados.columns.tolist()[0], showbackground=True),
            yaxis=dict(title=dados.columns.tolist()[1], showbackground=True),
            zaxis=dict(title=dados.columns.tolist()[2], showbackground=True),
        ),
        height=700,
        margin=dict(l=5, r=5, b=5, t=5),
        template=theme,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
for key in ["df", "pls2", "pls3", "RMSE", "C", "Y_pred", "trans", "NC",
            "Ccalc", "Ccalc_r", "var1", "var2", "computed"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "computed" not in st.session_state:
    st.session_state["computed"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📊 Módulo de Visualização de Superfícies</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Gere superfícies bonitas interpolando via PLS — faça upload de um arquivo Excel para começar</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar: Smart data loader + Setup
# ─────────────────────────────────────────────────────────────────────────────

def col_letter(idx):
    """Convert 0-based column index to Excel letter (0→A, 25→Z, 26→AA)."""
    result = ""
    while True:
        result = chr(idx % 26 + ord('A')) + result
        idx = idx // 26 - 1
        if idx < 0:
            break
    return result


def load_raw_sheet(uploaded_file, sheet_name):
    """Load a sheet with no header so every row is visible as-is."""
    uploaded_file.seek(0)
    raw = pd.read_excel(uploaded_file, sheet_name=sheet_name,
                        header=None, engine="openpyxl", dtype=str)
    return raw


with st.sidebar:
    st.header("⚙️ Configuração")

    uploaded_file = st.file_uploader(
        "Carregar arquivo Excel (.xlsx)",
        type=["xlsx"],
        help="Faça upload de um arquivo .xlsx com os dados para gerar a superfície.",
    )

    if uploaded_file is not None:
        # ── Step 1: Sheet selector ──
        uploaded_file.seek(0)
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        sheet_names = xls.sheet_names

        if len(sheet_names) > 1:
            sheet = st.selectbox("📄 Selecione a planilha", options=sheet_names, key="sel_sheet")
        else:
            sheet = sheet_names[0]
            st.caption(f"📄 Planilha: **{sheet}**")

        raw = load_raw_sheet(uploaded_file, sheet)
        total_rows, total_cols = raw.shape

        # ── Step 2: Raw preview ──
        with st.expander("👀 Pré-visualização do arquivo (primeiras 20 linhas)", expanded=True):
            preview = raw.head(20).copy()
            # Show Excel-style row numbers (1-based) and column letters
            preview.index = range(1, len(preview) + 1)
            preview.columns = [col_letter(i) for i in range(preview.shape[1])]
            st.dataframe(preview, use_container_width=True, height=300)
            st.caption(f"Arquivo: {total_rows} linhas × {total_cols} colunas")

        st.divider()

        # ── Step 3: Header row ──
        st.subheader("📍 Localização dos dados")
        header_row = st.number_input(
            "Linha do cabeçalho (nomes das colunas)",
            min_value=1, max_value=total_rows, value=1, step=1,
            help="Número da linha (como no Excel) onde estão os nomes das variáveis.",
            key="sel_header_row",
        )

        # ── Step 4: Data start row ──
        data_start = st.number_input(
            "Linha de início dos dados",
            min_value=header_row + 1, max_value=total_rows, value=header_row + 1, step=1,
            help="Primeira linha com dados numéricos (logo abaixo do cabeçalho, normalmente).",
            key="sel_data_start",
        )

        # ── Step 5: Column range ──
        col_letters = [col_letter(i) for i in range(total_cols)]
        col_a, col_b = st.columns(2)
        with col_a:
            start_col = st.selectbox("Coluna inicial", options=col_letters, index=0, key="sel_col_start")
        with col_b:
            end_col = st.selectbox("Coluna final", options=col_letters,
                                   index=total_cols - 1, key="sel_col_end")

        start_col_idx = col_letters.index(start_col)
        end_col_idx = col_letters.index(end_col)

        if end_col_idx < start_col_idx:
            st.error("A coluna final deve ser igual ou posterior à coluna inicial.")
            st.stop()

        # ── Step 6: Parse data ──
        # header_row and data_start are 1-based (Excel-style)
        header_idx = header_row - 1  # 0-based for pandas
        data_start_idx = data_start - 1

        # Get column names from the header row
        header_values = raw.iloc[header_idx, start_col_idx:end_col_idx + 1].values
        col_names = [str(c).strip() if pd.notna(c) else f"Col_{col_letter(start_col_idx + i)}"
                     for i, c in enumerate(header_values)]

        # Get data rows
        data_slice = raw.iloc[data_start_idx:, start_col_idx:end_col_idx + 1].copy()
        data_slice.columns = col_names

        # Convert to numeric where possible, drop all-NaN rows
        for c in data_slice.columns:
            data_slice[c] = pd.to_numeric(data_slice[c], errors="coerce")
        data_slice = data_slice.dropna(how="all").reset_index(drop=True)

        # Store the parsed dataframe
        st.session_state["df"] = data_slice
        df = data_slice

        # ── Confirm parsed table ──
        with st.expander("✅ Tabela interpretada", expanded=False):
            st.dataframe(df.head(15), use_container_width=True, height=250)
            st.caption(f"Dados: {df.shape[0]} linhas × {df.shape[1]} colunas  •  "
                       f"Cabeçalho na linha {header_row}  •  "
                       f"Dados a partir da linha {data_start}  •  "
                       f"Colunas {start_col}–{end_col}")

        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) < 3:
            st.warning("São necessárias pelo menos 3 colunas numéricas para gerar a superfície.")
            st.stop()

        st.divider()

        # ── Model variables ──
        cols = numeric_cols

        st.subheader("Variáveis do modelo")
        x1 = st.selectbox("Eixo x₁", options=cols, index=0, key="sel_x1")
        x2 = st.selectbox("Eixo x₂", options=cols, index=min(1, len(cols) - 1), key="sel_x2")
        x3 = st.selectbox("Eixo z (resposta)", options=cols, index=min(2, len(cols) - 1), key="sel_x3")

        st.divider()
        st.subheader("Ordem do interpolador")
        order = st.selectbox("Ordem do polinômio", options=[2, 3, 4, 5], index=1, key="sel_order")

        st.divider()
        st.subheader("Restrição de viabilidade")
        rest_options = cols + ["Problema sem restrições"]
        rest_name = st.selectbox("Variável com restrição", options=rest_options,
                                 index=len(rest_options) - 1, key="sel_rest")

        rest_value = None
        if rest_name != "Problema sem restrições":
            rest_value = st.number_input("Valor mínimo da restrição", value=0.0, key="sel_rest_val")

        st.divider()
        st.subheader("Aparência")
        theme = st.selectbox("Template da figura", options=themes, index=0, key="sel_theme")
        colormap = st.selectbox("Mapa de cores", options=mapcolors_list,
                                index=mapcolors_list.index("viridis") if "viridis" in mapcolors_list else 0,
                                key="sel_cmap")

        st.divider()

        # ── Compute button ──
        compute_btn = st.button("🚀 Calcular superfície", use_container_width=True, type="primary")

        if compute_btn:
            tags = [x1, x2, x3]

            # Validate no NaNs in selected columns
            df_clean = df[tags].dropna()
            if df_clean.shape[0] < 5:
                st.error(f"Dados insuficientes: apenas {df_clean.shape[0]} linhas válidas nas colunas selecionadas.")
                st.stop()

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

                # Store everything in session
                st.session_state.update(dict(
                    pls2=pls2, pls3=pls3, RMSE=RMSE, C=C,
                    Y_pred=Y_pred, Ccalc=Ccalc, Ccalc_r=Ccalc_r,
                    var1=var1, var2=var2, computed=True,
                    df_clean=df_clean,
                ))

            st.success("Superfície calculada com sucesso!")

    else:
        st.info("Faça upload de um arquivo Excel para começar.")
        st.session_state["df"] = None

    df = st.session_state.get("df")


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

if df is not None and st.session_state.get("computed"):
    tags = [st.session_state["sel_x1"], st.session_state["sel_x2"], st.session_state["sel_x3"]]
    df_clean = st.session_state.get("df_clean", df[tags].dropna())
    theme = st.session_state["sel_theme"]
    colormap = st.session_state["sel_cmap"]
    RMSE = st.session_state["RMSE"]
    Y_pred = st.session_state["Y_pred"]
    Ccalc = st.session_state["Ccalc"]
    Ccalc_r = st.session_state["Ccalc_r"]
    var1 = st.session_state["var1"]
    var2 = st.session_state["var2"]

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
        x1_val = st.number_input(f"Valor de {tags[0]}", value=float(df_clean.iloc[:, 0].mean()), key="x1_pred")
    with col_p2:
        x2_val = st.number_input(f"Valor de {tags[1]}", value=float(df_clean.iloc[:, 1].mean()), key="x2_pred")
    with col_p3:
        predict_btn = st.button("Calcular predição", use_container_width=True)

    if predict_btn:
        trans = st.session_state["trans"]
        NC = st.session_state["NC"]
        pls3 = st.session_state["pls3"]
        rest_name = st.session_state["sel_rest"]
        rest_value = st.session_state.get("sel_rest_val")

        Chat = np.zeros((1, NC))
        Chat[0, :] = trans(x1_val, x2_val)
        y_hat = pls3.predict(Chat)
        y_hat = round(float(np.asarray(y_hat).ravel()[0]), 4)

        st.metric(label=f"ẑ = f(x₁, x₂)", value=y_hat)

        if rest_name != "Problema sem restrições" and rest_value is not None:
            if y_hat >= rest_value:
                st.success(f"O estado estimado **ẑ = {y_hat}** corresponde a um estado **viável** (≥ {rest_value}).")
            else:
                st.error(f"O estado estimado **ẑ = {y_hat}** corresponde a um estado **não viável** (< {rest_value}).")

    st.divider()

    # ── Equation info + Downloads ──
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
                "mean": x_mean,
                "std": x_std,
                "coef": coefs[:n],
                "intercept": intercept_col,
            })

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                parametros_xls.to_excel(writer, sheet_name="parameters-PLS", index=False)
            buffer.seek(0)

            st.download_button(
                "📥 Baixar parâmetros (.xlsx)",
                data=buffer,
                file_name="parametros_PLS.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Não foi possível gerar parâmetros: {e}")

    with col_d2:
        try:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for idx, (fig, name) in enumerate([
                    (fig1, "RMSE_components"),
                    (fig2, "real_vs_predicted"),
                    (fig3, "3D_surface"),
                ], start=1):
                    html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
                    zipf.writestr(f"{name}.html", html_str)
            zip_buffer.seek(0)

            st.download_button(
                "📥 Baixar figuras (.zip)",
                data=zip_buffer,
                file_name="figuras.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Não foi possível gerar as imagens: {e}")

else:
    # Empty state
    st.markdown("---")
    col_e1, col_e2, col_e3 = st.columns([1, 2, 1])
    with col_e2:
        st.markdown(
            """
            <div style="text-align:center; padding: 4rem 2rem; opacity: 0.6;">
                <p style="font-size: 3rem;">📂</p>
                <h3>Nenhum dado carregado</h3>
                <p>Use a barra lateral para fazer upload de um arquivo Excel e configurar o modelo.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
