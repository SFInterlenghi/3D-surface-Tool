# 📊 3D Surface Viewer — PLS Regression Tool

Interactive tool for generating 3D surface visualizations from Excel data using PLS (Partial Least Squares) regression.

Upload an Excel file, select your variables, choose the polynomial order, and get beautiful interactive 3D surfaces with model validation plots — all in the browser.

## Features

- **Excel upload** — load `.xlsx` files directly in the browser
- **PLS regression** — polynomial surface fitting (orders 2–5)
- **Interactive 3D surface** — rotatable Plotly surface with customisable color maps and themes
- **Feasibility constraints** — optional constraint variable to mask infeasible regions
- **Point prediction** — enter x₁, x₂ values to get ẑ estimates with feasibility feedback
- **Export** — download PLS parameters as `.xlsx` and all figures as a `.zip` of PNGs

## Quick Start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Excel File Format

Your Excel file should have columns representing the process variables. For example:

| Flow_Rate | Temperature | Conversion |
|-----------|-------------|------------|
| 10.5      | 350         | 0.85       |
| 12.0      | 375         | 0.91       |
| ...       | ...         | ...        |

Select any two columns as inputs (x₁, x₂) and one as the response (z).

## License

Internal team tool to generate figures.
