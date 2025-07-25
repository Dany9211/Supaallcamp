import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Allcamp Viewer", layout="wide")
st.title("Analisi Tabella allcamp")

# --- Funzione connessione ---
@st.cache_data
def run_query(query: str):
    conn = psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        sslmode="require"
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- Caricamento dati ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
except Exception as e:
    st.error(f"Errore durante il caricamento: {e}")
    st.stop()

# --- Aggiunta colonne risultato_ft e risultato_ht ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)

if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

filters = {}

# --- FILTRO LEAGUE ---
if "league" in df.columns:
    leagues = ["Tutte"] + sorted(df["league"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona League", leagues)
    if selected_league != "Tutte":
        filters["league"] = selected_league

# --- FILTRO ANNO ---
if "anno" in df.columns:
    anni = ["Tutti"] + sorted(df["anno"].dropna().unique())
    selected_anno = st.sidebar.selectbox("Seleziona Anno", anni)
    if selected_anno != "Tutti":
        filters["anno"] = selected_anno

# --- FILTRO GIORNATA ---
if "giornata" in df.columns:
    giornata_min = int(df["giornata"].min())
    giornata_max = int(df["giornata"].max())
    giornata_range = st.sidebar.slider(
        "Seleziona Giornata",
        min_value=giornata_min,
        max_value=giornata_max,
        value=(giornata_min, giornata_max)
    )
    filters["giornata"] = giornata_range

# --- FILTRO RISULTATO HT ---
if "risultato_ht" in df.columns:
    ht_results = ["Tutti"] + sorted(df["risultato_ht"].dropna().unique())
    selected_ht = st.sidebar.selectbox("Seleziona Risultato HT", ht_results)
    if selected_ht != "Tutti":
        filters["risultato_ht"] = selected_ht

# --- FILTRO SQUADRA HOME ---
if "home_team" in df.columns:
    home_teams = ["Tutte"] + sorted(df["home_team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Squadra Home", home_teams)
    if selected_home != "Tutte":
        filters["home_team"] = selected_home

# --- FILTRO SQUADRA AWAY ---
if "away_team" in df.columns:
    away_teams = ["Tutte"] + sorted(df["away_team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Squadra Away", away_teams)
    if selected_away != "Tutte":
        filters["away_team"] = selected_away

# --- FILTRI QUOTE MANUALI ---
def add_range_filter(col_name, label=None):
    if col_name in df.columns:
        col_temp = pd.to_numeric(df[col_name].astype(str).str.replace(",", "."), errors="coerce")
        col_min = float(col_temp.min(skipna=True))
        col_max = float(col_temp.max(skipna=True))
        st.sidebar.write(f"Range attuale {col_name}: {col_min} - {col_max}")

        min_val = st.sidebar.text_input(f"Min {label or col_name}", value="")
        max_val = st.sidebar.text_input(f"Max {label or col_name}", value="")

        if min_val.strip() != "" and max_val.strip() != "":
            try:
                min_val = float(min_val)
                max_val = float(max_val)
                filters[col_name] = (min_val, max_val)
            except:
                st.sidebar.warning(f"Valori non validi per {col_name}")

st.sidebar.header("Filtri Quote")
for col in ["odd_home", "odd_draw", "odd_away"]:
    add_range_filter(col)

# --- PULSANTE RESET ---
if st.sidebar.button("🔄 Reset Filtri"):
    st.session_state.clear()
    st.rerun()

# --- APPLICA FILTRI ---
filtered_df = df.copy()

for col, val in filters.items():
    if col in ["odd_home", "odd_draw", "odd_away"]:
        mask = pd.to_numeric(filtered_df[col].astype(str).str.replace(",", "."), errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    elif col == "giornata":
        mask = pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]

if filters == {}:
    st.info("Nessun filtro attivo: vengono mostrati tutti i risultati.")

st.subheader("Dati Filtrati")
st.dataframe(filtered_df.head(50))
st.write(f"**Righe visualizzate:** {len(filtered_df)}")

# --- FUNZIONI PER WINRATE ---
def calcola_winrate(df, col_risultato):
    df_valid = df[df[col_risultato].notna() & (df[col_risultato].str.contains("-"))]
    risultati = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
    for ris in df_valid[col_risultato]:
        try:
            home, away = map(int, ris.split("-"))
            if home > away:
                risultati["1 (Casa)"] += 1
            elif home < away:
                risultati["2 (Trasferta)"] += 1
            else:
                risultati["X (Pareggio)"] += 1
        except:
            continue
    totale = len(df_valid)
    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale) * 100, 2) if totale > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    return stats, totale

def mostra_winrate_combinato(df):
    if "risultato_ht" not in df.columns or "risultato_ft" not in df.columns:
        st.warning("Colonne risultato_ht o risultato_ft mancanti.")
        return
    
    stats_ht, totale_ht = calcola_winrate(df, "risultato_ht")
    stats_ft, totale_ft = calcola_winrate(df, "risultato_ft")

    combined_data = []
    for i in range(3):
        combined_data.append([
            stats_ft[i][0], 
            stats_ht[i][1], stats_ht[i][2], stats_ht[i][3], 
            stats_ft[i][1], stats_ft[i][2], stats_ft[i][3]
        ])

    df_combined = pd.DataFrame(combined_data, columns=[
        "Esito",
        "Conteggio HT", "WinRate HT %", "Odd Minima HT",
        "Conteggio FT", "WinRate FT %", "Odd Minima FT"
    ])

    st.subheader("WinRate Combinato (HT & FT)")
    st.write(f"Totale partite (HT): {totale_ht} - Totale partite (FT): {totale_ft}")
    st.table(df_combined)

# --- RISULTATI ESATTI ---
def mostra_risultati_esatti(df, col_risultato, titolo):
    risultati_interessanti = [
        "0-0", "0-1", "0-2", "0-3",
        "1-0", "1-1", "1-2", "1-3",
        "2-0", "2-1", "2-2", "2-3",
        "3-0", "3-1", "3-2", "3-3"
    ]
    
    df_valid = df[df[col_risultato].notna() & (df[col_risultato].str.contains("-"))].copy()

    def classifica_risultato(ris):
        try:
            home, away = map(int, ris.split("-"))
        except:
            return "Altro"
        if ris in risultati_interessanti:
            return ris
        if home > away:
            return "Altro risultato casa vince"
        elif home < away:
            return "Altro risultato ospite vince"
        else:
            return "Altro pareggio"

    df_valid["classificato"] = df_valid[col_risultato].apply(classifica_risultato)
    distribuzione = df_valid["classificato"].value_counts().reset_index()
    distribuzione.columns = [titolo, "Conteggio"]
    distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df_valid) * 100).round(2)
    distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

    st.subheader(f"Risultati Esatti {titolo}")
    st.table(distribuzione)

# --- BTTS ---
def calcola_btts(df):
    btts = (df["home_g_ft"] > 0) & (df["away_g_ft"] > 0)
    count_btts = btts.sum()
    perc_btts = round(btts.mean() * 100, 2)
    odd_btts = round(100 / perc_btts, 2) if perc_btts > 0 else "-"
    st.subheader("BTTS (Both Teams To Score)")
    st.write(f"Partite BTTS SI: {count_btts}")
    st.write(f"Percentuale BTTS SI: {perc_btts}%")
    st.write(f"Odd Minima BTTS: {odd_btts}")

# --- OVER HT ---
def calcola_over_ht(df):
    if "risultato_ht" not in df.columns:
        st.warning("Colonna risultato_ht non trovata.")
        return
    
    temp_ht = df["risultato_ht"].str.split("-", expand=True)
    temp_ht = temp_ht.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    df["home_g_ht"], df["away_g_ht"] = temp_ht[0], temp_ht[1]
    df["tot_goals_ht"] = df["home_g_ht"] + df["away_g_ht"]

    st.subheader("Over Goals (HT)")
    over_data = []
    for t in [0.5, 1.5, 2.5]:
        count = (df["tot_goals_ht"] > t).sum()
        perc = round((count / len(df)) * 100, 2)
        over_data.append([f"Over {t} HT", count, perc, round(100/perc, 2) if perc > 0 else "-"])
    st.table(pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

# --- MEDIA GOL ---
def calcola_media_gol(df):
    st.subheader("Media Gol (HT, FT, SH)")
    media_home_ht = df["home_g_ht"].mean() if "home_g_ht" in df else np.nan
    media_away_ht = df["away_g_ht"].mean() if "away_g_ht" in df else np.nan
    media_home_ft = df["home_g_ft"].mean() if "home_g_ft" in df else np.nan
    media_away_ft = df["away_g_ft"].mean() if "away_g_ft" in df else np.nan

    media_home_sh = media_home_ft - media_home_ht if not np.isnan(media_home_ft) else np.nan
    media_away_sh = media_away_ft - media_away_ht if not np.isnan(media_away_ft) else np.nan

    media_df = pd.DataFrame({
        "": ["Home", "Away"],
        "HT": [round(media_home_ht, 2), round(media_away_ht, 2)],
        "FT": [round(media_home_ft, 2), round(media_away_ft, 2)],
        "SH": [round(media_home_sh, 2), round(media_away_sh, 2)]
    })

    st.table(media_df)
    st.write(f"**Totale Medio Gol HT:** {round(media_home_ht + media_away_ht, 2)}")
    st.write(f"**Totale Medio Gol FT:** {round(media_home_ft + media_away_ft, 2)}")
    st.write(f"**Totale Medio Gol SH:** {round(media_home_sh + media_away_sh, 2)}")

# --- CALCOLO RIMONTA ---
def calcola_rimonta(df):
    st.subheader("TOP Squadre Home e Away che Recuperano (da svantaggio HT a non-sconfitta FT)")

    # --- Home ---
    home_df = df[df["gol_home_ht"] < df["gol_away_ht"]]
    if not home_df.empty:
        home_stats = home_df.groupby("home_team").apply(
            lambda x: pd.Series({
                "Partite Svantaggio HT": len(x),
                "Non Perse FT": (x["gol_home_ft"] >= x["gol_away_ft"]).sum(),
                "Winrate Recupero %": round((x["gol_home_ft"] >= x["gol_away_ft"]).mean() * 100, 2)
            })
        ).sort_values("Winrate Recupero %", ascending=False)
        st.write("**Home:**")
        st.table(home_stats.reset_index())
    else:
        st.write("Nessuna squadra Home con svantaggio a HT nel dataset filtrato.")

    # --- Away ---
    away_df = df[df["gol_away_ht"] < df["gol_home_ht"]]
    if not away_df.empty:
        away_stats = away_df.groupby("away_team").apply(
            lambda x: pd.Series({
                "Partite Svantaggio HT": len(x),
                "Non Perse FT": (x["gol_away_ft"] >= x["gol_home_ft"]).sum(),
                "Winrate Recupero %": round((x["gol_away_ft"] >= x["gol_home_ft"]).mean() * 100, 2)
            })
        ).sort_values("Winrate Recupero %", ascending=False)
        st.write("**Away:**")
        st.table(away_stats.reset_index())
    else:
        st.write("Nessuna squadra Away con svantaggio a HT nel dataset filtrato.")

# --- DISTRIBUZIONE GOL PER TIMEFRAME ---
def distribuzione_gol_per_partita(df):
    st.subheader("Distribuzione Gol per Timeframe (max 2 gol per fascia)")
    intervalli = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
    risultati = []

    total_partite = len(df)

    for (start, end) in intervalli:
        partite_con_gol = 0
        for _, row in df.iterrows():
            # Gol Home
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_home_in_range = [g for g in gol_home if start <= g <= end][:2]  # max 2 gol

            # Gol Away
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            gol_away_in_range = [g for g in gol_away if start <= g <= end][:2]  # max 2 gol

            # Se c’è almeno un gol nel timeframe
            if gol_home_in_range or gol_away_in_range:
                partite_con_gol += 1

        perc = round((partite_con_gol / total_partite) * 100, 2) if total_partite > 0 else 0
        risultati.append([f"{start}-{end}", partite_con_gol, perc])

    df_ris = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %"])
    st.table(df_ris)

# --- STATISTICHE ---
if not filtered_df.empty and "risultato_ft" in filtered_df.columns:
    temp_ft = filtered_df["risultato_ft"].str.split("-", expand=True)
    temp_ft = temp_ft.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    filtered_df["home_g_ft"], filtered_df["away_g_ft"] = temp_ft[0], temp_ft[1]
    filtered_df["tot_goals_ft"] = filtered_df["home_g_ft"] + filtered_df["away_g_ft"]

    mostra_winrate_combinato(filtered_df)
    mostra_risultati_esatti(filtered_df, "risultato_ht", "HT")
    mostra_risultati_esatti(filtered_df, "risultato_ft", "FT")
    calcola_btts(filtered_df)
    calcola_over_ht(filtered_df)
    calcola_media_gol(filtered_df)

    st.subheader("Over Goals (FT)")
    over_data = []
    for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
        count = (filtered_df["tot_goals_ft"] > t).sum()
        perc = round((count / len(filtered_df)) * 100, 2)
        over_data.append([f"Over {t}", count, perc, round(100/perc, 2) if perc > 0 else "-"])
    st.table(pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"]))

    distribuzione_gol_per_partita(filtered_df)

calcola_rimonta(filtered_df)
