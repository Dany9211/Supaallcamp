import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

# Configurazione della pagina
st.set_page_config(page_title="Analisi Next Gol e stats live", layout="wide")
st.title("Analisi Tabella allcamp")

# --- Funzione connessione al database ---
@st.cache_data
def run_query(query: str):
    """
    Esegue una query SQL e restituisce i risultati come DataFrame.
    La funzione è cacheata per evitare di riconnettersi al database
    ogni volta che l'applicazione si aggiorna.
    """
    try:
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
    except Exception as e:
        st.error(f"Errore di connessione al database: {e}")
        st.stop()
        return pd.DataFrame()

# --- Caricamento dati iniziali ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    if df.empty:
        st.warning("Il DataFrame caricato dal database è vuoto.")
        st.stop()
    st.write(f"**Righe iniziali nel dataset:** {len(df)}")
except Exception as e:
    st.error(f"Errore durante il caricamento del database: {e}")
    st.stop()

# --- Aggiunta colonne risultato_ft e risultato_ht ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

filters = {}

# --- FILTRI INIZIALI ---
st.sidebar.header("Filtri Dati")

# Filtro League (Campionato) - Deve essere il primo per filtrare le squadre
if "league" in df.columns:
    leagues = ["Tutte"] + sorted(df["league"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)
    if selected_league != "Tutte":
        filters["league"] = selected_league
    
    # Crea un DataFrame temporaneo per filtrare le squadre in base al campionato
    if selected_league != "Tutte":
        filtered_teams_df = df[df["league"] == selected_league]
    else:
        filtered_teams_df = df.copy()
else:
    filtered_teams_df = df.copy()
    selected_league = "Tutte"

# Filtro Anno
if "anno" in df.columns:
    anni = ["Tutti"] + sorted(df["anno"].dropna().unique())
    selected_anno = st.sidebar.selectbox("Seleziona Anno", anni)
    if selected_anno != "Tutti":
        filters["anno"] = selected_anno

# Filtro Giornata
if "giornata" in df.columns:
    giornata_min = int(df["giornata"].min()) if not df["giornata"].isnull().all() else 1
    giornata_max = int(df["giornata"].max()) if not df["giornata"].isnull().all() else 38
    giornata_range = st.sidebar.slider(
        "Seleziona Giornata",
        min_value=giornata_min,
        max_value=giornata_max,
        value=(giornata_min, giornata_max)
    )
    filters["giornata"] = giornata_range

# --- FILTRI SQUADRE (ora dinamici) ---
if "home_team" in filtered_teams_df.columns:
    home_teams = ["Tutte"] + sorted(filtered_teams_df["home_team"].dropna().unique())
    selected_home = st.sidebar.selectbox("Seleziona Squadra Home", home_teams)
    if selected_home != "Tutte":
        filters["home_team"] = selected_home

if "away_team" in filtered_teams_df.columns:
    away_teams = ["Tutte"] + sorted(filtered_teams_df["away_team"].dropna().unique())
    selected_away = st.sidebar.selectbox("Seleziona Squadra Away", away_teams)
    if selected_away != "Tutte":
        filters["away_team"] = selected_away

# --- NUOVO FILTRO: Risultato HT ---
if "risultato_ht" in df.columns:
    ht_results = sorted(df["risultato_ht"].dropna().unique())
    selected_ht_results = st.sidebar.multiselect("Seleziona Risultato HT", ht_results, default=None)
    if selected_ht_results:
        filters["risultato_ht"] = selected_ht_results

# --- FUNZIONE per filtri range ---
def add_range_filter(col_name, label=None):
    if col_name in df.columns:
        col_temp = pd.to_numeric(df[col_name].astype(str).str.replace(",", "."), errors="coerce")
        col_min = float(col_temp.min(skipna=True))
        col_max = float(col_temp.max(skipna=True))
        
        st.sidebar.write(f"Range attuale {label or col_name}: {col_min} - {col_max}")
        min_val = st.sidebar.text_input(f"Min {label or col_name}", value="")
        max_val = st.sidebar.text_input(f"Max {label or col_name}", value="")
        
        if min_val.strip() != "" and max_val.strip() != "":
            try:
                filters[col_name] = (float(min_val), float(max_val))
            except ValueError:
                st.sidebar.warning(f"Valori non validi per {label or col_name}. Inserisci numeri.")

st.sidebar.header("Filtri Quote")
for col in ["odd_home", "odd_draw", "odd_away"]:
    add_range_filter(col)

# --- APPLICA FILTRI AL DATAFRAME PRINCIPALE ---
filtered_df = df.copy()
for col, val in filters.items():
    if col in ["odd_home", "odd_draw", "odd_away"]:
        mask = pd.to_numeric(filtered_df[col].astype(str).str.replace(",", "."), errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    elif col == "giornata":
        mask = pd.to_numeric(filtered_df[col], errors="coerce").between(val[0], val[1])
        filtered_df = filtered_df[mask.fillna(True)]
    elif col == "risultato_ht":
        filtered_df = filtered_df[filtered_df[col].isin(val)]
    else:
        filtered_df = filtered_df[filtered_df[col] == val]

st.subheader("Dati Filtrati")
st.write(f"**Righe visualizzate:** {len(filtered_df)}")
st.dataframe(filtered_df.head(50))


# --- FUNZIONE WINRATE ---
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
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])

# --- FUNZIONE CALCOLO FIRST TO SCORE ---
def calcola_first_to_score(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Home Team": 0, "Away Team": 0, "No Goals": 0}
    totale_partite = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit()]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else: 
            if min_home_goal == float('inf'):
                risultati["No Goals"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- NUOVA FUNZIONE CALCOLO FIRST TO SCORE HT ---
def calcola_first_to_score_ht(df_to_analyze):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
    
    risultati = {"Home Team": 0, "Away Team": 0, "No Goals": 0}
    totale_partite = len(df_to_analyze)

    for _, row in df_to_analyze.iterrows():
        gol_home_str = str(row.get("minutaggio_gol", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        # Considera solo i gol segnati nel primo tempo (minuto <= 45)
        gol_home = [int(x) for x in gol_home_str.split(";") if x.isdigit() and int(x) <= 45]
        gol_away = [int(x) for x in gol_away_str.split(";") if x.isdigit() and int(x) <= 45]

        min_home_goal = min(gol_home) if gol_home else float('inf')
        min_away_goal = min(gol_away) if gol_away else float('inf')
        
        if min_home_goal < min_away_goal:
            risultati["Home Team"] += 1
        elif min_away_goal < min_home_goal:
            risultati["Away Team"] += 1
        else: 
            if min_home_goal == float('inf'):
                risultati["No Goals"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

# --- FUNZIONE RISULTATI ESATTI ---
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

    st.subheader(f"Risultati Esatti {titolo} ({len(df_valid)} partite)")
    styled_df = distribuzione.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (15 MIN) ---
def mostra_distribuzione_timeband(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 15 minuti è vuoto.")
        return
    intervalli = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (91, 150)]
    label_intervalli = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]
    risultati = []
    totale_partite = len(df_to_analyze)
    for (start, end), label in zip(intervalli, label_intervalli):
        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)

# --- NUOVA FUNZIONE RIUTILIZZABILE PER DISTRIBUZIONE TIMEBAND (5 MIN) ---
def mostra_distribuzione_timeband_5min(df_to_analyze):
    if df_to_analyze.empty:
        st.warning("Il DataFrame per l'analisi a 5 minuti è vuoto.")
        return
    intervalli = [(0,5), (6,10), (11,15), (16,20), (21,25), (26,30), (31,35), (36,40), (41,45), (46,50), (51,55), (56,60), (61,65), (66,70), (71,75), (76,80), (81,85), (86,90), (91, 150)]
    label_intervalli = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", "71-75", "76-80", "81-85", "86-90", "90+"]
    risultati = []
    totale_partite = len(df_to_analyze)
    for (start, end), label in zip(intervalli, label_intervalli):
        partite_con_gol = 0
        for _, row in df_to_analyze.iterrows():
            gol_home = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
            gol_away = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
            if any(start <= g <= end for g in gol_home + gol_away):
                partite_con_gol += 1
        perc = round((partite_con_gol / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        risultati.append([label, partite_con_gol, perc, odd_min])
    df_result = pd.DataFrame(risultati, columns=["Timeframe", "Partite con Gol", "Percentuale %", "Odd Minima"])
    styled_df = df_result.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %'])
    st.dataframe(styled_df)


# --- NUOVA FUNZIONE CALCOLO NEXT GOAL con logica dinamica ---
def calcola_stats_dinamiche_next_goal(df_to_analyze, start_min, end_min, risultato_di_partenza):
    if df_to_analyze.empty:
        return pd.DataFrame(columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])

    risultati = {"Next Gol: Home": 0, "Next Gol: Away": 0, "Nessun prossimo gol": 0}
    totale_partite = 0
    
    # Filtra il dataframe in base al risultato di partenza e all'intervallo di minuti
    for _, row in df_to_analyze.iterrows():
        # Calcolo del risultato al minuto 'start_min'
        gol_home_str = str(row.get("minutaggio_gol", ""))
        gol_away_str = str(row.get("minutaggio_gol_away", ""))

        gol_home_all = [int(x) for x in gol_home_str.split(";") if x.isdigit()]
        gol_away_all = [int(x) for x in gol_away_str.split(";") if x.isdigit()]
        
        gol_home_at_start = sum(1 for g in gol_home_all if g < start_min)
        gol_away_at_start = sum(1 for g in gol_away_all if g < start_min)
        
        risultato_attuale = f"{gol_home_at_start}-{gol_away_at_start}"
        
        if risultato_attuale == risultato_di_partenza:
            totale_partite += 1
            # Analisi del prossimo gol nell'intervallo
            next_home_goal = min([g for g in gol_home_all if start_min <= g <= end_min] or [float('inf')])
            next_away_goal = min([g for g in gol_away_all if start_min <= g <= end_min] or [float('inf')])

            if next_home_goal < next_away_goal:
                risultati["Next Gol: Home"] += 1
            elif next_away_goal < next_home_goal:
                risultati["Next Gol: Away"] += 1
            else:
                if next_home_goal == float('inf'):
                    risultati["Nessun prossimo gol"] += 1

    stats = []
    for esito, count in risultati.items():
        perc = round((count / totale_partite) * 100, 2) if totale_partite > 0 else 0
        odd_min = round(100 / perc, 2) if perc > 0 else "-"
        stats.append((esito, count, perc, odd_min))
    
    return pd.DataFrame(stats, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])


# --- SEZIONE PRINCIPALE PER IL LAYOUT ---
st.header("Analisi Next Gol")

# UI per i filtri dinamici del Next Goal
st.sidebar.subheader("Filtri Dinamici Next Gol")

# Filtro Risultato di partenza
# Seleziona tutti i possibili risultati parziali per l'input utente
all_partial_results = [f"{h}-{a}" for h in range(10) for a in range(10)]
selected_start_result = st.sidebar.selectbox("Risultato di Partenza", all_partial_results)


start_min, end_min = st.sidebar.slider(
    'Seleziona intervallo di tempo (minuti)',
    0, 150, (45, 90)
)

st.subheader(f"Statistiche Next Goal: Intervallo {start_min}-{end_min} con risultato di partenza {selected_start_result}")

if not filtered_df.empty:
    df_next_goal_dinamico = calcola_stats_dinamiche_next_goal(filtered_df, start_min, end_min, selected_start_result)
    st.dataframe(df_next_goal_dinamico.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
else:
    st.warning("Nessun dato da analizzare per il Next Goal dinamico.")


# --- VECCHIE FUNZIONI DI ANALISI - mantenute per riferimento ---

st.header("Statistiche Generali")
st.subheader("WinRate Finale")
df_winrate = calcola_winrate(filtered_df, "risultato_ft")
st.dataframe(df_winrate)

st.subheader("First to Score (Partita Intera)")
df_first_to_score = calcola_first_to_score(filtered_df)
st.dataframe(df_first_to_score.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

st.subheader("First to Score (Primo Tempo)")
df_first_to_score_ht = calcola_first_to_score_ht(filtered_df)
st.dataframe(df_first_to_score_ht.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

mostra_risultati_esatti(filtered_df, "risultato_ft", "FT")
mostra_risultati_esatti(filtered_df, "risultato_ht", "HT")

st.header("Distribuzione Gol per Timeband")
st.subheader("Distribuzione gol a 15 minuti")
mostra_distribuzione_timeband(filtered_df)

st.subheader("Distribuzione gol a 5 minuti")
mostra_distribuzione_timeband_5min(filtered_df)

# --- BACKTESTING ---
st.header("Simulatore di Backtest")
if "risultato_ft" in filtered_df.columns and "odd_over_2_5_ft" in filtered_df.columns:
    def esegui_backtest(df, market, strategy, stake):
        vincite = 0
        perdite = 0
        numero_scommesse = 0
        profit_loss = 0
        odd_minima = np.inf
        
        # Filtra solo le partite con quote valide
        colonna_odd = None
        colonna_esito = None
        if market == "1 (Casa)": colonna_odd, colonna_esito = "odd_home", "1"
        elif market == "X (Pareggio)": colonna_odd, colonna_esito = "odd_draw", "X"
        elif market == "2 (Trasferta)": colonna_odd, colonna_esito = "odd_away", "2"
        elif market == "Over 2.5 FT": colonna_odd, colonna_esito = "odd_over_2_5_ft", "Over 2.5"
        elif market == "BTTS SI FT": colonna_odd, colonna_esito = "odd_btts_si_ft", "BTTS SI"
        
        if colonna_odd is None:
            return 0, 0, 0, 0, 0, 0, 0
        
        # Gestione del risultato per il backtest
        def get_esito(row, market):
            try:
                if market == "1 (Casa)":
                    home, away = map(int, str(row["risultato_ft"]).split("-"))
                    return "1" if home > away else ("X" if home == away else "2")
                elif market == "X (Pareggio)":
                    home, away = map(int, str(row["risultato_ft"]).split("-"))
                    return "X" if home == away else ("1" if home > away else "2")
                elif market == "2 (Trasferta)":
                    home, away = map(int, str(row["risultato_ft"]).split("-"))
                    return "2" if home < away else ("X" if home == away else "1")
                elif market == "Over 2.5 FT":
                    home, away = map(int, str(row["risultato_ft"]).split("-"))
                    return "Over 2.5" if (home + away) > 2.5 else "Under 2.5"
                elif market == "BTTS SI FT":
                    home, away = map(int, str(row["risultato_ft"]).split("-"))
                    return "BTTS SI" if home > 0 and away > 0 else "BTTS NO"
            except:
                return "Errore"
        
        df_backtest = df[df[colonna_odd].notna() & (df[colonna_odd] != 0)].copy()

        for _, row in df_backtest.iterrows():
            quota = float(str(row[colonna_odd]).replace(",", "."))
            risultato_partita = get_esito(row, market)
            
            if strategy == "Back":
                if risultato_partita == colonna_esito:
                    vincite += 1
                    profit_loss += (quota - 1) * stake
                else:
                    perdite += 1
                    profit_loss -= stake
            elif strategy == "Lay":
                if risultato_partita == colonna_esito:
                    perdite += 1
                    profit_loss -= (quota - 1) * stake
                else:
                    vincite += 1
                    profit_loss += stake
            
            numero_scommesse += 1
            if quota < odd_minima:
                odd_minima = quota

        roi = (profit_loss / (numero_scommesse * stake)) * 100 if numero_scommesse > 0 else 0
        win_rate = (vincite / numero_scommesse) * 100 if numero_scommesse > 0 else 0
        
        return vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima

    # UI per il backtest
    backtest_market = st.selectbox(
        "Seleziona un mercato da testare",
        ["1 (Casa)", "X (Pareggio)", "2 (Trasferta)", "Over 2.5 FT", "BTTS SI FT"]
    )
    backtest_strategy = st.selectbox(
        "Seleziona la strategia",
        ["Back", "Lay"]
    )
    stake = st.number_input("Stake per scommessa", min_value=1.0, value=1.0, step=0.5)
    
    if st.button("Avvia Backtest"):
        vincite, perdite, numero_scommesse, profit_loss, roi, win_rate, odd_minima = esegui_backtest(filtered_df, backtest_market, backtest_strategy, stake)
        
        if numero_scommesse > 0:
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            col_met1.metric("Numero Scommesse", numero_scommesse)
            col_met2.metric("Vincite", vincite)
            col_met3.metric("Perdite", perdite)
            col_met4.metric("Profitto/Perdita", f"{profit_loss:.2f} €")
            
            col_met5, col_met6 = st.columns(2)
            col_met5.metric("ROI", f"{roi:.2f} %")
            col_met6.metric("Win Rate", f"{win_rate:.2f} %")
            
            st.write(f"Odd Minima Selezionata: {odd_minima:.2f}")
        else:
            st.warning("Nessuna scommessa trovata con i filtri e il mercato selezionato.")
