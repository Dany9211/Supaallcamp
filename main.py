import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime

# --- Configurazione della pagina ---
st.set_page_config(page_title="Analisi Squadre Combinate", layout="wide")
st.title("Analisi Statistiche Combinate per Squadra")
st.write("Seleziona due squadre per analizzare le loro performance combinate.")

# --- Funzione di connessione al database (cacheata per efficienza) ---
@st.cache_data
def run_query(query: str):
    """
    Esegue una query SQL sul database PostgreSQL e restituisce i risultati come DataFrame.
    La funzione è cacheata per evitare di riconnettersi al database ad ogni aggiornamento.
    """
    try:
        conn = psycopg2.connect(**st.secrets["postgres"], sslmode="require")
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Errore di connessione al database: {e}")
        return pd.DataFrame()

# --- Caricamento dati iniziali dal database ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    if df.empty:
        st.warning("Il DataFrame caricato dal database è vuoto.")
        st.stop()
except Exception as e:
    st.error(f"Errore durante il caricamento del database: {e}")
    st.stop()

# --- Aggiunta di colonne calcolate per facilitare le analisi ---
if "gol_home_ft" in df.columns and "gol_away_ft" in df.columns:
    df["risultato_ft"] = df.apply(
        lambda row: "1" if row["gol_home_ft"] > row["gol_away_ft"] else
                    "X" if row["gol_home_ft"] == row["gol_away_ft"] else "2",
        axis=1
    )
    df["totale_gol_ft"] = df["gol_home_ft"] + df["gol_away_ft"]
else:
    st.error("Le colonne 'gol_home_ft' o 'gol_away_ft' non sono presenti nel DataFrame.")
    st.stop()

# --- Interfaccia utente per la selezione delle squadre e dei filtri ---
all_teams = sorted(list(set(df["home_team"].unique()) | set(df["away_team"].unique())))
home_team_selected = st.sidebar.selectbox("Seleziona Squadra Casa", [""] + all_teams)
away_team_selected = st.sidebar.selectbox("Seleziona Squadra Trasferta", [""] + all_teams)

# Filtro per l'anno
years = sorted(df["season"].unique(), reverse=True)
year_selected = st.sidebar.multiselect("Seleziona Anno/i", options=years, default=years)

# Filtro per il campionato
all_competitions = sorted(df["competition_name"].unique())
competition_selected = st.sidebar.multiselect("Seleziona Campionato/i", options=all_competitions, default=all_competitions)

# Filtro per i risultati finali (1, X, 2) - MODIFICA RICHIESTA
st.sidebar.subheader("Filtra per Risultato Finale")
selected_results = []
if st.sidebar.checkbox("1 (Vittoria Casa)", value=True):
    selected_results.append("1")
if st.sidebar.checkbox("X (Pareggio)", value=True):
    selected_results.append("X")
if st.sidebar.checkbox("2 (Vittoria Trasferta)", value=True):
    selected_results.append("2")

# --- Funzioni di calcolo delle statistiche ---
def calcola_stats_statiche(df_filtered, df_totale, team_name, team_type):
    """
    Calcola e visualizza le statistiche statiche per una squadra.
    """
    if df_filtered.empty:
        st.info(f"Nessuna partita trovata per {team_name} come {team_type}.")
        return

    st.subheader(f"Statistiche di base per {team_name} come {team_type}")
    
    # Calcola il numero totale di partite filtrate
    num_partite = len(df_filtered)
    st.write(f"Partite totali analizzate: {num_partite}")

    # Calcola la media dei gol fatti e subiti
    if team_type == "casa":
        gol_fatti_col = "gol_home_ft"
        gol_subiti_col = "gol_away_ft"
    else:
        gol_fatti_col = "gol_away_ft"
        gol_subiti_col = "gol_home_ft"

    avg_gol_fatti = df_filtered[gol_fatti_col].mean()
    avg_gol_subiti = df_filtered[gol_subiti_col].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Media Gol Fatti", f"{avg_gol_fatti:.2f}")
    with col2:
        st.metric(f"Media Gol Subiti", f"{avg_gol_subiti:.2f}")

def calcola_prob_next_goal_statico(df_filtered, home_team, away_team):
    """
    Calcola e visualizza la probabilità del prossimo gol in modo statico.
    """
    if df_filtered.empty:
        st.warning("Nessuna partita trovata per calcolare le probabilità statiche.")
        return

    st.subheader("Probabilità del prossimo gol (Analisi Statica)")

    total_matches = len(df_filtered)
    if total_matches == 0:
        st.info("Nessuna partita trovata per l'analisi statica.")
        return

    # Gol segnati e subiti dalla squadra di casa (somma totale)
    gol_home = df_filtered["gol_home_ft"].sum()
    gol_away = df_filtered["gol_away_ft"].sum()

    # Probabilità di un gol per ogni squadra
    prob_home_next = (gol_home / (gol_home + gol_away)) if (gol_home + gol_away) > 0 else 0
    prob_away_next = (gol_away / (gol_home + gol_away)) if (gol_home + gol_away) > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Probabilità prossimo gol {home_team}", f"{prob_home_next:.2%}")
    with col2:
        st.metric(f"Probabilità prossimo gol {away_team}", f"{prob_away_next:.2%}")

def calcola_timeband_stats(df_filtered, time_bands, analisi_type, home_team, away_team):
    """
    Calcola e visualizza le statistiche dei gol per bande temporali.
    """
    if df_filtered.empty:
        st.info(f"Nessuna partita trovata per l'analisi {analisi_type}.")
        return

    st.subheader(f"Statistiche Gol per Bande Temporali ({analisi_type})")
    
    # Crea un DataFrame per i risultati
    stats_df = pd.DataFrame(columns=["Banda", "Gol Fatti Casa", "Gol Subiti Casa", "Gol Fatti Trasferta", "Gol Subiti Trasferta"])
    
    for start_min, end_min in time_bands:
        # Filtra gli eventi gol all'interno della banda temporale
        gol_home_count = len(df_filtered[
            (df_filtered["minuto_gol_home"] >= start_min) & (df_filtered["minuto_gol_home"] < end_min)
        ])
        gol_away_count = len(df_filtered[
            (df_filtered["minuto_gol_away"] >= start_min) & (df_filtered["minuto_gol_away"] < end_min)
        ])

        # Aggiungi i risultati al DataFrame
        new_row = pd.DataFrame([{
            "Banda": f"{start_min}-{end_min} min",
            f"Gol Fatti {home_team}": gol_home_count,
            f"Gol Subiti {away_team}": gol_home_count,
            f"Gol Fatti {away_team}": gol_away_count,
            f"Gol Subiti {home_team}": gol_away_count
        }])
        stats_df = pd.concat([stats_df, new_row], ignore_index=True)
    
    st.table(stats_df)

# --- Logica principale dell'applicazione ---
if home_team_selected and away_team_selected:
    # Filtra il DataFrame iniziale in base alle squadre e ai filtri selezionati
    filtered_df_static = df[
        (df["home_team"] == home_team_selected) &
        (df["away_team"] == away_team_selected) &
        (df["season"].isin(year_selected)) &
        (df["competition_name"].isin(competition_selected)) &
        (df["risultato_ft"].isin(selected_results))
    ].copy()

    # --- Analisi Statica ---
    st.header(f"Analisi Statica: {home_team_selected} vs {away_team_selected}")
    if not filtered_df_static.empty:
        num_partite_statiche = len(filtered_df_static)
        st.info(f"Analisi basata su {num_partite_statiche} partite.")
        calcola_prob_next_goal_statico(filtered_df_static, home_team_selected, away_team_selected)
    else:
        st.warning("Nessuna partita trovata per l'analisi statica con i filtri selezionati.")

    # --- Analisi Dinamica (filtrando per ogni partita) ---
    st.header(f"Analisi Dinamica: {home_team_selected} vs {away_team_selected}")
    
    # Sliders per i filtri dinamici
    gol_home_dynamic = st.sidebar.slider(f"Gol attuali {home_team_selected}", 0, 10, 0)
    gol_away_dynamic = st.sidebar.slider(f"Gol attuali {away_team_selected}", 0, 10, 0)
    start_min = st.sidebar.slider("Minuto Iniziale", 0, 90, 0)

    # Filtra il DataFrame in base allo stato attuale della partita
    filtered_df_dynamic = df[
        (df["home_team"] == home_team_selected) &
        (df["away_team"] == away_team_selected) &
        (df["gol_home_ft"] > gol_home_dynamic) &
        (df["gol_away_ft"] > gol_away_dynamic) &
        (df["season"].isin(year_selected)) &
        (df["competition_name"].isin(competition_selected)) &
        (df["risultato_ft"].isin(selected_results))
    ].copy()

    if not filtered_df_dynamic.empty:
        st.info(f"Analisi dinamica basata su {len(filtered_df_dynamic)} partite.")

        # Genera le bande temporali e calcola le statistiche
        st.subheader(f"Bande temporali ogni 5 minuti (dal {start_min}° min)")
        time_bands_5min_dynamic = [(i, i + 5) for i in range(start_min, 90, 5)]
        if time_bands_5min_dynamic:
            calcola_timeband_stats(filtered_df_dynamic, time_bands_5min_dynamic, "Dinamica 5 min", home_team_selected, away_team_selected)
        else:
            st.info("Nessun intervallo di 5 minuti dopo il minuto iniziale selezionato.")

        st.subheader(f"Bande temporali ogni 15 minuti (dal {start_min}° min)")
        time_bands_15min_dynamic = [(i, i + 15) for i in range(start_min, 90, 15)]
        if time_bands_15min_dynamic:
            calcola_timeband_stats(filtered_df_dynamic, time_bands_15min_dynamic, "Dinamica 15 min", home_team_selected, away_team_selected)
        else:
            st.info("Nessun intervallo di 15 minuti dopo il minuto iniziale selezionato.")
    else:
        st.warning("Nessuna partita trovata per l'analisi dinamica con i filtri selezionati.")

else:
    st.warning("Seleziona una squadra 'Casa' e una 'Trasferta' per iniziare l'analisi.")
