import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import ast # Libreria per convertire stringhe di liste in liste Python

# Configurazione della pagina
st.set_page_config(page_title="Analisi Squadre Combinate", layout="wide")
st.title("Analisi Statistiche Combinate per Squadra")

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
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + " - " + df["gol_away_ft"].astype(str)
    df["totale_gol_ft"] = df["gol_home_ft"] + df["gol_away_ft"]
    df["gol_home_ht"] = df["gol_home_ht"].fillna(0).astype(int)
    df["gol_away_ht"] = df["gol_away_ht"].fillna(0).astype(int)
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + " - " + df["gol_away_ht"].astype(str)

# --- Controlli iniziali sui dati ---
if 'home_team' not in df.columns or 'away_team' not in df.columns or 'league' not in df.columns:
    st.error("Le colonne 'home_team', 'away_team' o 'league' non sono presenti nel DataFrame.")
    st.stop()

# --- Interfaccia utente per i filtri ---
st.sidebar.header("Filtri di Analisi")
leagues = sorted(df["league"].unique())
league_selected = st.sidebar.selectbox("Seleziona Campionato", leagues)

filtered_df_league = df[df["league"] == league_selected]

home_teams = sorted(filtered_df_league["home_team"].unique())
away_teams = sorted(filtered_df_league["away_team"].unique())

home_team_selected = st.sidebar.selectbox("Seleziona Squadra di Casa", home_teams)
away_team_selected = st.sidebar.selectbox("Seleziona Squadra in Trasferta", away_teams)

# --- Filtraggio dati combinato ---
df_combined = filtered_df_league[
    ((filtered_df_league["home_team"] == home_team_selected) & (filtered_df_league["away_team"] == away_team_selected)) |
    ((filtered_df_league["home_team"] == away_team_selected) & (filtered_df_league["away_team"] == home_team_selected))
]

st.write(f"**Partite trovate per {home_team_selected} vs {away_team_selected} nel campionato {league_selected}:** {len(df_combined)}")

# --- Funzione per calcolare le statistiche per intervallo specifico ---
def calcola_stats_intervallo_specifico(df_in, start_minute, end_minute, home_team_name, away_team_name):
    st.subheader(f"Statistiche tra il minuto {start_minute} e {end_minute}")
    
    # Inizializzazione contatori
    gol_home_count = 0
    gol_away_count = 0
    gol_totali = 0
    
    # Controlla se le colonne minute_goal_home_list e minute_goal_away_list esistono
    if 'minute_goal_home_list' not in df_in.columns or 'minute_goal_away_list' not in df_in.columns:
        st.warning("Colonne 'minute_goal_home_list' o 'minute_goal_away_list' mancanti. Impossibile eseguire l'analisi per intervallo.")
        return

    # Itera su ogni partita
    for _, row in df_in.iterrows():
        # Usa ast.literal_eval per convertire la stringa della lista in una lista Python in modo sicuro
        # Controlla se il team di casa è quello selezionato per determinare quali liste di gol usare
        if row['home_team'] == home_team_name:
            gol_home_list = ast.literal_eval(row['minute_goal_home_list'])
            gol_away_list = ast.literal_eval(row['minute_goal_away_list'])
        else: # Se il team di casa è l'away_team_selected
            gol_home_list = ast.literal_eval(row['minute_goal_away_list'])
            gol_away_list = ast.literal_eval(row['minute_goal_home_list'])
        
        # Controlla i gol del team di casa
        for minute in gol_home_list:
            if start_minute <= minute <= end_minute:
                gol_home_count += 1
                
        # Controlla i gol del team in trasferta
        for minute in gol_away_list:
            if start_minute <= minute <= end_minute:
                gol_away_count += 1
    
    gol_totali = gol_home_count + gol_away_count
    
    # Visualizzazione dei risultati
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Gol di {home_team_name}", gol_home_count)
    col2.metric(f"Gol di {away_team_name}", gol_away_count)
    col3.metric("Totale Gol in Intervallo", gol_totali)

# --- Funzione per calcolare le statistiche pre-match timeband ---
def calcola_prematch_timeband(df_in, start_minute, end_minute):
    """
    Calcola le statistiche di Next Goal per un intervallo di minuti specifico
    basandosi solo sulle partite filtrate.
    """
    st.subheader(f"Analisi Pre-Match per Intervallo tra il minuto {start_minute} e {end_minute}")

    # Inizializzazione contatori
    matches_with_goal_in_timeband = 0
    total_matches = len(df_in)

    if total_matches == 0:
        st.info("Nessuna partita trovata per l'analisi.")
        return

    # Controlla se le colonne necessarie esistono
    if 'minute_goal_home_list' not in df_in.columns or 'minute_goal_away_list' not in df_in.columns:
        st.warning("Colonne 'minute_goal_home_list' o 'minute_goal_away_list' mancanti. Impossibile eseguire l'analisi pre-match.")
        return

    # Itera su ogni partita
    for _, row in df_in.iterrows():
        gol_trovati_in_band = False
        
        # Gestione sicura delle liste di gol (si assume siano stringhe che rappresentano liste)
        try:
            gol_home_list = ast.literal_eval(str(row['minute_goal_home_list']))
            gol_away_list = ast.literal_eval(str(row['minute_goal_away_list']))
        except (ValueError, SyntaxError):
            st.warning("Errore nella formattazione delle liste di gol. Le righe potrebbero essere saltate.")
            continue
            
        # Controlla i gol segnati dal team di casa
        for minute in gol_home_list:
            if start_minute <= minute <= end_minute:
                gol_trovati_in_band = True
                break
        if gol_trovati_in_band:
            matches_with_goal_in_timeband += 1
            continue # Passa alla prossima partita

        # Controlla i gol segnati dal team in trasferta
        for minute in gol_away_list:
            if start_minute <= minute <= end_minute:
                gol_trovati_in_band = True
                break
        if gol_trovati_in_band:
            matches_with_goal_in_timeband += 1
    
    # Calcolo della probabilità
    prob_goal = (matches_with_goal_in_timeband / total_matches) * 100 if total_matches > 0 else 0
    
    # Visualizzazione dei risultati
    col_prob, col_matches = st.columns(2)
    with col_prob:
        st.metric("Probabilità di almeno 1 gol", f"{prob_goal:.2f}%")
    with col_matches:
        st.metric("Partite con almeno 1 gol in intervallo", matches_with_goal_in_timeband)
    
    st.info(f"Analisi basata su {total_matches} partite storiche.")

# --- Sezione principale dell'analisi ---
if not df_combined.empty:
    # --- ESECUZIONE E VISUALIZZAZIONE STATS PER INTERVALLO SPECIFICO ---
    st.markdown("---")
    st.header("Analisi per Intervallo Specifico")
    st.subheader("Analisi dei gol segnati solo in un intervallo di minuti personalizzato")
    
    col_min_start, col_min_end = st.columns(2)
    with col_min_start:
        start_minute_custom = st.number_input("Minuto di inizio", min_value=1, max_value=90, value=40, key="start_minute_custom")
    with col_min_end:
        end_minute_custom = st.number_input("Minuto di fine", min_value=1, max_value=90, value=65, key="end_minute_custom")

    if start_minute_custom < end_minute_custom:
        calcola_stats_intervallo_specifico(df_combined, start_minute_custom, end_minute_custom, home_team_selected, away_team_selected)
    else:
        st.warning("Il minuto di inizio deve essere inferiore a quello di fine.")
    
    # --- NUOVA SEZIONE: Analisi Pre-Match Timeband ---
    st.markdown("---")
    st.header("Analisi Pre-Match Timeband")
    st.subheader("Probabilità che si verifichi un gol in un intervallo di minuti, basata su dati storici")
    
    col_min_start_pm, col_min_end_pm = st.columns(2)
    with col_min_start_pm:
        start_minute_prematch = st.number_input("Minuto di inizio", min_value=1, max_value=90, value=75, key="start_minute_prematch")
    with col_min_end_pm:
        end_minute_prematch = st.number_input("Minuto di fine", min_value=1, max_value=90, value=90, key="end_minute_prematch")
    
    if start_minute_prematch >= end_minute_prematch:
        st.warning("Il minuto di inizio deve essere inferiore a quello di fine per l'analisi pre-match.")
    else:
        if st.button("Avvia Analisi Pre-Match"):
            calcola_prematch_timeband(df_combined, start_minute_prematch, end_minute_prematch)

else:
    st.warning("Nessuna partita trovata per la combinazione di squadre e campionato selezionata. Seleziona altre opzioni.")

