import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

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
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)
    # Aggiungi le colonne che indicano se una squadra ha segnato nel primo e secondo tempo
    df["home_ht_gol"] = df["gol_home_ht"] > 0
    df["away_ht_gol"] = df["gol_away_ht"] > 0
    df["home_ft_gol"] = (df["gol_home_ft"] - df["gol_home_ht"]) > 0
    df["away_ft_gol"] = (df["gol_away_ft"] - df["gol_away_ht"]) > 0

# --- UI per la selezione delle squadre e del campionato ---
st.header("Seleziona le squadre e il campionato")
campionato_options = ["Tutti i campionati"] + sorted(df["campionato"].unique())
campionato_selected = st.selectbox("Seleziona Campionato", campionato_options)

df_campionato = df if campionato_selected == "Tutti i campionati" else df[df["campionato"] == campionato_selected]

squadre_possibili = sorted(pd.concat([df_campionato["home_team"], df_campionato["away_team"]]).unique())

col_home_team, col_away_team = st.columns(2)
with col_home_team:
    home_team_selected = st.selectbox("Squadra in Casa", squadre_possibili)
with col_away_team:
    away_team_selected = st.selectbox("Squadra in Trasferta", squadre_possibili)

# --- Funzioni di calcolo ---

def calcola_stats_intervallo_specifico(df_filtered, start_minute, end_minute, home_team, away_team):
    """
    Calcola e visualizza la probabilità che venga segnato un gol in un intervallo di minuti specifico.
    Nota: Per questa analisi, supponiamo l'esistenza di una colonna che elenca i minuti dei gol (ad es., 'gol_minuti').
    Verrà utilizzata una logica semplificata basata sui punteggi HT/FT.
    """
    total_games = len(df_filtered)
    
    # Simula la colonna `gol_minuti` per l'analisi
    # Questa è una semplificazione, ma serve a dimostrare la logica.
    # Se hai dati più granulari sui minuti dei gol, la logica andrebbe adattata.
    
    # Conteggio dei gol in base al minuto di inizio e fine
    # Se start_minute > 45 e end_minute <= 90, cerca i gol nel secondo tempo.
    # Se start_minute <= 45 e end_minute <= 45, cerca i gol nel primo tempo.
    
    # Questo approccio è molto basilare e potrebbe non essere preciso senza dati esatti sui minuti.
    
    if start_minute >= 46:
        # Analisi secondo tempo
        goals_in_interval = df_filtered[(df_filtered["gol_home_ft"] > df_filtered["gol_home_ht"]) |
                                        (df_filtered["gol_away_ft"] > df_filtered["gol_away_ht"])]
    elif end_minute <= 45:
        # Analisi primo tempo
        goals_in_interval = df_filtered[(df_filtered["gol_home_ht"] > 0) |
                                        (df_filtered["gol_away_ht"] > 0)]
    else:
        # Analisi che attraversa l'intervallo HT/FT
        goals_in_interval = df_filtered[(df_filtered["gol_home_ft"] > 0) |
                                        (df_filtered["gol_away_ft"] > 0)]
                                        
    games_with_goals = len(goals_in_interval)
    
    if total_games > 0:
        probabilita = (games_with_goals / total_games) * 100
        st.subheader(f"Statistiche per l'intervallo {start_minute}'-{end_minute}'")
        col1, col2, col3 = st.columns(3)
        col1.metric("Partite Analizzate", total_games)
        col2.metric("Partite con Gol", games_with_goals)
        col3.metric("Probabilità di Gol", f"{probabilita:.2f}%")
    else:
        st.warning("Nessuna partita trovata per questo intervallo.")


def calcola_stats_to_score(df_filtered, home_team, away_team):
    """
    Calcola e visualizza la probabilità che una squadra segni in entrambi i tempi.
    """
    total_games = len(df_filtered)
    
    # Calcola per la squadra di casa
    home_scored_ht_and_ft = df_filtered[(df_filtered["home_ht_gol"] == True) & (df_filtered["home_ft_gol"] == True)]
    home_prob = (len(home_scored_ht_and_ft) / total_games) * 100 if total_games > 0 else 0

    # Calcola per la squadra in trasferta
    away_scored_ht_and_ft = df_filtered[(df_filtered["away_ht_gol"] == True) & (df_filtered["away_ft_gol"] == True)]
    away_prob = (len(away_scored_ht_and_ft) / total_games) * 100 if total_games > 0 else 0

    st.subheader("Probabilità di 'To Score' (Segnare in entrambi i tempi)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Partite Analizzate", total_games)
    col2.metric(f"Probabilità {home_team} To Score", f"{home_prob:.2f}%")
    col3.metric(f"Probabilità {away_team} To Score", f"{away_prob:.2f}%")
    
# --- Visualizzazione dei risultati ---
if home_team_selected and away_team_selected:
    df_combined = df_campionato[
        (df_campionato["home_team"] == home_team_selected) &
        (df_campionato["away_team"] == away_team_selected)
    ]

    if not df_combined.empty:
        st.success(f"Trovate {len(df_combined)} partite tra {home_team_selected} e {away_team_selected}.")

        # --- ESECUZIONE E VISUALIZZAZIONE STATS TO SCORE ---
        st.markdown("---")
        calcola_stats_to_score(df_combined, home_team_selected, away_team_selected)

        # --- ESECUZIONE E VISUALIZZAZIONE STATS PER INTERVALLO SPECIFICO ---
        st.markdown("---")
        st.header("Analisi per Intervallo Specifico (Pre-match)")
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

        # --- NUOVE SEZIONI: ANALISI PER TIMEBAND PRE-MATCH ---
        st.markdown("---")
        st.header("Analisi Timeband 5 e 15 Minuti (Pre-match)")

        # Analisi Timeband 5 Minuti
        st.subheader("Analisi Timeband 5 Minuti")
        col_start_5, col_end_5 = st.columns(2)
        with col_start_5:
            start_minute_5 = st.number_input("Minuto di inizio (5 min)", min_value=1, max_value=85, value=75)
        with col_end_5:
            end_minute_5 = start_minute_5 + 5
            st.markdown(f"Minuto di fine: **{end_minute_5}'**")
        
        calcola_stats_intervallo_specifico(df_combined, start_minute_5, end_minute_5, home_team_selected, away_team_selected)
        
        # Analisi Timeband 15 Minuti
        st.subheader("Analisi Timeband 15 Minuti")
        col_start_15, col_end_15 = st.columns(2)
        with col_start_15:
            start_minute_15 = st.number_input("Minuto di inizio (15 min)", min_value=1, max_value=75, value=75)
        with col_end_15:
            end_minute_15 = start_minute_15 + 15
            st.markdown(f"Minuto di fine: **{end_minute_15}'**")

        calcola_stats_intervallo_specifico(df_combined, start_minute_15, end_minute_15, home_team_selected, away_team_selected)
        
    else:
        st.warning("Nessuna partita trovata per la combinazione di squadre e campionato selezionata.")
else:
    st.info("Seleziona due squadre per avviare l'analisi.")
