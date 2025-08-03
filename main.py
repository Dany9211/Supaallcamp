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
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + " - " + df["gol_away_ft"].astype(str)

# --- Funzioni di calcolo ---
def calcola_stats_dinamico(df_filtered, starting_score_str, end_minute):
    """
    Calcola e visualizza le statistiche dinamiche per i gol successivi.
    """
    if starting_score_str and "-" in starting_score_str:
        home_score_start, away_score_start = map(int, starting_score_str.split(" - "))

        df_filtered_score = df_filtered[
            (df_filtered[f"gol_home_{end_minute}"] == home_score_start) &
            (df_filtered[f"gol_away_{end_minute}"] == away_score_start)
        ].copy()

        if not df_filtered_score.empty:
            total_matches_with_score = len(df_filtered_score)
            
            # Calcolo dei gol successivi in base al risultato di partenza
            df_filtered_score["gol_home_dopo_minuto"] = df_filtered_score["gol_home_ft"] - home_score_start
            df_filtered_score["gol_away_dopo_minuto"] = df_filtered_score["gol_away_ft"] - away_score_start

            matches_with_next_goal = df_filtered_score[
                (df_filtered_score["gol_home_dopo_minuto"] > 0) | 
                (df_filtered_score["gol_away_dopo_minuto"] > 0)
            ]
            
            matches_with_next_goal_home = df_filtered_score[df_filtered_score["gol_home_dopo_minuto"] > 0]
            matches_with_next_goal_away = df_filtered_score[df_filtered_score["gol_away_dopo_minuto"] > 0]

            perc_next_goal = (len(matches_with_next_goal) / total_matches_with_score) * 100 if total_matches_with_score > 0 else 0
            perc_next_goal_home = (len(matches_with_next_goal_home) / total_matches_with_score) * 100 if total_matches_with_score > 0 else 0
            perc_next_goal_away = (len(matches_with_next_goal_away) / total_matches_with_score) * 100 if total_matches_with_score > 0 else 0

            st.markdown(f"**Risultati per {total_matches_with_score} partite con punteggio di {starting_score_str} al minuto {end_minute}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Prossimo gol in Home", f"{perc_next_goal_home:.2f}%")
            col2.metric("Prossimo gol in Away", f"{perc_next_goal_away:.2f}%")
            col3.metric("Prossimo gol totale", f"{perc_next_goal:.2f}%")
            
        else:
            st.warning(f"Nessuna partita trovata in cui il punteggio era {starting_score_str} al minuto {end_minute}.")
    else:
        st.info("Inserisci un risultato di partenza valido per avviare l'analisi dinamica.")


def calcola_stats_prematch(df, time_band_minutes):
    """
    Calcola e visualizza le statistiche dei gol per una specifica time band prematch.
    """
    total_matches = len(df)
    
    # Crea una lista di nomi di colonne da controllare (es. 'gol_home_1', 'gol_away_1', ecc.)
    home_cols = [f'gol_home_{i}' for i in range(1, time_band_minutes + 1)]
    away_cols = [f'gol_away_{i}' for i in range(1, time_band_minutes + 1)]
    
    # Assumiamo che ci sia una colonna 'minuto_primo_gol'
    # Se non è presente, si può usare un approccio più complesso
    if 'minuto_primo_gol' in df.columns:
        matches_with_goal = df[df['minuto_primo_gol'] <= time_band_minutes]
    else:
        st.warning("Colonna 'minuto_primo_gol' non trovata. Impossibile calcolare le statistiche prematch.")
        return

    num_matches_with_goal = len(matches_with_goal)
    percentage = (num_matches_with_goal / total_matches) * 100 if total_matches > 0 else 0
    
    st.markdown(f"**Statistiche per la time band: 0-{time_band_minutes} minuti**")
    col1, col2 = st.columns(2)
    col1.metric("Numero di partite con gol", num_matches_with_goal)
    col2.metric("Percentuale di partite con gol", f"{percentage:.2f}%")

# --- UI per i filtri ---
campioni = sorted(df["campionato"].unique())
campionato_selezionato = st.selectbox("Seleziona un campionato", campioni)

df_camp = df[df["campionato"] == campionato_selezionato]

if not df_camp.empty:
    squadre_home = sorted(df_camp["home_team"].unique())
    squadre_away = sorted(df_camp["away_team"].unique())

    col_squadre1, col_squadre2 = st.columns(2)
    with col_squadre1:
        home_team_selected = st.selectbox("Squadra Casa", squadre_home)
    with col_squadre2:
        away_team_selected = st.selectbox("Squadra Trasferta", squadre_away)

    df_combined = df_camp[
        (df_camp["home_team"] == home_team_selected) & 
        (df_camp["away_team"] == away_team_selected)
    ]
    
    if not df_combined.empty:
        st.write(f"**Partite trovate per {home_team_selected} vs {away_team_selected}:** {len(df_combined)}")

        # --- ESECUZIONE E VISUALIZZAZIONE STATS DINAMICHE ---
        st.markdown("---")
        st.header("Analisi Dinamica")
        st.subheader("Analisi del prossimo gol in base al minuto e al risultato di partenza")
        
        col_start_min, col_start_score = st.columns(2)
        with col_start_min:
            end_minute = st.number_input("Minuto di analisi", min_value=1, max_value=90, value=65)
        with col_start_score:
            starting_score_str = st.text_input("Risultato di partenza (es. 0 - 0)", "0 - 0")

        calcola_stats_dinamico(df_combined, starting_score_str, end_minute)
        
        # --- ESECUZIONE E VISUALIZZAZIONE STATS PREMATCH ---
        st.markdown("---")
        st.header("Analisi Prematch")
        st.subheader("Statistiche gol per time band (5 e 15 minuti) su tutti i match")
        
        # Chiamata alla funzione per la time band di 5 minuti
        calcola_stats_prematch(df_combined, 5)
        
        st.markdown("---")
        
        # Chiamata alla funzione per la time band di 15 minuti
        calcola_stats_prematch(df_combined, 15)

    else:
        st.warning("Nessuna partita trovata per la combinazione di squadre e campionato selezionata.")
else:
    st.warning("Nessuna partita trovata per il campionato selezionato.")
