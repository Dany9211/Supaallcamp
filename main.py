import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
import json

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
        # st.secrets["postgres"] contiene le credenziali del database
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
if "gol_home_ht" in df.columns and "gol_away_ht" in df.columns:
    df["risultato_ht"] = df["gol_home_ht"].astype(str) + "-" + df["gol_away_ht"].astype(str)

if all(col in df.columns for col in ["gol_home_ft", "gol_away_ft", "gol_home_ht", "gol_away_ht"]):
    df["gol_home_sh"] = pd.to_numeric(df["gol_home_ft"], errors='coerce').fillna(0) - pd.to_numeric(df["gol_home_ht"], errors='coerce').fillna(0)
    df["gol_away_sh"] = pd.to_numeric(df["gol_away_ft"], errors='coerce').fillna(0) - pd.to_numeric(df["gol_away_ht"], errors='coerce').fillna(0)
    df["risultato_sh"] = df["gol_home_sh"].astype(int).astype(str) + "-" + df["gol_away_sh"].astype(int).astype(str)
else:
    st.sidebar.warning("Colonne mancanti per il calcolo delle statistiche del Secondo Tempo.")

# Identifica la colonna della data per il filtro delle ultime N partite
date_col_name = "data" if "data" in df.columns else "date" if "date" in df.columns else None
has_date_column = date_col_name is not None
if has_date_column:
    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
else:
    st.sidebar.warning("Le colonne 'data' o 'date' non sono presenti. Il filtro per le ultime N partite non sarà disponibile.")


# --- Selettori nella Sidebar per la configurazione dell'analisi ---
st.sidebar.header("Seleziona Squadre")

# 1. Selettore Campionato
leagues = ["Seleziona..."] + sorted(df["league"].dropna().unique())
selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)

df_filtered_by_league = df if selected_league == "Seleziona..." else df[df["league"] == selected_league]
all_teams = sorted(list(set(df_filtered_by_league['home_team'].dropna().unique()) | set(df_filtered_by_league['away_team'].dropna().unique())))

# 2. Selettori Squadre
home_team_selected = st.sidebar.selectbox("Seleziona Squadra CASA", ["Seleziona..."] + all_teams)
away_team_selected = st.sidebar.selectbox("Seleziona Squadra TRASFERTA", ["Seleziona..."] + all_teams)

if home_team_selected != "Seleziona..." and away_team_selected != "Seleziona...":
    
    # --- Selettore per le ultime N partite ---
    num_to_filter = None
    if has_date_column:
        st.sidebar.header("Filtra Partite per Numero")
        num_partite_options = ["Tutte", "Ultime 5", "Ultime 10", "Ultime 15", "Ultime 20", "Ultime 30", "Ultime 40", "Ultime 50"]
        selected_num_partite_str = st.sidebar.selectbox("Numero di partite da analizzare", num_partite_options)
        
        if selected_num_partite_str != "Tutte":
            num_to_filter = int(selected_num_partite_str.split(' ')[1])

    # --- FILTRAGGIO E COMBINAZIONE DATI PRE-ANALISI ---
    
    # Filtra le partite in casa della squadra selezionata
    df_home = df_filtered_by_league[df_filtered_by_league["home_team"] == home_team_selected]
    
    # Filtra le partite in trasferta della squadra selezionata
    df_away = df_filtered_by_league[df_filtered_by_league["away_team"] == away_team_selected]
    
    # Ordina per data e applica il filtro per il numero di partite
    if has_date_column:
        df_home = df_home.sort_values(by=date_col_name, ascending=False)
        df_away = df_away.sort_values(by=date_col_name, ascending=False)

    if num_to_filter is not None:
        df_home = df_home.head(num_to_filter)
        df_away = df_away.head(num_to_filter)

    # Combina i due DataFrame per le statistiche pre-partita
    df_combined = pd.concat([df_home, df_away], ignore_index=True)
    
    st.header(f"Analisi Combinata: {home_team_selected} (Casa) vs {away_team_selected} (Trasferta)")
    st.write(f"Basata su **{len(df_home)}** partite casalinghe di '{home_team_selected}' e **{len(df_away)}** partite in trasferta di '{away_team_selected}'.")
    st.write(f"**Totale Partite Analizzate:** {len(df_combined)}")

    if not df_combined.empty:
        
        # --- INIZIO FUNZIONI STATISTICHE ---
        
        def get_outcome(home_score, away_score):
            """Determina l'esito (1, X, 2) da un punteggio."""
            if home_score > away_score:
                return '1'
            elif home_score < away_score:
                return '2'
            else:
                return 'X'

        def calcola_winrate(df_to_analyze, col_risultato, title):
            """
            Calcola e mostra il WinRate basato sui risultati complessivi.
            """
            st.subheader(f"WinRate {title} ({len(df_to_analyze)} partite)")
            df_valid = df_to_analyze[df_to_analyze[col_risultato].notna() & (df_to_analyze[col_risultato].str.contains("-"))].copy()
            
            risultati = {"1 (Casa)": 0, "X (Pareggio)": 0, "2 (Trasferta)": 0}
            for ris in df_valid[col_risultato]:
                try:
                    home, away = map(int, ris.split("-"))
                    outcome = get_outcome(home, away)
                    if outcome == '1':
                        risultati["1 (Casa)"] += 1
                    elif outcome == '2':
                        risultati["2 (Trasferta)"] += 1
                    else:
                        risultati["X (Pareggio)"] += 1
                except ValueError:
                    continue
            
            totale = len(df_valid)
            stats = []
            for esito, count in risultati.items():
                perc = round((count / totale) * 100, 2) if totale > 0 else 0
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                stats.append((esito, count, perc, odd_min))
            
            df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['WinRate %']))

        def calcola_doppia_chance(df_to_analyze, home_team_name, away_team_name, title, col_home_goals, col_away_goals):
            """
            Calcola e visualizza le statistiche della doppia chance.
            """
            if df_to_analyze.empty:
                st.info(f"Nessuna partita trovata per calcolare le statistiche della doppia chance per {title}.")
                return

            st.subheader(f"Doppia Chance {title}")

            total_matches = len(df_to_analyze)
            if total_matches == 0:
                st.info("Nessuna partita trovata per questa analisi.")
                return

            # Determinare i risultati per il periodo specificato
            wins_home = (pd.to_numeric(df_to_analyze[col_home_goals], errors='coerce').fillna(0) > pd.to_numeric(df_to_analyze[col_away_goals], errors='coerce').fillna(0)).sum()
            draws = (pd.to_numeric(df_to_analyze[col_home_goals], errors='coerce').fillna(0) == pd.to_numeric(df_to_analyze[col_away_goals], errors='coerce').fillna(0)).sum()
            wins_away = (pd.to_numeric(df_to_analyze[col_home_goals], errors='coerce').fillna(0) < pd.to_numeric(df_to_analyze[col_away_goals], errors='coerce').fillna(0)).sum()
            
            # Calcolo delle doppie chance
            chance_1x = wins_home + draws
            chance_x2 = draws + wins_away
            chance_12 = wins_home + wins_away
            
            # Calcolo delle percentuali
            perc_1x = round((chance_1x / total_matches) * 100, 2)
            perc_x2 = round((chance_x2 / total_matches) * 100, 2)
            perc_12 = round((chance_12 / total_matches) * 100, 2)
            
            # Calcolo delle quote minime
            odd_1x = round(100 / perc_1x, 2) if perc_1x > 0 else "-"
            odd_x2 = round(100 / perc_x2, 2) if perc_x2 > 0 else "-"
            odd_12 = round(100 / perc_12, 2) if perc_12 > 0 else "-"
            
            data = {
                "Esito": ["1X", "X2", "12"],
                "Conteggio": [chance_1x, chance_x2, chance_12],
                "Percentuale %": [perc_1x, perc_x2, perc_12],
                "Odd Minima": [odd_1x, odd_x2, odd_12]
            }

            df_stats = pd.DataFrame(data)
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def mostra_risultati_esatti(df_to_analyze, col_risultato, titolo):
            """Mostra la distribuzione dei risultati esatti."""
            st.subheader(f"Risultati Esatti {titolo} ({len(df_to_analyze)} partite)")
            risultati_interessanti = [
                "0-0", "0-1", "0-2", "0-3", "1-0", "1-1", "1-2", "1-3",
                "2-0", "2-1", "2-2", "2-3", "3-0", "3-1", "3-2", "3-3"
            ]
            df_valid = df_to_analyze[df_to_analyze[col_risultato].notna() & (df_to_analyze[col_risultato].str.contains("-"))].copy()

            def classifica_risultato(ris):
                if ris in risultati_interessanti: return ris
                try:
                    home, away = map(int, ris.split("-"))
                    if home > away: return "Altro risultato casa vince"
                    elif home < away: return "Altro risultato ospite vince"
                    else: return "Altro pareggio"
                except: return "Altro"

            df_valid["classificato"] = df_valid[col_risultato].apply(classifica_risultato)
            distribuzione = df_valid["classificato"].value_counts().reset_index()
            distribuzione.columns = [titolo, "Conteggio"]
            distribuzione["Percentuale %"] = (distribuzione["Conteggio"] / len(df_valid) * 100).round(2)
            distribuzione["Odd Minima"] = distribuzione["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
            st.dataframe(distribuzione.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_over_goals(df_to_analyze, col_gol_home, col_gol_away, title):
            """Calcola la distribuzione Over/Under goals."""
            st.subheader(f"Over/Under Goals {title} ({len(df_to_analyze)} partite)")
            df_copy = df_to_analyze.copy()
            df_copy["tot_goals"] = pd.to_numeric(df_copy[col_gol_home], errors='coerce').fillna(0) + pd.to_numeric(df_copy[col_gol_away], errors='coerce').fillna(0)
            
            over_data = []
            for t in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                count = (df_copy["tot_goals"] > t).sum()
                perc = round((count / len(df_copy)) * 100, 2)
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                over_data.append([f"Over {t}", count, perc, odd_min])
            
            df_over = pd.DataFrame(over_data, columns=["Mercato", "Conteggio", "Percentuale %", "Odd Minima"])
            st.dataframe(df_over.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_btts(df_to_analyze, col_gol_home, col_gol_away, title):
            """Calcola la statistica GG/NG."""
            st.subheader(f"Entrambe le Squadre a Segno (GG/NG) {title} ({len(df_to_analyze)} partite)")
            df_copy = df_to_analyze.copy()
            df_copy["home_scored"] = pd.to_numeric(df_copy[col_gol_home], errors='coerce').fillna(0) > 0
            df_copy["away_scored"] = pd.to_numeric(df_copy[col_gol_away], errors='coerce').fillna(0) > 0
            
            gg_count = (df_copy["home_scored"] & df_copy["away_scored"]).sum()
            ng_count = (~(df_copy["home_scored"] & df_copy["away_scored"])).sum()
            
            total = len(df_copy)
            gg_perc = round((gg_count / total) * 100, 2) if total > 0 else 0
            ng_perc = round((ng_count / total) * 100, 2) if total > 0 else 0

            data = {
                "Esito": ["GG (Sì)", "NG (No)"],
                "Conteggio": [gg_count, ng_count],
                "Percentuale %": [gg_perc, ng_perc]
            }
            
            df_stats = pd.DataFrame(data)
            df_stats["Odd Minima"] = df_stats["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_clean_sheets(df_home_to_analyze, df_away_to_analyze, home_team_name, away_team_name, col_gol_home, col_gol_away, title):
            """Calcola le Clean Sheets e il Fallimento a Segnare per le squadre selezionate."""
            st.subheader(f"Clean Sheets / Fail to Score {title} (Home: {len(df_home_to_analyze)} partite, Away: {len(df_away_to_analyze)} partite)")
            
            # Calcolo per la squadra di casa (in casa)
            home_clean_sheets = (pd.to_numeric(df_home_to_analyze[col_gol_away], errors='coerce').fillna(0) == 0).sum()
            home_fail_to_score = (pd.to_numeric(df_home_to_analyze[col_gol_home], errors='coerce').fillna(0) == 0).sum()
            total_home_matches = len(df_home_to_analyze)
            
            # Calcolo per la squadra in trasferta (fuori casa)
            away_clean_sheets = (pd.to_numeric(df_away_to_analyze[col_gol_home], errors='coerce').fillna(0) == 0).sum()
            away_fail_to_score = (pd.to_numeric(df_away_to_analyze[col_gol_away], errors='coerce').fillna(0) == 0).sum()
            total_away_matches = len(df_away_to_analyze)
            
            data = {
                "Squadra": [home_team_name, away_team_name],
                "Clean Sheets": [home_clean_sheets, away_clean_sheets],
                "Fail to Score": [home_fail_to_score, away_fail_to_score]
            }
            
            df_stats = pd.DataFrame(data)
            df_stats["% Clean Sheets"] = [
                round((home_clean_sheets / total_home_matches) * 100, 2) if total_home_matches > 0 else 0,
                round((away_clean_sheets / total_away_matches) * 100, 2) if total_away_matches > 0 else 0
            ]
            df_stats["% Fail to Score"] = [
                round((home_fail_to_score / total_home_matches) * 100, 2) if total_home_matches > 0 else 0,
                round((away_fail_to_score / total_away_matches) * 100, 2) if total_away_matches > 0 else 0
            ]
            
            st.dataframe(df_stats.style.background_gradient(cmap='Blues', subset=['% Clean Sheets']).background_gradient(cmap='Reds', subset=['% Fail to Score']))

        def calcola_media_gol(df_home, df_away, home_team_name, away_team_name, title, col_home_goals, col_away_goals):
            """Calcola la media gol fatti e subiti per le squadre selezionate in un intervallo specifico (HT, FT, etc)."""
            st.subheader(f"Media Gol Fatti e Subiti {title} (Home: {len(df_home)} partite, Away: {len(df_away)} partite)")
            
            home_goals_mean = pd.to_numeric(df_home[col_home_goals], errors='coerce').mean()
            home_conceded_mean = pd.to_numeric(df_home[col_away_goals], errors='coerce').mean()
            away_goals_mean = pd.to_numeric(df_away[col_away_goals], errors='coerce').mean()
            away_conceded_mean = pd.to_numeric(df_away[col_home_goals], errors='coerce').mean()
            
            data = {
                "Squadra": [home_team_name, away_team_name],
                "Gol Fatti": [f"{home_goals_mean:.2f}", f"{away_goals_mean:.2f}"],
                "Gol Subiti": [f"{home_conceded_mean:.2f}", f"{away_conceded_mean:.2f}"]
            }
            
            df_stats = pd.DataFrame(data)
            st.dataframe(df_stats)

        def calcola_primo_gol_stats(df_combined, home_team_name, away_team_name, title, start_minute, end_minute):
            """Calcola le probabilità di chi segna il primo gol della partita/periodo in un intervallo di tempo."""
            st.subheader(f"Chi segna il prossimo gol? {title} ({len(df_combined)} partite)")
            first_goal_stats = defaultdict(int)
            total_matches = len(df_combined)
            for _, row in df_combined.iterrows():
                home_goals = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                away_goals = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                # Trova il primo gol nel range di minuti specificato
                first_goal_minute_home = min([g for g in home_goals if start_minute <= g <= end_minute] or [float('inf')])
                first_goal_minute_away = min([g for g in away_goals if start_minute <= g <= end_minute] or [float('inf')])
                # Assegna il gol alla squadra selezionata
                if first_goal_minute_home < first_goal_minute_away:
                    if row["home_team"] == home_team_name:
                        first_goal_stats[home_team_name] += 1
                    else:
                        first_goal_stats[away_team_name] += 1
                elif first_goal_minute_away < first_goal_minute_home:
                    if row["away_team"] == away_team_name:
                        first_goal_stats[away_team_name] += 1
                    else:
                        first_goal_stats[home_team_name] += 1
            
            # Calcola le percentuali e le quote
            stats = []
            total_goals = sum(first_goal_stats.values())
            for team, count in first_goal_stats.items():
                perc = round((count / total_goals) * 100, 2) if total_goals > 0 else 0
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                stats.append((team, count, perc, odd_min))
            
            # Gestione del caso in cui non ci sono gol
            if total_goals == 0:
                stats.append(("Nessun Gol nel Periodo", total_matches, 100, 1.0))
            
            df_stats = pd.DataFrame(stats, columns=["Squadra", "Conteggio", "Probabilità %", "Odd Minima"])
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Probabilità %']))

        # --- Sezione principale delle statistiche pre-partita (statica) ---
        with st.expander("Statistiche Pre-partita (FT, HT, SH)", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Partita Completa (FT)", "Primo Tempo (HT)", "Secondo Tempo (SH)"])

            with tab1:
                st.subheader("Statistiche Partita Completa (Full Time)")
                col1, col2 = st.columns(2)
                with col1:
                    calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected, "FT", "gol_home_ft", "gol_away_ft")
                    calcola_winrate(df_combined, "risultato_ft", "FT")
                with col2:
                    calcola_over_goals(df_combined, "gol_home_ft", "gol_away_ft", "FT")
                    calcola_btts(df_combined, "gol_home_ft", "gol_away_ft", "FT")
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ft", "gol_away_ft", "FT")
                calcola_doppia_chance(df_combined, home_team_selected, away_team_selected, "FT", "gol_home_ft", "gol_away_ft")
                mostra_risultati_esatti(df_combined, "risultato_ft", "FT")

            with tab2:
                st.subheader("Statistiche Primo Tempo (Half Time)")
                if "gol_home_ht" in df.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected, "HT", "gol_home_ht", "gol_away_ht")
                        calcola_winrate(df_combined, "risultato_ht", "HT")
                    with col2:
                        calcola_over_goals(df_combined, "gol_home_ht", "gol_away_ht", "HT")
                        calcola_btts(df_combined, "gol_home_ht", "gol_away_ht", "HT")
                    calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ht", "gol_away_ht", "HT")
                    calcola_doppia_chance(df_combined, home_team_selected, away_team_selected, "HT", "gol_home_ht", "gol_away_ht")
                    mostra_risultati_esatti(df_combined, "risultato_ht", "HT")
                else:
                    st.warning("Dati del Primo Tempo non disponibili.")

            with tab3:
                st.subheader("Statistiche Secondo Tempo (Second Half)")
                if "risultato_sh" in df.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected, "SH", "gol_home_sh", "gol_away_sh")
                        calcola_winrate(df_combined, "risultato_sh", "SH")
                    with col2:
                        calcola_over_goals(df_combined, "gol_home_sh", "gol_away_sh", "SH")
                        calcola_btts(df_combined, "gol_home_sh", "gol_away_sh", "SH")
                    calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_sh", "gol_away_sh", "SH")
                    calcola_doppia_chance(df_combined, home_team_selected, away_team_selected, "SH", "gol_home_sh", "gol_away_sh")
                    mostra_risultati_esatti(df_combined, "risultato_sh", "SH")
                else:
                    st.warning("Dati del Secondo Tempo non disponibili.")

        # --- Sezione per l'analisi dinamica (live) ---
        with st.expander("Analisi Dinamica (Live)", expanded=False):
            st.write("Analizza le probabilità basate su un punteggio e un minuto specifici.")
            
            col_din_1, col_din_2, col_din_3 = st.columns(3)
            with col_din_1:
                current_home_score = st.number_input(f"Gol {home_team_selected} (attuali)", min_value=0, value=0)
            with col_din_2:
                current_away_score = st.number_input(f"Gol {away_team_selected} (attuali)", min_value=0, value=0)
            with col_din_3:
                current_minute = st.number_input("Minuto di Gioco (attuale)", min_value=0, max_value=90, value=0)

            # Filtra il DataFrame dinamico in base allo stato live della partita
            def get_score_at_minute(row, target_minute):
                """Calcola il punteggio a un minuto specifico, gestendo i valori NaN."""
                minutaggio_gol_home = str(row.get("minutaggio_gol", ""))
                minutaggio_gol_away = str(row.get("minutaggio_gol_away", ""))
                
                try:
                    home_goals_count = len([int(m) for m in minutaggio_gol_home.split(';') if m.isdigit() and int(m) <= target_minute])
                    away_goals_count = len([int(m) for m in minutaggio_gol_away.split(';') if m.isdigit() and int(m) <= target_minute])
                    return home_goals_count, away_goals_count
                except:
                    return 0, 0

            df_combined_dynamic = df_combined[
                df_combined.apply(lambda row: get_score_at_minute(row, current_minute) == (current_home_score, current_away_score), axis=1)
            ]

            if not df_combined_dynamic.empty:
                st.write(f"Trovate {len(df_combined_dynamic)} partite storiche con punteggio di {current_home_score}-{current_away_score} al minuto {current_minute}.")

                # Genera le statistiche dinamiche
                with st.expander("Statistiche Dinamiche FT", expanded=True):
                    calcola_doppia_chance(df_combined_dynamic, home_team_selected, away_team_selected, "Dinamica FT", "gol_home_ft", "gol_away_ft")
                    calcola_winrate(df_combined_dynamic, "risultato_ft", "Dinamica FT")
                    calcola_over_goals(df_combined_dynamic, "gol_home_ft", "gol_away_ft", "Dinamica FT")
                    calcola_btts(df_combined_dynamic, "gol_home_ft", "gol_away_ft", "Dinamica FT")
                
                with st.expander("Probabilità Prossimo Gol", expanded=True):
                    calcola_primo_gol_stats(df_combined_dynamic, home_team_selected, away_team_selected, "Dinamica", current_minute + 1, 90)
                
                with st.expander("Bande Temporali Dinamiche FT", expanded=False):
                    dynamic_time_bands_5min_ft = [(i, i + 5) for i in range(current_minute + 1, 90, 5)]
                    if dynamic_time_bands_5min_ft:
                        st.subheader(f"Dinamica 5 min (dal {current_minute+1}')")
                        # Ho aggiornato il titolo per riflettere il minuto di inizio corretto
                        calcola_primo_gol_stats(df_combined_dynamic, home_team_selected, away_team_selected, "Dinamica", current_minute + 1, 90)
                    
                    dynamic_time_bands_15min_ft = [(i, i + 15) for i in range(current_minute + 1, 90, 15)]
                    if dynamic_time_bands_15min_ft:
                        st.subheader(f"Dinamica 15 min (dal {current_minute+1}')")
                        # Ho aggiornato il titolo per riflettere il minuto di inizio corretto
                        calcola_primo_gol_stats(df_combined_dynamic, home_team_selected, away_team_selected, "Dinamica", current_minute + 1, 90)
            else:
                st.warning("Nessuna partita storica trovata con i parametri dinamici specificati. Prova a modificare minuto e punteggio.")
    else:
        st.warning("Seleziona una squadra 'CASA' e una 'TRASFERTA' per iniziare l'analisi.")

