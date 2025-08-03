import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
from collections import defaultdict

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

# Calcolo dei gol e del risultato del Secondo Tempo (SH)
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
                
                # CORREZIONE: Assegna il gol alla squadra selezionata, indipendentemente se è "home" o "away" nel record storico
                if first_goal_minute_home < first_goal_minute_away:
                    if row["home_team"] == home_team_name:
                        first_goal_stats[home_team_name] += 1
                    else: # Questo significa che la squadra "away_team_selected" ha segnato il primo gol (in uno dei suoi match storici)
                        first_goal_stats[away_team_name] += 1
                elif first_goal_minute_away < first_goal_minute_home:
                    if row["away_team"] == away_team_name:
                        first_goal_stats[away_team_name] += 1
                    else: # Questo significa che la squadra "home_team_selected" ha segnato il primo gol (in uno dei suoi match storici)
                        first_goal_stats[home_team_name] += 1
                else:
                    first_goal_stats["Nessun gol"] += 1
            
            data = []
            for team, count in first_goal_stats.items():
                perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                data.append([team, count, perc, odd_min])
            
            df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_last_to_score(df_to_analyze, start_minute, end_minute, home_team_name, away_team_name):
            """Calcola chi segna l'ultimo gol della partita/periodo in un intervallo di tempo."""
            st.subheader(f"Chi segna l'ultimo gol? (Last to Score) ({len(df_to_analyze)} partite)")
            
            last_goal_stats = defaultdict(int)
            total_matches = len(df_to_analyze)
            
            for _, row in df_to_analyze.iterrows():
                home_goals = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                away_goals = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                
                # Trova l'ultimo gol nel range di minuti specificato
                last_goal_minute_home = max([g for g in home_goals if start_minute <= g <= end_minute] or [-1])
                last_goal_minute_away = max([g for g in away_goals if start_minute <= g <= end_minute] or [-1])

                # CORREZIONE: Assegna il gol alla squadra selezionata
                if last_goal_minute_home > last_goal_minute_away:
                    if row["home_team"] == home_team_name:
                        last_goal_stats[home_team_name] += 1
                    else:
                        last_goal_stats[away_team_name] += 1
                elif last_goal_minute_away > last_goal_minute_home:
                    if row["away_team"] == away_team_name:
                        last_goal_stats[away_team_name] += 1
                    else:
                        last_goal_stats[home_team_name] += 1
                else:
                    last_goal_stats["Nessun gol"] += 1
            
            data = []
            for team, count in last_goal_stats.items():
                perc = round((count / total_matches) * 100, 2) if total_matches > 0 else 0
                odd_min = round(100 / perc, 2) if perc > 0 else "-"
                data.append([team, count, perc, odd_min])

            df_stats = pd.DataFrame(data, columns=["Esito", "Conteggio", "Percentuale %", "Odd Minima"])
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

        def calcola_timeband_stats(df_to_analyze, time_bands, title, home_team_name, away_team_name):
            """
            Calcola le statistiche dei gol segnati per intervalli di tempo specifici.
            Inclusa la percentuale di partite con almeno 1 e 2 gol e colorazione a gradiente.
            """
            total_matches = len(df_to_analyze)
            st.subheader(f"Statistiche per intervallo: {title} ({total_matches} partite)")
            
            stats = []
            
            for start, end in time_bands:
                
                home_scored_count = 0
                home_conceded_count = 0
                away_scored_count = 0
                away_conceded_count = 0
                matches_at_least_1_goal = 0
                matches_at_least_2_goals = 0
                
                for _, row in df_to_analyze.iterrows():
                    
                    # Estrae i minuti dei gol, gestendo i valori NaN e stringhe vuote
                    home_goal_minutes_str = str(row.get("minutaggio_gol", ""))
                    away_goal_minutes_str = str(row.get("minutaggio_gol_away", ""))

                    home_goals_in_match = [int(x) for x in home_goal_minutes_str.split(";") if x.isdigit()]
                    away_goals_in_match = [int(x) for x in away_goal_minutes_str.split(";") if x.isdigit()]
                    
                    goals_by_home_team_in_band = [g for g in home_goals_in_match if start <= g < end]
                    goals_by_away_team_in_band = [g for g in away_goals_in_match if start <= g < end]
                    
                    # Calcola i gol fatti e subiti in base alla squadra selezionata
                    if row["home_team"] == home_team_name:
                        home_scored_count += len(goals_by_home_team_in_band)
                        home_conceded_count += len(goals_by_away_team_in_band)
                        away_scored_count += len(goals_by_away_team_in_band)
                        away_conceded_count += len(goals_by_home_team_in_band)
                    else: # Squadra selezionata gioca fuori casa
                        home_scored_count += len(goals_by_away_team_in_band)
                        home_conceded_count += len(goals_by_home_team_in_band)
                        away_scored_count += len(goals_by_home_team_in_band)
                        away_conceded_count += len(goals_by_away_team_in_band)
                        
                    # Calcola le metriche di gol totali per la banda temporale
                    total_goals_in_band = len(goals_by_home_team_in_band) + len(goals_by_away_team_in_band)
                    if total_goals_in_band >= 1:
                        matches_at_least_1_goal += 1
                    if total_goals_in_band >= 2:
                        matches_at_least_2_goals += 1

                if total_matches > 0:
                    perc_1_goal = round((matches_at_least_1_goal / total_matches) * 100, 2)
                    odd_min_1_goal = round(100 / perc_1_goal, 2) if perc_1_goal > 0 else "-"
                    perc_2_goals = round((matches_at_least_2_goals / total_matches) * 100, 2)
                    odd_min_2_goals = round(100 / perc_2_goals, 2) if perc_2_goals > 0 else "-"
                else:
                    perc_1_goal, odd_min_1_goal = 0, "-"
                    perc_2_goals, odd_min_2_goals = 0, "-"
                
                stats.append([
                    f"{start}'-{end}'", 
                    home_scored_count,
                    home_conceded_count,
                    away_scored_count,
                    away_conceded_count,
                    perc_1_goal,
                    odd_min_1_goal,
                    perc_2_goals,
                    odd_min_2_goals
                ])

            df_stats = pd.DataFrame(stats, columns=[
                "Intervallo", 
                f"Gol Fatti ({home_team_name})", 
                f"Gol Subiti ({home_team_name})", 
                f"Gol Fatti ({away_team_name})", 
                f"Gol Subiti ({away_team_name})", 
                "% Winrate >= 1 Gol", 
                "Odd Minima >= 1 Gol",
                "% Winrate >= 2 Gol",
                "Odd Minima >= 2 Gol"
            ])
            st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=[
                '% Winrate >= 1 Gol', '% Winrate >= 2 Gol'
            ]))

        # --- SEZIONE ANALISI PRE-PARTITA ---
        st.header("Analisi Pre-Partita")
        st.markdown("---")

        with st.expander("Statistiche Primo Tempo (HT)", expanded=False):
            if not df_combined.empty:
                calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected, "HT", "gol_home_ht", "gol_away_ht")
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ht", "gol_away_ht", "HT")
                calcola_primo_gol_stats(df_combined, home_team_selected, away_team_selected, "HT", 0, 45)
                calcola_last_to_score(df_combined, 0, 45, home_team_selected, away_team_selected)
                calcola_winrate(df_combined, "risultato_ht", "HT")
                mostra_risultati_esatti(df_combined, "risultato_ht", "HT")
                calcola_over_goals(df_combined, "gol_home_ht", "gol_away_ht", "HT")
                calcola_btts(df_combined, "gol_home_ht", "gol_away_ht", "HT")
        
        with st.expander("Statistiche Secondo Tempo (SH)", expanded=False):
            if not df_combined.empty:
                calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected, "SH", "gol_home_sh", "gol_away_sh")
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_sh", "gol_away_sh", "SH")
                calcola_primo_gol_stats(df_combined, home_team_selected, away_team_selected, "SH", 46, 90)
                calcola_last_to_score(df_combined, 46, 90, home_team_selected, away_team_selected)
                calcola_winrate(df_combined, "risultato_sh", "SH")
                mostra_risultati_esatti(df_combined, "risultato_sh", "SH")
                calcola_over_goals(df_combined, "gol_home_sh", "gol_away_sh", "SH")
                calcola_btts(df_combined, "gol_home_sh", "gol_away_sh", "SH")

        with st.expander("Statistiche Fine Partita (FT)", expanded=False):
            if not df_combined.empty:
                calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected, "FT", "gol_home_ft", "gol_away_ft")
                calcola_clean_sheets(df_home, df_away, home_team_selected, away_team_selected, "gol_home_ft", "gol_away_ft", "FT")
                calcola_primo_gol_stats(df_combined, home_team_selected, away_team_selected, "FT", 0, 90)
                calcola_last_to_score(df_combined, 0, 90, home_team_selected, away_team_selected)
                calcola_winrate(df_combined, "risultato_ft", "FT")
                mostra_risultati_esatti(df_combined, "risultato_ft", "FT")
                calcola_over_goals(df_combined, "gol_home_ft", "gol_away_ft", "FT")
                calcola_btts(df_combined, "gol_home_ft", "gol_away_ft", "FT")
        
        with st.expander("Analisi per Bande Temporali (Pre-partita)", expanded=False):
            if not df_combined.empty:
                st.subheader("Bande temporali ogni 5 minuti (0-90)")
                time_bands_5min = [(i, i + 5) for i in range(0, 90, 5)]
                calcola_timeband_stats(df_combined, time_bands_5min, f"Pre-partita 5 min", home_team_selected, away_team_selected)
                
                st.subheader("Bande temporali ogni 15 minuti (0-90)")
                time_bands_15min = [(i, i + 15) for i in range(0, 90, 15)]
                calcola_timeband_stats(df_combined, time_bands_15min, f"Pre-partita 15 min", home_team_selected, away_team_selected)

        # --- SEZIONE ANALISI NEXT GOAL (DINAMICA) ---
        st.header("Analisi Next Goal (In Play)")
        st.markdown("---")

        # Funzione per calcolare lo score a un minuto specifico
        def get_score_at_minute(row, target_minute):
            home_goals_minutes_str = str(row.get("minutaggio_gol", ""))
            away_goals_minutes_str = str(row.get("minutaggio_gol_away", ""))

            home_goals_in_match = [int(x) for x in home_goals_minutes_str.split(";") if x.isdigit() and int(x) <= target_minute]
            away_goals_in_match = [int(x) for x in away_goals_minutes_str.split(";") if x.isdigit() and int(x) <= target_minute]
            
            return len(home_goals_in_match), len(away_goals_in_match)
        
        # --- Sezione dinamica HT ---
        with st.expander("Analisi Dinamica Primo Tempo (HT)", expanded=False):
            use_dynamic_analysis_ht = st.checkbox("Abilita Analisi Dinamica HT")

            if use_dynamic_analysis_ht:
                st.info("Filtra per lo stato attuale della partita nel primo tempo (es. 0-0 al 30').")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_minute_dynamic_ht = st.number_input("Minuto attuale HT", min_value=1, max_value=44, value=30)
                with col2:
                    home_score_dynamic_ht = st.number_input(f"Gol {home_team_selected} (HT)", min_value=0, value=0)
                with col3:
                    away_score_dynamic_ht = st.number_input(f"Gol {away_team_selected} (HT)", min_value=0, value=0)
                
                st.markdown(f"**Filtro applicato:** `Punteggio {home_score_dynamic_ht}-{away_score_dynamic_ht} al {current_minute_dynamic_ht}'`")

                # Filtra i dati
                df_home_dynamic_ht = df_home[df_home.apply(lambda row: get_score_at_minute(row, current_minute_dynamic_ht) == (home_score_dynamic_ht, away_score_dynamic_ht), axis=1)]
                df_away_dynamic_ht = df_away[df_away.apply(lambda row: get_score_at_minute(row, current_minute_dynamic_ht) == (home_score_dynamic_ht, away_score_dynamic_ht), axis=1)]
                df_combined_dynamic_ht = pd.concat([df_home_dynamic_ht, df_away_dynamic_ht], ignore_index=True)

                if not df_combined_dynamic_ht.empty:
                    st.write(f"Trovate **{len(df_combined_dynamic_ht)}** partite storiche con punteggio di {home_score_dynamic_ht}-{away_score_dynamic_ht} al minuto {current_minute_dynamic_ht} (HT).")
                    
                    # Chiamata alle funzioni statistiche con i dati filtrati
                    calcola_over_goals(df_combined_dynamic_ht, "gol_home_ht", "gol_away_ht", f"HT (dopo {current_minute_dynamic_ht}')")
                    calcola_winrate(df_combined_dynamic_ht, "risultato_ht", f"HT (dopo {current_minute_dynamic_ht}')")
                    calcola_btts(df_combined_dynamic_ht, "gol_home_ht", "gol_away_ht", f"HT (dopo {current_minute_dynamic_ht}')")
                    calcola_primo_gol_stats(df_combined_dynamic_ht, home_team_selected, away_team_selected, f"Prossimo gol (dopo {current_minute_dynamic_ht}')", current_minute_dynamic_ht + 1, 45)
                    calcola_last_to_score(df_combined_dynamic_ht, current_minute_dynamic_ht + 1, 45, home_team_selected, away_team_selected)
                    
                    with st.expander("Bande Temporali Dinamiche HT", expanded=False):
                        dynamic_time_bands_5min_ht = [(i, i + 5) for i in range(current_minute_dynamic_ht, 45, 5)]
                        if dynamic_time_bands_5min_ht:
                            calcola_timeband_stats(df_combined_dynamic_ht, dynamic_time_bands_5min_ht, f"Dinamica 5 min (dal {current_minute_dynamic_ht}')", home_team_selected, away_team_selected)
                        
                        dynamic_time_bands_15min_ht = [(i, i + 15) for i in range(current_minute_dynamic_ht, 45, 15)]
                        if dynamic_time_bands_15min_ht:
                            calcola_timeband_stats(df_combined_dynamic_ht, dynamic_time_bands_15min_ht, f"Dinamica 15 min (dal {current_minute_dynamic_ht}')", home_team_selected, away_team_selected)
                else:
                    st.warning("Nessuna partita storica trovata con i parametri dinamici specificati per il primo tempo. Prova a modificare minuto e punteggio.")
        
        # --- Sezione dinamica FT (originaria) ---
        with st.expander("Analisi Dinamica Fine Partita (FT)", expanded=False):
            use_dynamic_analysis_ft = st.checkbox("Abilita Analisi Dinamica FT")

            if use_dynamic_analysis_ft:
                st.info("Filtra per lo stato attuale della partita (es. 0-0 al 60').")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_minute_dynamic_ft = st.number_input("Minuto attuale FT", min_value=1, max_value=89, value=60)
                with col2:
                    home_score_dynamic_ft = st.number_input(f"Gol {home_team_selected} (FT)", min_value=0, value=0)
                with col3:
                    away_score_dynamic_ft = st.number_input(f"Gol {away_team_selected} (FT)", min_value=0, value=0)
                
                st.markdown(f"**Filtro applicato:** `Punteggio {home_score_dynamic_ft}-{away_score_dynamic_ft} al {current_minute_dynamic_ft}'`")

                # Filtra i dati
                df_home_dynamic_ft = df_home[df_home.apply(lambda row: get_score_at_minute(row, current_minute_dynamic_ft) == (home_score_dynamic_ft, away_score_dynamic_ft), axis=1)]
                df_away_dynamic_ft = df_away[df_away.apply(lambda row: get_score_at_minute(row, current_minute_dynamic_ft) == (home_score_dynamic_ft, away_score_dynamic_ft), axis=1)]
                df_combined_dynamic_ft = pd.concat([df_home_dynamic_ft, df_away_dynamic_ft], ignore_index=True)

                if not df_combined_dynamic_ft.empty:
                    st.write(f"Trovate **{len(df_combined_dynamic_ft)}** partite storiche con punteggio di {home_score_dynamic_ft}-{away_score_dynamic_ft} al minuto {current_minute_dynamic_ft}.")
                    
                    # Chiamata alle funzioni statistiche con i dati filtrati
                    calcola_over_goals(df_combined_dynamic_ft, "gol_home_ft", "gol_away_ft", f"FT (dopo {current_minute_dynamic_ft}')")
                    calcola_winrate(df_combined_dynamic_ft, "risultato_ft", f"FT (dopo {current_minute_dynamic_ft}')")
                    calcola_btts(df_combined_dynamic_ft, "gol_home_ft", "gol_away_ft", f"FT (dopo {current_minute_dynamic_ft}')")
                    calcola_primo_gol_stats(df_combined_dynamic_ft, home_team_selected, away_team_selected, f"Prossimo gol (dopo {current_minute_dynamic_ft}')", current_minute_dynamic_ft + 1, 90)
                    calcola_last_to_score(df_combined_dynamic_ft, current_minute_dynamic_ft + 1, 90, home_team_selected, away_team_selected)
                    
                    with st.expander("Bande Temporali Dinamiche FT", expanded=False):
                        dynamic_time_bands_5min_ft = [(i, i + 5) for i in range(current_minute_dynamic_ft, 90, 5)]
                        if dynamic_time_bands_5min_ft:
                            calcola_timeband_stats(df_combined_dynamic_ft, dynamic_time_bands_5min_ft, f"Dinamica 5 min (dal {current_minute_dynamic_ft}')", home_team_selected, away_team_selected)
                        
                        dynamic_time_bands_15min_ft = [(i, i + 15) for i in range(current_minute_dynamic_ft, 90, 15)]
                        if dynamic_time_bands_15min_ft:
                            calcola_timeband_stats(df_combined_dynamic_ft, dynamic_time_bands_15min_ft, f"Dinamica 15 min (dal {current_minute_dynamic_ft}')", home_team_selected, away_team_selected)
                else:
                    st.warning("Nessuna partita storica trovata con i parametri dinamici specificati. Prova a modificare minuto e punteggio.")
    else:
        st.warning("Seleziona una squadra 'CASA' e una 'TRASFERTA' per avviare l'analisi.")
else:
    st.info("Seleziona un campionato e due squadre per iniziare.")
