import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisi Squadre Combinate", layout="wide")
st.title("Analisi Statistiche Combinate per Squadra")

# --- Funzione connessione al database ---
@st.cache_data
def run_query(query: str):
    """
    Esegue una query SQL e restituisce i risultati come DataFrame.
    La funzione è cacheata per evitare di riconnettersi al database
    ogni volta che l'applicazione si aggiorna.
    """
    try:
        conn = psycopg2.connect(**st.secrets["postgres"], sslmode="require")
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Errore di connessione al database: {e}")
        return pd.DataFrame()

# --- Caricamento dati iniziali ---
try:
    df = run_query('SELECT * FROM "allcamp";')
    if df.empty:
        st.warning("Il DataFrame caricato dal database è vuoto.")
        st.stop()
except Exception as e:
    st.error(f"Errore durante il caricamento del database: {e}")
    st.stop()

# --- Aggiunta colonne calcolate e controllo esistenza colonna 'data' ---
# Controllo se la colonna 'data' o 'date' esiste prima di usarla
date_col_name = None
if "data" in df.columns:
    date_col_name = "data"
elif "date" in df.columns:
    date_col_name = "date"

has_date_column = date_col_name is not None
if has_date_column:
    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
else:
    st.sidebar.warning("Le colonne 'data' o 'date' non sono presenti nel database. Il filtro per le ultime N partite non sarà disponibile.")

# --- Selettori Sidebar ---
st.sidebar.header("Seleziona Squadre")

# 1. Selettore Campionato
leagues = ["Seleziona..."] + sorted(df["league"].dropna().unique())
selected_league = st.sidebar.selectbox("Seleziona Campionato", leagues)

if selected_league != "Seleziona...":
    # Filtra le squadre in base al campionato selezionato
    league_teams_df = df[df["league"] == selected_league]
    all_teams = sorted(list(set(league_teams_df['home_team'].dropna().unique()) | set(league_teams_df['away_team'].dropna().unique())))

    # 2. Selettori Squadre
    home_team_selected = st.sidebar.selectbox("Seleziona Squadra CASA", ["Seleziona..."] + all_teams)
    away_team_selected = st.sidebar.selectbox("Seleziona Squadra TRASFERTA", ["Seleziona..."] + all_teams)

    if home_team_selected != "Seleziona..." and away_team_selected != "Seleziona...":

        # --- Selettore per le ultime N partite, mostrato solo se la colonna 'data' o 'date' esiste ---
        num_to_filter = None
        if has_date_column:
            st.sidebar.header("Filtra Partite per Numero")
            num_partite_options = ["Tutte", "Ultime 5", "Ultime 10", "Ultime 15", "Ultime 20", "Ultime 30", "Ultime 40", "Ultime 50"]
            selected_num_partite_str = st.sidebar.selectbox("Numero di partite da analizzare", num_partite_options)

            # Converti la selezione in un numero intero
            if selected_num_partite_str != "Tutte":
                num_to_filter = int(selected_num_partite_str.split(' ')[1])

        # --- Nuovi selettori dinamici per minutaggio ---
        st.sidebar.header("Imposta Intervallo Minutaggio")
        start_minute = st.sidebar.slider("Minuto Inizio", 0, 90, 0)
        end_minute = st.sidebar.slider("Minuto Fine", start_minute, 90, 90)

        # --- FILTRAGGIO E COMBINAZIONE DATI ---
        
        # Prendi tutte le partite in casa della squadra di casa
        df_home = df[(df["home_team"] == home_team_selected) & (df["league"] == selected_league)]

        # Prendi tutte le partite in trasferta della squadra in trasferta
        df_away = df[(df["away_team"] == away_team_selected) & (df["league"] == selected_league)]

        # Ordina per data decrescente e filtra per il numero di partite, solo se la colonna esiste
        if has_date_column:
            df_home = df_home.sort_values(by=date_col_name, ascending=False)
            df_away = df_away.sort_values(by=date_col_name, ascending=False)

        if num_to_filter is not None:
            df_home = df_home.head(num_to_filter)
            df_away = df_away.head(num_to_filter)

        # Combina i due DataFrame
        df_combined = pd.concat([df_home, df_away], ignore_index=True)
        
        st.header(f"Analisi Combinata: {home_team_selected} (Casa) vs {away_team_selected} (Trasferta)")
        st.write(f"Basata su **{len(df_home)}** partite casalinghe di '{home_team_selected}' e **{len(df_away)}** partite in trasferta di '{away_team_selected}'.")
        st.write(f"**Totale Partite Analizzate:** {len(df_combined)}")

        if not df_combined.empty:

            # --- Funzione per calcolare il punteggio al minuto 'end_minute' ---
            def get_scores_at_time_range(df_to_analyze, end_min):
                df_copy = df_to_analyze.copy()
                df_copy["gol_home_time"] = 0
                df_copy["gol_away_time"] = 0
                
                for index, row in df_copy.iterrows():
                    gol_home_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                    gol_away_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                    
                    # Conteggio gol della squadra di casa al minuto 'end_min'
                    home_goals_count = sum(1 for g in gol_home_minutes if g <= end_min)
                    # Conteggio gol della squadra in trasferta al minuto 'end_min'
                    away_goals_count = sum(1 for g in gol_away_minutes if g <= end_min)
                    
                    df_copy.loc[index, "gol_home_time"] = home_goals_count
                    df_copy.loc[index, "gol_away_time"] = away_goals_count
                
                df_copy["risultato_time"] = df_copy["gol_home_time"].astype(str) + "-" + df_copy["gol_away_time"].astype(str)
                return df_copy


            # --- Preparazione del DataFrame dinamico ---
            df_dynamic = get_scores_at_time_range(df_combined, end_minute)
            
            # --- INIZIO FUNZIONI STATISTICHE AGGIORNATE ---
            def get_outcome(home_score, away_score):
                """Determina l'esito (1, X, 2) da un punteggio."""
                if home_score > away_score:
                    return '1'
                elif home_score < away_score:
                    return '2'
                else:
                    return 'X'

            def calcola_winrate(df_to_analyze, col_risultato, title):
                st.subheader(f"WinRate ({title}) ({len(df_to_analyze)} partite)")
                df_valid = df_to_analyze[df_to_analyze[col_risultato].notna() & (df_to_analyze[col_risultato].str.contains("-"))]
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
                
                df_stats = pd.DataFrame(stats, columns=["Esito", "Conteggio", "WinRate %", "Odd Minima"])
                st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['WinRate %']))

            def mostra_risultati_esatti(df_to_analyze, col_risultato, titolo):
                st.subheader(f"Risultati Esatti ({titolo}) ({len(df_to_analyze)} partite)")
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
                st.subheader(f"Over/Under Goals ({title}) ({len(df_to_analyze)} partite)")
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
                st.subheader(f"Entrambe le Squadre a Segno (GG/NG) ({title}) ({len(df_to_analyze)} partite)")
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
                st.subheader(f"Clean Sheets / Fail to Score ({title}) (Home: {len(df_home_to_analyze)} partite, Away: {len(df_away_to_analyze)} partite)")
                
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
            
            def calcola_ht_ft_combo(df_to_analyze, col_home_ht, col_away_ht, col_home_ft, col_away_ft):
                st.subheader(f"Risultato Parziale/Finale (HT/FT) ({len(df_to_analyze)} partite)")
                df_copy = df_to_analyze.copy()

                df_copy["ht_outcome"] = df_copy.apply(lambda row: get_outcome(row[col_home_ht], row[col_away_ht]), axis=1)
                df_copy["ft_outcome"] = df_copy.apply(lambda row: get_outcome(row[col_home_ft], row[col_away_ft]), axis=1)

                df_copy["combo"] = df_copy["ht_outcome"] + "/" + df_copy["ft_outcome"]
                combo_counts = df_copy["combo"].value_counts().reset_index()
                combo_counts.columns = ["Risultato", "Conteggio"]
                
                total = len(df_to_analyze)
                combo_counts["Percentuale %"] = (combo_counts["Conteggio"] / total * 100).round(2)
                combo_counts["Odd Minima"] = combo_counts["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                st.dataframe(combo_counts.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))

            def calcola_margine_vittoria(df_to_analyze, col_gol_home, col_gol_away, title):
                st.subheader(f"Margine di Vittoria ({title}) ({len(df_to_analyze)} partite)")
                df_copy = df_to_analyze.copy()
                df_copy["gol_diff"] = pd.to_numeric(df_copy[col_gol_home], errors='coerce').fillna(0) - pd.to_numeric(df_copy[col_gol_away], errors='coerce').fillna(0)
                
                def classify_margin(diff):
                    if diff > 0:
                        if diff == 1: return "Casa vince di 1"
                        elif diff == 2: return "Casa vince di 2"
                        else: return "Casa vince di 3+"
                    elif diff < 0:
                        if diff == -1: return "Trasferta vince di 1"
                        elif diff == -2: return "Trasferta vince di 2"
                        else: return "Trasferta vince di 3+"
                    else:
                        return "Pareggio"
                
                df_copy["margine"] = df_copy["gol_diff"].apply(classify_margin)
                
                margin_counts = df_copy["margine"].value_counts().reset_index()
                margin_counts.columns = ["Margine", "Conteggio"]
                total = len(df_to_analyze)
                margin_counts["Percentuale %"] = (margin_counts["Conteggio"] / total * 100).round(2)
                margin_counts["Odd Minima"] = margin_counts["Percentuale %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                st.dataframe(margin_counts.style.background_gradient(cmap='RdYlGn', subset=['Percentuale %']))
            
            def mostra_progressione_combinata(df_combined, home_team_selected, away_team_selected, start_min, end_min):
                st.subheader(f"Progressione Risultati (Combinata tra min {start_min} e {end_min})")
                total_matches = len(df_combined)

                if total_matches == 0:
                    st.write("Nessun dato disponibile per l'analisi della progressione.")
                    return

                home_went_1_0 = 0
                home_went_1_1_after_1_0 = 0
                home_went_2_0 = 0

                away_went_0_1 = 0
                away_went_1_1_after_0_1 = 0
                away_went_0_2 = 0
                
                total_matches_with_goals = 0

                for _, row in df_combined.iterrows():
                    gol_home_minutes = [int(x) for x in str(row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                    gol_away_minutes = [int(x) for x in str(row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                    
                    # Filtra i gol nel range di tempo
                    gol_home_range = [m for m in gol_home_minutes if start_min <= m <= end_min]
                    gol_away_range = [m for m in gol_away_minutes if start_min <= m <= end_min]
                    
                    if gol_home_range or gol_away_range:
                        total_matches_with_goals += 1
                        
                        is_home_selected_at_home = row["home_team"] == home_team_selected
                        
                        goal_events = []
                        if is_home_selected_at_home:
                            goal_events = [(m, "home") for m in gol_home_range] + [(m, "away") for m in gol_away_range]
                        else:
                            goal_events = [(m, "away") for m in gol_away_range] + [(m, "home") for m in gol_home_range]
                        
                        goal_events.sort(key=lambda x: x[0])

                        current_home_score_analysis = 0
                        current_away_score_analysis = 0
                        
                        has_been_1_0 = False
                        has_been_1_1_after_1_0 = False
                        has_been_2_0 = False
                        
                        has_been_0_1 = False
                        has_been_1_1_after_0_1 = False
                        has_been_0_2 = False

                        for _, team in goal_events:
                            if team == "home":
                                current_home_score_analysis += 1
                            else:
                                current_away_score_analysis += 1

                            if is_home_selected_at_home:
                                if current_home_score_analysis == 1 and current_away_score_analysis == 0:
                                    has_been_1_0 = True
                                if has_been_1_0 and current_home_score_analysis == 1 and current_away_score_analysis == 1:
                                    has_been_1_1_after_1_0 = True
                                if current_home_score_analysis == 2 and current_away_score_analysis == 0:
                                    has_been_2_0 = True
                            else:
                                if current_home_score_analysis == 0 and current_away_score_analysis == 1:
                                    has_been_0_1 = True
                                if has_been_0_1 and current_home_score_analysis == 1 and current_away_score_analysis == 1:
                                    has_been_1_1_after_0_1 = True
                                if current_home_score_analysis == 0 and current_away_score_analysis == 2:
                                    has_been_0_2 = True
                        
                        if is_home_selected_at_home:
                            if has_been_1_0: home_went_1_0 += 1
                            if has_been_1_1_after_1_0: home_went_1_1_after_1_0 += 1
                            if has_been_2_0: home_went_2_0 += 1
                        else:
                            if has_been_0_1: away_went_0_1 += 1
                            if has_been_1_1_after_0_1: away_went_1_1_after_0_1 += 1
                            if has_been_0_2: away_went_0_2 += 1
                
                denominator = total_matches_with_goals if total_matches_with_goals > 0 else 1
                
                home_progression_data = {
                    "Scenario": [
                        f"{home_team_selected} va in vantaggio 1-0",
                        f"Il punteggio diventa 1-1 dopo l'1-0",
                        f"{home_team_selected} va in vantaggio 2-0"
                    ],
                    "Conteggio": [
                        home_went_1_0,
                        home_went_1_1_after_1_0,
                        home_went_2_0
                    ],
                    "Percentuale su partite con gol %": [
                        round((home_went_1_0 / denominator) * 100, 2),
                        round((home_went_1_1_after_1_0 / denominator) * 100, 2),
                        round((home_went_2_0 / denominator) * 100, 2)
                    ]
                }
                df_home_prog = pd.DataFrame(home_progression_data)
                df_home_prog["Odd Minima"] = df_home_prog["Percentuale su partite con gol %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                away_progression_data = {
                    "Scenario": [
                        f"{away_team_selected} va in vantaggio 0-1",
                        f"Il punteggio diventa 1-1 dopo lo 0-1",
                        f"{away_team_selected} va in vantaggio 0-2"
                    ],
                    "Conteggio": [
                        away_went_0_1,
                        away_went_1_1_after_0_1,
                        away_went_0_2
                    ],
                    "Percentuale su partite con gol %": [
                        round((away_went_0_1 / denominator) * 100, 2),
                        round((away_went_1_1_after_0_1 / denominator) * 100, 2),
                        round((away_went_0_2 / denominator) * 100, 2)
                    ]
                }
                df_away_prog = pd.DataFrame(away_progression_data)
                df_away_prog["Odd Minima"] = df_away_prog["Percentuale su partite con gol %"].apply(lambda x: round(100/x, 2) if x > 0 else "-")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{home_team_selected}**")
                    st.dataframe(df_home_prog.style.background_gradient(cmap='RdYlGn', subset=['Percentuale su partite con gol %']))
                with col2:
                    st.write(f"**{away_team_selected}**")
                    st.dataframe(df_away_prog.style.background_gradient(cmap='RdYlGn', subset=['Percentuale su partite con gol %']))

            
            # --- ESECUZIONE E VISUALIZZAZIONE STATS ---
            st.markdown("---")
            st.header(f"Statistiche Dinamiche (punteggio finale al minuto {end_minute})")
            
            calcola_winrate(df_dynamic, "risultato_time", f"al min {end_minute}")
            mostra_risultati_esatti(df_dynamic, "risultato_time", f"al min {end_minute}")
            calcola_over_goals(df_dynamic, "gol_home_time", "gol_away_time", f"al min {end_minute}")
            calcola_btts(df_dynamic, "gol_home_time", "gol_away_time", f"al min {end_minute}")
            calcola_margine_vittoria(df_dynamic, "gol_home_time", "gol_away_time", f"al min {end_minute}")

            st.markdown("---")
            st.header("Statistiche sui Gol e Analisi Temporale")
            
            # La funzione di progressione ora usa l'intervallo dinamico
            mostra_progressione_combinata(df_combined, home_team_selected, away_team_selected, start_minute, end_minute)

            # Il resto delle funzioni non dinamiche rimane invariato
            st.markdown("---")
            st.header("Analisi Temporale dei Gol")
            
            # Analisi per intervalli di 5 minuti
            mostra_distribuzione_timeband(df_combined, "(5 Min)", home_team_selected, away_team_selected)

            # Analisi per intervalli di 15 minuti
            mostra_distribuzione_timeband(df_combined, "(15 Min)", home_team_selected, away_team_selected, timeframe=15)

        else:
            st.warning("Nessuna partita trovata per la combinazione di squadre e campionato selezionata.")
else:
    st.info("Per iniziare, seleziona un campionato dalla barra laterale.")

