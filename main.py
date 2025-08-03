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

if selected_league != "Seleziona...":
    # Filtra le squadre disponibili in base al campionato selezionato
    league_teams_df = df[df["league"] == selected_league]
    all_teams = sorted(list(set(league_teams_df['home_team'].dropna().unique()) | set(league_teams_df['away_team'].dropna().unique())))

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
        df_home = df[(df["home_team"] == home_team_selected) & (df["league"] == selected_league)]
        
        # Filtra le partite in trasferta della squadra selezionata
        df_away = df[(df["away_team"] == away_team_selected) & (df["league"] == selected_league)]
        
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
            
            # --- Funzione di supporto per ottenere il punteggio a un minuto specifico ---
            def get_scores_at_minute(df_row, selected_min, home_team_name, away_team_name):
                """
                Calcola il punteggio di una partita a un minuto specificato.
                Restituisce una stringa nel formato 'gol_home-gol_away'.
                """
                gol_home_minutes = [int(x) for x in str(df_row.get("minutaggio_gol", "")).split(";") if x.isdigit()]
                gol_away_minutes = [int(x) for x in str(df_row.get("minutaggio_gol_away", "")).split(";") if x.isdigit()]
                
                # Calcola i gol segnati dalla squadra di casa e da quella in trasferta selezionate
                # fino al minuto specificato.
                if df_row["home_team"] == home_team_name:
                    home_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min)
                    away_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min)
                else: # Questo caso si verifica quando la squadra "di casa" selezionata gioca in trasferta nella riga attuale
                    home_goals_count = sum(1 for g in gol_away_minutes if g <= selected_min)
                    away_goals_count = sum(1 for g in gol_home_minutes if g <= selected_min)
                
                return f"{home_goals_count}-{away_goals_count}"


            # --- INIZIO FUNZIONI STATISTICHE (Pre-partita e dinamiche) ---
            
            def get_outcome(home_score, away_score):
                """Determina l'esito (1, X, 2) da un punteggio."""
                if home_score > away_score:
                    return '1'
                elif home_score < away_score:
                    return '2'
                else:
                    return 'X'

            def calcola_winrate(df_to_analyze, col_risultato, title):
                """Calcola e mostra il WinRate basato sui risultati complessivi."""
                st.subheader(f"WinRate {title} ({len(df_to_analyze)} partite)")
                df_valid = df_to_analyze[df_to_analyze[col_risultato].notna() & (df_to_analyze[col_risultato].str.contains("-"))]
                
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

            def calcola_media_gol(df_home, df_away, home_team_name, away_team_name):
                """Calcola la media gol fatti e subiti per le squadre selezionate."""
                st.subheader(f"Media Gol Fatti e Subiti (Home: {len(df_home)} partite, Away: {len(df_away)} partite)")
                
                home_goals_ht = pd.to_numeric(df_home["gol_home_ht"], errors='coerce').mean()
                home_conceded_ht = pd.to_numeric(df_home["gol_away_ht"], errors='coerce').mean()
                away_goals_ht = pd.to_numeric(df_away["gol_away_ht"], errors='coerce').mean()
                away_conceded_ht = pd.to_numeric(df_away["gol_home_ht"], errors='coerce').mean()

                home_goals_ft = pd.to_numeric(df_home["gol_home_ft"], errors='coerce').mean()
                home_conceded_ft = pd.to_numeric(df_home["gol_away_ft"], errors='coerce').mean()
                away_goals_ft = pd.to_numeric(df_away["gol_away_ft"], errors='coerce').mean()
                away_conceded_ft = pd.to_numeric(df_away["gol_home_ft"], errors='coerce').mean()

                home_goals_sh = pd.to_numeric(df_home["gol_home_sh"], errors='coerce').mean()
                home_conceded_sh = pd.to_numeric(df_home["gol_away_sh"], errors='coerce').mean()
                away_goals_sh = pd.to_numeric(df_away["gol_away_sh"], errors='coerce').mean()
                away_conceded_sh = pd.to_numeric(df_away["gol_home_sh"], errors='coerce').mean()

                data = {
                    "Squadra": [home_team_name, away_team_name],
                    "Gol Fatti (PT)": [f"{home_goals_ht:.2f}", f"{away_goals_ht:.2f}"],
                    "Gol Subiti (PT)": [f"{home_conceded_ht:.2f}", f"{away_conceded_ht:.2f}"],
                    "Gol Fatti (ST)": [f"{home_goals_sh:.2f}", f"{away_goals_sh:.2f}"],
                    "Gol Subiti (ST)": [f"{home_conceded_sh:.2f}", f"{away_conceded_sh:.2f}"],
                    "Gol Fatti (FT)": [f"{home_goals_ft:.2f}", f"{away_goals_ft:.2f}"],
                    "Gol Subiti (FT)": [f"{home_conceded_ft:.2f}", f"{away_conceded_ft:.2f}"]
                }
                
                df_media = pd.DataFrame(data)
                st.table(df_media)

            # --- ESECUZIONE E VISUALIZZAZIONE STATS PRE-PARTITA (FISSE) ---
            st.markdown("---")
            st.header("Statistiche Pre-partita")
            
            calcola_media_gol(df_home, df_away, home_team_selected, away_team_selected)
            
            # WinRate PT, ST, FT
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_winrate(df_combined, "risultato_ht", "PT")
            with col2:
                calcola_winrate(df_combined, "risultato_sh", "ST")
            with col3:
                calcola_winrate(df_combined, "risultato_ft", "FT")

            # Risultati Esatti PT, ST, FT
            col1, col2, col3 = st.columns(3)
            with col1:
                mostra_risultati_esatti(df_combined, "risultato_ht", "PT")
            with col2:
                mostra_risultati_esatti(df_combined, "risultato_sh", "ST")
            with col3:
                mostra_risultati_esatti(df_combined, "risultato_ft", "FT")
            
            st.markdown("---")
            st.header("Statistiche sui Gol Pre-partita")
            
            # Over/Under PT, ST, FT
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_over_goals(df_combined, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_over_goals(df_combined, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_over_goals(df_combined, "gol_home_ft", "gol_away_ft", "FT")

            # GG/NG PT, ST, FT
            col1, col2, col3 = st.columns(3)
            with col1:
                calcola_btts(df_combined, "gol_home_ht", "gol_away_ht", "PT")
            with col2:
                calcola_btts(df_combined, "gol_home_sh", "gol_away_sh", "ST")
            with col3:
                calcola_btts(df_combined, "gol_home_ft", "gol_away_ft", "FT")

            # --- ESECUZIONE E VISUALIZZAZIONE STATS DINAMICHE ---
            st.markdown("---")
            st.header("Statistiche Dinamiche (basate su un intervallo di minutaggio)")

            col_sliders_1, col_sliders_2 = st.columns(2)
            with col_sliders_1:
                # Cursore con due maniglie per l'intervallo di minutaggio
                minutaggio_range = st.slider("Intervallo di minutaggio", 0, 90, (45, 90), key="minute_range_slider")
                start_minute = minutaggio_range[0]
                end_minute = minutaggio_range[1]
            with col_sliders_2:
                starting_score_str = st.text_input("Risultato di Partenza (es. 1-0)", "0-0", key="score_input")

            # Validazione e parsing del risultato di partenza
            try:
                if "-" in starting_score_str:
                    # Filtra il DataFrame in base al minutaggio e al risultato di partenza
                    # L'applicazione calcola il punteggio al minuto finale dell'intervallo e lo confronta
                    # con il risultato di partenza inserito dall'utente.
                    df_dynamic_filtered = df_combined[df_combined.apply(
                        lambda row: get_scores_at_minute(row, end_minute, home_team_selected, away_team_selected) == starting_score_str,
                        axis=1
                    )]
                else:
                    st.warning("Formato risultato non valido. Usa il formato 'X-Y'.")
                    df_dynamic_filtered = pd.DataFrame()
            except ValueError:
                st.warning("Formato risultato non valido. Usa il formato 'X-Y'.")
                df_dynamic_filtered = pd.DataFrame()
            
            if not df_dynamic_filtered.empty:
                st.write(f"Analisi basata su **{len(df_dynamic_filtered)}** partite in cui il punteggio era **{starting_score_str}** al minuto **{end_minute}**.")
                
                # Le funzioni dinamiche usano ora il DataFrame filtrato
                calcola_winrate(df_dynamic_filtered, "risultato_ft", "Finale (per partite con punteggio specificato)")
                mostra_risultati_esatti(df_dynamic_filtered, "risultato_ft", "Finale (per partite con punteggio specificato)")
                calcola_over_goals(df_dynamic_filtered, "gol_home_ft", "gol_away_ft", "Finale (per partite con punteggio specificato)")
                calcola_btts(df_dynamic_filtered, "gol_home_ft", "gol_away_ft", "Finale (per partite con punteggio specificato)")

            else:
                if starting_score_str and "-" in starting_score_str:
                    st.warning(f"Nessuna partita trovata in cui il punteggio era {starting_score_str} al minuto {end_minute}.")
                else:
                    st.info("Inserisci un risultato di partenza valido per avviare l'analisi dinamica.")

        else:
            st.warning("Nessuna partita trovata per la combinazione di squadre e campionato selezionata.")
else:
    st.info("Per iniziare, seleziona un campionato dalla barra laterale.")

