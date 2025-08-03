import streamlit as st
import psycopg2
import pandas as pd
import numpy as np

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Analisi Next Gol e stats live", layout="wide")
st.title("Analisi Tabella allcamp")

# --- Funzione connessione al database (cacheata per efficienza) ---
@st.cache_data
def run_query(query: str):
    """
    Esegue una query SQL e restituisce i risultati come DataFrame.
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

# --- VALIDAZIONE: VERIFICA CHE TUTTE LE COLONNE ESSENZIALI SIANO PRESENTI ---
required_columns = [
    'campionato', 'score_start_next_goal', 'goal_next_team_live',
    'gol_home_ft', 'gol_away_ft', 'gol_home_ht', 'gol_away_ht',
    'minuto', 'minute_goal', 'gol_home_live', 'gol_away_live',
    'odd_next_goal_home', 'odd_next_goal_away', 'odd_next_goal_no',
    'odd_home_live', 'odd_draw_live', 'odd_away_live', 'home_team', 'away_team',
    'penalties_home', 'penalties_away', 'data', 'odd_over_2_5', 'odd_btts_si'
]

missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    st.error(
        f"Errore critico: Le seguenti colonne essenziali non sono state trovate nel database: "
        f"{', '.join(missing_cols)}. "
        f"L'applicazione non può funzionare senza queste colonne. "
        f"Assicurati che la tabella 'allcamp' contenga tutte le colonne necessarie."
    )
    st.stop()

# --- PRE-ELABORAZIONE DEI DATI ---
# Questa sezione viene eseguita solo se tutte le colonne necessarie sono presenti.
try:
    # Conversione di colonne numeriche
    for col in [
        'minuto', 'minute_goal', 'gol_home_live', 'gol_away_live', 'gol_home_ft',
        'gol_away_ft', 'gol_home_ht', 'gol_away_ht', 'penalties_home', 'penalties_away'
    ]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Conversione di colonne di quote
    for col in [
        'odd_next_goal_home', 'odd_next_goal_away', 'odd_next_goal_no',
        'odd_home_live', 'odd_draw_live', 'odd_away_live', 'odd_over_2_5', 'odd_btts_si'
    ]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1.0)
    
    # Calcolo dei gol del secondo tempo (second half)
    df['gol_home_sh'] = df['gol_home_ft'] - df['gol_home_ht']
    df['gol_away_sh'] = df['gol_away_ft'] - df['gol_away_ht']
    df['gol_home_sh'] = df['gol_home_sh'].apply(lambda x: max(x, 0))
    df['gol_away_sh'] = df['gol_away_sh'].apply(lambda x: max(x, 0))
    
    # Aggiunta di colonne calcolate per facilitare le analisi
    df["risultato_ft"] = df["gol_home_ft"].astype(str) + "-" + df["gol_away_ft"].astype(str)

    # Rimozione righe con valori 'nan' nelle quote cruciali
    df.dropna(subset=['odd_next_goal_home'], inplace=True)

except Exception as e:
    st.error(f"Errore durante la pre-elaborazione dei dati: {e}")
    st.stop()


# --- Filtri per l'utente ---
st.sidebar.header("Filtri Dati")
camp_selezionato = st.sidebar.selectbox("Seleziona Campionato", df["campionato"].unique())
df_filtered_camp = df[df["campionato"] == camp_selezionato]

punteggi_unici = sorted(df_filtered_camp["score_start_next_goal"].unique())
score_start_next_goal_selezionato = st.sidebar.selectbox("Seleziona Punteggio di partenza", punteggi_unici)
df_filtered_punteggio = df_filtered_camp[df_filtered_camp["score_start_next_goal"] == score_start_next_goal_selezionato]

st.header(f"Dati per il campionato: {camp_selezionato}")
st.write(f"**Partite dopo il filtro del campionato:** {len(df_filtered_camp)}")
st.write(f"**Partite dopo il filtro del punteggio:** {len(df_filtered_punteggio)}")

# --- Analisi dei Dati (Stats Next Gol) ---
if not df_filtered_punteggio.empty:
    st.subheader("Statistiche sul 'Next Gol' a un determinato punteggio")
    
    # Calcola il numero di volte che il gol successivo è stato segnato
    gol_successivo_df = df_filtered_punteggio[
        (df_filtered_punteggio["goal_next_team_live"] == "Home") | 
        (df_filtered_punteggio["goal_next_team_live"] == "Away")
    ]
    numero_gol_successivi = len(gol_successivo_df)
    
    # Calcola il numero di volte che NON è stato segnato un gol
    no_gol_successivo_df = df_filtered_punteggio[df_filtered_punteggio["goal_next_team_live"] == "No Goal"]
    numero_no_gol_successivo = len(no_gol_successivo_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Partite analizzate", len(df_filtered_punteggio))
    with col2:
        st.metric("Totale Next Gol", numero_gol_successivi)
    with col3:
        st.metric("Totale No Next Gol", numero_no_gol_successivo)
    with col4:
        # Calcola la percentuale di next gol
        if len(df_filtered_punteggio) > 0:
            percentuale_next_gol = (numero_gol_successivi / len(df_filtered_punteggio)) * 100
            st.metric("Percentuale Next Gol", f"{percentuale_next_gol:.2f}%")
        else:
            st.metric("Percentuale Next Gol", "N/A")
            
    # Analisi dei risultati per il "Next Gol" a casa e in trasferta
    st.markdown("### Dettaglio Next Gol")
    if numero_gol_successivi > 0:
        gol_home_successivo = len(gol_successivo_df[gol_successivo_df["goal_next_team_live"] == "Home"])
        gol_away_successivo = len(gol_successivo_df[gol_successivo_df["goal_next_team_live"] == "Away"])

        col_next1, col_next2 = st.columns(2)
        with col_next1:
            st.metric("Gol Successivo Squadra di Casa", gol_home_successivo)
        with col_next2:
            st.metric("Gol Successivo Squadra in Trasferta", gol_away_successivo)

        # Calcola le quote medie per il next gol
        avg_odd_home = gol_successivo_df[gol_successivo_df["goal_next_team_live"] == "Home"]["odd_next_goal_home"].mean()
        avg_odd_away = gol_successivo_df[gol_successivo_df["goal_next_team_live"] == "Away"]["odd_next_goal_away"].mean()
        
        col_odd1, col_odd2 = st.columns(2)
        with col_odd1:
            st.metric("Quota Media Next Gol Casa", f"{avg_odd_home:.2f}")
        with col_odd2:
            st.metric("Quota Media Next Gol Trasferta", f"{avg_odd_away:.2f}")
    
    # Analisi del "No Next Gol"
    if numero_no_gol_successivo > 0:
        st.markdown("### Dettaglio No Next Gol")
        avg_odd_no_gol = no_gol_successivo_df["odd_next_goal_no"].mean()
        st.metric("Quota Media No Next Gol", f"{avg_odd_no_gol:.2f}")
    else:
        st.info("Nessun 'Next Gol' o 'No Next Gol' trovato per l'analisi.")

    # --- Analisi dei risultati finali (FT) dopo un certo punteggio live ---
    st.markdown("---")
    st.subheader("Risultati Finali dopo un determinato punteggio live")
    
    # Calcola i risultati FT (casa, pareggio, trasferta)
    risultati_ft_casa = len(df_filtered_punteggio[
        df_filtered_punteggio["gol_home_ft"] > df_filtered_punteggio["gol_away_ft"]
    ])
    risultati_ft_pareggio = len(df_filtered_punteggio[
        df_filtered_punteggio["gol_home_ft"] == df_filtered_punteggio["gol_away_ft"]
    ])
    risultati_ft_trasferta = len(df_filtered_punteggio[
        df_filtered_punteggio["gol_home_ft"] < df_filtered_punteggio["gol_away_ft"]
    ])
    
    col_ft1, col_ft2, col_ft3 = st.columns(3)
    with col_ft1:
        st.metric("Vittorie FT Casa", risultati_ft_casa)
    with col_ft2:
        st.metric("Pareggi FT", risultati_ft_pareggio)
    with col_ft3:
        st.metric("Vittorie FT Trasferta", risultati_ft_trasferta)

else:
    st.warning("Nessun dato trovato per i filtri selezionati.")

# --- Sezione Backtest per il "Next Gol" ---
st.markdown("---")
st.header("Simulatore di Backtest 'Next Gol'")

# Preparazione del DataFrame per il backtest
df_backtest = df.copy()
df_backtest = df_backtest[
    (df_backtest['score_start_next_goal'] == score_start_next_goal_selezionato) &
    (df_backtest['campionato'] == camp_selezionato)
]
df_backtest.dropna(subset=['odd_next_goal_home', 'odd_next_goal_away', 'odd_next_goal_no'], inplace=True)
df_backtest = df_backtest.sort_values(by='data')

def esegui_backtest_next_gol(df_data, strategy_next_gol, odd_min, stake):
    vincite = 0
    perdite = 0
    profit_loss = 0.0
    scommesse_piazzate = 0
    
    for _, row in df_data.iterrows():
        odd_scommessa = None
        esito_vincente = False
        
        if strategy_next_gol == 'Next Gol Home':
            odd_scommessa = row['odd_next_goal_home']
            esito_vincente = row['goal_next_team_live'] == 'Home'
        elif strategy_next_gol == 'Next Gol Away':
            odd_scommessa = row['odd_next_goal_away']
            esito_vincente = row['goal_next_team_live'] == 'Away'
        else: # 'Next Gol No'
            odd_scommessa = row['odd_next_goal_no']
            esito_vincente = row['goal_next_team_live'] == 'No Goal'
        
        if odd_scommessa >= odd_min:
            scommesse_piazzate += 1
            if esito_vincente:
                vincite += 1
                profit_loss += (odd_scommessa - 1) * stake
            else:
                perdite += 1
                profit_loss -= stake
                
    if scommesse_piazzate > 0:
        roi = (profit_loss / (scommesse_piazzate * stake)) * 100 if (scommesse_piazzate * stake) > 0 else 0
        win_rate = (vincite / scommesse_piazzate) * 100
    else:
        roi = 0
        win_rate = 0
        
    return vincite, perdite, scommesse_piazzate, profit_loss, roi, win_rate

st.markdown("---")
st.subheader("Analisi Backtest Next Gol")
col_strategy, col_odd, col_stake = st.columns(3)
with col_strategy:
    strategy_next_gol = st.selectbox(
        "Strategia 'Next Gol'",
        ['Next Gol Home', 'Next Gol Away', 'Next Gol No']
    )
with col_odd:
    odd_min = st.number_input("Quota Minima", min_value=1.0, value=1.5, step=0.1)
with col_stake:
    stake_next_gol = st.number_input("Stake per scommessa", min_value=0.5, value=1.0, step=0.5)

if st.button("Avvia Backtest Next Gol"):
    if not df_backtest.empty:
        vincite, perdite, numero_scommesse, profit_loss, roi, win_rate = esegui_backtest_next_gol(
            df_backtest, strategy_next_gol, odd_min, stake_next_gol
        )
        
        if numero_scommesse > 0:
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Numero Scommesse", numero_scommesse)
            col_res2.metric("Vincite", vincite)
            col_res3.metric("Perdite", perdite)
            
            col_res4, col_res5, col_res6 = st.columns(3)
            col_res4.metric("Profitto/Perdita", f"{profit_loss:.2f} €")
            col_res5.metric("ROI", f"{roi:.2f}%")
            col_res6.metric("Win Rate", f"{win_rate:.2f}%")
        else:
            st.warning("Nessuna scommessa piazzata con i criteri selezionati.")
    else:
        st.warning("DataFrame per il backtest è vuoto. Rivedi i filtri.")

# --- Analisi stats live ---
st.markdown("---")
st.header("Analisi Statistiche Live")
st.subheader("Filtra i dati in base alle statistiche live")

# Aggiungi un filtro per la squadra di casa e in trasferta
squadre_casa_unici = sorted(df["home_team"].unique())
squadre_away_unici = sorted(df["away_team"].unique())
home_team_selected = st.selectbox("Seleziona Squadra di Casa", squadre_casa_unici)
away_team_selected = st.selectbox("Seleziona Squadra in Trasferta", squadre_away_unici)

df_home = df[(df["home_team"] == home_team_selected) & (df["campionato"] == camp_selezionato)]
df_away = df[(df["away_team"] == away_team_selected) & (df["campionato"] == camp_selezionato)]

st.write(f"Partite trovate per {home_team_selected} in casa: {len(df_home)}")
st.write(f"Partite trovate per {away_team_selected} in trasferta: {len(df_away)}")

# Funzione per calcolare le stats
def calcola_stats(df_data, nome_squadra, tipo_squadra):
    if df_data.empty:
        return
    
    st.markdown(f"### Statistiche di {nome_squadra} ({tipo_squadra})")
    
    # Statistiche sui gol
    if tipo_squadra == 'casa':
        gol_fatti = df_data['gol_home_live'].sum()
        gol_subiti = df_data['gol_away_live'].sum()
        rigori_fatti = df_data['penalties_home'].sum()
        rigori_subiti = df_data['penalties_away'].sum()
    else: # tipo_squadra == 'trasferta'
        gol_fatti = df_data['gol_away_live'].sum()
        gol_subiti = df_data['gol_home_live'].sum()
        rigori_fatti = df_data['penalties_away'].sum()
        rigori_subiti = df_data['penalties_home'].sum()

    st.write(f"Gol fatti: {gol_fatti}")
    st.write(f"Gol subiti: {gol_subiti}")
    st.write(f"Rigori fatti: {rigori_fatti}")
    st.write(f"Rigori subiti: {rigori_subiti}")

# Funzione per calcolare To Score HT e FT
def calcola_to_score_ht_ft(df_home, df_away, home_team, away_team):
    st.markdown("### To Score HT/FT")
    
    total_home_matches = len(df_home)
    total_away_matches = len(df_away)

    # Analisi To Score Home HT/FT
    if total_home_matches > 0:
        to_score_ht_home_matches = len(df_home[df_home["gol_home_ht"] > 0])
        to_score_sh_home_matches = len(df_home[df_home["gol_home_sh"] > 0])
        to_score_ht_sh_home_matches = len(df_home[(df_home["gol_home_ht"] > 0) & (df_home["gol_home_sh"] > 0)])
        
        st.write(f"**{home_team} (casa)**:")
        st.write(f"Partite in cui ha segnato nel 1° tempo: {to_score_ht_home_matches} su {total_home_matches} ({to_score_ht_home_matches/total_home_matches*100:.2f}%)")
        st.write(f"Partite in cui ha segnato nel 2° tempo: {to_score_sh_home_matches} su {total_home_matches} ({to_score_sh_home_matches/total_home_matches*100:.2f}%)")
        st.write(f"Partite in cui ha segnato in entrambi i tempi: {to_score_ht_sh_home_matches} su {total_home_matches} ({to_score_ht_sh_home_matches/total_home_matches*100:.2f}%)")
    else:
        st.write(f"**{home_team} (casa)**: Nessun dato disponibile.")

    # Analisi To Score Away HT/FT
    if total_away_matches > 0:
        to_score_ht_away_matches = len(df_away[df_away["gol_away_ht"] > 0])
        to_score_sh_away_matches = len(df_away[df_away["gol_away_sh"] > 0])
        to_score_ht_sh_away_matches = len(df_away[(df_away["gol_away_ht"] > 0) & (df_away["gol_away_sh"] > 0)])
        
        st.write(f"**{away_team} (trasferta)**:")
        st.write(f"Partite in cui ha segnato nel 1° tempo: {to_score_ht_away_matches} su {total_away_matches} ({to_score_ht_away_matches/total_away_matches*100:.2f}%)")
        st.write(f"Partite in cui ha segnato nel 2° tempo: {to_score_sh_away_matches} su {total_away_matches} ({to_score_sh_away_matches/total_away_matches*100:.2f}%)")
        st.write(f"Partite in cui ha segnato in entrambi i tempi: {to_score_ht_sh_away_matches} su {total_away_matches} ({to_score_ht_sh_away_matches/total_away_matches*100:.2f}%)")
    else:
        st.write(f"**{away_team} (trasferta)**: Nessun dato disponibile.")


if not df_home.empty and not df_away.empty:
    calcola_stats(df_home, home_team_selected, "casa")
    calcola_stats(df_away, away_team_selected, "trasferta")
    
    calcola_to_score_ht_ft(df_home, df_away, home_team_selected, away_team_selected)
else:
    st.warning("Selezionare due squadre e un campionato con dati disponibili.")

# --- Sezione Backtest per il mercato 1x2 e Over/Under ---
st.markdown("---")
st.header("Simulatore di Backtest per Mercati Tradizionali")
st.subheader("Backtest su risultati finali (1X2, Over/Under)")

# Prepara il DataFrame per il backtest
filtered_df = df[(df['campionato'] == camp_selezionato) & 
                 (df['home_team'] == home_team_selected) & 
                 (df['away_team'] == away_team_selected)].copy()
        
filtered_df.dropna(subset=['odd_home_live', 'odd_draw_live', 'odd_away_live', 'odd_over_2_5', 'odd_btts_si'], inplace=True)
filtered_df = filtered_df.sort_values(by='data')


def esegui_backtest(df_data, market, strategy, stake):
    vincite = 0
    perdite = 0
    numero_scommesse = 0
    profit_loss = 0.0
    
    for _, row in df_data.iterrows():
        odd_scommessa = None
        esito_vincente = False
        
        # Logica per il mercato e la strategia
        if market == "1 (Casa)":
            odd_scommessa = row['odd_home_live']
            if row['gol_home_ft'] > row['gol_away_ft']:
                esito_vincente = True
        elif market == "X (Pareggio)":
            odd_scommessa = row['odd_draw_live']
            if row['gol_home_ft'] == row['gol_away_ft']:
                esito_vincente = True
        elif market == "2 (Trasferta)":
            odd_scommessa = row['odd_away_live']
            if row['gol_home_ft'] < row['gol_away_ft']:
                esito_vincente = True
        elif market == "Over 2.5 FT":
            odd_scommessa = row['odd_over_2_5']
            if (row['gol_home_ft'] + row['gol_away_ft']) > 2: # Over 2.5
                esito_vincente = True
        elif market == "BTTS SI FT":
            odd_scommessa = row['odd_btts_si']
            if row['gol_home_ft'] > 0 and row['gol_away_ft'] > 0:
                esito_vincente = True
        else:
            continue
            
        if odd_scommessa and odd_scommessa > 1:
            numero_scommesse += 1
            if strategy == "Back":
                if esito_vincente:
                    vincite += 1
                    profit_loss += (odd_scommessa - 1) * stake
                else:
                    perdite += 1
                    profit_loss -= stake
            elif strategy == "Lay":
                if not esito_vincente:
                    vincite += 1
                    profit_loss += stake 
                else:
                    perdite += 1
                    profit_loss -= (odd_scommessa - 1) * stake

    if numero_scommesse > 0:
        roi = (profit_loss / (numero_scommesse * stake)) * 100 if (numero_scommesse * stake) > 0 else 0
        win_rate = (vincite / numero_scommesse) * 100
    else:
        roi = 0
        win_rate = 0
        
    return vincite, perdite, numero_scommesse, profit_loss, roi, win_rate

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
    if not filtered_df.empty:
        vincite, perdite, numero_scommesse, profit_loss, roi, win_rate = esegui_backtest(filtered_df, backtest_market, backtest_strategy, stake)
        
        if numero_scommesse > 0:
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            col_met1.metric("Numero Scommesse", numero_scommesse)
            col_met2.metric("Vincite", vincite)
            col_met3.metric("Perdite", perdite)
            col_met4.metric("Profitto/Perdita", f"{profit_loss:.2f} €")
            
            col_met5, col_met6 = st.columns(2)
            col_met5.metric("ROI", f"{roi:.2f}%")
            col_met6.metric("Win Rate", f"{win_rate:.2f}%")
        else:
            st.warning("Nessuna scommessa piazzata con i criteri selezionati.")
    else:
        st.warning("DataFrame per il backtest è vuoto. Rivedi i filtri per il campionato e le squadre.")
