import numpy as np
import pandas as pd

from scipy.signal import find_peaks


from data_fetcher import fetch_historical_data, convert_to_weekly


def calculate_sma(data: pd.DataFrame, window: int) -> float:
    '''Oblicza ostatnią wartość SMA dla podanego okna.

    Args:
        data (pd.DataFrame): DataFrame zawierający dane.
        window (int): Okno (ilość świec) dla obliczeń SMA.

    Returns:
        float: Ostatnia wartość SMA.
    '''
    if len(data) < window:
        return None
    sma = data['Close'].rolling(window=window).mean()
    return float(sma.iloc[-1])


def is_uptrend(ticker: str) -> bool:
    '''Weryfikuje, czy dany ticker znajduje się w trendzie wzrostowym.

    Trend wzrostowy jest zdefiniowany jako spełnienie dwóch warunków:
    1. Cena zamknięcia na interwale dziennym (D1) jest powyżej prostej średniej ruchomej SMA 200.
    2. Cena zamknięcia na interwale tygodniowym (W1) jest powyżej prostej średniej ruchomej SMA 50.

    Args:
        ticker (str): Symbol tickera, np. 'PKO.WA'.

    Returns:
        bool: True, jeśli ticker spełnia warunki trendu wzrostowego, False w przeciwnym razie.
    '''
    try:
        data_d1 = fetch_historical_data(ticker)
        data_w1 = convert_to_weekly(data_d1.copy())
    except Exception:
        return False  # Zwróć False, jeśli nie można pobrać danych

    # Obliczenia
    sma200_d1 = calculate_sma(data_d1, 200)
    sma50_w1 = calculate_sma(data_w1, 50)

    if sma200_d1 is None or sma50_w1 is None:
        return False

    current_close_d1 = float(data_d1['Close'].iloc[-1])
    current_close_w1 = float(data_w1['Close'].iloc[-1])

    # Sprawdzenie warunków
    d1_ok = current_close_d1 > sma200_d1
    w1_ok = current_close_w1 > sma50_w1

    return d1_ok, w1_ok

def get_precise_point(approx_date, search_type, df_d1):
    '''Doprecyzowuje datę i cenę szczytu/dołka na podstawie danych dziennych.

    Args:
        approx_date (pd.Timestamp): Przybliżona data szczytu/dołka z danych tygodniowych.
        search_type (str): Typ punktu do wyszukania ('Peak' dla szczytu, 'Trough' dla dołka).
        df_d1 (pd.DataFrame): DataFrame z danymi dziennymi, z którego pobierane są precyzyjne wartości.

    Returns:
        tuple: Para (precyzyjna data, precyzyjna cena) lub (approx_date, None) jeśli brak danych.
    '''
    start_search = approx_date - pd.Timedelta(days=7)
    end_search = approx_date + pd.Timedelta(days=7)
    mask = (df_d1.index >= start_search) & (df_d1.index <= end_search)
    daily_slice = df_d1.loc[mask]

    if daily_slice.empty:
        return approx_date, None

    if search_type == 'Peak':
        precise_date = daily_slice['High'].idxmax()
        precise_price = round(float(daily_slice.loc[precise_date, 'High']), 2)
    else:
        precise_date = daily_slice['Low'].idxmin()
        precise_price = round(float(daily_slice.loc[precise_date, 'Low']), 2)
    return precise_date, precise_price

def identify_pivots(data_w1: pd.DataFrame, data_d1: pd.DataFrame, distance: int = 2, prominence_pct: float = 0.03) -> pd.DataFrame:
    '''
    Identyfikuje szczyty i dołki na W1 i doprecyzowuje ich daty/ceny na podstawie D1.
    Główny szczyt jest zawsze wyznaczany jako absolutne maksimum z danych dziennych.

    Args:
        data_w1 (pd.DataFrame): DataFrame zawierający dane tygodniowe.
        data_d1 (pd.DataFrame): DataFrame zawierający dane dzienne.
        distance (int): Minimalna odległość (w liczbie świec) między dwoma kolejnymi szczytami/dołkami.
        prominence_pct (float): Minimalna 'wybitność' szczytu/dołka jako procentowa zmiana ceny.

    Returns:
        pd.DataFrame: DataFrame zawierający zidentyfikowane punkty zwrotne z kolumnami 'Date', 'Price', 'Type', 'Label'.
    '''
    df_w1 = data_w1.copy()
    df_d1 = data_d1.copy()

    # Ograniczamy analizę do ostatnich 2 lat, aby skupić się na aktualnym cyklu
    two_years_ago = df_d1.index.max() - pd.DateOffset(years=2)
    df_d1 = df_d1[df_d1.index >= two_years_ago]
    df_w1 = df_w1[df_w1.index >= two_years_ago]

    if df_d1.empty or df_w1.empty:
        return pd.DataFrame()

    # Używamy logarytmu naturalnego do uzyskania stałej procentowej czułości
    log_high = np.log(df_w1['High'])
    log_low = np.log(df_w1['Low'])

    # 1. Znajdź punkty na W1 (tylko dołki nas tu realnie interesują jako baza)
    # prominence_pct np. 0.05 odpowiada ok. 5% ruchowi po zlogarytmowaniu
    peaks_idx, _ = find_peaks(log_high, distance=distance, prominence=prominence_pct)
    troughs_idx, _ = find_peaks(-log_low, distance=distance, prominence=prominence_pct)

    pivots = []

    # Mapowanie wszystkich znalezionych punktów
    for idx in peaks_idx:
        d, p = get_precise_point(df_w1.index[idx], 'Peak', df_d1)
        if p: pivots.append({'Date': d, 'Price': p, 'Type': 'Peak'})

    for idx in troughs_idx:
        d, p = get_precise_point(df_w1.index[idx], 'Trough', df_d1)
        if p: pivots.append({'Date': d, 'Price': p, 'Type': 'Trough'})

    # 2. WYMUSZENIE ABSOLUTNEGO SZCZYTU (Zielona strzałka)
    # To jest nasz nadrzędny punkt odniesienia.
    abs_max_date = df_d1['High'].idxmax()
    abs_max_price = round(float(df_d1['High'].max()), 2)

    # Dodajemy go jako Peak (jeśli już jest o tej samej dacie, drop_duplicates go wyczyści)
    pivots.append({'Date': abs_max_date, 'Price': abs_max_price, 'Type': 'Peak'})

    pivots_df = pd.DataFrame(pivots).sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)

    # Etykietowanie trendu
    lp, lt = None, None
    for i, row in pivots_df.iterrows():
        if row['Type'] == 'Peak':
            pivots_df.at[i, 'Label'] = ('HH' if row['Price'] > lp else 'LH') if lp else 'High'
            lp = row['Price']
        else:
            pivots_df.at[i, 'Label'] = ('HL' if row['Price'] > lt else 'LL') if lt else 'Low'
            lt = row['Price']

    return pivots_df

def get_fibo_targets(pivots_df: pd.DataFrame, min_dist_pct: float = 0.10, min_time_days: int = 60) -> dict:
    '''
    Cofa się od ABSOLUTNIE NAJWYŻSZEGO szczytu i szuka coraz niższych dołków.
    Ignoruje dołki zbyt bliskie cenowo (min_dist_pct) lub czasowo (min_time_days).

    Args:
        pivots_df (pd.DataFrame): DataFrame zawierający zidentyfikowane punkty zwrotne (szczyty i dołki).
        min_dist_pct (float): Minimalna procentowa różnica ceny między kolejnymi dołkami Fibonacciego.
        min_time_days (int): Minimalna różnica w dniach między kolejnymi dołkami Fibonacciego.

    Returns:
        dict: Słownik zawierający 'peak' (pd.Series) dla absolutnego szczytu
              oraz 'troughs' (pd.DataFrame) dla wybranych dołków Fibonacciego.
    '''
    if pivots_df.empty:
        return {'peak': None, 'troughs': pd.DataFrame()}

    # 1. Znajdź szczyt o najwyższej cenie (Peak z max Price)
    all_peaks = pivots_df[pivots_df['Type'] == 'Peak']
    if all_peaks.empty:
        return {'peak': None, 'troughs': pd.DataFrame()}

    # Wybieramy ten, który ma najwyższą cenę w całym zbiorze
    absolute_peak = all_peaks.loc[all_peaks['Price'].idxmax()]

    # 2. Szukamy dołków tylko przed datą tego szczytu, idąc wstecz, z limitem 3 lat wstecz
    three_years_ago = absolute_peak['Date'] - pd.DateOffset(years=3)
    points_before = pivots_df[(pivots_df['Date'] < absolute_peak['Date']) &
                               (pivots_df['Date'] >= three_years_ago)].sort_values('Date', ascending=False)

    selected_troughs = []
    current_min_price = float('inf')

    for _, row in points_before.iterrows():
        if len(selected_troughs) >= 4:
            break

        if row['Type'] == 'Trough':
            if row['Price'] < current_min_price:
                is_duplicate = False
                if selected_troughs:
                    last_added = selected_troughs[-1]
                    price_dist = abs(row['Price'] - last_added['Price']) / last_added['Price']
                    time_dist_days = (last_added['Date'] - row['Date']).days

                    if price_dist < min_dist_pct or time_dist_days < min_time_days:
                        is_duplicate = True
                        if row['Price'] < last_added['Price']:
                            selected_troughs[-1] = row
                            current_min_price = row['Price']

                if not is_duplicate:
                    selected_troughs.append(row)
                    current_min_price = row['Price']

    return {
        'peak': absolute_peak,
        'troughs': pd.DataFrame(selected_troughs).sort_values('Date')
    }