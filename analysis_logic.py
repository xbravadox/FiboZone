import numpy as np
import pandas as pd
import streamlit as st 
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
    peaks_idx, _ = find_peaks(
        log_high, distance=distance, prominence=prominence_pct)
    troughs_idx, _ = find_peaks(-log_low,
                                distance=distance, prominence=prominence_pct)

    pivots = []

    # Mapowanie wszystkich znalezionych punktów
    for idx in peaks_idx:
        d, p = get_precise_point(df_w1.index[idx], 'Peak', df_d1)
        if p:
            pivots.append({'Date': d, 'Price': p, 'Type': 'Peak'})

    for idx in troughs_idx:
        d, p = get_precise_point(df_w1.index[idx], 'Trough', df_d1)
        if p:
            pivots.append({'Date': d, 'Price': p, 'Type': 'Trough'})

    # 2. WYMUSZENIE ABSOLUTNEGO SZCZYTU (Zielona strzałka)
    # To jest nasz nadrzędny punkt odniesienia.
    abs_max_date = df_d1['High'].idxmax()
    abs_max_price = round(float(df_d1['High'].max()), 2)

    # Dodajemy go jako Peak (jeśli już jest o tej samej dacie, drop_duplicates go wyczyści)
    pivots.append(
        {'Date': abs_max_date, 'Price': abs_max_price, 'Type': 'Peak'})

    pivots_df = pd.DataFrame(pivots).sort_values(
        'Date').drop_duplicates(subset=['Date']).reset_index(drop=True)

    # Etykietowanie trendu
    lp, lt = None, None
    for i, row in pivots_df.iterrows():
        if row['Type'] == 'Peak':
            pivots_df.at[i, 'Label'] = (
                'HH' if row['Price'] > lp else 'LH') if lp else 'High'
            lp = row['Price']
        else:
            pivots_df.at[i, 'Label'] = (
                'HL' if row['Price'] > lt else 'LL') if lt else 'Low'
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
    points_before = pivots_df[
        (pivots_df['Date'] < absolute_peak['Date']) &
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
                    price_dist = abs(
                        row['Price'] - last_added['Price']) / last_added['Price']
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


def calculate_fibonacci_levels(peak_price: float, trough_price: float) -> dict:
    '''
    Oblicza standardowe poziomy zniesienia Fibonacciego.

    Args:
        peak_price (float): Cena szczytu.
        trough_price (float): Cena dołka.

    Returns:
        dict: Słownik zawierający obliczone poziomy Fibonacciego.
    '''
    fib_levels = {
        '38.2%': 0.382,
        '50%': 0.50,
        '61.8%': 0.618,
        '78.6%': 0.786
    }

    calculated_levels = {}
    price_range = peak_price - trough_price

    for label, level in fib_levels.items():
        if peak_price > trough_price:  # Uptrend: retracement from peak towards trough
            fib_value = peak_price - (price_range * level)
        # Downtrend: retracement from trough towards peak (or extension below trough)
        else:
            # This calculates retracement from trough to peak
            fib_value = trough_price + (abs(price_range) * level)
        calculated_levels[label] = round(fib_value, 2)

    return calculated_levels


def find_fibonacci_confluences(peak_price: float, troughs_df: pd.DataFrame) -> list:
    '''
    Wyszukuje konfluencje Fibonacciego, gdzie co najmniej dwa poziomy Fibonacciego
    pochodzące z różnych dołków znajdują się w strefie 0-0.2% od siebie.
    Dodaje informację o sumie punktów dla każdej konfluencji.

    Args:
        peak_price (float): Cena szczytu, od którego obliczane są poziomy Fibonacciego.
        troughs_df (pd.DataFrame): DataFrame zawierający dołki, z których obliczane są poziomy Fibonacciego.
                                    Oczekiwane kolumny to 'Date' i 'Price'.

    Returns:
        list: Lista konfluencji. Każda konfluencja to słownik zawierający:
              - 'levels': Lista poziomów Fibonacciego tworzących konfluencję.
              - 'total_score': Suma punktów dla tej konfluencji.
    '''
    if troughs_df.empty or len(troughs_df) < 2:  # Potrzebujemy przynajmniej dwóch dołków do konfluencji
        return []

    all_fib_levels = []

    # Oblicz poziomy Fibonacciego dla każdego dołka
    for _, trough_row in troughs_df.iterrows():
        trough_date = trough_row['Date']
        trough_price = trough_row['Price']

        # Użyj istniejącej funkcji calculate_fibonacci_levels
        fib_levels_for_trough = calculate_fibonacci_levels(
            peak_price, trough_price)

        for label, value in fib_levels_for_trough.items():
            all_fib_levels.append({
                'level_value': value,
                'label': label,
                'trough_date': trough_date,
                'trough_price': trough_price,
                'peak_price': peak_price
            })

    confluences = []
    # Sortujemy wszystkie poziomy Fibonacciego po wartości, aby ułatwić wyszukiwanie bliskich punktów
    all_fib_levels.sort(key=lambda x: x['level_value'])

    # Wyszukaj konfluencje
    # Iterujemy po każdym poziomie Fibonacciego
    for i in range(len(all_fib_levels)):
        current_level = all_fib_levels[i]
        # Tworzymy grupę konfluencji dla bieżącego poziomu
        current_confluence_group = [current_level]

        # Porównujemy bieżący poziom z kolejnymi poziomami
        for j in range(i + 1, len(all_fib_levels)):
            next_level = all_fib_levels[j]

            # Upewnij się, że poziomy pochodzą z różnych dołków
            if current_level['trough_date'] == next_level['trough_date']:
                continue

            # Sprawdź warunek konfluencji: różnica do 0.2% względem niższego poziomu
            min_val = min(
                current_level['level_value'],
                next_level['level_value']
            )
            max_val = max(
                current_level['level_value'],
                next_level['level_value']
            )

            if min_val == 0:  # Unikaj dzielenia przez zero
                continue

            percentage_diff = (max_val - min_val) / min_val

            # Jeśli różnica mieści się w zakresie do 0.2% (i jest większa od 0, by uniknąć identycznych poziomów z różnych dołków)
            if 0 < percentage_diff <= 0.002:
                current_confluence_group.append(next_level)
            # Jeśli następny poziom jest zbyt daleko, aby tworzyć konfluencję z bieżącą grupą,
            # możemy przerwać wewnętrzną pętlę, ponieważ lista jest posortowana.
            elif percentage_diff > 0.002:  # Changed from 0.015 to 0.002
                break

        # Jeśli znaleziono konfluencję (co najmniej dwa poziomy z różnych dołków)
        # i upewnij się, że nie jest to ten sam poziom wielokrotnie dodany
        if len(current_confluence_group) >= 2:
            # Sprawdź, czy w grupie są poziomy z co najmniej dwóch różnych dołków
            unique_troughs = set(
                item['trough_date'] for item in current_confluence_group)
            if len(unique_troughs) >= 2:
                # Sprawdź, czy ta konfluencja nie została już dodana (aby unikać duplikatów)
                # Można to zrobić poprzez posortowanie poziomów wewnątrz konfluencji i przekształcenie na krotki
                sorted_group_items = [
                    frozenset(level.items()) for level in current_confluence_group]
                sorted_group_tuple = tuple(sorted(sorted_group_items))

                is_duplicate = False
                for existing_confluence in confluences:
                    existing_levels = existing_confluence.get('levels', [])
                    existing_group_items = [
                        frozenset(level.items()) for level in existing_levels]
                    existing_group_tuple = tuple(sorted(existing_group_items))
                    if sorted_group_tuple == existing_group_tuple:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    # Oblicz sumę punktów dla tej konfluencji
                    score_mapping = {
                        '61.8%': 3,
                        '78.6%': 3,
                        '50%': 2,
                        '38.2%': 2
                    }
                    total_score = 0
                    for level_data in current_confluence_group:
                        label = level_data.get('label')
                        if label in score_mapping:
                            total_score += score_mapping[label]

                    # Dodaj konfluencję z jej poziomami i obliczoną sumą punktów
                    confluences.append({
                        'levels': current_confluence_group,
                        'total_score': total_score
                    })

    return confluences


def check_last_d1_low_against_confluences(df_d1: pd.DataFrame, confluences: list) -> tuple:
    '''
    Sprawdza położenie Low ostatniej świecy D1 względem środka strefy konfluencji.

    Args:
        df_d1 (pd.DataFrame): DataFrame zawierający dane dzienne z kolumną 'Low'.
        confluences (list): Lista konfluencji Fibonacciego zwrócona przez `find_fibonacci_confluences`.

    Returns:
        tuple: (True, dict) jeśli Low ostatniej świecy D1 znajduje się w strefie sygnału konfluencji,
               w przeciwnym razie (False, None).
               Dict zawiera szczegóły znalezionego sygnału.
    '''
    # Usunięto wszystkie linie st.write z tej funkcji

    if df_d1.empty or 'Low' not in df_d1.columns:
        # st.error("Brak danych D1 lub brak kolumny 'Low'.") # Usunięto interakcję z UI
        return False, None

    # Sprawdzenie typu danych w kolumnie 'Low' przed konwersją
    if not pd.api.types.is_numeric_dtype(df_d1['Low']):
        # st.error(f"Kolumna 'Low' w df_d1 nie jest typu numerycznego. Typ: {df_d1['Low'].dtype}") # Usunięto interakcję z UI
        return False, None

    last_d1_low = float(df_d1['Low'].iloc[-1])

    for conf_group in confluences:
        # Sprawdzenie, czy conf_group jest słownikiem i zawiera klucz 'levels'
        if not isinstance(conf_group, dict) or 'levels' not in conf_group:
            # st.warning(f"Nieprawidłowy format grupy konfluencji: {conf_group}. Pomijanie.") # Usunięto interakcję z UI
            continue

        levels_in_group = conf_group.get('levels', [])
        if not levels_in_group:
            # st.warning("Grupa konfluencji bez poziomów. Pomijanie.") # Usunięto interakcję z UI
            continue

        # Sprawdzenie, czy wszystkie elementy w levels_in_group są słownikami i mają 'level_value'
        valid_levels = []
        for item in levels_in_group:
            if isinstance(item, dict) and 'level_value' in item:
                # Dodatkowe sprawdzenie typu 'level_value'
                if isinstance(item['level_value'], (int, float)):
                    valid_levels.append(item)
                else:
                    # st.warning(f"Nieprawidłowy typ danych dla 'level_value' w elemencie: {item}. Oczekiwano liczby, otrzymano {type(item['level_value'])}. Pomijanie.") # Usunięto interakcję z UI
                    pass # Pomijamy element z nieprawidłowym typem
            else:
                # st.warning(f"Nieprawidłowy element w 'levels' lub brak klucza 'level_value': {item}. Pomijanie.") # Usunięto interakcję z UI
                pass # Pomijamy element o nieprawidłowej strukturze

        if not valid_levels:
            # st.warning("Brak poprawnych poziomów w grupie konfluencji. Pomijanie.") # Usunięto interakcję z UI
            continue

        # Upewnij się, że obliczenia min/max są bezpieczne
        try:
            min_conf_level = min(item['level_value'] for item in valid_levels)
            max_conf_level = max(item['level_value'] for item in valid_levels)
        except ValueError:
            # st.error("Błąd podczas obliczania min/max poziomów konfluencji.") # Usunięto interakcję z UI
            continue # Przejdź do następnej grupy konfluencji

        confluence_center = (min_conf_level + max_conf_level) / 2
        upper_signal_limit = confluence_center * 1.05

        if confluence_center <= last_d1_low <= upper_signal_limit:
            signal_details = {
                'last_d1_low': last_d1_low,
                'confluence_min_level': round(min_conf_level, 2),
                'confluence_max_level': round(max_conf_level, 2),
                'confluence_center': round(confluence_center, 2),
                'upper_signal_limit': round(upper_signal_limit, 2),
                'confluence_details': valid_levels
            }
            return True, signal_details

    return False, None