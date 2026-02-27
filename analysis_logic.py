import pandas as pd

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
