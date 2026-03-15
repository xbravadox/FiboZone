import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from curl_cffi import requests as cffi_requests


_session = cffi_requests.Session(impersonate="chrome")


class TickerNotFoundError(Exception):
    pass


class NoDataError(Exception):
    pass


class TimeoutError(Exception):
    pass


def fetch_historical_data(ticker: str) -> pd.DataFrame:
    '''Pobiera dane historyczne dla podanego tickera z Yahoo Finance.

    Args:
        ticker (str): Symbol tickera, np. 'PKO.WA'.

    Returns:
        pd.DataFrame: DataFrame zawierający dane historyczne (Close, High, Low, Open, Volume).

    Raises:
        TimeoutError: Jeśli wystąpi błąd połączenia lub przekroczenie czasu oczekiwania.
        NoDataError: Jeśli brak danych dla podanego tickera.
        TickerNotFoundError: Jeśli ticker nie istnieje lub brakuje kolumn danych.
    '''
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20 * 365)

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='1d',
            progress=False,
            auto_adjust=False,
            timeout=10,
            session=_session
        )
    except Exception as e:
        raise TimeoutError(f'Timeout lub błąd połączenia dla {ticker}: {e}')

    if data.empty:
        raise NoDataError(f'Brak danych dla tickera: {ticker}')

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    expected_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    missing = [c for c in expected_cols if c not in data.columns]
    if missing:
        raise TickerNotFoundError(
            f'Ticker "{ticker}" nie istnieje lub brak kolumn: {missing}')

    data = data[expected_cols]
    data.columns.name = None

    return data


def convert_to_weekly(data: pd.DataFrame) -> pd.DataFrame:
    '''Konwertuje dane D1 na interwał tygodniowy (W1).

    Args:
        data (pd.DataFrame): DataFrame zawierający dane dzienne z kolumnami 'Open', 'High', 'Low', 'Close', 'Volume' i indeksem 'Date'.

    Returns:
        pd.DataFrame: DataFrame zawierający dane tygodniowe z kolumnami 'Open', 'High', 'Low', 'Close', 'Volume' i indeksem 'Date'.
    '''
    data = data.copy()
    data.reset_index(inplace=True)
    data.columns = [col[1] if isinstance(
        col, tuple) else col for col in data.columns]
    data.set_index('Date', inplace=True)

    weekly_data = data.resample('W').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    return weekly_data
