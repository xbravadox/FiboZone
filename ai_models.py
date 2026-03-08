import openai
import streamlit as st
import re # Dodany import dla regex, potrzebny do parsowania wyjścia modelu

openai_api_key = st.secrets.get("OPENAI_API_KEY")

client = None
client_initialized = False
if openai_api_key:
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        test_model = "gpt-3.5-turbo"
        client.models.list() # Sprawdza autentykację
        client_initialized = True
    except openai.AuthenticationError:
        st.error("Błąd uwierzytelnienia OpenAI: Klucz API jest nieprawidłowy lub wygasł.")
    except openai.RateLimitError:
        st.error("Limit zapytań OpenAI API został przekroczony. Spróbuj ponownie później.")
    except openai.APIConnectionError as e:
        st.error(f"Błąd połączenia z OpenAI API: {e}")
    except Exception as e:
        st.error(f"Wystąpił nieoczekiwany błąd podczas inicjalizacji OpenAI: {e}")
else:
    st.warning("Klucz API OpenAI nie został znaleziony w Streamlit Secrets. Funkcje AI mogą nie działać.")

def test_openai_connection():
    '''
    Funkcja testująca połączenie z OpenAI API.
    '''
    if client_initialized:
        return "Połączenie z OpenAI API jest aktywne."
    else:
        return "Połączenie z OpenAI API nie jest aktywne. Sprawdź klucz API i konfigurację."

def analyze_fundamental_with_gpt4o(ticker: str) -> str:
    '''
    Przeprowadza analizę fundamentalną spółki przy użyciu GPT-4o.
    Zadaje proste pytanie o analizę fundamentalną dla podanego tickera.

    Args:
        ticker (str): Symbol tickera spółki.

    Returns:
        str: Odpowiedź LLM dotycząca analizy fundamentalnej.
    '''
    if not client:
        return "Klucz API OpenAI nie jest skonfigurowany. Nie można wykonać analizy."

    try:
        prompt = f"Podaj krótką analizę fundamentalną dla spółki o tickerze {ticker}."

        response = client.chat.completions.create(
            model="gpt-4o-mini", # Używamy GPT-4o zgodnie z projektem
            messages=[
                {"role": "system", "content": "Jesteś pomocnym asystentem udzielającym analiz finansowych."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )

        # Zwracamy tylko tekst odpowiedzi
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Wystąpił błąd podczas zapytania do OpenAI: {e}"

def analyze_technical_with_gpt4o_mini(pivots_data: dict, trend_data: dict) -> str:
    '''
    Generuje ustrukturyzowaną analizę techniczną jako tekst przy użyciu GPT-4o-mini.

    Args:
        pivots_data (dict): Dane dotyczące punktów zwrotnych (pivotów).
        trend_data (dict): Dane dotyczące trendu.

    Returns:
        str: Ustrukturyzowana analiza techniczna jako tekst.
    '''
    if not client:
        return "Klucz API OpenAI nie jest skonfigurowany. Nie można wykonać analizy."

    try:
        import json
        trend_data_str = json.dumps(trend_data)
        pivots_data_str = json.dumps(pivots_data)

        # Nowy prompt podkreślający format tekstowy z dokładnymi nagłówkami
        prompt_template = '''
Przeanalizuj podane dane techniczne akcji giełdowych. Zwróć analizę w następującym, ścisłym formacie tekstowym:

Analiza Techniczna:
[Tutaj szczegółowa analiza]

Rekomendacja:
[Tutaj rekomendacja]

Informacje o użyciu AI:
Model: [nazwa modelu]
Tokeny promptu: [liczba]
Tokeny odpowiedzi: [liczba]
Łącznie tokenów: [liczba]
*(Koszty mogą się różnić w zależności od cennika OpenAI)*

Wykorzystaj poniższe dane do analizy:
1. Potwierdzenie trendu wzrostowego (dane: {trend_data_str}).
2. Szczegóły najbliższej strefy konfluencji Fibonacciego (dane: {pivots_data_str}), w tym ostatnie Low świecy D1, granice strefy konfluencji, środek strefy i górną granicę sygnału.

Jeśli brakuje danych dla którejś sekcji (np. rekomendacji lub szczegółów analizy), użyj komunikatu "Brak danych.", ale zachowaj ogólną strukturę.
'''

        formatted_prompt = prompt_template.format(
            trend_data_str=trend_data_str,
            pivots_data_str=pivots_data_str
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini", # Używamy GPT-4o-mini zgodnie z projektem
            messages=[
                # Zmieniono rolę systemową na bardziej ogólną, aby pasowała do formatu tekstowego
                {"role": "system", "content": "Jesteś asystentem analizy technicznej. Zwracaj wyniki w określonym formacie tekstowym."},
                {"role": "user", "content": formatted_prompt}
            ],
            max_tokens=400 # Zwiększono max_tokens dla potencjalnie bardziej złożonej odpowiedzi
            # Usunięto response_format, ponieważ nie potrzebujemy już JSON
        )

        # Zwracamy cały surowy tekst odpowiedzi. Parsowanie sekcji (analiza, rekomendacja, usage_info)
        # odbędzie się w pliku app.py
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Handle potential errors during API call
        return f"Wystąpił błąd podczas zapytania do OpenAI API dla analizy technicznej: {e}"
