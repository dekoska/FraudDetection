# Wykrywanie nadużyć w służbie zdrowia – projekt ML


## O projekcie

Projekt dotyczy **wykrywania oszustw** w roszczeniach medycznych przy użyciu uczenia maszynowego. Problem został zdefiniowany jako **klasyfikacja binarna** na silnie niezbalansowanym zbiorze danych (ok. **8,3%** przypadków pozytywnych).

Głównym celem było stworzenie narzędzia wspomagającego analityków ds. nadużyć, z priorytetem na **czułość (Recall)** — minimalizację niewykrytych oszustw — przy zachowaniu akceptowalnej **precyzji**, aby nie przeciążyć zespołu fałszywymi alarmami.

Źródło danych: [Healthcare Fraud Detection Dataset](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset) (Kaggle).

## Metodologia

- **Algorytm:** XGBoost z optymalizacją hiperparametrów (Optuna, 50 prób, maksymalizacja F1 na zbiorze walidacyjnym).
- **Podział czasowy (time-based split):** trening 2021–2022, walidacja 2023, test 2024 — bez losowego podziału, aby uniknąć wycieku informacji i lepiej odzwierciedlić wdrożenie produkcyjne.
- **Preprocessing:** imputacja medianą (cechy numeryczne), kategoria `Unknown` (braki tekstowe), usunięcie cech powodujących **data leakage** (`Claim_Status`, `Approved_Amount`, `Number_of_Claims_Per_Provider_Monthly`), frequency encoding (wysoka kardynalność) oraz one-hot encoding (niska kardynalność).
- **Próg decyzyjny:** **0,3** (dobrany na zbiorze walidacyjnym) — kompromis między kosztem niewykrytego oszustwa a kosztem fałszywej weryfikacji.

## Wyniki na zbiorze testowym (2024)

### XGBoost (model końcowy)

| Metryka | Wartość |
|---------|---------|
| Precision (klasa fraud) | **0,60** |
| Recall (klasa fraud) | **0,62** |
| F1-score | **0,61** |
| Accuracy | **94%** |
| Average Precision (AUPRC) | **0,72** |

**Macierz pomyłek:** 2160 poprawnych negatywów, 121 wykrytych fraudów, 82 fałszywe alarmy (FP), **73 niewykryte oszustwa (FN)**.

### Random Forest (baseline)

Na tym samym pipeline i progu 0,3:

| Metryka | Wartość |
|---------|---------|
| Recall | **1,00** |
| Precision | **0,35** |
| F1-score | **0,52** |

RF wykrywał praktycznie wszystkie oszustwa, ale generował **365 fałszywych alarmów** (vs 82 w XGBoost). XGBoost zapewnia lepszy kompromis precision/recall i bardziej zrównoważone wykorzystanie cech.

> Po optymalizacji Optuna parametr `scale_pos_weight` wyniósł ok. **1,0** — regularyzacja, inżynieria cech i obniżony próg okazały się wystarczające bez agresywnego ważenia klasy mniejszościowej.

## Najważniejsze cechy (feature importance)

1. **Claim_Amount** — kwota roszczenia  
2. **Patient_Age** — wiek pacjenta  
3. **Days_Between_Service_and_Claim** — opóźnienie zgłoszenia  
4. **Provider_ID_freq** — częstość dostawcy (frequency encoding)  
5. **Length_of_Stay** — długość hospitalizacji  

## Struktura plików

| Plik | Opis |
|------|------|
| `raport.pdf` | Pełna dokumentacja (analiza biznesowa i techniczna) |
| `healthcare_fraud_detection.csv` | Zbiór danych |
| `preprocessing.py` | Czyszczenie, kodowanie i podział czasowy danych |
| `xgb.py` | Trening XGBoost, Optuna, wykresy (macierz pomyłek, PRC, importance) |
| `random_forest.py` | Model bazowy Random Forest |

## Wnioski i ograniczenia

- Realistyczny **podział czasowy** i eliminacja wycieku danych dają **niższe, ale wiarygodniejsze** metryki niż przy losowym podziale — model jest oceniany na przyszłych roszczeniach (concept drift w 2024).
- Model dobrze wspiera decyzje antyfraudowe, ale ma ograniczenia: mała liczba fraudów w zbiorze, brak analizy grafowej relacji pacjent–świadczeniodawca, potrzeba okresowego retrenowania przy nowych schematach oszustw.
- Możliwy rozwój: analiza grafowa, **SHAP** (XAI), dynamiczny próg decyzyjny, uczenie przyrostowe.

Szczegóły, uzasadnienia metryk, hiperparametry i wykresy — w [`raport.pdf`](raport.pdf).
