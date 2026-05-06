# 🏥 Healthcare Fraud Detection – Machine Learning Project

## O projekcie
Projekt dotyczy **wykrywania nadużyć (oszustw)** w roszczeniach medycznych przy użyciu algorytmów uczenia maszynowego. Problem został zdefiniowany jako **klasyfikacja binarna** na silnie niezbalansowanym zbiorze danych (ok. **8.3%** przypadków pozytywnych).

Głównym celem było stworzenie modelu o wysokiej **czułości (Recall)**, który minimalizuje straty finansowe ubezpieczyciela.

## Wyniki i Modele
W projekcie porównano dwa podejścia:
* **Random Forest (Baseline):** Służył jako punkt odniesienia, wykazał tendencję do przeuczenia i generował dużo fałszywych alarmów (**Precision: 0.52**).
* **XGBoost (Final Model):** Dzięki optymalizacji wag klas (**scale_pos_weight=11**) oraz dostrojeniu progu decyzyjnego do poziomu **0.3**, model osiągnął:
    * **Recall (Czułość):** 0.97
    * **Precision (Precyzja):** 0.82
    * **AUPRC:** 0.99

##Struktura plików
* **`raport.pdf`** – Pełna dokumentacja projektowa z analizą biznesową i techniczną.
* **`healthcare_fraud_detection.csv`** – Zbiór danych wykorzystany w badaniu.
* **`main.py`** – Skrypt trenujący modele i generujący wykresy.
* **`preprocessing.py`** – Funkcje do czyszczenia i transformacji danych.

## Kluczowe Wnioski
* **Najważniejsze cechy:** Największy wpływ na wykrycie oszustwa miały cechy **Claim_Amount** (kwota roszczenia) oraz **Days_Between_Service_and_Claim** (czas zgłoszenia).
* **Optymalizacja progu:** Obniżenie progu decyzyjnego pozwoliło na niemal całkowite wyeliminowanie pominiętych oszustw (tylko **5 przypadków False Negative** w zbiorze testowym).
* **Wyjaśnialność:** Zastosowanie metod **XAI (SHAP)** pozwala analitykom zrozumieć powody oflagowania konkretnego roszczenia.
