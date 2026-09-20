import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from typesafe_sdk import TypeSafeClient

from data_preprocessing import load_and_preprocess_data
from jev_client import klasifikuj_pacijenta

load_dotenv(dotenv_path="../.env")


def _ucitaj_sirovi_test_skup():
    """Isto ciscenje i split kao data_preprocessing.py, ali BEZ one-hot
    enkodiranja i BEZ skaliranja - da Jev dobije citljive klinicke vrednosti.
    Isti random_state/test_size/stratify garantuju iste redove kao
    load_and_preprocess_data()."""
    df = pd.read_csv('../data/heart.csv')
    df.drop(df[df["RestingBP"] == 0].index, inplace=True)
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].mean())

    x = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    _, x_test_sirovo, _, y_test_sirovo = train_test_split(
        x, y, test_size=0.1, random_state=42, stratify=y
    )
    return x_test_sirovo.reset_index(drop=True), y_test_sirovo.reset_index(drop=True)


def _oceni_sklearn_model(x_test_skalirano):
    putanja_do_modela = os.path.join('models', 'best_model.pkl')
    model = joblib.load(putanja_do_modela)

    predikcije = []
    latencije_ms = []
    for red in x_test_skalirano:
        pocetak = time.perf_counter()
        predikcija = model.predict(red.reshape(1, -1))[0]
        latencije_ms.append((time.perf_counter() - pocetak) * 1000)
        predikcije.append(predikcija)

    return np.array(predikcije), float(np.mean(latencije_ms))


def _oceni_jev(x_test_sirovo, jev_model):
    predikcije = []
    latencije_ms = []

    with TypeSafeClient(model=jev_model) as client:
        for _, red in x_test_sirovo.iterrows():
            rezultat = klasifikuj_pacijenta(client, red.to_dict())
            predikcije.append(rezultat.predikcija)
            latencije_ms.append(rezultat.latencija_ms)

    return np.array(predikcije), float(np.mean(latencije_ms))


def _izracunaj_metrike(y_test, y_pred):
    return {
        "precision": metrics.precision_score(y_test, y_pred),
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "f1": metrics.f1_score(y_test, y_pred),
    }


def _sacuvaj_izvestaj(n, jev_model, metrike_sklearn, latencija_sklearn, metrike_jev, latencija_jev):
    os.makedirs('results', exist_ok=True)
    putanja = os.path.join('results', 'jev_comparison.txt')

    vreme = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = (
        f"\n==================================================\n"
        f"Vreme: {vreme}\n"
        f"Test skup: {n} pacijenata (heart.csv, random_state=42, test_size=0.1)\n"
        f"Jev model: {jev_model}\n"
        f"Metodologija: Jev dobija samo sirove klinicke podatke jednog\n"
        f"pacijenta po pozivu (zero-shot, bez uvida u train skup, bez\n"
        f"few-shot primera i bez fine-tuninga).\n"
        f"==================================================\n"
    )

    red_format = "{:<20} | {:>9} | {:>8} | {:>6} | {:>6} | {:>18}\n"
    tabela = red_format.format("Model", "Precision", "Accuracy", "Recall", "F1", "Latencija (ms)")
    tabela += "-" * (len(tabela) - 1) + "\n"
    tabela += red_format.format(
        "LogisticRegression",
        f"{metrike_sklearn['precision']:.2f}",
        f"{metrike_sklearn['accuracy']:.2f}",
        f"{metrike_sklearn['recall']:.2f}",
        f"{metrike_sklearn['f1']:.2f}",
        f"{latencija_sklearn:.2f}",
    )
    tabela += red_format.format(
        jev_model,
        f"{metrike_jev['precision']:.2f}",
        f"{metrike_jev['accuracy']:.2f}",
        f"{metrike_jev['recall']:.2f}",
        f"{metrike_jev['f1']:.2f}",
        f"{latencija_jev:.2f}",
    )

    with open(putanja, 'a', encoding='utf-8') as f:
        f.write(header)
        f.write(tabela)

    print(header, end="")
    print(tabela, end="")


def uporedi_modele(jev_model: str = None):
    jev_model = jev_model or os.getenv("JEV_MODEL", "jev-latest")

    _, x_test_skalirano, _, y_test = load_and_preprocess_data()
    x_test_sirovo, y_test_sirovo = _ucitaj_sirovi_test_skup()

    assert y_test.tolist() == y_test_sirovo.tolist(), (
        "Test skupovi (skalirani i sirovi) nisu poravnati - proveriti da li su "
        "parametri split-a u ovoj funkciji isti kao u data_preprocessing.py"
    )

    y_pred_sklearn, latencija_sklearn = _oceni_sklearn_model(x_test_skalirano)
    y_pred_jev, latencija_jev = _oceni_jev(x_test_sirovo, jev_model)

    metrike_sklearn = _izracunaj_metrike(y_test, y_pred_sklearn)
    metrike_jev = _izracunaj_metrike(y_test_sirovo, y_pred_jev)

    _sacuvaj_izvestaj(
        len(y_test), jev_model, metrike_sklearn, latencija_sklearn, metrike_jev, latencija_jev
    )


if __name__ == "__main__":
    uporedi_modele()
