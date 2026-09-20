import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from typesafe_sdk import TypeSafeClient

from compare_jev import _izracunaj_metrike, _oceni_sklearn_model
from data_preprocessing import load_and_preprocess_data
from jev_client import klasifikuj_sa_primerima

load_dotenv(dotenv_path="../.env")

BROJ_PRIMERA_PO_KLASI = 15
FEWSHOT_SEED = 7


def _ucitaj_sirove_podatke():
    """Isto ciscenje i split kao data_preprocessing.py, ali BEZ one-hot
    enkodiranja i BEZ skaliranja, i vraca i train i test skup (za few-shot
    primere je potreban train skup, za koji sklearn model nema potrebe)."""
    df = pd.read_csv('../data/heart.csv')
    df.drop(df[df["RestingBP"] == 0].index, inplace=True)
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].mean())

    x = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    x_train_sirovo, x_test_sirovo, y_train_sirovo, y_test_sirovo = train_test_split(
        x, y, test_size=0.1, random_state=42, stratify=y
    )
    return (
        x_train_sirovo.reset_index(drop=True), y_train_sirovo.reset_index(drop=True),
        x_test_sirovo.reset_index(drop=True), y_test_sirovo.reset_index(drop=True),
    )


def _izaberi_fewshot_primere(x_train, y_train, po_klasi=BROJ_PRIMERA_PO_KLASI, seed=FEWSHOT_SEED):
    """Stratifikovan uzorak iz train skupa - podjednako iz obe klase."""
    primeri = []
    for klasa in (0, 1):
        indeksi = y_train[y_train == klasa].index
        izabrani = pd.Series(indeksi).sample(n=po_klasi, random_state=seed)
        for i in izabrani:
            primeri.append((x_train.loc[i].to_dict(), int(y_train.loc[i])))
    random.Random(seed).shuffle(primeri)
    return primeri


def _oceni_jev_fewshot(x_test_sirovo, primeri, jev_model):
    predikcije = []
    latencije_ms = []

    with TypeSafeClient(model=jev_model) as client:
        for _, red in x_test_sirovo.iterrows():
            rezultat = klasifikuj_sa_primerima(client, red.to_dict(), primeri)
            predikcije.append(rezultat.predikcija)
            latencije_ms.append(rezultat.latencija_ms)

    return np.array(predikcije), float(np.mean(latencije_ms))


def _sacuvaj_izvestaj(n, broj_primera, jev_model, metrike_sklearn, latencija_sklearn, metrike_jev, latencija_jev):
    os.makedirs('results', exist_ok=True)
    putanja = os.path.join('results', 'jev_fewshot_comparison.txt')

    vreme = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = (
        f"\n==================================================\n"
        f"Vreme: {vreme}\n"
        f"Test skup: {n} pacijenata (heart.csv, random_state=42, test_size=0.1)\n"
        f"Jev model: {jev_model}\n"
        f"Metodologija: Jev dobija {broj_primera} few-shot primera iz train\n"
        f"skupa (stratifikovano, {broj_primera // 2} po klasi, seed={FEWSHOT_SEED})\n"
        f"uz svaki poziv, pre nego sto klasifikuje jednog test pacijenta.\n"
        f"==================================================\n"
    )

    red_format = "{:<24} | {:>9} | {:>8} | {:>6} | {:>6} | {:>18}\n"
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
        f"{jev_model} (few-shot)",
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


def uporedi_modele_fewshot(jev_model: str = None, broj_primera_po_klasi: int = BROJ_PRIMERA_PO_KLASI):
    jev_model = jev_model or os.getenv("JEV_MODEL", "jev-latest")

    _, x_test_skalirano, _, y_test = load_and_preprocess_data()
    x_train_sirovo, y_train_sirovo, x_test_sirovo, y_test_sirovo = _ucitaj_sirove_podatke()

    assert y_test.tolist() == y_test_sirovo.tolist(), (
        "Test skupovi (skalirani i sirovi) nisu poravnati - proveriti da li su "
        "parametri split-a u ovoj funkciji isti kao u data_preprocessing.py"
    )

    primeri = _izaberi_fewshot_primere(x_train_sirovo, y_train_sirovo, po_klasi=broj_primera_po_klasi)

    y_pred_sklearn, latencija_sklearn = _oceni_sklearn_model(x_test_skalirano)
    y_pred_jev, latencija_jev = _oceni_jev_fewshot(x_test_sirovo, primeri, jev_model)

    metrike_sklearn = _izracunaj_metrike(y_test, y_pred_sklearn)
    metrike_jev = _izracunaj_metrike(y_test_sirovo, y_pred_jev)

    _sacuvaj_izvestaj(
        len(y_test), len(primeri), jev_model, metrike_sklearn, latencija_sklearn, metrike_jev, latencija_jev
    )


if __name__ == "__main__":
    uporedi_modele_fewshot()
