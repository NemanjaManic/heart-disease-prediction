import time
from dataclasses import dataclass

from typesafe_sdk import Noul, NoulCriteria

UPUTSTVO = "Based on the given clinical measurements, does this patient have heart disease?"

KRITERIJUMI = NoulCriteria(
    true=(
        "Klinicke vrednosti ukazuju na povisen rizik od srcane bolesti "
        "(npr. ST depresija/Oldpeak > 0, ST_Slope Flat ili Down, "
        "ExerciseAngina Y, atipican/asimptomatski bol u grudima, "
        "nizak MaxHR za dato doba)."
    ),
    false="Klinicke vrednosti su u skladu sa zdravim pacijentom.",
)


@dataclass
class JevRezultat:
    predikcija: int
    verovatnoca: float
    latencija_ms: float


def _formatiraj_vrednost(vrednost):
    if isinstance(vrednost, float):
        return round(vrednost, 2)
    return vrednost


def _formatiraj_pacijenta(obelezja: dict) -> str:
    redovi = [f"{naziv}: {_formatiraj_vrednost(vrednost)}" for naziv, vrednost in obelezja.items()]
    return "Klinicki podaci pacijenta:\n" + "\n".join(redovi)


def klasifikuj_pacijenta(client, obelezja: dict) -> JevRezultat:
    """Salje SAMO ovog jednog pacijenta Jev-u (zero-shot, bez train skupa)."""
    stanje = _formatiraj_pacijenta(obelezja)

    pocetak = time.perf_counter()
    odgovor = client.system_one(
        state=stanje,
        questions={
            "srcana_bolest": Noul(instructions=UPUTSTVO, criteria=KRITERIJUMI),
        },
    )
    latencija_ms = (time.perf_counter() - pocetak) * 1000

    verovatnoca = odgovor.nouls["srcana_bolest"].noul
    predikcija = int(verovatnoca >= 0.5)

    return JevRezultat(predikcija=predikcija, verovatnoca=verovatnoca, latencija_ms=latencija_ms)


def _formatiraj_primer(obelezja: dict, dijagnoza: int) -> str:
    opis = ", ".join(f"{naziv}={_formatiraj_vrednost(vrednost)}" for naziv, vrednost in obelezja.items())
    return f"{opis} -> HeartDisease: {dijagnoza}"


def klasifikuj_sa_primerima(client, obelezja: dict, primeri: list) -> JevRezultat:
    """Isto kao klasifikuj_pacijenta, ali state sadrzi i few-shot primere iz
    train skupa (lista (obelezja_dict, dijagnoza) parova) pre pacijenta koji
    se klasifikuje."""
    linije_primera = [
        f"{i + 1}. {_formatiraj_primer(o, d)}" for i, (o, d) in enumerate(primeri)
    ]
    stanje = (
        "Primeri iz istorije (dijagnoza je vec poznata, samo za referencu):\n"
        + "\n".join(linije_primera)
        + "\n\nPacijent za klasifikaciju (dijagnoza NIJE poznata):\n"
        + "\n".join(f"{naziv}: {_formatiraj_vrednost(vrednost)}" for naziv, vrednost in obelezja.items())
    )

    pocetak = time.perf_counter()
    odgovor = client.system_one(
        state=stanje,
        questions={
            "srcana_bolest": Noul(instructions=UPUTSTVO, criteria=KRITERIJUMI),
        },
    )
    latencija_ms = (time.perf_counter() - pocetak) * 1000

    verovatnoca = odgovor.nouls["srcana_bolest"].noul
    predikcija = int(verovatnoca >= 0.5)

    return JevRezultat(predikcija=predikcija, verovatnoca=verovatnoca, latencija_ms=latencija_ms)


if __name__ == "__main__":
    # Rucni smoke-test: python jev_client.py (pokrenuti iz src/, sa .env popunjenim)
    from dotenv import load_dotenv
    from typesafe_sdk import TypeSafeClient
    import os

    load_dotenv(dotenv_path="../.env")

    primeri = [
        {
            "Age": 63, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
            "Cholesterol": 233, "FastingBS": 1, "RestingECG": "Normal",
            "MaxHR": 150, "ExerciseAngina": "N", "Oldpeak": 2.3, "ST_Slope": "Down",
        },
        {
            "Age": 37, "Sex": "F", "ChestPainType": "NAP", "RestingBP": 130,
            "Cholesterol": 211, "FastingBS": 0, "RestingECG": "Normal",
            "MaxHR": 142, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up",
        },
    ]

    with TypeSafeClient(model=os.getenv("JEV_MODEL", "jev-latest")) as client:
        for pacijent in primeri:
            rezultat = klasifikuj_pacijenta(client, pacijent)
            print(pacijent, "->", rezultat)
