# Plan: Poređenje najboljeg sklearn modela sa Jev modelom

## Cilj

Uporediti trenutno najbolji model iz ovog projekta (LogisticRegression,
`src/models/best_model.pkl`) sa **Jev**-om — TypeSafe AI-jevim "System One"
decision/classification modelom, dostupnim preko Vercel AI Gateway-a — na
istom zadatku (binarna klasifikacija rizika od srčane bolesti), i rezultat
objaviti u README-u.

Jev nije opšti LLM (ne generiše slobodan tekst) već brz model specijalizovan
za klasifikaciju/rutiranje/skorovanje sa tipizovanim izlazom, pa je prirodan
kandidat za poređenje sa klasičnim ML klasifikatorima na tabelarnim
podacima.

## Ključno metodološko pravilo (dogovoreno)

**Jev dobija samo test podatke, red po red, i sam odlučuje.** Nema few-shot
primera iz train skupa, nema fine-tuninga, nema ikakvog uvida u
`x_train`/`y_train`. Jev za svaki poziv dobija isključivo klinička obeležja
jednog pacijenta iz test skupa i mora sam da proceni rizik — isključivo na
osnovu svog opšteg znanja i tipizovane šeme odgovora koju mu damo.

Ovo je namerna asimetrija u odnosu na sklearn model (koji *jeste* treniran
na `x_train`/`y_train`) i treba da bude eksplicitno navedena u README-u kao
deo metodologije, da rezultat ne bude pogrešno protumačen kao "fer" u smislu
istog pristupa učenju — fer je u smislu da oba modela vide iste test
instance i ne vide odgovore unapred.

## Grana

`feature/jev-comparison`, ogranak od `master`-a. Na kraju se merge-uje nazad
u `master` (kod + rezultati + README sekcija). Ništa u postojećem pipeline-u
se ne menja:

- `src/data_preprocessing.py` — bez izmena
- `src/train.py` — bez izmena
- `src/save_best_model.py` — bez izmena
- `src/models_config.py` — bez izmena
- `src/main.py` (FastAPI servis) — bez izmena

## Novi fajlovi

Nema posebnog export skripta niti persistovanih test-skup fajlova. Sve se
računa u memoriji, pri svakom pokretanju `compare_jev.py`, uz ponovnu
upotrebu postojećeg koda gde god je moguće.

### `src/jev_client.py`
Tanak wrapper oko Vercel AI Gateway REST poziva za Jev:
- Čita API ključ iz environment varijable (učitane preko `.env` uz
  `python-dotenv`).
- Definiše tipizovanu šemu odgovora: `{"rizik": 0|1, "confidence": float}`.
- Jedna funkcija tipa `classify_patient(features: dict) -> JevResult`
  (label, confidence, latencija poziva u ms).
- Minimalan retry (npr. 1 pokušaj ponovo na transient grešku/timeout) —
  bez preterane robusnosti, ovo je eksperiment, ne produkcioni servis.

### `src/compare_jev.py`
Glavni skript za poređenje, pokreće se ručno (`python src/compare_jev.py`),
sve računa **u memoriji**, bez ijednog persistovanog međufajla:

1. **Importuje** postojeću `load_and_preprocess_data()` iz
   `data_preprocessing.py` (bez ijedne izmene tog fajla) da dobije
   skalirani `x_test, y_test` — ovo je isti test skup na kom je i
   `best_model.pkl` ocenjivan tokom treninga, ponovna upotreba postojećeg
   koda, nula duplikacije.
2. Malom lokalnom pomoćnom funkcijom (u istom fajlu), ponavlja identično
   čišćenje + `train_test_split(test_size=0.1, random_state=42,
   stratify=y)` kao u `data_preprocessing.py`, ali **pre** `StandardScaler`
   koraka, da dobije sirove, čitljive kliničke vrednosti istog test skupa.
   Pošto je `random_state` fiksiran, redovi i redosled se garantovano
   poklapaju sa skaliranom verzijom iz koraka 1.
3. Za sklearn model: koristi skalirani `x_test` iz koraka 1, pušta
   `src/models/best_model.pkl` da predikuje, meri latenciju lokalne
   inference po redu (avg ms).
4. Za Jev: nad sirovim redovima iz koraka 2, za svaki red pojedinačno
   (samo taj jedan uzorak — bez ikakvog konteksta iz train skupa) zove
   `jev_client.classify_patient(...)`, hvata predikciju i latenciju poziva
   (avg ms).
5. **Ista funkcija za metrike** (`compute_metrics(y_true, y_pred) ->
   {precision, accuracy, recall, f1}`) se poziva nad oba niza predikcija —
   garantuje da su metrike računate identično za oba modela, nema
   razlike u definiciji ili zaokruživanju.
6. Rezultat se prikazuje kao **jedna uporedna tabela** (ne dva odvojena
   izveštaja): red po red model, kolone precision/accuracy/recall/F1 +
   avg latencija — direktno pogodno za copy-paste u README.

### Rezultati
- `src/results/jev_comparison.txt` — izveštaj u istom stilu kao
  `train_results.txt` (dodaje se, ne prepisuje postojeće fajlove).
- Opciono: `src/results/jev_comparison.json` sa sirovim podacima
  (po-red predikcije + latencije) za dalju obradu/grafove.

## Secrets / konfiguracija

- Novi `.env` fajl u root-u (van git-a) sa Vercel AI Gateway API ključem.
- `.env.example` sa placeholder vrednošću, commit-uje se.
- Dodati `.env` u `.gitignore` (trenutno ga nema — samo `.venv/`,
  `__pycache__/`, `.idea/`, `.vscode/`, `*.pyc`).
- U `requirements.txt` dodati (append, ništa se ne uklanja/menja):
  `python-dotenv`, `requests`.

## Van obima (namerno)

- Bez async/batch poziva ka Jev-u — test skup je mali (~90 redova nakon
  10% split-a), sekvencijalno je dovoljno brzo i jednostavnije za čitanje.
- Bez automatizovanih unit testova za eksterni API poziv — samo ručni
  smoke-test sa 2-3 primera pre pune evaluacije (provera da parsiranje
  odgovora radi kako treba).
- Bez ikakvog fine-tuninga ili prompt-engineering optimizacije Jev-a preko
  više iteracija na test skupu (to bi bilo "curenje" test podataka u
  proces odlučivanja o promptu) — prompt/šema se fiksira pre pokretanja
  pune evaluacije.

## Koraci (checklist)

1. [ ] `git checkout -b feature/jev-comparison`
2. [ ] Dodati `.env` u `.gitignore`, napraviti `.env.example`
3. [ ] Dodati `python-dotenv`, `requests` u `requirements.txt`
4. [ ] Napisati `src/jev_client.py` (poziv + tipizovana šema + retry +
   merenje latencije)
5. [ ] Ručni smoke-test `jev_client.py` na 2-3 ručno uneta pacijenta,
   proveriti da parsiranje odgovora radi
6. [ ] Napisati `src/compare_jev.py` (import postojeće
   `load_and_preprocess_data()` za skalirani test skup + lokalna pomoćna
   funkcija za sirovi test skup, sklearn evaluacija, Jev evaluacija,
   zajednička `compute_metrics` funkcija, latencija)
7. [ ] Pokrenuti punu evaluaciju, generisati
   `src/results/jev_comparison.txt` (uporedna tabela) + opciono `.json`
8. [ ] Napisati README sekciju "Model Comparison: LogisticRegression vs.
   Jev" — metodologija (uključujući napomenu o zero-shot pristupu za Jev),
   uporedna tabela metrika + latencije, kratak zaključak
9. [ ] Review rezultata sa korisnikom pre merge-a
10. [ ] Merge `feature/jev-comparison` → `master`

## Otvoreno / za proveru pri implementaciji

- Tačan naziv env varijable i format Vercel AI Gateway REST poziva za Jev
  (proveriti trenutnu dokumentaciju u trenutku implementacije — API je nov
  i može se menjati).
- Da li Jev vraća i confidence/probability (za eventualno poređenje ROC-AUC
  pored praga 0/1), ili samo tvrdu labelu.
