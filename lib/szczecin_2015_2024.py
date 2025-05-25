import re
from pathlib import Path

import numpy as np
import pandas as pd

from lib.data import DATA_PATH

PATH = DATA_PATH / "Szczecin_2015_2024"


def get_min_max_square_meters_area_from_path(path: Path) -> tuple[int, int]:
    stem = path.stem
    if match := re.search(r'_(\d+)_(\d+)m2', stem):
        return int(match.group(1)), int(match.group(2))
    if match := re.search(r'od_(\d+)m2', stem):
        return int(match.group(1)), None
    if match := re.search(r'pon_(\d+)m2', stem):
        return 0, int(match.group(1))
    raise ValueError(f"Could not extract area range from filename: {path}")


def get_details_df() -> pd.DataFrame:
    taxpayers_parts = []
    area_parts = []

    for csv_path in PATH.glob('*_o_pow_*.csv'):
        raw_df = pd.read_csv(csv_path, sep=';')
        if csv_path.stem.startswith('Liczba_podatnikow_os_fiz_o_pow_'):
            area_min, area_max = get_min_max_square_meters_area_from_path(csv_path)
            assert len(raw_df.columns) == 4
            assert all(raw_df.iloc[:, 0] == 'Fizyczne')
            assert all(raw_df.iloc[:, 1] == 'Osoba fizyczna')
            taxpayers_parts.append(
                raw_df
                .iloc[:, 2:]
                .rename(columns={'rok': 'year', 'ilosc_podatników': 'n_taxpayers'})
                .assign(area_min=area_min, area_max=area_max)
            )
        elif csv_path.stem.startswith('Powierzchnia_opodatk_os_fiz_przedzial_o_pow_'):
            area_min, area_max = get_min_max_square_meters_area_from_path(csv_path)
            assert len(raw_df.columns) == 4
            assert all(raw_df.iloc[:, 0] == 'Fizyczne')
            area_parts.append(
                raw_df
                .iloc[:, 1:]
                .rename(columns={
                    'rok': 'year',
                    'suma_powierzchni_opodatkowanej': 'total_area',
                    'ilosc_kont': 'n_accounts',
                })
                .assign(area_min=area_min, area_max=area_max)
            )
    merged_df = pd.merge(
        (
            pd.concat(taxpayers_parts)
            .reset_index(drop=True)
            [['area_min', 'area_max', 'year', 'n_taxpayers']]
            .set_index(['year', 'area_min', 'area_max'])
            .sort_index()
        ),
        (
            pd.concat(area_parts)
            .reset_index(drop=True)
            [['area_min', 'area_max', 'year', 'total_area', 'n_accounts']]
            .set_index(['year', 'area_min', 'area_max'])
            .sort_index()
        ),
        on=['year', 'area_min', 'area_max'],
        how='outer',
    )
    return (
        merged_df
        .reset_index()
        .astype({
            'area_min': np.float64,
            'area_max': np.float64,
            'year': np.int64,
            'n_taxpayers': np.int64,
            'n_accounts': np.int64,
        })
        .assign(total_area=lambda df: df['total_area'].str.replace(',', '.').astype(np.float64))
    )


def get_summary_df() -> pd.DataFrame:
    people_part = pd.merge(
        (
            pd.read_csv(PATH / 'Suma_liczba_podatników_osoby_fizyczne.csv', sep=';')
            .iloc[:, 1:]
            .rename(columns={'typ_osoby': 'taxpayer_type', 'rok': 'year', 'ilosc_podatników': 'n_taxpayers'})
            .set_index(['year', 'taxpayer_type'])
        ),
        (
            pd.read_csv(PATH / 'Suma_powierzchnia_opodatkowana_osoby_fizyczne.csv', sep=';')
            .iloc[:, 1:]
            .rename(columns={
                'rok': 'year',
                'suma_powierzchni_opodatkowanej': 'total_area',
                'ilosc_kont': 'n_accounts',
            })
            .assign(taxpayer_type='Osoba fizyczna')
            .set_index(['year', 'taxpayer_type'])
        ),
        how='outer',
        on=['year', 'taxpayer_type'],
    )
    legal_part = pd.merge(
        (
            pd.read_csv(PATH / 'Suma_liczba_podatników_osoby_prawne.csv', sep=';')
            .iloc[:, 1:]
            .rename(columns={'typ_osoby': 'taxpayer_type', 'rok': 'year', 'ilosc_podatników': 'n_taxpayers'})
            .set_index(['year', 'taxpayer_type'])
        ),
        (
            pd.read_csv(PATH / 'Suma_powierzchnia_opodatkowana_osoby_prawne.csv', sep=';')
            .iloc[:, 1:]
            .rename(columns={
                'rok': 'year',
                'suma_powierzchni_opodatkowanej': 'total_area',
                'ilosc_kont': 'n_accounts',
            })
            .assign(taxpayer_type='Osoba prawna')
            .set_index(['year', 'taxpayer_type'])
        ),
        how='outer',
        on=['year', 'taxpayer_type'],
    )
    return (
        pd.concat([people_part, legal_part])
        .reset_index()
        .assign(total_area=lambda df: df['total_area'].str.replace(',', '.').astype(np.float64))
    )
