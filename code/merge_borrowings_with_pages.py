"""Merge borrowings and media inventory to add pages information.
Creates "data/processed/borrowings_with_pages.csv" by matching on normalized ISBN and falling back to Barcode/EAN when ISBN matching is not available.
"""

from pathlib import Path
import re
import pandas as pd


def normalize_isbn(val):
    if pd.isna(val):
        return None
    s = str(val)
    # remove non-digits and keep X (for ISBN-10)
    s = re.sub(r"[^0-9Xx]", "", s)
    s = s.upper()
    return s if s else None


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / 'data'
    borrowings_file = data_dir / 'processed' / 'borrowings_2019_2025.csv'
    media_file = data_dir / 'raw' / 'Stadtbücherei Tübingen Medienbestand.csv'
    #out_file = data_dir / 'processed' / 'borrowings_with_pages.csv'
    out_file = borrowings_file  # Überschreibe die Originaldatei

    if not borrowings_file.exists():
        raise FileNotFoundError(f'Borrowings file not found: {borrowings_file}')
    if not media_file.exists():
        raise FileNotFoundError(f'Media inventory file not found: {media_file}')

    print('Loading files...')
    df = pd.read_csv(borrowings_file, sep=';', quotechar='"', encoding='utf-8', low_memory=False)
    med = pd.read_csv(media_file, sep=';', quotechar='"', encoding='utf-8', low_memory=False)

    # Directly set column names based on known structure
    borrow_isbn_col = 'ISBN'
    borrow_barcode_col = 'Barcode'
    med_isbn_col = 'ISBN_ISSN_EAN'
    med_barcode_col = 'Barcodes'
    pages_col = 'Seitenzahl'

    print('Using columns:')
    print(' borrowings ISBN:', borrow_isbn_col)
    print(' borrowings barcode:', borrow_barcode_col)
    print(' media ISBN:', med_isbn_col)
    print(' media barcode:', med_barcode_col)
    print(' media pages:', pages_col)

    # normalize and create helper columns
    if borrow_isbn_col:
        df['ISBN_norm'] = df[borrow_isbn_col].apply(normalize_isbn)
    else:
        df['ISBN_norm'] = None

    if med_isbn_col:
        med['ISBN_norm'] = med[med_isbn_col].apply(normalize_isbn)
    else:
        med['ISBN_norm'] = None

    if borrow_barcode_col:
        df['Barcode_str'] = df[borrow_barcode_col].astype(str).where(df[borrow_barcode_col].notna(), pd.NA).str.strip()
    else:
        df['Barcode_str'] = None

    if med_barcode_col:
        med['Barcode_str'] = med[med_barcode_col].astype(str).where(med[med_barcode_col].notna(), pd.NA).str.strip()
    else:
        med['Barcode_str'] = None

    # pages numeric
    if pages_col:
        med['_pages_num'] = pd.to_numeric(med[pages_col].astype(str).str.extract(r'([0-9]+)', expand=False), errors='coerce')
    else:
        med['_pages_num'] = pd.NA

    # Use memory-friendly mapping instead of a full DataFrame merge
    # Build ISBN -> pages map (take first non-null pages for each ISBN)
    isbn_med = med.loc[med['ISBN_norm'].notna() & med['_pages_num'].notna(), ['ISBN_norm', '_pages_num']]
    isbn_map = isbn_med.groupby('ISBN_norm')['_pages_num'].first().to_dict()

    # Map ISBN to borrowings
    df['_pages_num'] = df['ISBN_norm'].map(isbn_map)
    matched_isbn = df['_pages_num'].notna().sum()
    print(f'Matched by ISBN (mapping): {matched_isbn}')

    # Fallback by barcode for rows still missing pages
    need_pages_mask = df['_pages_num'].isna()
    if med_barcode_col and borrow_barcode_col:
        med_bar = med.loc[med['Barcode_str'].notna() & med['_pages_num'].notna(), ['Barcode_str', '_pages_num']].drop_duplicates(subset=['Barcode_str'])
        bar_map = med_bar.set_index('Barcode_str')['_pages_num'].to_dict()
        # map only for rows that still need pages
        df.loc[need_pages_mask, '_pages_num'] = df.loc[need_pages_mask, 'Barcode_str'].map(bar_map)
        matched_bar = df['_pages_num'].notna().sum() - matched_isbn
        print(f'Additional matched by Barcode (mapping): {matched_bar}')
    else:
        print('Barcode fallback skipped (column missing)')

    df['_pages_num'] = pd.to_numeric(df['_pages_num'], errors='coerce')
    merged = df

    # Entferne Hilfsspalten
    merged = merged.drop(['ISBN_norm', 'Barcode_str'], axis=1, errors='ignore')

    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_file, index=False, sep=';', quotechar='"', encoding='utf-8')

    print('Saved merged file to', out_file)
    print('Total rows:', len(merged))
    print('Rows with pages info:', merged['_pages_num'].notna().sum())
    print('Fraction with pages info: {:.2%}'.format(merged['_pages_num'].notna().mean()))


if __name__ == '__main__':
    main()
