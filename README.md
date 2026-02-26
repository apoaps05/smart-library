# Smart Library OCR Scanner

Smart Library scans a book barcode from your camera, reads the ISBN using OCR, fetches metadata, and saves results to `books_list.csv`.

## Features

- Detects barcode-like rectangular regions before scanning
- Extracts ISBN from camera frame with Tesseract OCR
- Looks up title/author from online book metadata sources
- Prevents duplicate ISBN entries in CSV

## Requirements

- Windows
- Python 3.10+
- Tesseract OCR installed and available in `PATH`

## Setup

1. Clone the repository and open it.

```powershell
git clone <your-repo-url>
cd <your-repo-folder>
```

2. Verify Python.

```powershell
py -V
```

3. Install Tesseract OCR.

```powershell
winget install --id tesseract-ocr.tesseract -e
```

4. Verify Tesseract.

```powershell
tesseract --version
```

5. Install Python dependencies.

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Run

```powershell
py app_ocr.py --camera 0
```

Controls:
- `S`: scan current detected barcode region
- `Q` or `Esc`: quit

## Output

- Results are written to `books_list.csv`
- CSV columns: `ISBN`, `Book Title`, `Book Author`
- Duplicate ISBN values are skipped automatically

## Troubleshooting

- If `tesseract --version` is not found, add this folder to `PATH`:
  `C:\Program Files\Tesseract-OCR`
- Restart your terminal after updating `PATH`
- If needed, set explicit path in code:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```
