import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen
import cv2
import pytesseract


@dataclass
class AppConfig:
    camera_index: int
    csv_path: Path


@dataclass
class BookInfo:
    title: str
    author: str


@dataclass
class ShapeMeasurement:
    contour: list[tuple[int, int]]
    bounds: tuple[int, int, int, int]


class IsbnUtil:
    @staticmethod
    def _keep_isbn_chars(text: str) -> str:
        return "".join(ch for ch in text if ch.isdigit() or ch.upper() == "X")

    @staticmethod
    def is_valid_isbn13(value: str) -> bool:
        if len(value) != 13 or not value.isdigit():
            return False
        total = 0
        for i, ch in enumerate(value[:12]):
            digit = int(ch)
            total += digit if i % 2 == 0 else digit * 3
        check = (10 - (total % 10)) % 10
        return check == int(value[-1])

    @staticmethod
    def is_valid_isbn10(value: str) -> bool:
        if len(value) != 10:
            return False
        if not value[:9].isdigit():
            return False
        if not (value[9].isdigit() or value[9].upper() == "X"):
            return False
        total = 0
        for i, ch in enumerate(value[:9], start=1):
            total += i * int(ch)
        check_digit = 10 if value[9].upper() == "X" else int(value[9])
        total += 10 * check_digit
        return total % 11 == 0

    @staticmethod
    def isbn10_to_isbn13(isbn10: str) -> str:
        base = "978" + isbn10[:9]
        total = 0
        for i, ch in enumerate(base):
            digit = int(ch)
            total += digit if i % 2 == 0 else digit * 3
        check = (10 - (total % 10)) % 10
        return base + str(check)

    @classmethod
    def normalize_to_isbn13(cls, raw_value: str) -> Optional[str]:
        value = cls._keep_isbn_chars(raw_value)
        if len(value) == 13 and cls.is_valid_isbn13(value):
            return value
        if len(value) == 10 and cls.is_valid_isbn10(value):
            return cls.isbn10_to_isbn13(value)
        return None


class BooksCsv:
    """Simple CSV storage with dedupe by ISBN."""

    HEADER = ["ISBN", "Book Title", "Book Author"]

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.known_isbns: set[str] = set()
        self._prepare_file()

    @staticmethod
    def _normalize_row(row: list[str], trusted_header: bool) -> Optional[list[str]]:
        if not row:
            return None

        isbn = row[0].strip()
        if not trusted_header and not isbn and len(row) > 1:
            isbn = row[1].strip()

        if not isbn:
            return None

        title = row[1].strip() if len(row) > 1 else ""
        author = row[2].strip() if len(row) > 2 else ""
        if not trusted_header and len(row) > 3 and not title:
            title = row[2].strip()
            author = row[3].strip()

        return [
            isbn,
            title or "Unknown Title",
            author or "Unknown Author",
        ]

    def _prepare_file(self) -> None:
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(self.HEADER)
            return

        try:
            with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        except OSError:
            rows = []

        if not rows:
            rows = [self.HEADER]

        header = [col.strip() for col in rows[0]]
        trusted_header = header == self.HEADER
        body = rows[1:] if trusted_header else rows
        cleaned_rows: list[list[str]] = []
        seen: set[str] = set()

        for row in body:
            normalized = self._normalize_row(row, trusted_header=trusted_header)
            if not normalized:
                continue
            isbn = normalized[0]
            if isbn in seen:
                continue
            seen.add(isbn)
            cleaned_rows.append(normalized)

        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.HEADER)
            writer.writerows(cleaned_rows)

        self.known_isbns = seen

    def has_isbn(self, isbn13: str) -> bool:
        return isbn13 in self.known_isbns

    def add_book(self, isbn13: str, info: BookInfo) -> bool:
        if isbn13 in self.known_isbns:
            return False
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    isbn13,
                    info.title,
                    info.author,
                ]
            )
        self.known_isbns.add(isbn13)
        return True


class MetadataService:
    """Fetches book title/author using online ISBN metadata APIs."""

    @staticmethod
    def _fetch_json(url: str, timeout: float = 3.0) -> Optional[dict]:
        try:
            with urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def _from_google_books(self, isbn13: str) -> Optional[BookInfo]:
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn13}"
        payload = self._fetch_json(url)
        if not payload:
            return None
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            return None

        info = items[0].get("volumeInfo", {})
        if not isinstance(info, dict):
            return None
        title = str(info.get("title", "")).strip()
        authors = info.get("authors", [])
        if isinstance(authors, list):
            author = ", ".join(str(name).strip() for name in authors if str(name).strip())
        else:
            author = ""

        if not title:
            return None
        return BookInfo(
            title=title,
            author=author or "Unknown Author",
        )

    def _from_open_library(self, isbn13: str) -> Optional[BookInfo]:
        url = f"https://openlibrary.org/isbn/{isbn13}.json"
        payload = self._fetch_json(url)
        if not payload:
            return None

        title = str(payload.get("title", "")).strip()
        author = str(payload.get("by_statement", "")).strip()
        if not title:
            return None
        return BookInfo(
            title=title,
            author=author or "Unknown Author",
        )

    def lookup(self, isbn13: str) -> BookInfo:
        return (
            self._from_google_books(isbn13)
            or self._from_open_library(isbn13)
            or BookInfo(
                title="Unknown Title",
                author="Unknown Author",
            )
        )


class ShapeDetector:
    """Detects a barcode-like rectangular region in a frame."""

    def __init__(self, min_area_ratio: float = 0.02) -> None:
        self.min_area_ratio = min_area_ratio

    def detect_barcode_region(self, frame) -> Optional[ShapeMeasurement]:
        h_full, w_full = frame.shape[:2]
        scale = 1.0
        if max(h_full, w_full) > 900:
            scale = 900.0 / float(max(h_full, w_full))
        work = frame
        if scale < 1.0:
            work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)
        blur = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_area = work.shape[0] * work.shape[1]
        min_area = int(frame_area * self.min_area_ratio)
        best = None
        best_score = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(contour)
            w, h = rect[1]
            if w < 1 or h < 1:
                continue
            long_side = max(w, h)
            short_side = min(w, h)
            ratio = long_side / max(short_side, 1.0)
            if ratio < 1.3:
                continue
            score = area * ratio
            if score > best_score:
                best_score = score
                best = contour

        if best is None:
            return None

        rect = cv2.minAreaRect(best)
        box = cv2.boxPoints(rect)
        inv_scale = 1.0 / scale
        points = [(int(x * inv_scale), int(y * inv_scale)) for x, y in box]
        x, y, w, h = cv2.boundingRect(best)
        x = int(x * inv_scale)
        y = int(y * inv_scale)
        w = int(w * inv_scale)
        h = int(h * inv_scale)

        return ShapeMeasurement(
            contour=points,
            bounds=(x, y, x + w, y + h),
        )


class OcrIsbnReader:
    """Extracts ISBN text from an image ROI using Tesseract OCR."""

    def __init__(self) -> None:
        self._configure_tesseract()

    def _configure_tesseract(self) -> None:
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise RuntimeError(
                "Tesseract OCR engine not found in PATH. Install Tesseract and restart terminal."
            ) from exc

    @staticmethod
    def _variants(roi) -> list:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        h, w = gray.shape[:2]
        if min(h, w) < 40:
            return []

        target_width = 1100
        if w < target_width:
            scale = target_width / float(max(w, 1))
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        elif w > 1700:
            scale = 1700.0 / float(w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        inv = cv2.bitwise_not(otsu)
        adaptive = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        )
        return [gray, otsu, inv, adaptive]

    @staticmethod
    def _normalize_ocr_text(text: str) -> str:
        mapping = str.maketrans(
            {
                "O": "0",
                "o": "0",
                "Q": "0",
                "D": "0",
                "I": "1",
                "l": "1",
                "|": "1",
                "S": "5",
                "s": "5",
                "B": "8",
                "Z": "2",
                "G": "6",
            }
        )
        return text.translate(mapping)

    @classmethod
    def _extract_isbn13(cls, text: str) -> Optional[str]:
        normalized = cls._normalize_ocr_text(re.sub(r"[ \t]+", " ", text))

        raw_candidates = re.findall(r"[0-9Xx][0-9Xx\-\s]{8,28}[0-9Xx]", normalized)
        digit_stream = "".join(ch for ch in normalized if ch.isdigit() or ch in "Xx")
        if len(digit_stream) >= 10:
            raw_candidates.append(digit_stream)

        for token in raw_candidates:
            cleaned = re.sub(r"[^0-9Xx]", "", token)
            isbn13 = IsbnUtil.normalize_to_isbn13(cleaned)
            if isbn13:
                return isbn13

            # Recover valid ISBNs from noisy longer strings.
            for window in (13, 10):
                if len(cleaned) < window:
                    continue
                chunks = [cleaned[i : i + window] for i in range(0, len(cleaned) - window + 1)]
                if window == 13:
                    chunks = sorted(chunks, key=lambda chunk: (0 if chunk.startswith(("978", "979")) else 1))
                for chunk in chunks:
                    isbn13 = IsbnUtil.normalize_to_isbn13(chunk)
                    if isbn13:
                        return isbn13
        return None

    @staticmethod
    def _barcode_text_regions(roi) -> list:
        h, _ = roi.shape[:2]
        regions = [roi]
        if h >= 60:
            regions.append(roi[int(h * 0.60) : h, :])  # human-readable barcode digits
            regions.append(roi[int(h * 0.68) : h, :])  # tighter digit strip
            regions.append(roi[: int(h * 0.35), :])    # top ISBN text (if present)
        return [region for region in regions if region.size > 0]

    def read_isbn13(self, roi) -> Optional[str]:
        configs = [
            r"--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789Xx- ",
            r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789Xx- ",
            r"--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789Xx- ",
        ]
        for region in self._barcode_text_regions(roi):
            variants = self._variants(region)
            for variant in variants:
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(variant, config=config)
                    except pytesseract.TesseractError:
                        continue
                    isbn13 = self._extract_isbn13(text)
                    if isbn13:
                        return isbn13
        return None


class SmartLibraryOcrApp:
    WINDOW_NAME = "Smart Library - OCR ISBN Scanner"
    SHAPE_MARGIN_PX = 10

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.csv = BooksCsv(config.csv_path)
        self.metadata = MetadataService()
        self.shape_detector = ShapeDetector()
        self.ocr = OcrIsbnReader()
        self.status_text = "Show one book barcode to camera."
        self.status_until = 0.0
        self.last_scan_text = ""
        self.visible_lock_isbn: Optional[str] = None
        self.frame_index = 0
        self.last_shape: Optional[ShapeMeasurement] = None
        self.last_shape_frame = -999

    def _set_status(self, text: str, hold_seconds: float = 0.0) -> None:
        self.status_text = text
        self.status_until = time.time() + max(0.0, hold_seconds)

    def _crop_shape_region(self, frame, shape: ShapeMeasurement):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = shape.bounds
        x1 = max(0, x1 - self.SHAPE_MARGIN_PX)
        y1 = max(0, y1 - self.SHAPE_MARGIN_PX)
        x2 = min(w, x2 + self.SHAPE_MARGIN_PX)
        y2 = min(h, y2 + self.SHAPE_MARGIN_PX)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _expanded_roi(self, frame, bounds: tuple[int, int, int, int], expand_ratio: float = 0.20):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bounds
        bw = x2 - x1
        bh = y2 - y1
        dx = int(round(bw * expand_ratio))
        dy = int(round(bh * expand_ratio))
        ex1 = max(0, x1 - dx)
        ey1 = max(0, y1 - dy)
        ex2 = min(w, x2 + dx)
        ey2 = min(h, y2 + dy)
        return frame[ey1:ey2, ex1:ex2]

    def _save_if_needed(self, isbn13: str) -> None:
        if self.csv.has_isbn(isbn13):
            self._set_status(f"Already in list: {isbn13}", hold_seconds=1.5)
            self.last_scan_text = f"{isbn13} (already saved)"
            print(f"[SKIP] {isbn13} already in list")
            return
        info = self.metadata.lookup(isbn13)
        self.csv.add_book(isbn13, info)
        self._set_status(f"Saved: {info.title}", hold_seconds=2.0)
        self.last_scan_text = f"{isbn13} | {info.title} - {info.author}"
        print(f"[SAVED] {isbn13} | {info.title} - {info.author}")

    def _get_stable_shape(self, frame) -> Optional[ShapeMeasurement]:
        detected_shape = self.shape_detector.detect_barcode_region(frame)
        if detected_shape is not None:
            self.last_shape = detected_shape
            self.last_shape_frame = self.frame_index
            return detected_shape

        if self.last_shape and self.frame_index - self.last_shape_frame <= 8:
            return self.last_shape
        return None

    def _update_live_status(self, shape: Optional[ShapeMeasurement]) -> None:
        if time.time() < self.status_until:
            return
        if shape is None:
            self.status_text = "No barcode-like shape found."
            self.visible_lock_isbn = None
            return
        self.status_text = "Shape found. Press S to OCR scan."

    def _scan_current_shape(self, frame, shape: Optional[ShapeMeasurement]) -> None:
        print("[SCAN] Manual OCR triggered.")
        self._set_status("Scanning OCR...", hold_seconds=0.8)

        if shape is None:
            self._set_status("No barcode shape to scan. Align barcode and try again.", hold_seconds=1.5)
            return

        roi, bounds = self._crop_shape_region(frame, shape)
        if roi.size == 0:
            self._set_status("Invalid scan region. Reposition barcode.", hold_seconds=1.5)
            return

        isbn13 = self.ocr.read_isbn13(roi)
        if not isbn13:
            wide_roi = self._expanded_roi(frame, bounds, expand_ratio=0.25)
            if wide_roi.size > 0:
                isbn13 = self.ocr.read_isbn13(wide_roi)

        if not isbn13:
            self._set_status("OCR failed. Hold steady and press S again.", hold_seconds=1.5)
            return

        if self.visible_lock_isbn == isbn13:
            self._set_status("Same ISBN already scanned in view. Move away first.", hold_seconds=1.5)
            return

        self._save_if_needed(isbn13)
        self.visible_lock_isbn = isbn13

    def _draw(self, frame, shape: Optional[ShapeMeasurement]):
        display = frame.copy()
        if shape is not None and len(shape.contour) >= 4:
            for i in range(len(shape.contour)):
                p1 = shape.contour[i]
                p2 = shape.contour[(i + 1) % len(shape.contour)]
                cv2.line(display, p1, p2, (255, 180, 0), 2)
            cv2.putText(
                display,
                "Shape detected: barcode region",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 120),
                2,
                cv2.LINE_AA,
            )

        if self.last_scan_text:
            cv2.putText(
                display,
                f"Last: {self.last_scan_text[:100]}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        h, w = display.shape[:2]
        cv2.rectangle(display, (0, h - 38), (w, h), (0, 0, 0), -1)
        cv2.putText(
            display,
            self.status_text,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return display

    @staticmethod
    def _open_camera(index: int):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened() and hasattr(cv2, "CAP_DSHOW"):
            cap.release()
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        return cap

    def run_camera(self) -> None:
        print("OCR scanner running. Press S to scan, Q or Esc to quit.")
        cap = self._open_camera(self.config.camera_index)
        try:
            while True:
                self.frame_index += 1
                frame_ok, frame = cap.read()
                if not frame_ok:
                    self._set_status("Camera read failed.", hold_seconds=1.0)
                    cv2.waitKey(30)
                    continue

                shape = self._get_stable_shape(frame)
                self._update_live_status(shape)

                cv2.imshow(self.WINDOW_NAME, self._draw(frame, shape))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("s"), ord("S")):
                    self._scan_current_shape(frame, shape)
        finally:
            cap.release()
            cv2.destroyAllWindows()

def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Smart Library - OCR + Shape ISBN Scanner")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--csv", type=Path, default=Path("books_list.csv"), help="CSV output file")
    args = parser.parse_args()
    return AppConfig(camera_index=args.camera, csv_path=args.csv)


def main() -> None:
    try:
        SmartLibraryOcrApp(parse_args()).run_camera()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")


main()
