# M2A PROJECT PROPOSAL

**Group Name:** Ginebra San Miguel  
**Members:** Vince Laude, Andrei Maquinto, Irron Panganiban  
**Section:** CIS302  
**Date:** February 26, 2026

## Project Title
Smart Library: ISBN Barcode Quick Cataloger

## Concepts to be Implemented
**Combination:** Barcode Detection + Metadata Lookup

1. **Barcode Detection**
The system detects and decodes a book barcode (EAN-13 / ISBN-13) from a single book shown to a webcam or uploaded photo.

2. **Metadata Lookup**
After ISBN capture, the system retrieves **Title** and **Author** from a metadata source (API).  
If online lookup fails, the system uses a local fallback dataset/cache when available.

## Problem Statement
When books are donated or sold in bulk, encoding titles manually is slow and error-prone.  
This project replaces manual typing with barcode scanning: the user shows a barcode to the camera, and the system automatically records ISBN, title, and author into a CSV log.

## Project Overview
### Input
- Live camera feed (USB webcam/laptop camera), or uploaded image
- One book barcode at a time
- Recommended distance: ~8 to 12 inches from camera

### Processing Pipeline
1. **Frame Preprocessing**
   - Convert frame to grayscale
   - Apply contrast/brightness enhancement for barcode clarity
2. **Barcode Detection and Decode**
   - Locate barcode region
   - Decode ISBN value from barcode
3. **ISBN Validation**
   - Validate ISBN-13 checksum
   - If ISBN-10 is detected, convert to ISBN-13
4. **Metadata Lookup**
   - Fetch title and author using ISBN
   - Use fallback source if online API is unavailable
5. **Logging**
   - Append timestamp, ISBN, title, and author to `books_list.csv`

### Output
1. **Real-time Overlay**
   - Green bounding box around detected barcode
   - On-screen message:
   `ISBN Detected: [ISBN] | [Title] - [Author]`
2. **CSV Running Log**
   - File: `books_list.csv`
   - Columns: `timestamp, isbn, title, author, source`

## Scope
### In Scope
- Single-book barcode scanning
- ISBN extraction and validation
- Metadata lookup and CSV logging
- Duplicate-scan suppression (short cooldown per ISBN)

### Out of Scope
- Multi-book simultaneous scanning
- Recovery of severely damaged barcodes
- Full library management system (borrowing, returns, accounts)

## Error Handling Rules
- If barcode is unreadable: show prompt to refocus/reposition
- If ISBN is invalid: do not log; show validation error
- If metadata lookup fails: log ISBN with `Unknown Title` / `Unknown Author`
- If duplicate ISBN is scanned within cooldown window: ignore duplicate log entry

## Expected Outcome
- Average successful scan-to-log time: **under 2 seconds** on clear barcodes
- Success rate target:
  - **95%+** on clear, well-lit, unblurred barcodes
  - **Lower** under glare, blur, extreme tilt, or low light

## Testing Plan
The team will test at least these scenarios:
1. Clear and centered barcode
2. Slightly tilted barcode
3. Low-light barcode
4. Slight blur / hand movement
5. Duplicate scan attempts
6. No internet / API unavailable

## Deliverables
1. Python application (camera + optional image mode)
2. CSV logger (`books_list.csv`)
3. Short user guide (setup and scan steps)
4. Final report with test results and limitations
