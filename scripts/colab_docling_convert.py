"""
Run this in Google Colab (GPU runtime) - local Docling conversion of
even ONE 160-page 10-K took over 5 minutes of CPU time and still hadn't
finished (still in model-loading/OCR-init phase), so converting all 84
needed documents locally would likely take hours. Docling's layout +
table-structure + OCR models benefit from GPU the same way mpnet
embedding did earlier in this project.

Restricts each PDF to a page range around its actual evidence pages
(see compute_page_ranges.py) instead of converting the full 100-250
page document - local testing showed this cuts per-document time by
roughly 3x (296s -> 96s for 3M's 10-K) since most of a 10-K's length is
legal boilerplate/exhibits that isn't needed for financial QA anyway.

Setup in Colab, before running:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload the data/raw/pdfs_needed/ folder (84 PDFs) AND
     data/interim/page_ranges.json to your Google Drive, e.g.
     MyDrive/evidence_aware_rag_2/pdfs_needed/ and
     MyDrive/evidence_aware_rag_2/page_ranges.json

Paste each ### CELL ### block below into its own Colab cell, in order.
Output (one .md file per PDF) is saved to Drive, then zipped for one
clean download.

Resumable: skips any PDF whose .md output already exists, so a dropped
Colab session doesn't lose completed conversions.
"""

### CELL 1 - install + mount Drive ###
"""
!pip install -q docling

from google.colab import drive
drive.mount('/content/drive')
"""

### CELL 2 - config: EDIT THIS PATH to match where you put the files in Drive ###
"""
DRIVE_DIR = '/content/drive/MyDrive/evidence_aware_rag_2'
PDF_DIR = f'{DRIVE_DIR}/pdfs_needed'
PAGE_RANGES_PATH = f'{DRIVE_DIR}/page_ranges.json'
MARKDOWN_DIR = f'{DRIVE_DIR}/docling_markdown'
OUTPUT_ZIP = f'{DRIVE_DIR}/docling_markdown.zip'

import json
import os
os.makedirs(MARKDOWN_DIR, exist_ok=True)
page_ranges = json.load(open(PAGE_RANGES_PATH))
"""

### CELL 3 - convert all PDFs (resumable, GPU-accelerated) ###
"""
import time
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption

pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options.device = 'cuda'
# these are digitally-generated PDFs with a real text layer (confirmed
# earlier - pypdf could extract text directly), so OCR is unnecessary
# and was almost certainly the actual bottleneck at 5 min/PDF, since
# OCR doesn't benefit from the GPU setting the way layout/table models do
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

pdf_paths = sorted(Path(PDF_DIR).glob('*.pdf'))
print(f'{len(pdf_paths)} PDFs to convert')

t_start = time.time()
for i, pdf_path in enumerate(pdf_paths, 1):
    out_path = Path(MARKDOWN_DIR) / (pdf_path.stem + '.md')
    if out_path.exists():
        continue
    t0 = time.time()
    start, end = page_ranges[pdf_path.stem]
    result = converter.convert(str(pdf_path), page_range=(start, end))
    md = result.document.export_to_markdown()
    out_path.write_text(md)
    elapsed = time.time() - t0
    total_elapsed = time.time() - t_start
    print(f'[{i}/{len(pdf_paths)}] {pdf_path.name} done in {elapsed:.0f}s | total {total_elapsed/60:.1f} min', flush=True)

print('all conversions done')
"""

### CELL 4 - zip for download ###
"""
import shutil
shutil.make_archive(OUTPUT_ZIP.replace('.zip', ''), 'zip', MARKDOWN_DIR)
print('zipped:', OUTPUT_ZIP)
"""
