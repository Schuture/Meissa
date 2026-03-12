"""
preprocess_mimic.py — Build multimodal OSCE scenarios by linking:
  - MIMIC-IV (admissions, diagnoses, labs, patients)
  - MIMIC-IV-Note (discharge summaries, radiology reports)
  - MIMIC-CXR-JPG (images, CheXpert labels, metadata)

Output: datasets/mimic_cxr_train.jsonl — one OSCE scenario per line.

Usage:
    python preprocess_mimic.py \
        --output datasets/mimic_cxr_train.jsonl \
        --num_samples 999999 \
        --use_resized
"""

import os
import sys
import json
import re
import argparse
import logging
import gzip
import csv
from collections import defaultdict

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Paths
# ============================================================

# Set these paths to your local PhysioNet data directories.
# Download from: https://physionet.org/content/mimiciv/
#                https://physionet.org/content/mimic-cxr-jpg/
#                https://physionet.org/content/mimic-iv-note/
MIMIC_IV_DIR = os.environ.get("MIMIC_IV_DIR", "/path/to/physionet/mimiciv/3.1")
MIMIC_CXR_DIR = os.environ.get("MIMIC_CXR_DIR", "/path/to/physionet/mimic-cxr-jpg/2.1.0")
MIMIC_NOTE_DIR = os.environ.get("MIMIC_NOTE_DIR", "/path/to/physionet/mimic-iv-note/2.2/note")


# ============================================================
# Discharge Summary Parsing
# ============================================================

# Common section headers in MIMIC discharge summaries
SECTION_HEADERS = [
    "Name", "Unit No", "Admission Date", "Discharge Date", "Date of Birth",
    "Sex", "Service", "Allergies",
    "Chief Complaint", "Major Surgical or Invasive Procedure",
    "History of Present Illness", "History of Present illness",
    "Past Medical History", "Past medical History",
    "Social History", "Family History",
    "Physical Exam", "Physical Examination",
    "Admission Physical Exam", "Discharge Physical Exam",
    "Pertinent Results", "Brief Hospital Course",
    "Medications on Admission", "Discharge Medications",
    "Discharge Disposition", "Discharge Diagnosis",
    "Discharge Condition", "Discharge Instructions",
    "Followup Instructions", "Follow-up Instructions",
]


def parse_discharge_summary(text):
    """
    Parse a MIMIC discharge summary into sections.
    Returns dict of {section_name: section_text}.
    """
    if not text or not isinstance(text, str):
        return {}

    sections = {}
    # Build regex from known headers (case-insensitive)
    header_pattern = re.compile(
        r'^(' + '|'.join(re.escape(h) for h in SECTION_HEADERS) + r')\s*:\s*$',
        re.MULTILINE | re.IGNORECASE
    )

    matches = list(header_pattern.finditer(text))
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        # Normalize section name
        normalized = section_name.replace("_", " ").strip()
        sections[normalized] = content

    return sections


def extract_patient_presentation(sections):
    """Extract patient presentation fields from parsed discharge sections."""
    # Chief complaint
    cc = ""
    for key in ["Chief Complaint"]:
        if key in sections:
            cc = sections[key]
            break

    # History of Present Illness
    hpi = ""
    for key in ["History of Present Illness", "History of Present illness"]:
        if key in sections:
            hpi = sections[key]
            break

    # Past Medical History
    pmh = ""
    for key in ["Past Medical History", "Past medical History"]:
        if key in sections:
            pmh = sections[key]
            break

    # Social History
    social = sections.get("Social History", "")

    # Family History
    family = sections.get("Family History", "")

    # Allergies
    allergies = sections.get("Allergies", "")

    return {
        "chief_complaint": cc,
        "hpi": hpi,
        "pmh": pmh,
        "social_history": social,
        "family_history": family,
        "allergies": allergies,
    }


def extract_physical_exam(sections):
    """Extract physical exam from discharge summary."""
    for key in ["Physical Exam", "Physical Examination",
                "Admission Physical Exam"]:
        if key in sections:
            return sections[key]
    return ""


def extract_discharge_diagnosis(sections):
    """Extract discharge diagnosis from summary."""
    for key in ["Discharge Diagnosis"]:
        if key in sections:
            return sections[key]
    return ""


# ============================================================
# Lab Results Grouping
# ============================================================

def group_lab_results(labs_df, d_labitems_map):
    """
    Group lab events into categories (vectorized — no iterrows).
    Returns dict of {category: {test_name: "value unit (ref: low-high)"}}.
    """
    if labs_df.empty:
        return {}

    # Map itemid to category and label via vectorized lookup
    itemids = labs_df["itemid"].values
    categories = []
    labels = []
    for iid in itemids:
        info = d_labitems_map.get(iid, {})
        categories.append(info.get("category", "Other"))
        labels.append(info.get("label", f"Item_{iid}"))

    # Build display strings vectorized
    values = labs_df["value"].fillna("").astype(str).values
    uoms = labs_df["valueuom"].fillna("").astype(str).values
    ref_lows = labs_df["ref_range_lower"].values if "ref_range_lower" in labs_df.columns else [None] * len(labs_df)
    ref_highs = labs_df["ref_range_upper"].values if "ref_range_upper" in labs_df.columns else [None] * len(labs_df)

    groups = {}
    for i in range(len(labs_df)):
        cat = categories[i]
        lab = labels[i]
        v = values[i]
        u = uoms[i]
        display = f"{v} {u}".strip() if v else "N/A"
        rl, rh = ref_lows[i], ref_highs[i]
        if pd.notna(rl) and pd.notna(rh):
            display += f" (ref: {rl}-{rh})"
        if cat not in groups:
            groups[cat] = {}
        if lab not in groups[cat]:
            groups[cat][lab] = display

    return groups


# ============================================================
# CXR Path Builder
# ============================================================

def build_cxr_path(subject_id, study_id, dicom_id, use_resized=True):
    """Build the file path to a CXR JPG image."""
    prefix = "files_resized_1024" if use_resized else "files"
    p_group = f"p{str(subject_id)[:2]}"
    return os.path.join(
        MIMIC_CXR_DIR, prefix,
        p_group, f"p{subject_id}", f"s{study_id}",
        f"{dicom_id}.jpg"
    )


# ============================================================
# CheXpert Label Processing
# ============================================================

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]


def format_chexpert_labels(row):
    """Convert CheXpert label row to readable dict."""
    labels = {}
    for label in CHEXPERT_LABELS:
        val = row.get(label)
        if pd.notna(val):
            if val == 1.0:
                labels[label] = "Positive"
            elif val == 0.0:
                labels[label] = "Negative"
            elif val == -1.0:
                labels[label] = "Uncertain"
    return labels


# ============================================================
# Build OSCE Scenario
# ============================================================

def build_osce_scenario(
    subject_id, hadm_id, study_id, dicom_id,
    patient_row, presentation, physical_exam_text,
    lab_groups, radiology_text, chexpert_labels, image_path,
    discharge_diagnosis
):
    """Build a single OSCE scenario dict."""
    # Demographics
    gender = patient_row.get("gender", "Unknown")
    age = patient_row.get("anchor_age", "Unknown")
    gender_str = "male" if gender == "M" else "female" if gender == "F" else gender
    demographics = f"{age}-year-old {gender_str}"

    # Parse physical exam text into sub-sections
    pe_findings = {}
    if physical_exam_text:
        # Try to split by lines that look like headers
        current_key = "General"
        current_lines = []
        for line in physical_exam_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Check if line looks like a header (e.g., "HEENT:", "Lungs:", "CV:")
            header_match = re.match(r'^([A-Z][A-Za-z/\-\s]{1,30}):(.*)$', line)
            if header_match:
                if current_lines:
                    pe_findings[current_key] = "\n".join(current_lines)
                current_key = header_match.group(1).strip()
                rest = header_match.group(2).strip()
                current_lines = [rest] if rest else []
            else:
                current_lines.append(line)
        if current_lines:
            pe_findings[current_key] = "\n".join(current_lines)

    if not pe_findings and physical_exam_text:
        pe_findings["Physical_Exam"] = physical_exam_text

    # Build test results
    test_results = dict(lab_groups)
    if image_path and os.path.exists(image_path):
        test_results["Chest_XRay"] = {
            "image_path": image_path,
            "findings": radiology_text or "No radiology report available.",
        }

    # Build secondary symptoms from HPI (first sentence as primary, rest as secondary)
    cc = presentation.get("chief_complaint", "")
    hpi = presentation.get("hpi", "")

    scenario = {
        "OSCE_Examination": {
            "Objective_for_Doctor": "Diagnose the patient's condition based on clinical evaluation",
            "Patient_Actor": {
                "Demographics": demographics,
                "History": hpi[:2000] if hpi else "",  # Truncate very long HPI
                "Symptoms": {
                    "Primary_Symptom": cc,
                    "Secondary_Symptoms": [],
                },
                "Past_Medical_History": presentation.get("pmh", "")[:1000],
                "Social_History": presentation.get("social_history", ""),
                "Review_of_Systems": "",
            },
            "Physical_Examination_Findings": pe_findings,
            "Test_Results": test_results,
            "Correct_Diagnosis": discharge_diagnosis,
            "CheXpert_Labels": chexpert_labels,
            "image_url": image_path,
        },
        "_mimic_ids": {
            "subject_id": int(subject_id),
            "hadm_id": int(hadm_id),
            "study_id": int(study_id),
            "dicom_id": str(dicom_id),
        }
    }

    return scenario


# ============================================================
# Main Pipeline
# ============================================================

def load_processed_ids(output_file):
    """Load already-processed (subject_id, hadm_id, study_id) tuples."""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        ids = record.get("_mimic_ids", {})
                        key = (ids.get("subject_id"), ids.get("hadm_id"), ids.get("study_id"))
                        processed.add(key)
                    except json.JSONDecodeError:
                        pass
    return processed


def main():
    parser = argparse.ArgumentParser(description="Build MIMIC-IV + CXR multimodal OSCE scenarios")
    parser.add_argument("--output", type=str, default="datasets/mimic_cxr_train.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--num_samples", type=int, default=999999,
                        help="Max scenarios to build")
    parser.add_argument("--use_resized", action="store_true", default=True,
                        help="Use resized 1024px CXR images")
    parser.add_argument("--no_resized", action="store_true",
                        help="Use original resolution CXR images")
    parser.add_argument("--min_labs", type=int, default=3,
                        help="Minimum lab results required per admission")
    parser.add_argument("--lab_chunk_size", type=int, default=500000,
                        help="Chunk size for reading labevents")
    args = parser.parse_args()

    use_resized = not args.no_resized

    # Resume support
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    processed_ids = load_processed_ids(args.output)
    if processed_ids:
        logger.info(f"Resuming: {len(processed_ids)} already processed")

    # ----------------------------------------------------------
    # Step 1: Load CXR metadata + split + CheXpert labels
    # ----------------------------------------------------------
    logger.info("Loading MIMIC-CXR metadata...")
    metadata = pd.read_csv(os.path.join(MIMIC_CXR_DIR, "mimic-cxr-2.0.0-metadata.csv.gz"))
    split = pd.read_csv(os.path.join(MIMIC_CXR_DIR, "mimic-cxr-2.0.0-split.csv.gz"))
    chexpert = pd.read_csv(os.path.join(MIMIC_CXR_DIR, "mimic-cxr-2.0.0-chexpert.csv.gz"))

    # Filter to train split
    train_dicoms = set(split[split["split"] == "train"]["dicom_id"])
    metadata_train = metadata[metadata["dicom_id"].isin(train_dicoms)].copy()
    logger.info(f"Train split CXR images: {len(metadata_train)}")

    # Filter to PA or AP views (most informative chest views)
    metadata_train = metadata_train[metadata_train["ViewPosition"].isin(["PA", "AP"])].copy()
    logger.info(f"After PA/AP filter: {len(metadata_train)}")

    # Deduplicate: keep one image per study (prefer PA over AP)
    metadata_train["view_priority"] = metadata_train["ViewPosition"].map({"PA": 0, "AP": 1})
    metadata_train = metadata_train.sort_values("view_priority").drop_duplicates(
        subset=["subject_id", "study_id"], keep="first"
    ).drop(columns=["view_priority"])
    logger.info(f"After dedup (1 per study): {len(metadata_train)}")

    # ----------------------------------------------------------
    # Step 2: Load MIMIC-IV tables
    # ----------------------------------------------------------
    logger.info("Loading MIMIC-IV tables...")
    patients = pd.read_csv(os.path.join(MIMIC_IV_DIR, "hosp", "patients.csv.gz"))
    admissions = pd.read_csv(os.path.join(MIMIC_IV_DIR, "hosp", "admissions.csv.gz"),
                             parse_dates=["admittime", "dischtime"])
    diagnoses = pd.read_csv(os.path.join(MIMIC_IV_DIR, "hosp", "diagnoses_icd.csv.gz"))
    d_icd = pd.read_csv(os.path.join(MIMIC_IV_DIR, "hosp", "d_icd_diagnoses.csv.gz"))
    d_labitems = pd.read_csv(os.path.join(MIMIC_IV_DIR, "hosp", "d_labitems.csv.gz"))

    # Build d_labitems lookup
    d_labitems_map = {}
    for _, row in d_labitems.iterrows():
        d_labitems_map[row["itemid"]] = {
            "label": row["label"],
            "category": row.get("category", "Other"),
        }

    # Build ICD code to description map
    icd_map = {}
    for _, row in d_icd.iterrows():
        icd_map[(str(row["icd_code"]), int(row["icd_version"]))] = row["long_title"]

    logger.info(f"Patients: {len(patients)}, Admissions: {len(admissions)}")

    # ----------------------------------------------------------
    # Step 3: Load discharge summaries and radiology reports
    # ----------------------------------------------------------
    logger.info("Loading MIMIC-IV-Note discharge summaries...")
    discharge_notes = pd.read_csv(os.path.join(MIMIC_NOTE_DIR, "discharge.csv.gz"))
    logger.info(f"Discharge notes: {len(discharge_notes)}")

    logger.info("Loading MIMIC-IV-Note radiology reports...")
    radiology_notes = pd.read_csv(os.path.join(MIMIC_NOTE_DIR, "radiology.csv.gz"))
    logger.info(f"Radiology reports: {len(radiology_notes)}")

    # ----------------------------------------------------------
    # Step 4: Link CXR studies to admissions via subject_id + date
    # ----------------------------------------------------------
    logger.info("Linking CXR studies to hospital admissions...")

    # Convert CXR StudyDate to datetime for matching
    metadata_train["StudyDate"] = pd.to_datetime(metadata_train["StudyDate"], format="%Y%m%d", errors="coerce")

    # Merge with admissions on subject_id
    linked = metadata_train.merge(admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
                                  on="subject_id", how="inner")

    # Filter: CXR study date within admission window
    linked = linked[
        (linked["StudyDate"] >= linked["admittime"].dt.normalize()) &
        (linked["StudyDate"] <= linked["dischtime"].dt.normalize() + pd.Timedelta(days=1))
    ].copy()
    logger.info(f"CXR linked to admissions: {len(linked)}")

    # Deduplicate: one CXR per admission (keep earliest study)
    linked = linked.sort_values("StudyDate").drop_duplicates(
        subset=["subject_id", "hadm_id"], keep="first"
    )
    logger.info(f"After dedup (1 CXR per admission): {len(linked)}")

    # ----------------------------------------------------------
    # Step 5: Merge with discharge notes
    # ----------------------------------------------------------
    logger.info("Merging with discharge summaries...")
    # discharge notes have subject_id and hadm_id
    linked = linked.merge(
        discharge_notes[["subject_id", "hadm_id", "text"]].rename(columns={"text": "discharge_text"}),
        on=["subject_id", "hadm_id"],
        how="inner"
    )
    logger.info(f"With discharge notes: {len(linked)}")

    # ----------------------------------------------------------
    # Step 6: Merge with CheXpert labels
    # ----------------------------------------------------------
    logger.info("Merging with CheXpert labels...")
    linked = linked.merge(
        chexpert,
        on=["subject_id", "study_id"],
        how="left"
    )

    # ----------------------------------------------------------
    # Step 7: Merge with patient demographics
    # ----------------------------------------------------------
    linked = linked.merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id",
        how="left"
    )

    # ----------------------------------------------------------
    # Step 8: Get primary diagnoses
    # ----------------------------------------------------------
    logger.info("Getting primary diagnoses...")
    primary_diag = diagnoses[diagnoses["seq_num"] == 1].copy()
    primary_diag["diag_text"] = primary_diag.apply(
        lambda r: icd_map.get((str(r["icd_code"]), int(r["icd_version"])), str(r["icd_code"])),
        axis=1
    )
    linked = linked.merge(
        primary_diag[["subject_id", "hadm_id", "icd_code", "diag_text"]],
        on=["subject_id", "hadm_id"],
        how="left"
    )

    # Filter: must have a diagnosis
    linked = linked.dropna(subset=["diag_text"])
    logger.info(f"With primary diagnosis: {len(linked)}")

    # ----------------------------------------------------------
    # Step 9: Build radiology report lookup (vectorized — no iterrows)
    # ----------------------------------------------------------
    logger.info("Building radiology report index...")
    # Filter to rows with valid hadm_id, then group
    rad_valid = radiology_notes.dropna(subset=["subject_id", "hadm_id"]).copy()
    rad_valid["subject_id"] = rad_valid["subject_id"].astype(int)
    rad_valid["hadm_id"] = rad_valid["hadm_id"].astype(int)
    rad_valid["text"] = rad_valid["text"].fillna("").astype(str)
    rad_by_admission = {}
    for (sid, hid), grp in rad_valid.groupby(["subject_id", "hadm_id"]):
        rad_by_admission[(sid, hid)] = grp["text"].tolist()
    del rad_valid
    logger.info(f"Radiology index: {len(rad_by_admission)} admissions")

    # ----------------------------------------------------------
    # Step 10: Load lab events (chunked — large file, vectorized)
    # ----------------------------------------------------------
    logger.info("Building lab events index (chunked reading)...")
    # Only load labs for the hadm_ids we need
    needed_hadm_ids = set(linked["hadm_id"].dropna().astype(int))
    lab_cols = {"hadm_id", "itemid", "value", "valueuom", "ref_range_lower", "ref_range_upper"}
    lab_chunks_collected = []
    total_rows = 0

    labevents_path = os.path.join(MIMIC_IV_DIR, "hosp", "labevents.csv.gz")
    chunk_count = 0
    for chunk in pd.read_csv(labevents_path, chunksize=args.lab_chunk_size,
                             low_memory=False, usecols=lambda c: c in lab_cols):
        filtered = chunk[chunk["hadm_id"].isin(needed_hadm_ids)]
        if len(filtered) > 0:
            lab_chunks_collected.append(filtered.copy())
            total_rows += len(filtered)
        chunk_count += 1
        if chunk_count % 10 == 0:
            logger.info(f"  Read {chunk_count} lab chunks, collected {total_rows} events")

    # Concatenate all and group by hadm_id
    if lab_chunks_collected:
        all_labs = pd.concat(lab_chunks_collected, ignore_index=True)
        del lab_chunks_collected
        all_labs["hadm_id"] = all_labs["hadm_id"].astype(int)
        labs_by_hadm = {hid: grp for hid, grp in all_labs.groupby("hadm_id")}
        logger.info(f"Lab events indexed: {len(all_labs)} events for {len(labs_by_hadm)} admissions")
        del all_labs
    else:
        labs_by_hadm = {}
        logger.info("No lab events found for target admissions")

    # ----------------------------------------------------------
    # Step 11: Pre-compute image paths and check existence in batch
    # ----------------------------------------------------------
    logger.info("Pre-computing CXR image paths...")
    linked = linked.copy()
    linked["cxr_img_path"] = linked.apply(
        lambda r: build_cxr_path(int(r["subject_id"]), int(r["study_id"]),
                                 str(r["dicom_id"]), use_resized),
        axis=1
    )
    # Batch os.path.exists (still I/O but unavoidable; doing it once upfront)
    logger.info("Checking CXR image existence (batch)...")
    linked["cxr_img_exists"] = linked["cxr_img_path"].map(os.path.exists)
    n_exists = linked["cxr_img_exists"].sum()
    logger.info(f"CXR images found: {n_exists}/{len(linked)}")

    # Rename CheXpert columns with spaces → underscores for itertuples compatibility
    chexpert_col_rename = {}
    for label in CHEXPERT_LABELS:
        safe_name = label.replace(" ", "_")
        if safe_name != label and label in linked.columns:
            chexpert_col_rename[label] = safe_name
    if chexpert_col_rename:
        linked = linked.rename(columns=chexpert_col_rename)
    # Build reverse map: safe_name → original label name for output
    safe_to_label = {label.replace(" ", "_"): label for label in CHEXPERT_LABELS}

    # ----------------------------------------------------------
    # Step 12: Build OSCE scenarios
    # ----------------------------------------------------------
    logger.info(f"Building OSCE scenarios (max {args.num_samples})...")

    count = 0
    skipped = 0

    # Open output file once for appending
    out_f = open(args.output, 'a')

    # Use itertuples for ~10x speedup over iterrows
    for tup in linked.itertuples(index=False):
        if count >= args.num_samples:
            break

        sid = int(tup.subject_id)
        hid = int(tup.hadm_id)
        study_id = int(tup.study_id)
        dicom_id = str(tup.dicom_id)

        # Skip if already processed
        if (sid, hid, study_id) in processed_ids:
            continue

        # Check image existence (pre-computed)
        if not tup.cxr_img_exists:
            skipped += 1
            continue
        image_path = tup.cxr_img_path

        # Parse discharge summary
        discharge_text = str(getattr(tup, "discharge_text", ""))
        sections = parse_discharge_summary(discharge_text)
        presentation = extract_patient_presentation(sections)

        # Skip if no meaningful patient info
        if not presentation["chief_complaint"] and not presentation["hpi"]:
            skipped += 1
            continue

        # Physical exam from discharge summary
        pe_text = extract_physical_exam(sections)

        # Lab results — labs_by_hadm now stores DataFrames directly
        hadm_labs_df = labs_by_hadm.get(hid)
        if hadm_labs_df is None or len(hadm_labs_df) < args.min_labs:
            skipped += 1
            continue

        lab_groups = group_lab_results(hadm_labs_df, d_labitems_map)

        # Radiology report — match CXR reports, avoid CT/MRI reports
        rad_reports = rad_by_admission.get((sid, hid), [])
        radiology_text = ""
        _CXR_KEYWORDS = ("chest", "cxr", "x-ray", "xray", "radiograph", "pa and lateral", "portable")
        _NON_CXR_KEYWORDS = ("ct abdomen", "ct pelvis", "ct head", "ct neck", "ct spine",
                             "mri ", "mr ", "ultrasound", "echocardiogram", "nuclear")
        for rpt in rad_reports:
            rpt_lower = rpt[:500].lower()  # Check first 500 chars for keywords
            if any(kw in rpt_lower for kw in _CXR_KEYWORDS):
                # Exclude if it's clearly a non-CXR study
                if not any(kw in rpt_lower for kw in _NON_CXR_KEYWORDS):
                    radiology_text = rpt
                    break
        # No fallback to rad_reports[0] — better to have empty than wrong modality

        # CheXpert labels (columns renamed to safe_name with underscores)
        chexpert_labels = {}
        for safe_name, orig_label in safe_to_label.items():
            val = getattr(tup, safe_name, None)
            if val is not None and pd.notna(val):
                if val == 1.0:
                    chexpert_labels[orig_label] = "Positive"
                elif val == 0.0:
                    chexpert_labels[orig_label] = "Negative"
                elif val == -1.0:
                    chexpert_labels[orig_label] = "Uncertain"

        # Discharge diagnosis (prefer parsed, fallback to ICD)
        discharge_diag = extract_discharge_diagnosis(sections)
        if not discharge_diag:
            discharge_diag = str(getattr(tup, "diag_text", "Unknown"))

        # Build scenario — pass a dict-like object for patient_row
        patient_info = {"gender": getattr(tup, "gender", "Unknown"),
                        "anchor_age": getattr(tup, "anchor_age", "Unknown")}
        scenario = build_osce_scenario(
            sid, hid, study_id, dicom_id,
            patient_info, presentation, pe_text,
            lab_groups, radiology_text, chexpert_labels, image_path,
            discharge_diag
        )

        out_f.write(json.dumps(scenario, ensure_ascii=False) + '\n')

        count += 1
        if count % 500 == 0:
            out_f.flush()
            logger.info(f"Progress: {count} scenarios built, {skipped} skipped")

    out_f.close()
    logger.info(f"Done. Built {count} scenarios, skipped {skipped}")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
