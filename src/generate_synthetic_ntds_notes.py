import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


@dataclass
class Complication:
    id: str
    label: str
    short_definition: str
    positive_note_clues: List[str]


def load_complications(path: str) -> Dict[str, Complication]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    comps: Dict[str, Complication] = {}
    for item in data:
        comps[item["id"]] = Complication(
            id=item["id"],
            label=item["label"],
            short_definition=item["short_definition"],
            positive_note_clues=item.get("positive_note_clues", []),
        )
    return comps


MECHANISMS = [
    "a high-speed motor vehicle collision",
    "a rollover motor vehicle crash",
    "a motorcycle collision",
    "a fall from height",
    "being struck as a pedestrian",
    "an assault with blunt trauma",
]

SEX_WORDS = {
    "M": "male",
    "F": "female",
}


def sample_demographics(rng: np.random.Generator) -> Dict[str, Any]:
    age = int(rng.integers(18, 90))
    sex = "M" if rng.random() < 0.65 else "F"
    mechanism = rng.choice(MECHANISMS)
    iss = int(rng.integers(9, 35))
    return {
        "age": age,
        "sex": sex,
        "sex_word": SEX_WORDS[sex],
        "mechanism": mechanism,
        "iss": iss,
    }


def sample_complication_flags(
    comp_ids: List[str], rng: np.random.Generator, mean_events: float = 1.5
) -> Dict[str, int]:
    lam = max(mean_events, 0.01)
    k = rng.poisson(lam)
    k = int(min(k, len(comp_ids)))
    chosen: List[str] = []
    if k > 0:
        chosen = rng.choice(comp_ids, size=k, replace=False).tolist()
    flags = {cid: int(cid in chosen) for cid in comp_ids}
    return flags


COMP_SECTION_PREFS: Dict[str, List[str]] = {
    "aki": ["icu_day2", "icu_day3", "ward", "discharge"],
    "aws": ["icu_day1", "ward"],
    "ards": ["icu_day1", "icu_day2", "icu_day3", "discharge"],
    "cardiac_arrest_cpr": ["icu_day1", "icu_day2"],
    "cauti": ["icu_day2", "ward"],
    "delirium": ["icu_day2", "icu_day3", "ward"],
    "dvt": ["icu_day3", "ward", "discharge"],
    "mi": ["icu_day1", "icu_day2"],
    "osteomyelitis": ["ward", "discharge"],
    "pressure_ulcer": ["icu_day3", "ward", "discharge"],
    "pe": ["icu_day2", "icu_day3"],
    "severe_sepsis": ["icu_day1", "icu_day2"],
    "stroke_cva": ["icu_day1", "icu_day2", "discharge"],
    "superficial_ssi": ["ward", "discharge"],
    "unplanned_icu_admission": ["ward", "icu_day1"],
    "unplanned_intubation": ["ward", "icu_day1", "icu_day2"],
    "unplanned_or_visit": ["ward", "discharge"],
    "vap": ["icu_day2", "icu_day3"],
}

SECTION_ORDER = ["ed", "icu_day1", "icu_day2", "icu_day3", "ward", "discharge"]


def build_ed_section(meta: Dict[str, Any], rng: np.random.Generator) -> str:
    hr = int(rng.integers(90, 130))
    sbp = int(rng.integers(80, 130))
    map_val = int(sbp * 2 / 3 + 10)
    spo2 = int(rng.integers(88, 100))
    gcs = int(rng.integers(13, 15))
    template = (
        "{age}-year-old {sex_word} was brought in by EMS after {mechanism}. "
        "On arrival to the trauma bay, initial vital signs showed heart rate {hr} bpm, "
        "systolic blood pressure {sbp} mmHg (MAP approximately {map_val}), "
        "respiratory rate in the mid 20s, and oxygen saturation {spo2}% on supplemental oxygen. "
        "The primary survey was completed per ATLS protocol. "
        "The patient had a GCS of {gcs} with no obvious focal neurologic deficits. "
        "CT imaging demonstrated multiple injuries consistent with high-energy blunt trauma."
    )
    return template.format(
        age=meta["age"],
        sex_word=meta["sex_word"],
        mechanism=meta["mechanism"],
        hr=hr,
        sbp=sbp,
        map_val=map_val,
        spo2=spo2,
        gcs=gcs,
    )


def build_icu_day_section(day: int, rng: np.random.Generator) -> str:
    if day == 1:
        base = (
            "On ICU day 1, the patient remained hemodynamically tenuous but responsive to resuscitation. "
            "Continuous monitoring was maintained, and ventilatory and analgesia strategies were adjusted as needed."
        )
    elif day == 2:
        base = (
            "On ICU day 2, the patient's overall status evolved, with ongoing close monitoring of respiratory, "
            "cardiovascular, and renal function. Antibiotic therapy and venous thromboembolism prophylaxis were reassessed."
        )
    else:
        base = (
            "On ICU day 3, the team focused on stabilization, early mobilization when feasible, and evaluation for "
            "possible stepdown from the ICU depending on respiratory and hemodynamic trends."
        )
    return base


def build_ward_section(rng: np.random.Generator) -> str:
    base = (
        "After transfer out of the ICU, the patient continued recovery on the trauma ward. "
        "Pain control, physical therapy, and monitoring for hospital-acquired complications were emphasized. "
        "Nursing notes describe gradual improvement in mobility and participation in care."
    )
    return base


def build_discharge_section(rng: np.random.Generator) -> str:
    base = (
        "At the time of discharge, a detailed summary of the hospital course was documented, including injuries, "
        "operative and critical care management, and any complications encountered. "
        "Follow-up with trauma surgery and relevant specialty clinics was arranged."
    )
    return base


def create_empty_sections(meta: Dict[str, Any], rng: np.random.Generator) -> Dict[str, str]:
    sections = {
        "ed": build_ed_section(meta, rng),
        "icu_day1": build_icu_day_section(1, rng),
        "icu_day2": build_icu_day_section(2, rng),
        "icu_day3": build_icu_day_section(3, rng),
        "ward": build_ward_section(rng),
        "discharge": build_discharge_section(rng),
    }
    return sections


def inject_complication_sentences(
    sections: Dict[str, str],
    comp_flags: Dict[str, int],
    comps: Dict[str, Complication],
    rng: np.random.Generator,
    max_mentions_per_comp: int = 2,
) -> None:
    for comp_id, flag in comp_flags.items():
        if not flag:
            continue
        comp = comps.get(comp_id)
        if comp is None or not comp.positive_note_clues:
            continue
        preferred_sections = COMP_SECTION_PREFS.get(comp_id, ["icu_day1", "icu_day2", "ward"])
        n_mentions = int(rng.integers(1, max_mentions_per_comp + 1))
        n_mentions = min(n_mentions, len(preferred_sections))
        chosen_secs = rng.choice(preferred_sections, size=n_mentions, replace=False).tolist()
        for sec in chosen_secs:
            clue = rng.choice(comp.positive_note_clues)
            sections[sec] = sections[sec].rstrip() + " " + clue


def derive_stay_metadata(comp_flags: Dict[str, int], rng: np.random.Generator) -> Dict[str, Any]:
    requires_icu = 1
    requires_vent = int(
        comp_flags.get("ards", 0)
        or comp_flags.get("vap", 0)
        or comp_flags.get("unplanned_intubation", 0)
    )
    icu_los = int(rng.integers(2, 10)) if requires_icu else int(rng.integers(0, 3))
    vent_days = 0
    if requires_vent:
        vent_days = int(rng.integers(2, min(icu_los, 14) + 1))
    return {
        "icu_los_days": icu_los,
        "vent_days": vent_days,
        "ever_ventilated": int(vent_days > 0),
    }


def assemble_note(sections: Dict[str, str]) -> str:
    parts = []
    for sec in SECTION_ORDER:
        header = {
            "ed": "ED TRAUMA H&P:",
            "icu_day1": "ICU DAY 1 PROGRESS NOTE:",
            "icu_day2": "ICU DAY 2 PROGRESS NOTE:",
            "icu_day3": "ICU DAY 3 PROGRESS NOTE:",
            "ward": "WARD / FLOOR COURSE:",
            "discharge": "DISCHARGE SUMMARY:",
        }[sec]
        parts.append(header + "\n" + sections[sec].strip())
    return "\n\n".join(parts)


def paraphrase_with_gemini(
    client: Any,
    model: str,
    note_text: str,
    comp_flags: Dict[str, int],
) -> str:
    active = [cid for cid, v in comp_flags.items() if v == 1]
    if active:
        complications_str = ", ".join(active)
    else:
        complications_str = "none"
    prompt = (
        "You are a trauma ICU clinician writing a clinical note. "
        "Rewrite the following trauma hospitalization note to sound more natural, detailed, and realistic, "
        "while preserving all clinical facts. "
        "Do not remove, negate, or contradict any of the following complications, which must remain clearly "
        "inferable from the note: "
        + complications_str
        + ". "
        "Do not change whether the patient has each complication; you may elaborate but must not flip positives to negatives "
        "or negatives to positives. "
        "Return only the rewritten note text and nothing else.\n\n"
        "Original note:\n"
        + note_text
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if text is None:
            return note_text
        cleaned = text.strip()
        if not cleaned:
            return note_text
        return cleaned
    except Exception:
        return note_text


def generate_one_encounter(
    encounter_id: int,
    comps: Dict[str, Complication],
    rng: np.random.Generator,
    mean_events: float,
    gemini_client: Optional[Any] = None,
    gemini_model: Optional[str] = None,
) -> Dict[str, Any]:
    meta = sample_demographics(rng)
    comp_ids = list(comps.keys())
    comp_flags = sample_complication_flags(comp_ids, rng, mean_events=mean_events)
    stay_meta = derive_stay_metadata(comp_flags, rng)
    sections = create_empty_sections(meta, rng)
    inject_complication_sentences(sections, comp_flags, comps, rng)
    note_text = assemble_note(sections)
    if gemini_client is not None and gemini_model is not None:
        note_text = paraphrase_with_gemini(gemini_client, gemini_model, note_text, comp_flags)
    record: Dict[str, Any] = {
        "encounter_id": encounter_id,
        "age": meta["age"],
        "sex": meta["sex"],
        "mechanism": meta["mechanism"],
        "iss": meta["iss"],
        "icu_los_days": stay_meta["icu_los_days"],
        "vent_days": stay_meta["vent_days"],
        "ever_ventilated": stay_meta["ever_ventilated"],
        "note_text": note_text,
    }
    for cid in comp_ids:
        record[cid] = comp_flags[cid]
    return record


def generate_dataset(
    n: int,
    comps: Dict[str, Complication],
    seed: int,
    mean_events: float,
    gemini_client: Optional[Any] = None,
    gemini_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        records.append(
            generate_one_encounter(
                i,
                comps,
                rng,
                mean_events=mean_events,
                gemini_client=gemini_client,
                gemini_model=gemini_model,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic trauma ICU notes with NTDS/TQIP-style complications."
    )
    parser.add_argument(
        "--comp_json",
        type=str,
        required=True,
        help="Path to ntds_18_complications.json.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of synthetic encounters to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--mean_events",
        type=float,
        default=1.5,
        help="Average number of complications per encounter (Poisson mean).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="synthetic_ntds_trauma_notes.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default=None,
        help="Optional JSONL output file path.",
    )
    parser.add_argument(
        "--use_gemini",
        action="store_true",
        help="If set, use Gemini to paraphrase and polish note_text.",
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key; if not provided, the client will use GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name to use for paraphrasing.",
    )
    args = parser.parse_args()

    comps = load_complications(args.comp_json)

    gemini_client = None
    if args.use_gemini:
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed; install it with `pip install google-genai`."
            ) from e
        if args.gemini_api_key:
            gemini_client = genai.Client(api_key=args.gemini_api_key)
        else:
            gemini_client = genai.Client()

    records = generate_dataset(
        n=args.n_samples,
        comps=comps,
        seed=args.seed,
        mean_events=args.mean_events,
        gemini_client=gemini_client,
        gemini_model=args.gemini_model if args.use_gemini else None,
    )
    df = pd.DataFrame(records)
    df.to_csv(args.out_csv, index=False)
    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
