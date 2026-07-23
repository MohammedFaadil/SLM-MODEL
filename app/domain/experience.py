"""Deterministic experience math.

Years-of-experience (total and per-skill) is computed in code, not guessed by the
model: parse each role's start/end into absolute months, merge overlapping/adjacent
intervals (so concurrent roles aren't double-counted), and sum. Per-skill years =
merged intervals of the roles that evidence that skill. Reproducible and auditable;
the model only narrates these numbers.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import List, Optional, Tuple

from .schemas import CandidateProfile, ExperienceItem, SkillExperience
from .skills import normalize_skill

_MONTHS = {
    "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
    "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11,
}
_PRESENT = ("present", "current", "now", "till date", "to date", "ongoing", "till now")

_MIN_YEAR = 1950


def _today_index() -> int:
    d = _dt.date.today()
    return d.year * 12 + (d.month - 1)


def _valid_year(y: int) -> bool:
    return _MIN_YEAR <= y <= (_dt.date.today().year + 1)


def parse_month(s: Optional[str]) -> Optional[Tuple[int, bool]]:
    """Parse a resume date into (absolute_month_index, had_explicit_month).

    absolute_month_index = year*12 + (month-1). Returns None if unparseable.
    """
    if not s:
        return None
    t = str(s).strip().lower()
    if not t:
        return None
    if any(w in t for w in _PRESENT):
        return _today_index(), True

    # Month-name + year, e.g. "Jan 2020", "September, 2019".
    m = re.search(r"([a-z]{3,9})\.?\s*,?\s*(\d{4})", t)
    if m and m.group(1)[:3] in _MONTHS and _valid_year(int(m.group(2))):
        return int(m.group(2)) * 12 + _MONTHS[m.group(1)[:3]], True

    # MM/YYYY or MM-YYYY.
    m = re.search(r"\b(1[0-2]|0?[1-9])[/\-](\d{4})\b", t)
    if m and _valid_year(int(m.group(2))):
        return int(m.group(2)) * 12 + (int(m.group(1)) - 1), True

    # YYYY-MM or YYYY/MM.
    m = re.search(r"\b(\d{4})[/\-](1[0-2]|0?[1-9])\b", t)
    if m and _valid_year(int(m.group(1))):
        return int(m.group(1)) * 12 + (int(m.group(2)) - 1), True

    # Year only.
    m = re.search(r"\b(19|20)\d{2}\b", t)
    if m and _valid_year(int(m.group(0))):
        return int(m.group(0)) * 12, False

    return None


def role_interval(start: Optional[str], end: Optional[str]) -> Optional[Tuple[int, int]]:
    """Inclusive [start_month, end_month] absolute-month interval, or None."""
    ps = parse_month(start)
    if ps is None:
        return None
    start_idx = ps[0]

    pe = parse_month(end)
    if pe is None:
        end_idx = _today_index()  # ongoing / missing end
    else:
        end_idx, had_month = pe
        if not had_month:
            end_idx += 11  # a year-only end means through December
    end_idx = min(end_idx, _today_index())  # no future dates
    if end_idx < start_idx:
        end_idx = start_idx
    return start_idx, end_idx


def _merge_months(intervals: List[Tuple[int, int]]) -> int:
    """Total months covered by the union of inclusive intervals."""
    if not intervals:
        return 0
    ivs = sorted(intervals)
    total = 0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s <= cur_e + 1:  # overlapping or adjacent -> merge
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s + 1
            cur_s, cur_e = s, e
    total += cur_e - cur_s + 1
    return total


def total_years(experience: List[ExperienceItem]) -> Optional[float]:
    intervals = [iv for r in experience if (iv := role_interval(r.start, r.end))]
    if not intervals:
        return None
    return round(_merge_months(intervals) / 12.0, 1)


def annotate_durations(experience: List[ExperienceItem]) -> None:
    for r in experience:
        iv = role_interval(r.start, r.end)
        if iv:
            r.duration_years = round((iv[1] - iv[0] + 1) / 12.0, 1)


def _role_has_skill(role: ExperienceItem, skill_norm: str) -> bool:
    if len(skill_norm) < 2:
        return False
    for rs in role.skills or []:
        n = normalize_skill(rs)
        if n == skill_norm or skill_norm in n or n in skill_norm:
            return True
    text = ((role.title or "") + " " + " ".join(role.highlights or [])).lower()
    return re.search(r"(?<![a-z0-9])" + re.escape(skill_norm) + r"(?![a-z0-9])", text) is not None


def skill_experience(profile: CandidateProfile) -> List[SkillExperience]:
    """Per-skill evidenced years, from the roles that mention each skill."""
    out: List[SkillExperience] = []
    for skill in profile.skills:
        sk = normalize_skill(skill)
        intervals = [
            iv for role in profile.experience
            if _role_has_skill(role, sk) and (iv := role_interval(role.start, role.end))
        ]
        if intervals:
            years = round(_merge_months(intervals) / 12.0, 1)
            out.append(SkillExperience(skill=skill, years=years, evidenced=True))
        else:
            out.append(SkillExperience(skill=skill, years=0.0, evidenced=False))
    out.sort(key=lambda x: (x.evidenced, x.years), reverse=True)
    return out
