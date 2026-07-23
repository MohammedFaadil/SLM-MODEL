from app.domain.experience import parse_month, skill_experience, total_years
from app.domain.schemas import CandidateProfile, ExperienceItem


def test_parse_month_formats():
    assert parse_month("Jan 2020")[0] == 2020 * 12 + 0
    assert parse_month("January 2020")[0] == 2020 * 12 + 0
    assert parse_month("03/2019")[0] == 2019 * 12 + 2
    assert parse_month("2019-06")[0] == 2019 * 12 + 5
    assert parse_month("2021")[0] == 2021 * 12
    assert parse_month("2021")[1] is False          # year-only
    assert parse_month("Jan 2020")[1] is True        # explicit month
    assert parse_month("garbage") is None
    assert parse_month("Present")[0] == parse_month("current")[0]


def _profile():
    return CandidateProfile(
        skills=["python", "aws", "docker"],
        experience=[
            ExperienceItem(title="ML Engineer", start="Jan 2018", end="Dec 2019",
                           skills=["python", "aws"]),
            ExperienceItem(title="Senior ML Engineer", start="Jan 2020", end="Dec 2021",
                           skills=["python", "docker"]),
        ],
    )


def test_total_years_merges_contiguous():
    # Jan 2018 -> Dec 2021 continuous = 48 months = 4.0 years.
    assert total_years(_profile().experience) == 4.0


def test_per_skill_years():
    se = {s.skill: s for s in skill_experience(_profile())}
    assert se["python"].years == 4.0   # both roles, contiguous
    assert se["aws"].years == 2.0      # first role only
    assert se["docker"].years == 2.0   # second role only
    assert all(s.evidenced for s in se.values())


def test_overlapping_roles_not_double_counted():
    prof = CandidateProfile(
        skills=["python"],
        experience=[
            ExperienceItem(start="Jan 2019", end="Dec 2020", skills=["python"]),
            ExperienceItem(start="Jan 2020", end="Dec 2021", skills=["python"]),  # overlaps 2020
        ],
    )
    # Union Jan 2019 -> Dec 2021 = 36 months = 3.0 years (not 4).
    assert total_years(prof.experience) == 3.0
    assert skill_experience(prof)[0].years == 3.0


def test_education_role_excluded_but_internship_counts():
    prof = CandidateProfile(
        skills=["python"],
        experience=[
            ExperienceItem(title="Software Engineer", start="Jan 2021", end="Dec 2022",
                           skills=["python"]),                       # 2y job
            ExperienceItem(title="Software Engineering Intern", start="Jan 2020",
                           end="Dec 2020", skills=["python"]),       # 1y internship (counts)
            ExperienceItem(title="B.Tech Computer Science", start="Jan 2016",
                           end="Dec 2019", skills=["python"]),       # degree (excluded)
        ],
    )
    # 2y job + 1y internship = 3y; the 4y degree is NOT counted.
    assert total_years(prof.experience) == 3.0
    assert skill_experience(prof)[0].years == 3.0


def test_skill_evidenced_from_highlights_when_role_skills_missing():
    prof = CandidateProfile(
        skills=["kubernetes"],
        experience=[ExperienceItem(start="Jan 2021", end="Dec 2021",
                                   highlights=["Deployed services on Kubernetes"])],
    )
    se = skill_experience(prof)
    assert se[0].skill == "kubernetes" and se[0].evidenced and se[0].years == 1.0
