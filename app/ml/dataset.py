"""Labelled training-data generator for the civic complaint classifiers.

**Why generated data.** There is no public labelled corpus of municipal complaints, and
the spec explicitly allows "generated training examples or another defensible source"
provided the data is cleaned and validated. So we build one from composable parts:

    opener + issue phrasing + location + duration + impact + closer, then noise

Each *issue phrasing* is tagged with the categories it belongs to and the priority band
it can plausibly fall into. Sampling picks the priority first and then an issue that is
compatible with it, which keeps the priority classes balanced instead of letting them
follow whatever the category mix happens to be.

**Noise is deliberate.** Real complaints arrive lowercase, unpunctuated, with typos, and
with Hindi/English code-mixing. Training on clean prose alone produces a model that looks
excellent on a held-out split and falls over on the first real submission, so a
configurable share of samples is degraded on purpose.

**Honest limitation** (repeated in the README and the evaluation report): a model trained
on generated text learns the *generator's* view of how citizens write. Real-world
accuracy will be lower than the held-out numbers suggest. The held-out split measures
consistency, not real-world truth.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from app.models.enums import ComplaintCategory as Cat
from app.models.enums import ComplaintPriority as Pri


@dataclass(frozen=True)
class Issue:
    """One way of describing a problem, plus which priorities it can carry."""

    phrase: str
    category: Cat
    priorities: tuple[Pri, ...]
    #: Optional subject noun used by templates that need one ("the {subject} is broken").
    subject: str = ""
    tags: tuple[str, ...] = field(default=())


# --------------------------------------------------------------------------- issues
# Each entry is a natural phrasing a citizen might actually use. Categories get
# distinctive vocabulary so the classifier learns real signal rather than template shape.
ISSUES: list[Issue] = [
    # ------------------------------------------------------------------ ROAD
    Issue("there is a deep pothole", Cat.ROAD, (Pri.MEDIUM, Pri.HIGH), "pothole"),
    Issue("the road surface has completely broken up", Cat.ROAD, (Pri.HIGH, Pri.CRITICAL)),
    Issue("a large crater has formed on the road", Cat.ROAD, (Pri.HIGH, Pri.CRITICAL)),
    Issue("the road is full of potholes", Cat.ROAD, (Pri.MEDIUM, Pri.HIGH)),
    Issue("the footpath tiles are broken and lifted", Cat.ROAD, (Pri.LOW, Pri.MEDIUM)),
    Issue("the speed breaker has worn away", Cat.ROAD, (Pri.LOW, Pri.MEDIUM)),
    Issue("road markings and zebra crossing have faded", Cat.ROAD, (Pri.LOW,)),
    Issue("the newly laid road has already started cracking", Cat.ROAD, (Pri.MEDIUM,)),
    Issue("construction debris has been dumped on the road", Cat.ROAD, (Pri.MEDIUM, Pri.HIGH)),
    Issue("a portion of the road has caved in", Cat.ROAD, (Pri.CRITICAL,)),
    Issue("the flyover approach road is badly damaged", Cat.ROAD, (Pri.HIGH,)),
    Issue("there is no proper road for vehicles to pass", Cat.ROAD, (Pri.HIGH,)),
    # ----------------------------------------------------------------- WATER
    Issue("there is a large water leak from the pipeline", Cat.WATER, (Pri.HIGH, Pri.CRITICAL)),
    Issue("the water main has burst", Cat.WATER, (Pri.CRITICAL,)),
    Issue("we have had no water supply", Cat.WATER, (Pri.MEDIUM, Pri.HIGH, Pri.CRITICAL)),
    Issue("the tap water is coming out muddy and dirty", Cat.WATER, (Pri.HIGH, Pri.CRITICAL)),
    Issue("drinking water smells foul", Cat.WATER, (Pri.HIGH, Pri.CRITICAL)),
    Issue("water pressure is extremely low", Cat.WATER, (Pri.LOW, Pri.MEDIUM)),
    Issue("the public water tap is leaking continuously", Cat.WATER, (Pri.MEDIUM, Pri.HIGH)),
    Issue("the overhead water tank is overflowing and wasting water", Cat.WATER, (Pri.MEDIUM,)),
    Issue("the water meter is damaged", Cat.WATER, (Pri.LOW,)),
    Issue("the borewell pump is not working", Cat.WATER, (Pri.MEDIUM, Pri.HIGH)),
    Issue("water supply comes only for a few minutes", Cat.WATER, (Pri.MEDIUM,)),
    Issue("the pipeline is leaking and water is being wasted", Cat.WATER, (Pri.MEDIUM, Pri.HIGH)),
    # ----------------------------------------------------------------- WASTE
    Issue("the garbage bin is overflowing", Cat.WASTE, (Pri.MEDIUM, Pri.HIGH)),
    Issue("garbage has not been collected", Cat.WASTE, (Pri.MEDIUM, Pri.HIGH)),
    Issue("a huge pile of rubbish is rotting", Cat.WASTE, (Pri.HIGH, Pri.CRITICAL)),
    Issue("people are dumping waste illegally", Cat.WASTE, (Pri.MEDIUM, Pri.HIGH)),
    Issue("dead animal carcass is lying", Cat.WASTE, (Pri.HIGH, Pri.CRITICAL)),
    Issue("the dustbin has been broken", Cat.WASTE, (Pri.LOW, Pri.MEDIUM)),
    Issue("sweeping has not been done", Cat.WASTE, (Pri.LOW, Pri.MEDIUM)),
    Issue("medical and plastic waste is scattered", Cat.WASTE, (Pri.HIGH, Pri.CRITICAL)),
    Issue("the garbage truck has not visited", Cat.WASTE, (Pri.MEDIUM,)),
    Issue("construction waste is piling up", Cat.WASTE, (Pri.MEDIUM,)),
    Issue("stray dogs are tearing open the garbage bags", Cat.WASTE, (Pri.MEDIUM, Pri.HIGH)),
    # ----------------------------------------------------------- ELECTRICITY
    Issue("the streetlight is not working", Cat.ELECTRICITY, (Pri.MEDIUM, Pri.HIGH)),
    Issue("all streetlights are switched off", Cat.ELECTRICITY, (Pri.HIGH,)),
    Issue("a live electric wire is hanging low", Cat.ELECTRICITY, (Pri.CRITICAL,)),
    Issue("the electricity pole is sparking", Cat.ELECTRICITY, (Pri.CRITICAL,)),
    Issue("the transformer is making loud noises and smoking", Cat.ELECTRICITY, (Pri.CRITICAL,)),
    Issue("there has been a power cut", Cat.ELECTRICITY, (Pri.MEDIUM, Pri.HIGH)),
    Issue("voltage fluctuation is damaging appliances", Cat.ELECTRICITY, (Pri.MEDIUM, Pri.HIGH)),
    Issue("the streetlight stays on during the day", Cat.ELECTRICITY, (Pri.LOW,)),
    Issue("the electric pole is leaning dangerously", Cat.ELECTRICITY, (Pri.HIGH, Pri.CRITICAL)),
    Issue("the meter box is open and exposed", Cat.ELECTRICITY, (Pri.HIGH, Pri.CRITICAL)),
    Issue("the streetlight bulb has fused", Cat.ELECTRICITY, (Pri.LOW, Pri.MEDIUM)),
    # -------------------------------------------------------------- DRAINAGE
    Issue("the drain is completely blocked", Cat.DRAINAGE, (Pri.MEDIUM, Pri.HIGH)),
    Issue("sewage is overflowing onto the road", Cat.DRAINAGE, (Pri.HIGH, Pri.CRITICAL)),
    Issue("the manhole cover is missing", Cat.DRAINAGE, (Pri.CRITICAL,)),
    Issue("there is severe waterlogging", Cat.DRAINAGE, (Pri.HIGH, Pri.CRITICAL)),
    Issue("rainwater is not draining away", Cat.DRAINAGE, (Pri.MEDIUM, Pri.HIGH)),
    Issue("the open drain smells terrible", Cat.DRAINAGE, (Pri.MEDIUM, Pri.HIGH)),
    Issue("mosquitoes are breeding in the stagnant drain water", Cat.DRAINAGE, (Pri.MEDIUM, Pri.HIGH)),
    Issue("the storm water drain is choked with silt", Cat.DRAINAGE, (Pri.MEDIUM,)),
    Issue("sewage water has entered our homes", Cat.DRAINAGE, (Pri.CRITICAL,)),
    Issue("the drain cover is cracked", Cat.DRAINAGE, (Pri.LOW, Pri.MEDIUM)),
    Issue("the nallah has not been cleaned", Cat.DRAINAGE, (Pri.MEDIUM,)),
    # ---------------------------------------------------------------- SAFETY
    Issue("the park is unsafe after dark", Cat.SAFETY, (Pri.MEDIUM, Pri.HIGH)),
    Issue("an abandoned building has become a hideout for miscreants", Cat.SAFETY, (Pri.HIGH,)),
    Issue("stray dogs are attacking people", Cat.SAFETY, (Pri.HIGH, Pri.CRITICAL)),
    Issue("the boundary wall is about to collapse", Cat.SAFETY, (Pri.CRITICAL,)),
    Issue("there is no barricade around the open excavation", Cat.SAFETY, (Pri.CRITICAL,)),
    Issue("anti-social activity happens every night", Cat.SAFETY, (Pri.MEDIUM, Pri.HIGH)),
    Issue("the school crossing has no signage", Cat.SAFETY, (Pri.MEDIUM, Pri.HIGH)),
    Issue("a big tree branch is about to fall", Cat.SAFETY, (Pri.HIGH, Pri.CRITICAL)),
    Issue("the playground equipment is broken and children can get hurt", Cat.SAFETY, (Pri.MEDIUM, Pri.HIGH)),
    Issue("there is no proper fencing around the pond", Cat.SAFETY, (Pri.HIGH,)),
    Issue("illegal parking is blocking emergency access", Cat.SAFETY, (Pri.MEDIUM, Pri.HIGH)),
    # ----------------------------------------------------------------- OTHER
    Issue("the community hall booking system is not working", Cat.OTHER, (Pri.LOW, Pri.MEDIUM)),
    Issue("the property tax receipt has not been issued", Cat.OTHER, (Pri.LOW, Pri.MEDIUM)),
    Issue("the ward office staff were unhelpful", Cat.OTHER, (Pri.LOW, Pri.MEDIUM)),
    Issue("the public notice board has not been updated", Cat.OTHER, (Pri.LOW,)),
    Issue("the birth certificate application is pending", Cat.OTHER, (Pri.MEDIUM,)),
    Issue("stray cattle are roaming in the market", Cat.OTHER, (Pri.MEDIUM,)),
    Issue("illegal encroachment has narrowed the lane", Cat.OTHER, (Pri.MEDIUM, Pri.HIGH)),
    Issue("the public toilet is locked all the time", Cat.OTHER, (Pri.MEDIUM,)),
    Issue("noise from a nearby function goes on till late night", Cat.OTHER, (Pri.LOW, Pri.MEDIUM)),
    Issue("the bus stop shelter is damaged", Cat.OTHER, (Pri.LOW, Pri.MEDIUM)),
]

# ------------------------------------------------------------------- filler phrases
LOCATIONS = [
    "near the main road", "in Ward 12", "at MG Road", "behind the bus stand",
    "opposite the government school", "near the market area", "in Gandhi Nagar",
    "at the Sector 7 junction", "near the community hall", "in the residential colony",
    "close to the railway crossing", "outside the primary health centre",
    "in the industrial area", "near the temple street", "at the third cross road",
    "in front of our apartment", "near the vegetable market", "at the bus depot",
    "on the road leading to the hospital", "in the new layout", "near the park gate",
    "at the corner of the lane", "beside the police station", "in Shivaji Nagar",
]

DURATIONS = [
    "for the past two days", "since last week", "for more than a month",
    "since yesterday", "for the last three days", "for several weeks",
    "since the rains started", "for over ten days", "since Monday",
]

OPENERS = [
    "", "", "", "",  # weighted towards no opener
    "I want to report that ", "Please note that ", "Kindly look into this - ",
    "Complaint: ", "This is to inform you that ", "Reporting an issue - ",
    "Requesting urgent action. ", "Sir/Madam, ",
]

CLOSERS = [
    "", "", "",
    " Please take action soon.", " Kindly resolve this at the earliest.",
    " Requesting the concerned department to look into it.",
    " Hoping for a quick response.", " Thank you.",
    " We have already complained twice about this.",
]

#: Impact clauses, keyed by priority. This is the main lever that separates the four
#: priority classes — the issue phrasing sets the plausible band, this sets the level.
IMPACTS: dict[Pri, list[str]] = {
    Pri.CRITICAL: [
        " This is extremely dangerous and someone could be seriously injured.",
        " It is an emergency and needs immediate attention.",
        " There is a serious risk to life and traffic has come to a standstill.",
        " Children pass this way daily and a fatal accident could happen any time.",
        " The situation is unsafe and spreading rapidly. Please send a team urgently.",
        " An accident has already happened here once. This is urgent.",
    ],
    Pri.HIGH: [
        " Many residents in the area are badly affected.",
        " The whole street is facing difficulty because of this.",
        " It is causing serious inconvenience to everyone here.",
        " Traffic is getting blocked and people are struggling.",
        " Nearly fifty families are affected by this problem.",
        " This is affecting the hospital route and needs quick action.",
    ],
    Pri.MEDIUM: [
        " It is causing inconvenience to the residents.",
        " Please arrange for repair work.",
        " We request the department to attend to it.",
        " This has been a recurring problem in our area.",
        " It would help if this is fixed soon.",
    ],
    Pri.LOW: [
        " It is a minor issue but should be fixed eventually.",
        " Not very urgent, but please add it to your list.",
        " Just bringing it to your notice for future maintenance.",
        " There is no hurry, whenever convenient.",
        " A small problem, nothing serious for now.",
    ],
}

#: Code-mixed fragments — common in Indian civic complaint channels. Including a small
#: share teaches the character n-gram features to tolerate them.
CODE_MIXED = [
    " Kripya jaldi theek karwaiye.",
    " Bahut problem ho rahi hai.",
    " Please dekh lijiye.",
    " Yahan bohot dikkat hai.",
    " Turant action lijiye.",
]

_TYPO_TARGETS = "aeiounrst"


def _inject_typos(text: str, rng: random.Random, rate: float = 0.02) -> str:
    """Randomly drop, duplicate or swap characters to mimic phone typing."""
    chars = list(text)
    for index in range(len(chars) - 1, -1, -1):
        if chars[index] in _TYPO_TARGETS and rng.random() < rate:
            roll = rng.random()
            if roll < 0.4:
                del chars[index]
            elif roll < 0.7:
                chars.insert(index, chars[index])
            elif index > 0:
                chars[index - 1], chars[index] = chars[index], chars[index - 1]
    return "".join(chars)


def _degrade(text: str, rng: random.Random) -> str:
    """Apply realistic input noise to a share of samples."""
    roll = rng.random()
    if roll < 0.18:
        text = text.lower()
    if rng.random() < 0.12:
        text = text.replace(".", "").replace(",", "")
    if rng.random() < 0.15:
        text = _inject_typos(text, rng)
    if rng.random() < 0.08:
        text = text + rng.choice(CODE_MIXED)
    if rng.random() < 0.06:
        text = text.upper()
    return text


def _compose(
    issue: Issue, priority: Pri, rng: random.Random, impacts: dict[Pri, list[str]]
) -> str:
    """Assemble one complaint from its parts."""
    parts = [rng.choice(OPENERS), issue.phrase]

    if rng.random() < 0.85:
        parts.append(" " + rng.choice(LOCATIONS))
    if rng.random() < 0.45:
        parts.append(" " + rng.choice(DURATIONS))
    parts.append(".")

    # The impact clause is the primary priority signal; occasionally omit it so the
    # model cannot rely on it exclusively and must read the issue phrasing too.
    available_impacts = impacts.get(priority) or []
    if available_impacts and rng.random() < 0.88:
        parts.append(rng.choice(available_impacts))
    if rng.random() < 0.55:
        parts.append(rng.choice(CLOSERS))

    text = "".join(parts).strip()
    return text[0].upper() + text[1:] if text else text


def build_dataset(
    n_samples: int = 6000,
    *,
    seed: int = 42,
    noise_rate: float = 0.35,
    issues: list[Issue] | None = None,
    impacts: dict[Pri, list[str]] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Generate a labelled dataset.

    Returns ``(texts, category_labels, priority_labels)``. Sampling is stratified by
    priority first so all four priority classes are represented roughly equally,
    then by an issue compatible with that priority.

    Args:
        n_samples: How many complaints to generate.
        seed: RNG seed — the same seed always produces the same dataset.
        noise_rate: Share of samples degraded with typos/casing/code-mixing.
        issues: Restrict generation to this subset of issue phrasings. Used by the
            generalisation evaluation to build a test set whose phrasings the model has
            never seen — see ``split_issues``.
        impacts: Restrict generation to this subset of impact clauses, likewise.
    """
    rng = random.Random(seed)
    issue_pool = issues if issues is not None else ISSUES
    impact_pool = impacts if impacts is not None else IMPACTS

    by_priority: dict[Pri, list[Issue]] = {priority: [] for priority in Pri}
    for issue in issue_pool:
        for priority in issue.priorities:
            by_priority[priority].append(issue)

    texts: list[str] = []
    categories: list[str] = []
    priorities: list[str] = []
    seen: set[str] = set()

    # Only cycle through priorities that some available issue can actually carry —
    # otherwise a restricted issue pool would spin until max_attempts.
    priority_cycle = [priority for priority in Pri if by_priority[priority]]
    if not priority_cycle:
        raise ValueError("No issues available for any priority band.")

    attempts = 0
    max_attempts = n_samples * 12

    while len(texts) < n_samples and attempts < max_attempts:
        attempts += 1
        priority = priority_cycle[len(texts) % len(priority_cycle)]
        issue = rng.choice(by_priority[priority])

        text = _compose(issue, priority, rng, impact_pool)
        if rng.random() < noise_rate:
            text = _degrade(text, rng)

        # Exact duplicates add nothing and inflate held-out scores by leaking
        # identical strings across the train/test split.
        key = text.lower().strip()
        if key in seen:
            continue
        seen.add(key)

        texts.append(text)
        categories.append(issue.category.value)
        priorities.append(priority.value)

    return texts, categories, priorities


def split_issues(
    *, seed: int = 42, holdout_ratio: float = 0.25
) -> tuple[list[Issue], list[Issue]]:
    """Partition the issue phrasings into seen / unseen sets.

    This is what makes the evaluation honest. A plain random split over generated text
    puts the *same* issue phrasing in both train and test, so the classifier only has to
    memorise ~80 phrases and scores a meaningless 1.000. Holding out entire phrasings
    instead asks the real question: does the model recognise "the drain is choked with
    silt" as Drainage when it has only ever been trained on other ways of saying it?

    At least two issues per category are always kept in train, so no category is left
    with nothing to learn from.
    """
    rng = random.Random(seed)
    by_category: dict[Cat, list[Issue]] = {}
    for issue in ISSUES:
        by_category.setdefault(issue.category, []).append(issue)

    train_issues: list[Issue] = []
    holdout_issues: list[Issue] = []

    for category_issues in by_category.values():
        shuffled = category_issues[:]
        rng.shuffle(shuffled)
        n_holdout = min(
            max(1, int(round(len(shuffled) * holdout_ratio))),
            max(0, len(shuffled) - 2),
        )
        holdout_issues.extend(shuffled[:n_holdout])
        train_issues.extend(shuffled[n_holdout:])

    return train_issues, holdout_issues


def split_impacts(
    *, seed: int = 42, holdout_ratio: float = 0.34
) -> tuple[dict[Pri, list[str]], dict[Pri, list[str]]]:
    """Partition the impact clauses the same way, for the priority model.

    Priority is driven mainly by the impact clause, so holding out issue phrasings alone
    would leave the priority model's real signal fully visible in training. Holding out
    clauses too makes the priority generalisation number meaningful.
    """
    rng = random.Random(seed + 1)
    train_impacts: dict[Pri, list[str]] = {}
    holdout_impacts: dict[Pri, list[str]] = {}

    for priority, clauses in IMPACTS.items():
        shuffled = clauses[:]
        rng.shuffle(shuffled)
        n_holdout = min(max(1, int(round(len(shuffled) * holdout_ratio))), len(shuffled) - 1)
        holdout_impacts[priority] = shuffled[:n_holdout]
        train_impacts[priority] = shuffled[n_holdout:]

    return train_impacts, holdout_impacts


def dataset_summary(categories: list[str], priorities: list[str]) -> dict[str, dict[str, int]]:
    """Class counts, used by the training report to prove the split is balanced."""

    def counts(values: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for value in values:
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))

    return {"category": counts(categories), "priority": counts(priorities)}
