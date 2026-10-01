# W2R — Addendum: presenting both workloads in one table

2026-07-28, filed before the table is built. Appended to
`W2R-PREREGISTRATION.md` rather than edited into it.

## The question

Rule 5 of the preregistration bars comparing counts across the synthetic and
real-text workloads: different corpora, different dimensions, not commensurable.
It bars sentences of the form "35 in synthetic versus N in real text."

A results table with the two workloads in adjacent columns is not such a
sentence, but it invites the reader to make the comparison anyway. So the
decision has to be made explicitly rather than by drifting into it.

## The decision, and why

Both workloads go in one table, and the caption states what the reader must not
read into it. Two reasons this is safer than prose, not laxer:

1. **The near-agreement of the counts is arithmetic, and prose hides that.**
   B1's poisoned-retrieval count is set by the query rate, the measurement
   window, and how many of the eight crafted queries the poison reaches --
   six in both workloads. Those are all shared configuration. So B1 landing on
   the same median in both is close to forced, and carries no information about
   whether real text behaves like synthetic vectors. A reader who sees the two
   numbers only in separate paragraphs is MORE likely to read the agreement as
   independent corroboration, not less, because nothing tells them the counts
   were never free to disagree.

2. Sec. VII-C already states the rule in the text. A table whose caption
   repeats it puts the disclaimer where the tempting comparison happens.

## What the caption must say

That retrieval counts are set by the shared query rate, window and crafted-query
reach, so agreement between the two workloads' counts is arithmetic rather than
evidence; what the table shows is the ordering across baselines and the response
to backlog, per workload.

## What stays barred

Any sentence in the running text comparing a count from one workload with a
count from the other, in either direction, including "closely matches", "nearly
identical", and "reproduces the numbers". `check_numbers.py` already fails the
build on the sentence form; that check stays.
