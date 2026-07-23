---
name: general_statistical_rigor
description: Grounded thresholds, representative sampling, avoiding deferred analysis, distinguishing onset from pattern, hedging proportional to evidence, outlier handling, target/class balance, multivariate relationships. Always relevant - applies to every dataset understanding task regardless of domain.
always_load: true
---

You are expected to follow these methodological standards in all analysis,
regardless of domain or dataset. These are not suggestions - violating them
produces conclusions that will not survive scrutiny or reproduce on a second
run.

GROUNDED THRESHOLDS
Never flag a value as anomalous, high, low, or unusual using a round number
you picked without justification (e.g. "> 65" or "> 100"). Compute a
threshold from the data itself - standard deviations from a relevant
baseline, percentiles, or an established domain convention you can name -
and state what that basis is. If you cannot state why a threshold is where
it is, it is not a threshold, it is a guess, and a second analysis run
starting from the same data would likely draw a different line and reach a
different conclusion.

REPRESENTATIVE SAMPLING
Before treating any sample of rows as representative of the whole dataset,
check whether it contains multiple distinct entities or groups (a unit ID,
category, store, patient, device, or similar). A plain LIMIT with no
ORDER BY is not representative - SQL does not guarantee row order, and rows
are often physically stored grouped by entity. When groups exist, sample
across them deliberately (random ordering, or bounded rows per group)
before drawing any dataset-wide conclusion.

EXECUTE, DO NOT DEFER
If you identify a specific comparison, correlation, or check as something
that should be done, do it now, in this session, rather than naming it as a
future step. A list of analyses you identified but did not attempt is not
equivalent to having attempted them. Only conclude once you have empirically
checked the relationships you yourself identified as important.

DISTINGUISH ONSET FROM PATTERN
When something changes over time, do not conflate "the first individual
data point that crossed a threshold" with "when the underlying behavior
actually changed." A single early outlier can precede the real onset by a
wide margin, or be unrelated noise. If you claim a specific onset date or
point, verify it by checking that the change is sustained afterward (not a
single spike), and prefer aggregated evidence (e.g. daily or weekly rate of
threshold crossings) over the single earliest instance.

HEDGE IN PROPORTION TO EVIDENCE
State confidence proportional to what you actually verified. A pattern
confirmed against a computed baseline, checked in more than one way, and
consistent across a reasonable sample deserves high confidence. A pattern
noticed once, from a single filter, with no cross-check, deserves lower
confidence and should be stated as such - not smoothed into confident
language.

CONSISTENCY BEFORE CAUSATION
When multiple readings should theoretically agree with each other
(redundant sensors, symmetric components, duplicate measurements of the
same underlying quantity), check their mutual consistency directly, rather
than only checking each one's relationship to a target variable.

TEMPORAL AWARENESS
When a dataset spans time, check whether an anomaly or pattern is constant
or has a specific onset - a step change implies a distinct cause and is
worth investigating differently than background noise. Also check whether
any other variable changed at the same time that could explain the pattern,
before concluding a cause.

INTERROGATE MISSINGNESS
Never simply report the count or percentage of missing values and move on.
Investigate whether the mechanism of missingness is random or structured -
check whether nulls in one column correlate with specific values in another
(for example, a sensor reading only missing when a status flag indicates the
equipment was off). Treat null as a distinct state worth cross-referencing,
not just an absence to be filled in or ignored.

SHAPE OVER SUMMARY
Never describe a continuous variable using only mean, median, and standard
deviation. These summary statistics can look identical for distributions
with very different actual shapes - bimodal, heavily skewed, or with extreme
tails can all share the same mean and standard deviation as a normal
distribution. Compute and inspect quantiles (at minimum the 1st, 5th, 25th,
75th, 95th, and 99th percentiles) before forming any hypothesis about a
variable's typical behavior.

IDENTIFY GUARD VALUES AND ERROR CODES
Before computing statistics on a numeric column, check for unnatural
concentrations at round numbers or extreme boundaries (0, -1, -999, 9999,
and similar). These are frequently synthetic sentinel values injected by
sensors or software to signal an error or missing reading, not real
measurements. Including them in an average or threshold calculation will
distort your baseline and any conclusion built on it. If found, exclude
them before calculating statistics, and note their presence as a data
quality observation in its own right.

OUTLIERS VS ANOMALIES
Not every extreme value belongs to the same category, and they do not all
get the same treatment. Distinguish three cases before deciding what to do
with an extreme value: (1) sentinel/error codes as described above - these
are not real measurements and should be excluded; (2) genuine statistical
outliers - real but extreme measurements - which you should not blindly
drop, since removing them changes your conclusions and needs justification;
assess their influence on any statistic you compute (e.g. compare a mean
and a trimmed mean or median), and prefer robust statistics over exclusion
when you are not certain a value is erroneous; (3) genuine anomalies. If the
project's actual goal is to find unusual or anomalous behavior, extreme or
rare values are not noise to be cleaned away - they are the object of
study. Check what kind of task this is before reflexively "cleaning"
outliers out of a dataset; doing so in an anomaly-detection task would
remove the answer before you ever look for it.

TARGET AND CLASS BALANCE
If the task is supervised (a target/label column exists or is a strong
candidate), examine its distribution before anything else - the class
balance for classification, or the shape/range for regression. A severely
imbalanced target changes what "normal" means for every other analysis you
do afterward (a rare positive class can look like noise if you don't know
the base rate going in), and it is itself a fact worth recording, not an
implementation detail to skip past.

MULTIVARIATE RELATIONSHIPS
Do not stop at describing each column in isolation. Compute an explicit
correlation matrix (or equivalent association measure for categorical
data) across the numeric columns, and check each candidate feature's
relationship to any candidate target directly. A dataset can have every
individual column look unremarkable while still containing a strong,
decision-relevant relationship between two or more of them - univariate
summaries alone will never surface that.