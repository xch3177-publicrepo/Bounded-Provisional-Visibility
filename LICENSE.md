# Bounded Provisional Visibility — license scope
Effective for this public artifact release: 2026-10-04

Copyright (c) 2026 Bounded Provisional Visibility contributors

This artifact uses separate licenses for software and for experiment data
and independent documentation. The grants below apply only to material for
which the contributors hold the relevant rights, subject to the exclusions
in this file and NOTICE.md. They do not require material to be licensed
under both licenses at once.

1. Software: MIT

Contributor-owned software, including Python programs, tests, verification
helpers, plotting programs, and accompanying software configuration and
dependency specifications, is licensed under the MIT License. The complete
license text is in LICENSES/MIT.txt.

This scope includes contributor-owned software in tools/, source/prototype/,
source/paper/, and historical-runtime/. Comments and docstrings integral to
those software files follow MIT. Software configuration includes the owned
portions of config_manifest.json, Docker Compose files, requirements files,
and .gitignore. Third-party components and upstream material are excluded as
described below.

2. Experimental data and independent documentation: CC BY 4.0

Contributor-owned experiment observations, scores, labels, derived results,
result logs, and associated provenance metadata are licensed under Creative
Commons Attribution 4.0 International (CC BY 4.0), to the extent copyright
or similar rights apply and are held by the contributors.

This scope includes the owned material in source/prototype/results/ and
data-provenance/, plus PUBLIC_AUTHORITY.json, PUBLIC_PROVENANCE.json and
SHA256SUMS. Independent documentation, including README.md, NOTICE.md,
SANITIZATION.md, VALIDATION.md, release notes, preregistrations, amendments
and run sheets, follows CC BY 4.0. Code within a software file remains in
the MIT scope above; third-party quotations remain excluded.

The CC BY 4.0 notice, attribution guidance and controlling legal-code link
are in LICENSES/CC-BY-4.0.md:
https://creativecommons.org/licenses/by/4.0/legalcode.en

3. Explicit exclusions

Neither grant applies to the IEEE/CBDCom manuscript or publication, whether
PDF, LaTeX source, accepted manuscript, camera-ready manuscript, publisher
version, or an extracted/reproduced part of the paper. IEEE publication
templates and class files, including IEEEtran, are also excluded. Paper figures and
tables as publication content are excluded; contributor-owned plotting
software and separately released experimental data remain covered by the
applicable grants above. The paper is not included in this artifact.

Neither grant relicenses third-party material: corpus passages and query
text, third-party datasets or protected portions of their metadata, model
weights or model artifacts, embedding caches, dependencies, container images,
upstream templates/configuration excerpts, third-party code or quotations,
or trademarks. Such material remains subject to its own terms, whether
referenced, separately downloaded, or present as a clearly identified
third-party portion. No ownership of third-party inputs is asserted by
releasing observations derived from them.

The standard MIT and CC BY 4.0 texts retain their own terms; this scope
statement does not add restrictions to either standard license. If a file
has a specific third-party notice or license, that notice governs the
third-party portion. No rights in an employer's or another organization's
property are asserted by the collective contributor copyright notice.
