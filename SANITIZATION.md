# Public export transformations

This is an allowlist export of the experiment subtree at source revision `f7d2e90bf27da111bf52b9d86a2c8dd7b182cbcb`, plus four manuscript verification/plotting scripts. It is not a clone of the private repository.

- Included: tracked Python modules, configuration, requirements, preregistrations/amendments, all tracked result/attempt/archive JSON and experiment logs, and source-only historical runtime snapshots named by recorded commits.
- Excluded: unrelated projects/papers, original Git objects/history/config/remotes, compiled Python caches, raw input/embedding NPZ files, scorer input passages, model caches and model weights.
- Replaced: private home paths and workstation/account metadata with portable markers. A relative artifact suffix is retained when needed to locate evidence. Generic localhost/loopback runtime defaults and public upstream development credentials remain in archived code as functional configuration; they identify no private host/account.
- Redacted inside otherwise released JSON: raw query text, text passages and associated input embedding vectors. A marker and SHA256 of the original input value replaces each such leaf. These are inputs, not experiment outcomes. Redacted material must be independently reacquired before runtime regeneration.
- Preserved: all finite numeric/Boolean/null observation leaves outside the explicitly enumerated input redactions. Source and public numerical projections were compared during export. Original NaN/Infinity in non-authoritative historical artifacts are retained as explicit `_public_nonfinite` tags so public JSON remains strict; they are not converted to zero or silently treated as successful data.
- Canonicalized: JSON formatting/key order. Even a JSON with unchanged semantic content can have a new byte hash. The source SHA and public SHA are separate fields.

`PUBLIC_PROVENANCE.json` lists every exported file, original source SHA, public SHA, source commit, numerical projection hash and leaf count, exact input-redaction JSON pointers, and metadata-transformation locations. It also lists intentionally excluded source files with hashes. The record is an export attestation plus a checkable public distribution, not a proof that privately withheld bytes can be reconstructed from a hash.

`PUBLIC_AUTHORITY.json` maps each claim family to released files. Embedded historical SHA references are left intact as records of the original experiment. The public verifier checks the link from those historical hashes to the corresponding source SHA in the export record and verifies the new public bytes independently.

Runtime snapshots retain the original source bytes whenever no sensitive string needed removal. In particular, the accepted E3 runtime snapshot's six components remain suitable for the original runtime fingerprint comparison. Every file's `byte_identical_to_source` flag states whether that stronger comparison is possible.
