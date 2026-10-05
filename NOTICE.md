# Rights and external resources

As authorized for the 2026-10-04 licensing release, contributor-owned software is licensed under MIT, and contributor-owned experimental data and independent documentation are licensed under CC BY 4.0. `LICENSE.md` defines the scope; `LICENSES/MIT.txt` supplies the complete MIT text; `LICENSES/CC-BY-4.0.md` provides the CC BY notice, attribution guidance and official legal-code link. These grants apply only to rights held by the contributors.

The IEEE/CBDCom paper is excluded from both grants, including PDF, LaTeX manuscript, accepted/camera-ready and publisher versions, and paper content extracted or reproduced elsewhere. The paper is distributed separately and is not part of this reproducibility artifact. Contributor-owned plotting software and separately released experimental observations follow the scope in `LICENSE.md`.

No raw third-party corpus passages, original query text, embedding arrays, downloaded model weights or dependency source trees are distributed. The experiment's own source, observations and documentation are released by the author-directed export. External inputs must be obtained directly from their sources subject to applicable terms:

- 20 Newsgroups, acquired by the archived `scikit-learn` data loader.
- `sentence-transformers/all-MiniLM-L6-v2`, with the W2D model revision recorded in source and provenance.
- NumPy, scikit-learn, sentence-transformers, PyTorch/Transformers and their dependencies.
- PyMilvus/Milvus Lite, Milvus Standalone, etcd and MinIO container images.

Third-party material, including dependencies, IEEEtran and other upstream templates/class files, upstream configuration excerpts, quotations, protected corpus/dataset content, model artifacts and trademarks, remains subject to its own terms and is not relicensed by this release. A contributor-owned observation or label does not grant rights in the external material from which it was derived.

The archived Docker Compose file contains standard upstream example configuration. Its example development password is not an extracted private credential. Only contributor-owned portions fall within the MIT grant; upstream portions retain their applicable terms. Users should deploy any local reproduction according to the upstream security and licensing guidance. No claim of production deployment readiness is made.

The copyright notice names the Bounded Provisional Visibility contributors collectively; it does not assert ownership for any employer or other organization. See `RELEASE-LICENSING-2026-10-04.md` for the licensing-only change record.
