# Rights and external resources

The inspected project source did not include an explicit LICENSE file. This artifact does not manufacture a new license grant or label the project MIT/Apache licensed. Public inspection and download do not by themselves settle permission for redistribution, modification or commercial reuse; consult the authors for an explicit license if needed.

No raw third-party corpus passages, original query text, embedding arrays, downloaded model weights or dependency source trees are distributed. The experiment's own source, observations and documentation are released by the author-directed export. External inputs must be obtained directly from their sources subject to applicable terms:

- 20 Newsgroups, acquired by the archived `scikit-learn` data loader.
- `sentence-transformers/all-MiniLM-L6-v2`, with the W2D model revision recorded in source and provenance.
- NumPy, scikit-learn, sentence-transformers, PyTorch/Transformers and their dependencies.
- PyMilvus/Milvus Lite, Milvus Standalone, etcd and MinIO container images.

The archived Docker Compose file contains standard upstream example configuration. Its example development password is not an extracted private credential. Users should deploy any local reproduction according to the upstream security and licensing guidance. No claim of production deployment readiness is made.
