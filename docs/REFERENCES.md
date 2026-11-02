# References

## Papers and datasets

[1] A. Oliner and J. Stearley. "What Supercomputers Say: A Study of Five System Logs."
In *Proc. 37th Annual IEEE/IFIP International Conference on Dependable Systems and Networks
(DSN)*, 2007. Original source of the BGL and Thunderbird logs.

[2] J. Zhu, S. He, P. He, J. Liu, and M. R. Lyu. "Loghub: A Large Collection of System Log
Datasets for AI-driven Log Analytics." In *Proc. IEEE 34th International Symposium on Software
Reliability Engineering (ISSRE)*, 2023. Dataset download: https://zenodo.org/records/8196385
and https://github.com/logpai/loghub

[3] P. He, J. Zhu, Z. Zheng, and M. R. Lyu. "Drain: An Online Log Parsing Approach with Fixed
Depth Tree." In *Proc. IEEE International Conference on Web Services (ICWS)*, 2017.

[4] Drain3, a streaming implementation of Drain. https://github.com/logpai/Drain3

[5] F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine
Learning Research*, 12:2825–2830, 2011.

[6] P. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."
In *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[7] M. Du, F. Li, G. Zheng, and V. Srikumar. "DeepLog: Anomaly Detection and Diagnosis from
System Logs through Deep Learning." In *Proc. ACM SIGSAC Conference on Computer and
Communications Security (CCS)*, 2017.

[8] W. Xu, L. Huang, A. Fox, D. Patterson, and M. I. Jordan. "Detecting Large-Scale System
Problems by Mining Console Logs." In *Proc. ACM Symposium on Operating Systems Principles
(SOSP)*, 2009. Source of the HDFS log.

[9] J. S. Vitter. "Random Sampling with a Reservoir." *ACM Transactions on Mathematical
Software*, 11(1):37–57, 1985.

[10] T. Cover and P. Hart. "Nearest Neighbor Pattern Classification." *IEEE Transactions on
Information Theory*, 13(1):21–27, 1967.

[11] Z. Nussbaum, J. X. Morris, B. Duderstadt, and A. Mulyar. "Nomic Embed: Training a
Reproducible Long Context Text Embedder." arXiv:2402.01613, 2024. The `nomic-embed-text`
embedding model.

[12] A. Grattafiori et al. "The Llama 3 Herd of Models." arXiv:2407.21783, 2024. The
`llama3.2:3b` chat model belongs to this family.

## Software

| Component | Used for | Link |
|---|---|---|
| LangGraph | Agent graph and conditional routing | https://github.com/langchain-ai/langgraph |
| Ollama | Local model serving | https://ollama.com |
| Qdrant | Vector search | https://qdrant.tech |
| MongoDB Atlas | Incident and event storage | https://www.mongodb.com/atlas |
| FastAPI | HTTP API | https://fastapi.tiangolo.com |
| Pydantic | Request/response validation | https://docs.pydantic.dev |
| scikit-learn | Evaluation metrics | https://scikit-learn.org |
| pytest | Tests | https://pytest.org |

The BGL data is used under the terms given by Loghub [2]. Cite [1] and [2] when using
results derived from it.
