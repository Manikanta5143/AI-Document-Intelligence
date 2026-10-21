# Ground-truth audit

Temperature matrix follows the user-requested values `[0.0, 0.05, 0.1, 0.2, 0.3]`. Bundle manifest values 0.23 and 0.4 are excluded.

Three questions per document: Q01 short factual; Q13 long cross-section; Q11 unanswerable. Q01 and every Q13 atomic fact were checked against extracted text. Q11 is supported by explicit statements that claims-approval accuracy/real claim rates are not provided.

## synthetic_insurance_policy_1_page.pdf — 1 pages
Extracted characters: 2119; chunk count: 3; chunk_size=1000; overlap=200
SHA-256: `50e3e625aec3e0031fa72e1f886ceeb3d4658409c21afb1e7a1553624c333561`
Chunk ID / page / Q01 relevant / Q13 relevant:
- `0` / 1 / True / True
- `1` / 1 / False / True
- `2` / 1 / False / False

## synthetic_insurance_policy_2_pages.pdf — 2 pages
Extracted characters: 4751; chunk count: 7; chunk_size=1000; overlap=200
SHA-256: `09c26b0107b4af555bd4feee69d0cc2b388980a851dd16729b0b6ca6c953efd8`
Chunk ID / page / Q01 relevant / Q13 relevant:
- `0` / 1 / True / True
- `1` / 1 / False / True
- `2` / 1 / False / True
- `3` / 2 / True / False
- `4` / 2 / False / False
- `5` / 2 / False / False
- `6` / 2 / False / False

## synthetic_insurance_policy_5_pages.pdf — 5 pages
Extracted characters: 12207; chunk count: 16; chunk_size=1000; overlap=200
SHA-256: `5b24b93ee0861fda4ee688e41df585d7b1661e5e82974de80fc9360675b7b229`
Chunk ID / page / Q01 relevant / Q13 relevant:
- `0` / 1 / True / True
- `1` / 1 / False / True
- `2` / 1 / False / True
- `3` / 2 / False / True
- `4` / 2 / False / True
- `5` / 2 / False / True
- `6` / 3 / False / False
- `7` / 3 / False / True
- `8` / 3 / False / False
- `9` / 4 / False / True
- `10` / 4 / False / True
- `11` / 4 / False / True
- `12` / 4 / False / False
- `13` / 5 / False / False
- `14` / 5 / False / False
- `15` / 5 / False / False

## synthetic_insurance_policy_6_pages.pdf — 6 pages
Extracted characters: 14926; chunk count: 20; chunk_size=1000; overlap=200
SHA-256: `1382a245e5c119f764f208ecd9703d7e7dc20cdb0ad7811bf8b5e151076727b5`
Chunk ID / page / Q01 relevant / Q13 relevant:
- `0` / 1 / True / True
- `1` / 1 / False / True
- `2` / 1 / False / True
- `3` / 2 / False / True
- `4` / 2 / False / True
- `5` / 2 / False / True
- `6` / 3 / False / False
- `7` / 3 / False / True
- `8` / 3 / False / False
- `9` / 4 / False / True
- `10` / 4 / False / True
- `11` / 4 / False / True
- `12` / 4 / False / False
- `13` / 5 / False / False
- `14` / 5 / False / False
- `15` / 5 / False / False
- `16` / 6 / False / True
- `17` / 6 / False / False
- `18` / 6 / False / False
- `19` / 6 / False / False

## synthetic_insurance_policy_7_pages.pdf — 7 pages
Extracted characters: 17637; chunk count: 24; chunk_size=1000; overlap=200
SHA-256: `5eb952e16aba9f77bc31ca3c7c59440de0af6037392d5d28418a192dc6af8eea`
Chunk ID / page / Q01 relevant / Q13 relevant:
- `0` / 1 / True / True
- `1` / 1 / False / True
- `2` / 1 / False / True
- `3` / 2 / False / True
- `4` / 2 / False / True
- `5` / 2 / False / True
- `6` / 3 / False / False
- `7` / 3 / False / True
- `8` / 3 / False / False
- `9` / 4 / False / True
- `10` / 4 / False / True
- `11` / 4 / False / True
- `12` / 4 / False / False
- `13` / 5 / False / False
- `14` / 5 / False / False
- `15` / 5 / False / False
- `16` / 6 / False / True
- `17` / 6 / False / False
- `18` / 6 / False / False
- `19` / 6 / False / False
- `20` / 7 / False / False
- `21` / 7 / False / False
- `22` / 7 / False / False
- `23` / 7 / False / False

Relevance mapping is deterministic: Q01 chunks contain the policy number; Q13 chunks contain at least one verified atomic fact. Q11 has no relevant answer chunk; retrieval relevance metrics are N/A.