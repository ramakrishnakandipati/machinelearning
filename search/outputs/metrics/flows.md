### End-to-end flows (per-stage metrics against simulator ground truth)

Each example runs the real funnel. Stage variants share the same 200 hybrid candidates: **A** retrieval order, **B** + LTR ensemble, **C** + hard rules & penalties, **D** + margin re-rank, **E** + MMR (final page). Metrics are @10; revenue/margin are expected values per page from the latent model.

#### 1 · Vocabulary gap

Colloquial shopper language that never appears in the catalog. Stage 1 expands it; hybrid retrieval bridges the gap.

| Example | Stage 1: corrected / intent / facet acc. | Stage 2: relevant in 200 (R@200 sparse/dense/hybrid) | NDCG A→B→E | Margin A→E | Unfulfillable A→E | ms total |
|---|---|---|---|---|---|---|
| waterproof trail kicks | `waterproof trail kicks` / attribute_specific (99%) / 100% | 187/200 (0.93/0.93/0.94) | 1.000→1.000→1.000 | 67.7→161.4 | 0→0 | 65 |
| breathable gym tee | `breathable gym tee` / attribute_specific (99%) / 100% | 183/200 (0.92/0.91/0.92) | 1.000→1.000→1.000 | 25.9→50.7 | 0→0 | 63 |
| office chic heels | `office chic heels` / attribute_specific (97%) / 100% | 135/200 (0.93/0.52/0.68) | 1.000→1.000→1.000 | 49.7→124.8 | 1→0 | 61 |
| warm winter puffer | `warm winter puffer` / attribute_specific (100%) / 100% | 167/200 (0.92/0.71/0.83) | 1.000→1.000→1.000 | 88.4→228.1 | 0→0 | 61 |

Mean over this flow's examples:

| Stage | NDCG@10 | Precision@10 | Exp. revenue | Exp. margin | Unfulfillable | Avg return prob |
|---|---|---|---|---|---|---|
| A retrieval order | 1.000 | 1.000 | 154.1 | 57.9 | 0.2 | 0.201 |
| B + LTR ensemble | 1.000 | 1.000 | 260.9 | 109.8 | 0.2 | 0.174 |
| C + hard rules & penalties | 1.000 | 1.000 | 239.2 | 108.6 | 0.0 | 0.113 |
| D + margin re-rank | 1.000 | 1.000 | 285.0 | 142.1 | 0.0 | 0.096 |
| E + MMR (final page) | 1.000 | 1.000 | 286.1 | 141.2 | 0.0 | 0.101 |

#### 2 · Typos + price constraint

Misspellings are fixed against catalog vocabulary and 'under $N' becomes a hard price filter.

| Example | Stage 1: corrected / intent / facet acc. | Stage 2: relevant in 200 (R@200 sparse/dense/hybrid) | NDCG A→B→E | Margin A→E | Unfulfillable A→E | ms total |
|---|---|---|---|---|---|---|
| gortex snekers under $100 | `gore-tex sneakers under $100` / attribute_specific (100%) / 100% | 29/59 (0.49/0.34/0.49) | 0.571→0.816→0.919 | 30.7→70.5 | 0→0 | 57 |
| breathble gym tee under $30 | `breathable gym tee under $30` / attribute_specific (100%) / 100% | 55/115 (0.45/0.43/0.48) | 0.656→0.964→1.000 | 9.2→16.5 | 0→0 | 62 |
| warm pufer jaket below $200 | `warm puffer jacket below $200` / attribute_specific (100%) / 100% | 85/199 (0.46/0.39/0.43) | 0.894→0.671→0.724 | 70.7→121.3 | 0→0 | 63 |
| legings with pockets under $60 | `leggings with pockets under $60` / attribute_specific (100%) / 100% | 110/200 (0.52/0.54/0.55) | 0.776→1.000→1.000 | 21.4→43.3 | 0→0 | 57 |

Mean over this flow's examples:

| Stage | NDCG@10 | Precision@10 | Exp. revenue | Exp. margin | Unfulfillable | Avg return prob |
|---|---|---|---|---|---|---|
| A retrieval order | 0.724 | 0.475 | 94.0 | 33.0 | 0.0 | 0.218 |
| B + LTR ensemble | 0.863 | 0.825 | 145.6 | 60.9 | 0.8 | 0.192 |
| C + hard rules & penalties | 0.912 | 0.925 | 132.8 | 57.4 | 0.0 | 0.154 |
| D + margin re-rank | 0.912 | 0.900 | 137.4 | 61.9 | 0.0 | 0.148 |
| E + MMR (final page) | 0.911 | 0.900 | 138.0 | 62.9 | 0.0 | 0.146 |

#### 3 · Navigational (brand) intent

Brand-led queries: the intent classifier flags navigational intent, brand features dominate, and the brand diversity cap is lifted.

| Example | Stage 1: corrected / intent / facet acc. | Stage 2: relevant in 200 (R@200 sparse/dense/hybrid) | NDCG A→B→E | Margin A→E | Unfulfillable A→E | ms total |
|---|---|---|---|---|---|---|
| aeroflux running shoes | `aeroflux running shoes` / navigational (63%) / 100% | 143/143 (1.00/0.99/1.00) | 1.000→1.000→1.000 | 69.7→126.5 | 2→0 | 56 |
| terrapeak jackets | `terrapeak jackets` / attribute_specific (67%) / 100% | 188/200 (0.92/0.94/0.94) | 1.000→1.000→1.000 | 79.4→279.9 | 0→0 | 59 |
| hearthly bedding | `hearthly bedding` / attribute_specific (94%) / 100% | 78/78 (1.00/1.00/1.00) | 1.000→1.000→1.000 | 60.3→114.3 | 0→0 | 54 |
| voltra headphones | `voltra headphones` / navigational (60%) / 100% | 64/64 (1.00/1.00/1.00) | 1.000→1.000→1.000 | 73.3→143.1 | 0→0 | 56 |

Mean over this flow's examples:

| Stage | NDCG@10 | Precision@10 | Exp. revenue | Exp. margin | Unfulfillable | Avg return prob |
|---|---|---|---|---|---|---|
| A retrieval order | 1.000 | 1.000 | 187.1 | 70.7 | 0.5 | 0.112 |
| B + LTR ensemble | 1.000 | 1.000 | 281.1 | 117.7 | 0.5 | 0.097 |
| C + hard rules & penalties | 1.000 | 1.000 | 279.8 | 119.0 | 0.0 | 0.086 |
| D + margin re-rank | 1.000 | 1.000 | 386.4 | 167.3 | 0.0 | 0.080 |
| E + MMR (final page) | 1.000 | 1.000 | 387.3 | 166.0 | 0.0 | 0.082 |

#### 4 · Regional stock & BOPIS

Buy-online-pick-up-in-store: only items stocked at (or near) the shopper's store may be shown. Compare unfulfillable results before vs after rules.

| Example | Stage 1: corrected / intent / facet acc. | Stage 2: relevant in 200 (R@200 sparse/dense/hybrid) | NDCG A→B→E | Margin A→E | Unfulfillable A→E | ms total |
|---|---|---|---|---|---|---|
| running shoes · BOPIS Chicago (S3) | `running shoes` / navigational (43%) / 100% | 187/200 (0.93/0.94/0.94) | 1.000→1.000→1.000 | 84.1→136.0 | 6→0 | 56 |
| winter boots · BOPIS Denver (S5) | `warm winter boots` / attribute_specific (98%) / 100% | 151/200 (0.66/0.87/0.76) | 0.943→1.000→1.000 | 87.6→133.1 | 2→0 | 60 |
| office chair · BOPIS Austin (S2) | `back friendly desk chair` / attribute_specific (99%) / 100% | 90/200 (0.44/0.51/0.45) | 0.378→0.729→0.659 | 34.8→155.1 | 3→0 | 61 |

Mean over this flow's examples:

| Stage | NDCG@10 | Precision@10 | Exp. revenue | Exp. margin | Unfulfillable | Avg return prob |
|---|---|---|---|---|---|---|
| A retrieval order | 0.774 | 0.733 | 173.4 | 68.8 | 3.7 | 0.172 |
| B + LTR ensemble | 0.910 | 0.867 | 288.6 | 114.8 | 3.7 | 0.163 |
| C + hard rules & penalties | 0.879 | 0.833 | 269.0 | 110.9 | 0.0 | 0.150 |
| D + margin re-rank | 0.905 | 0.900 | 321.8 | 142.7 | 0.0 | 0.129 |
| E + MMR (final page) | 0.886 | 0.867 | 319.5 | 141.4 | 0.0 | 0.134 |

#### 5 · Cold-start SKUs

Brand-new drops with zero interaction history are retrieved from text+image+tabular content embeddings and still reach the page.

| Example | Stage 1: corrected / intent / facet acc. | Stage 2: relevant in 200 (R@200 sparse/dense/hybrid) | NDCG A→B→E | Margin A→E | Unfulfillable A→E | ms total |
|---|---|---|---|---|---|---|
| new drop: Aeroflux H-9211 Block Heel Pointed Toe Stiletto  | `navy heels aeroflux` / attribute_specific (85%) / 100% | 4/4 (1.00/1.00/1.00) | 1.000→0.854→0.928 | 63.1→82.5 | 0→0 | 64 |
| new drop: Quillon OC-2081 Breathable Mesh Back Adjustable  | `cobalt blue office chairs quillon` / attribute_specific (93%) / 100% | 6/6 (1.00/1.00/1.00) | 1.000→0.787→0.810 | 72.2→134.3 | 1→0 | 57 |
| new drop: TerraPeak RJ-5763 Pit Zips Waterproof Rain Shell | `black rain jackets terrapeak` / attribute_specific (88%) / 100% | 4/4 (1.00/1.00/1.00) | 0.907→0.767→0.923 | 116.2→148.2 | 0→0 | 59 |

| Cold SKU | impressions | rank sparse | rank dense | rank hybrid | rank in final page |
|---|---|---|---|---|---|
| Aeroflux H-9211 Block Heel Pointed Toe Stiletto Heels - Navy | 0 | 2 | 2 | 3 | 2 |
| Quillon OC-2081 Breathable Mesh Back Adjustable Armrests Mes | 0 | 6 | 6 | 6 | 8 |
| TerraPeak RJ-5763 Pit Zips Waterproof Rain Shell - Black | 0 | 1 | 2 | 1 | 1 |

Mean over this flow's examples:

| Stage | NDCG@10 | Precision@10 | Exp. revenue | Exp. margin | Unfulfillable | Avg return prob |
|---|---|---|---|---|---|---|
| A retrieval order | 0.969 | 0.500 | 189.2 | 83.8 | 0.3 | 0.143 |
| B + LTR ensemble | 0.802 | 0.333 | 224.6 | 96.3 | 0.3 | 0.129 |
| C + hard rules & penalties | 0.838 | 0.333 | 224.6 | 98.9 | 0.0 | 0.113 |
| D + margin re-rank | 0.901 | 0.400 | 268.6 | 122.4 | 0.0 | 0.102 |
| E + MMR (final page) | 0.887 | 0.367 | 267.3 | 121.7 | 0.0 | 0.109 |

#### 6 · Margin-aware re-ranking

Same relevance, higher expected profit: the margin optimiser (weight 0.6 here) re-orders relevant items by E[profit] without crossing the relevance floor.

| Example | Stage 1: corrected / intent / facet acc. | Stage 2: relevant in 200 (R@200 sparse/dense/hybrid) | NDCG A→B→E | Margin A→E | Unfulfillable A→E | ms total |
|---|---|---|---|---|---|---|
| running shoes · margin weight 0.6 | `running shoes` / navigational (43%) / 100% | 187/200 (0.93/0.94/0.94) | 1.000→1.000→1.000 | 60.2→111.1 | 1→0 | 57 |
| headphones · margin weight 0.6 | `headphones` / attribute_specific (50%) / 100% | 183/200 (0.91/0.94/0.92) | 1.000→1.000→1.000 | 26.2→160.6 | 0→0 | 59 |
| backpacks · margin weight 0.6 | `backpacks` / navigational (39%) / 100% | 181/200 (0.91/0.90/0.91) | 1.000→1.000→1.000 | 62.8→176.7 | 0→0 | 55 |

Mean over this flow's examples:

| Stage | NDCG@10 | Precision@10 | Exp. revenue | Exp. margin | Unfulfillable | Avg return prob |
|---|---|---|---|---|---|---|
| A retrieval order | 1.000 | 1.000 | 148.0 | 49.7 | 0.3 | 0.097 |
| B + LTR ensemble | 1.000 | 1.000 | 216.9 | 72.6 | 1.0 | 0.101 |
| C + hard rules & penalties | 1.000 | 1.000 | 218.4 | 74.2 | 0.0 | 0.091 |
| D + margin re-rank | 1.000 | 1.000 | 390.7 | 149.8 | 0.0 | 0.097 |
| E + MMR (final page) | 1.000 | 1.000 | 388.8 | 149.5 | 0.0 | 0.095 |

#### 7 · Personalisation

The same query for three different shoppers. User brand/price affinity features change the order; 'preferred-brand share' shows it.

| Example | Stage 1: corrected / intent / facet acc. | Stage 2: relevant in 200 (R@200 sparse/dense/hybrid) | NDCG A→B→E | Margin A→E | Unfulfillable A→E | ms total |
|---|---|---|---|---|---|---|
| running shoes · user U00001 | `running shoes` / navigational (43%) / 100% | 187/200 (0.93/0.94/0.94) | 1.000→1.000→1.000 | 91.1→165.3 | 1→0 | 66 |
| running shoes · user U00100 | `running shoes` / navigational (43%) / 100% | 187/200 (0.93/0.94/0.94) | 1.000→1.000→1.000 | 68.4→120.9 | 1→0 | 64 |
| running shoes · user U00200 | `running shoes` / navigational (43%) / 100% | 187/200 (0.93/0.94/0.94) | 1.000→1.000→1.000 | 52.7→96.3 | 1→0 | 75 |

| User | preferred-brand share@10 (A→E) | exp. revenue A→E |
|---|---|---|
| U00001 | 0.40→0.10 | 228.9→369.9 |
| U00100 | 0.50→0.80 | 170.0→249.0 |
| U00200 | 0.00→0.00 | 130.3→205.5 |

Mean over this flow's examples:

| Stage | NDCG@10 | Precision@10 | Exp. revenue | Exp. margin | Unfulfillable | Avg return prob |
|---|---|---|---|---|---|---|
| A retrieval order | 1.000 | 1.000 | 176.4 | 70.7 | 1.0 | 0.147 |
| B + LTR ensemble | 1.000 | 1.000 | 226.6 | 97.0 | 0.0 | 0.166 |
| C + hard rules & penalties | 1.000 | 1.000 | 221.5 | 98.5 | 0.0 | 0.146 |
| D + margin re-rank | 1.000 | 1.000 | 275.6 | 127.7 | 0.0 | 0.134 |
| E + MMR (final page) | 1.000 | 1.000 | 274.8 | 127.5 | 0.0 | 0.135 |
