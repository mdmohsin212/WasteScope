# End-to-end demonstration evidence

[Completed demo job](20261106T143655Z_cc2f9dff/job_status.json) contains five clip summaries. Each per-clip folder keeps its run_config.json, summary.json and status.json, including checkpoint/configuration identity and end-to-end duration.

Annotated videos, tracks and frame-level mask records remain on Drive. They are not packaged as public report data. Demo counts are predictions, not ground-truth accuracy measurements. In particular, public_stock_bins produced zero container tracks/counts and should not be presented as a successful detection example.

End-to-end timing includes rendering and encoding, and model loading when cold. It must not be compared directly with the warmed processing FPS in ../benchmarks/.
