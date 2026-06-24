# TODO

- [ ] Update `app.py` so the Detect button runs records in 100-record batches.
- [ ] On each click: evaluate next 100 records (in sorted order), accumulate results.
- [x] Stop after 2 batches (total 200 records) and show a final “Decision point” message; further clicks should be disabled.

- [ ] Ensure UI uses accumulated `scan_results['full_df']` / `scan_results['anomaly_df']`.
- [ ] Add a caption showing current batch progress.
- [ ] Smoke test with `streamlit run app.py`.

