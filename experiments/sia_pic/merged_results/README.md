# Merged results

Complete five-seed SIA/PIC result matrix.

- `scores.json`: 840 unique rows.
- `contrasts.csv`: recomputed seed-level contrasts and 95% Student-t intervals.
- `manifest.json`: source files and construction details.

Both KL-off and KL-on cover seeds `1, 2, 3, 5, 42`. For KL-on, MRL and
MRL+PIC are copied from the corresponding KL-off scores because attention-KL
is applied only to MRL+SIA and MIPIC.
