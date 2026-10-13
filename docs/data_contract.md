# Private Data Contract

The dataset is not included. Obtain access and permission from the data holder before using it. Keep the following files outside Git:

- dataset image and mask files;
- `canonical/metadata/all_1000_samples.csv`;
- the original train/validation/test split CSVs;
- any case-level or patient-level metadata.

The private dataset root must contain:

```text
<data-root>/canonical/metadata/all_1000_samples.csv
<data-root>/<image_relative_path from the split/metadata manifest>
<data-root>/<masks_relative_directory>/<semantic-id>_*.png
```

Split CSVs require `sample_id`, `image_relative_path`, and `view`. The private metadata manifest additionally requires `masks_relative_directory`, `annotation_complete`, `present_labels`, `missing_expected_labels`, and `unexpected_labels`. The split files must be the authorized paper split (625/89/179); the repository does not publish case identifiers or recreate a split from the test data.

Run `python data_processing/validate_manifest.py --data-root ... --split-root ...` before training. The checker prints only split counts and schema status; it does not print sample identifiers.

The dataset loader joins selected split rows to the private metadata CSV in memory. Training/evaluation then opens image and mask files only for rows in the selected split; the full metadata manifest itself must remain private and must never be copied into the repository.
