# Phase 2 data audit

Manifest SHA256: `94e3a2ce989ce6a298d7ea30e7ce70f22ac676092b29df808bc3d33b426bf242`

MAIN only; IDs from generator metadata, verified source/output hashes.

- evaluated_instances: 9898
- evaluated_unique_sources: 9898
- excluded_instances_total: 102
- excluded_not_in_textocr_or_no_gt: 85
- excluded_source_no_usable_valid_gt: 17
- gt_ignored_cropped_below_visibility: 1508
- gt_ignored_unreadable: 149050
- gt_outside_common_field_of_view: 866
- intersection_before_file_geometry_checks: 9915
- main_metadata_instances: 10000
- main_unique_sources: 10000
- perspective_instances: 3924
- repaired_gt_annotations: 177
- textocr_sources: 21778
- textvqa_unique_sources: 21953

Cyan: valid field of view. Green: valid GT. Gray: ignored/cropped GT.
Review overlays before full inference. This run does not fine-tune or generate images.
