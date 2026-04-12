# CSE-495-Code
Image inpainting pipeline for visual search eye-tracking analysis.

## Workflow
1. **download_dataset.py** - Download and prepare the dataset
2. **inpainting_plausible_realistic.py** - Generate plausible and realistic image inpaintings
3. **calculate_metrics_plausible_realistic.py** - Calculate metrics on inpainted images
4. **eye_measures/** - Compute eye-tracking measures:
   - first_saccade_initiation.py
   - nfix_to_target.py
   - second_fix_to_target_landing.py
   - target_verification_time.py
   - total_search_time.py
5. **original_rcnn/** - Run baseline comparisons on original dataset:
   - run_rcnn_on_target_bbox.py
   - correlate_rcnn_to_nfix.py


## Directories

### eye_measures/
Computes correlations between image quality metrics and eye-tracking measures. Outputs Excel/CSV files with statistical analysis:
- first_saccade_initiation.py - Time from stimulus onset to first saccade
- nfix_to_target.py - Number of fixations to first target landing
- second_fix_to_target_landing.py - Time from second fixation to first target landing
- target_verification_time.py - Time from first target landing to response
- total_search_time.py - Total search time (sum of above measures)

### original_rcnn/
Baseline comparisons using original (unmanipulated) dataset:
- run_rcnn_on_target_bbox.py - Run Mask R-CNN on original images to get detection scores
- correlate_rcnn_to_nfix.py - Correlate RCNN confidence scores to eye-tracking measures

### visualizations/
Scripts to generate figures and plots for paper publication and general analysis of results 
 
