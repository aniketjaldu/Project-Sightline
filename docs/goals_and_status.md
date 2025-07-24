# Project Goals and Status

## Main Goals
- **Build a robust pipeline for training and evaluating object detection/tracking models on Valorant gameplay.**
  - **Status:** Achieved. The pipeline supports data fetching, annotation, training, inference, and tracking.

- **Integrate Labelbox for annotation management.**
  - **Status:** Achieved. Labelbox is used for all annotation import/export and project management.

- **Support both static and dynamic gameplay elements.**
  - **Status:** Partially achieved. The project initially supported both, but ultimately focused on dynamic elements for better model relevance.

- **Develop custom tracking configurations for Valorant.**
  - **Status:** Achieved. Custom ByteTrack and BoT-SORT configs were created and integrated.

- **Organize all workflows and results for reproducibility.**
  - **Status:** Achieved. Each workflow run is stored in a timestamped directory with all inputs, outputs, and logs.

- **Enable both interactive and automated (batch) operation.**
  - **Status:** Achieved. Scripts support both modes.

## Additional Goals and Outcomes
- **Annotation Quality and Consistency:** Improved over time by adding metadata and reviewing all clips.
- **Model Performance:** Early results were promising for static scenes; dynamic tracking required more tuning.
- **Team Collaboration:** Regular meetings and clear division of tasks helped maintain progress.
- **Flexibility:** The pipeline design allowed for rapid pivots in project direction as needed.

## Unmet or Adjusted Goals
- **Static Element Focus:** The team decided not to pursue static-only labeling due to limited value for the model.
- **Automated Test Script:** Initial plans for a test script were deprioritized in favor of direct annotation and model training. 