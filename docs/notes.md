# Project Notes: Interesting, Weird, and Unexpected Items

- **Annotation Complexity:** The team initially underestimated the time and effort required to annotate gameplay clips, especially for dynamic elements. This led to several pivots in project scope.
- **Static vs. Dynamic Focus:** The project switched focus multiple times between static UI elements and dynamic gameplay elements, based on what was feasible and most valuable for the model.
- **Labelbox API Quirks:** Some API behaviors (e.g., label import/export, project configuration) were not well-documented and required trial and error.
- **Clip Consistency:** Using only LIVE round gameplay (not REPLAY) was necessary to ensure data consistency for training.
- **Metadata Tagging:** Adding round type and map metadata to clips improved dataset organization and model evaluation.
- **Tracking Challenges:** Tracking players in Valorant is difficult due to abrupt perspective changes and occlusions. Custom tracking configs and POV-based splicing were developed to address this.
- **Model Output:** Early model outputs worked well for static scenes but struggled with occluded or partially visible objects.
- **Pipeline Flexibility:** The pipeline was designed to allow for rapid changes in scope and workflow, which proved essential as project requirements evolved..
- **Tooling:** A basic tool and website were prototyped for gameplay collection, but the focus shifted to annotation and model training.
- **Labeling Speed:** Explored confidence-based tools and other methods to speed up the annotation process. 