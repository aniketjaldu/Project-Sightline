import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import labelbox as lb
import labelbox.types as lb_types
from core.config import (
    LABELBOX_API_KEY, LABELBOX_PROJECT_ID, LABELBOX_STATUS,
    CLASS_MAPPING, ensure_directories, INFERENCE_DIR,
    SIGHTLINE_TRACKER_CONFIG
)
from pipeline.sightline_tracker import SightlineTracker
from utils.logger import setup_logger, log_data_row_action
import traceback

class LabelboxAnnotationImporter:
    """Class to handle importing annotations back to Labelbox as actual labels"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
        self.sightline_tracker = SightlineTracker()
        
        if not LABELBOX_API_KEY:
            self.logger.warning("LABELBOX_API_KEY not set - Labelbox import functionality will be disabled")
            self.client = None
            self.project = None
            self.ontology = None
            self.tools = {}
            return
        
        self.client = lb.Client(api_key=LABELBOX_API_KEY)
        self.project = self.client.get_project(LABELBOX_PROJECT_ID) if LABELBOX_PROJECT_ID else None
        
        # Get project ontology
        if self.project:
            self.ontology = self.project.ontology()
            # Store tools with both lowercase keys and original names for case-insensitive lookup
            self.tools = {tool.name.lower(): tool for tool in self.ontology.tools()}
            # Also store a mapping of lowercase to original case for accurate retrieval
            self.tool_name_mapping = {tool.name.lower(): tool.name for tool in self.ontology.tools()}
        else:
            self.ontology = None
            self.tools = {}
            self.tool_name_mapping = {}
        
        self.logger.info("Labelbox annotation importer initialized successfully")
    
    def convert_yolo_to_labelbox_annotations(self, detections_data):
        """
        Convert YOLO detection data to Labelbox VideoObjectAnnotations.
        
        CORRECT APPROACH (based on Labelbox docs): Per-track segment indexing.
        Each object track gets its own segment indexing (0, 1, 2... for temporal gaps).
        Different objects can have the same segment_index values.
        
        Args:
            detections_data: Dictionary with frame numbers as keys and detection lists as values
            
        Returns:
            List of VideoObjectAnnotation arrays, one per object segment
        """
        try:
            print(f"🔄 Converting {len(detections_data)} frames to Labelbox format...")
            
            # Group detections by track_id to identify consecutive segments
            track_segments = self._identify_track_segments(detections_data)
            
            print(f"📊 Analysis Results:")
            print(f"  • Total tracks: {len(track_segments)}")
            total_segments = sum(len(segments) for segments in track_segments.values())
            print(f"  • Total segments: {total_segments}")
            
            # Convert each segment to annotation array
            all_annotation_arrays = []
            all_annotations = []  # Collect all annotations for final sorting
            
            for track_id, segments in track_segments.items():
                for segment_index, segment in enumerate(segments):
                    start_frame = segment['start_frame']
                    end_frame = segment['end_frame']
                    
                    # Get detection from start frame for bbox coordinates
                    start_detection = None
                    for detection in detections_data[start_frame]:
                        if detection.get('track_id', 0) == track_id:
                            start_detection = detection
                            break
                    
                    if not start_detection:
                        print(f"⚠️  Warning: No detection found for track {track_id} at start frame {start_frame}")
                        continue
                    
                    # Create annotations for ALL frames in this segment, not just keyframes
                    for frame_num in range(start_frame, end_frame + 1):
                        # Get detection for this specific frame
                        frame_detection = None
                        if frame_num in detections_data:
                            for detection in detections_data[frame_num]:
                                if detection.get('track_id', 0) == track_id:
                                    frame_detection = detection
                                    break
                        
                        if frame_detection:
                            # Bbox is already in pixel coordinates
                            bbox = frame_detection['bbox']
                            
                            # Map class name to ontology tool name
                            tool_name = self._map_class_to_tool_name(frame_detection['class_name'])
                            if not tool_name:
                                self.logger.warning(f"No ontology tool found for class: {frame_detection['class_name']}")
                                continue
                            
                            # Create annotation for this frame
                            annotation = lb_types.VideoObjectAnnotation(
                                name=tool_name,
                                keyframe=True,
                                frame=frame_num + 1,  # Convert from 0-indexed to 1-indexed for Labelbox
                                segment_index=segment_index,  # Per-track segment index (0, 1, 2... for temporal gaps)
                                value=lb_types.Rectangle(
                                    start=lb_types.Point(x=bbox["left"], y=bbox["top"]),
                                    end=lb_types.Point(x=bbox["left"] + bbox["width"], y=bbox["top"] + bbox["height"])
                                )
                            )
                            annotation._track_id = track_id  # For sorting
                            all_annotations.append(annotation)
                    
                    print(f"  📦 Track {track_id} Segment {segment_index}: frames {start_frame}-{end_frame} ({end_frame - start_frame + 1} annotations)")
            
            # Sort annotations by (frame, track_id) for consistent submission order
            all_annotations.sort(key=lambda x: (x.frame, getattr(x, '_track_id', 0)))
            
            print(f"📤 Submitting annotations to Labelbox...")
            print(f"📋 Annotations sorted by (frame, track_id)")
            
            # Group annotations by track_id to create one array per track
            track_annotation_arrays = {}
            
            for ann in all_annotations:
                track_id = getattr(ann, '_track_id', 0)
                if track_id not in track_annotation_arrays:
                    track_annotation_arrays[track_id] = []
                track_annotation_arrays[track_id].append(ann)
            
            # Convert to list format expected by Labelbox
            final_annotation_arrays = list(track_annotation_arrays.values())
            
            print(f"✅ Created {len(final_annotation_arrays)} annotation arrays (one per track)")
            print(f"📋 Total tracks from Sightline: {len(track_annotation_arrays)}")
            print(f"📋 Total annotations: {len(all_annotations)}")
            print(f"🔍 Each array contains all frames for one track with temporal gap segment_index logic")
            
            # Debug: Show structure of arrays being submitted
            print(f"\n🔍 Debug: Array structure being submitted to Labelbox:")
            for i, array in enumerate(final_annotation_arrays[:5]):  # Show first 5 arrays
                track_id = getattr(array[0], '_track_id', 'unknown') if array else 'empty'
                frames = [ann.frame for ann in array] if array else []
                segment_indices = [ann.segment_index for ann in array] if array else []
                print(f"  Array {i} (Track {track_id}): {len(array)} annotations")
                print(f"    Frames: {frames}")
                print(f"    Segment indices: {segment_indices}")
            
            # Debug: Specific analysis for frame 804
            print(f"\n🔍 Debug: Frame 804 analysis across all arrays:")
            frame_804_found = []
            for i, array in enumerate(final_annotation_arrays):
                track_id = getattr(array[0], '_track_id', 'unknown') if array else 'empty'
                for ann in array:
                    if ann.frame == 804:
                        frame_804_found.append({
                            'array_index': i,
                            'track_id': track_id,
                            'segment_index': ann.segment_index,
                            'frame': ann.frame
                        })
            
            print(f"  Frame 804 appears in {len(frame_804_found)} arrays:")
            for item in frame_804_found:
                print(f"    Array {item['array_index']} (Track {item['track_id']}): frame {item['frame']}, segment_index {item['segment_index']}")
            
            return final_annotation_arrays
            
        except Exception as e:
            print(f"❌ Error in convert_yolo_to_labelbox_annotations: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _identify_track_segments(self, detections_data):
        """
        Identify consecutive frame segments for each track.
        
        Returns:
            Dict[track_id, List[segment_info]] where segment_info contains:
            - start_frame: First frame of consecutive sequence
            - end_frame: Last frame of consecutive sequence
        """
        track_frames = {}
        
        # Collect all frames for each track
        for frame_num, detections in detections_data.items():
            for detection in detections:
                track_id = detection.get('track_id', 0)
                if track_id not in track_frames:
                    track_frames[track_id] = []
                track_frames[track_id].append(frame_num)
        
        # Sort frames and identify consecutive segments
        track_segments = {}
        
        for track_id, frames in track_frames.items():
            frames = sorted(frames)
            segments = []
            
            if not frames:
                continue
                
            # Identify consecutive sequences
            current_start = frames[0]
            current_end = frames[0]
            
            for i in range(1, len(frames)):
                if frames[i] == current_end + 1:  # Consecutive frame
                    current_end = frames[i]
                else:  # Gap detected - end current segment, start new one
                    segments.append({
                        'start_frame': current_start,
                        'end_frame': current_end
                    })
                    current_start = frames[i]
                    current_end = frames[i]
            
            # Add the last segment
            segments.append({
                'start_frame': current_start,
                'end_frame': current_end
            })
            
            track_segments[track_id] = segments
        
        return track_segments
    
    def _yolo_to_pixel_coords(self, bbox_norm, video_width, video_height):
        """
        Convert YOLO normalized coordinates to pixel coordinates.
        
        Args:
            bbox_norm: Dictionary with normalized coordinates {center_x, center_y, width, height}
            video_width: Video width in pixels
            video_height: Video height in pixels
            
        Returns:
            Dictionary with pixel coordinates {left, top, width, height}
        """
        # YOLO format: center_x, center_y, width, height (all normalized 0-1)
        center_x_norm = bbox_norm['center_x']
        center_y_norm = bbox_norm['center_y']
        width_norm = bbox_norm['width']
        height_norm = bbox_norm['height']
        
        # Convert to pixel coordinates
        width_px = width_norm * video_width
        height_px = height_norm * video_height
        center_x_px = center_x_norm * video_width
        center_y_px = center_y_norm * video_height
        
        # Convert center coordinates to top-left coordinates
        left = center_x_px - (width_px / 2)
        top = center_y_px - (height_px / 2)
        
        return {
            'left': int(left),
            'top': int(top),
            'width': int(width_px),
            'height': int(height_px)
        }
    
    def _convert_tracked_annotations(
        self, 
        yolo_annotations: Dict[str, Any], 
        global_key: str
    ) -> List[lb_types.VideoObjectAnnotation]:
        """
        Convert Sightline tracking annotations to VideoObjectAnnotation format with continuous temporal segments.
        Creates placeholder annotations to fill gaps between actual detections.
        """
        all_detections = []
        
        # Collect all detections from the frames
        for frame_num_str, frame_objects in yolo_annotations.get('frames', {}).items():
            frame_num = int(frame_num_str)
            
            for obj in frame_objects:
                track_id = obj.get('track_id')
                if track_id is None:
                    self.logger.warning(f"Object missing track_id in frame {frame_num}, skipping")
                    continue
                
                class_name = obj['class_name']
                confidence = obj['confidence']
                bbox = obj['bounding_box']
                
                # Map class name to ontology tool name
                tool_name = self._map_class_to_tool_name(class_name)
                if not tool_name:
                    self.logger.warning(f"No ontology tool found for class: {class_name}")
                    continue
                
                detection = {
                    'frame': frame_num,
                    'track_id': track_id,
                    'tool_name': tool_name,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                }
                all_detections.append(detection)
        
        if not all_detections:
            self.logger.warning("No valid detections found for conversion")
            return []
        
        # Group detections by track_id for temporal analysis
        tracks = {}
        for detection in all_detections:
            track_id = detection['track_id']
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append(detection)
        
        # Sort detections within each track by frame number
        for track_id in tracks:
            tracks[track_id].sort(key=lambda x: x['frame'])
        
        # Process each track to create continuous segments with placeholder annotations
            video_annotations = []
            
        for track_id, track_detections in tracks.items():
            current_segment_index = 0
            
            # Identify temporal segments (continuous sequences)
            segments = []
            current_segment = []
            
            for i, detection in enumerate(track_detections):
                if i == 0:
                    # First detection starts a new segment
                    current_segment = [detection]
                else:
                    prev_frame = track_detections[i-1]['frame']
                    curr_frame = detection['frame']
                    
                    if curr_frame - prev_frame == 1:
                        # Consecutive frame - continue current segment
                        current_segment.append(detection)
                    else:
                        # Gap detected - close current segment and start new one
                        segments.append(current_segment)
                        current_segment = [detection]
                        
                        self.logger.debug(f"Track {track_id}: Temporal gap between frames {prev_frame}-{curr_frame}, creating new segment")
                
                # Handle last detection
                if i == len(track_detections) - 1:
                    segments.append(current_segment)
            
            # Create continuous annotations for each segment
            for segment_index, segment_detections in enumerate(segments):
                if not segment_detections:
                    continue
                    
                start_frame = segment_detections[0]['frame']
                end_frame = segment_detections[-1]['frame']
                
                self.logger.debug(f"Track {track_id}, Segment {segment_index}: Creating continuous annotations from frame {start_frame} to {end_frame}")
                
                # Create keyframe annotations at start and end of segment
                start_detection = segment_detections[0]
                end_detection = segment_detections[-1]
                
                # Create start keyframe
                bbox = start_detection['bbox']
                start_annotation = lb_types.VideoObjectAnnotation(
                    name=start_detection['tool_name'],
                    keyframe=True,
                    frame=start_frame + 1,  # Convert from 0-indexed to 1-indexed for Labelbox
                    segment_index=segment_index,  # Use per-track segment index
                    value=lb_types.Rectangle(
                        start=lb_types.Point(x=bbox['left'], y=bbox['top']),
                        end=lb_types.Point(x=bbox['left'] + bbox['width'], y=bbox['top'] + bbox['height'])
                    )
                )
                video_annotations.append(start_annotation)
                
                # If segment has multiple frames, create end keyframe
                if start_frame != end_frame:
                    bbox = end_detection['bbox']
                    end_annotation = lb_types.VideoObjectAnnotation(
                        name=end_detection['tool_name'],
                        keyframe=True,
                        frame=end_frame + 1,  # Convert from 0-indexed to 1-indexed for Labelbox
                        segment_index=segment_index,  # Same segment index
                        value=lb_types.Rectangle(
                            start=lb_types.Point(x=bbox['left'], y=bbox['top']),
                            end=lb_types.Point(x=bbox['left'] + bbox['width'], y=bbox['top'] + bbox['height'])
                        )
                    )
                    video_annotations.append(end_annotation)
                
                # Create intermediate keyframes for any detections between start and end
                for detection in segment_detections[1:-1]:  # Skip first and last (already created)
                    bbox = detection['bbox']
                    intermediate_annotation = lb_types.VideoObjectAnnotation(
                        name=detection['tool_name'],
                        keyframe=True,
                        frame=detection['frame'] + 1,  # Convert from 0-indexed to 1-indexed for Labelbox
                        segment_index=segment_index,  # Same segment index
                        value=lb_types.Rectangle(
                            start=lb_types.Point(x=bbox['left'], y=bbox['top']),
                            end=lb_types.Point(x=bbox['left'] + bbox['width'], y=bbox['top'] + bbox['height'])
                        )
                    )
                    video_annotations.append(intermediate_annotation)
                
                # Note: Removed keyframe=False annotations as they were causing issues with Labelbox rendering
                # Labelbox can handle temporal gaps automatically when segments are properly separated
        
        # Apply GLOBAL segment_index assignment for Labelbox validation
        # Sort all annotations by frame, then by original segment_index for consistency
        video_annotations.sort(key=lambda x: (x.frame, x.segment_index))
        
        # Apply FRAME-LOCAL segment_index assignment (0, 1, 2... within each frame)
        final_annotations = []
        current_frame = None
        frame_segment_index = 0
        
        for ann in video_annotations:
            if ann.frame != current_frame:
                # New frame - reset segment_index to 0
                current_frame = ann.frame
                frame_segment_index = 0
            
            # Create new annotation with frame-local segment_index
            fixed_annotation = lb_types.VideoObjectAnnotation(
                name=ann.name,
                keyframe=ann.keyframe,
                frame=ann.frame + 1,  # Convert from 0-indexed to 1-indexed for Labelbox
                segment_index=frame_segment_index,  # Frame-local: 0, 1, 2... within each frame
                value=ann.value,
                classifications=ann.classifications
            )
            final_annotations.append(fixed_annotation)
            frame_segment_index += 1
        
        # Replace video_annotations with frame-locally indexed ones
        video_annotations = final_annotations
        
        # Sort annotations by frame, then by segment_index for consistent ordering
        video_annotations.sort(key=lambda x: (x.frame, x.segment_index))
        
        self.logger.debug(f"✅ Continuous segment creation + frame-local segment_index assignment applied successfully")
        
        self.logger.info(f"Applied Sightline tracking annotation processing with CONTINUOUS SEGMENTS + FRAME-LOCAL INDEXING")
        self.logger.info(f"Converted {len(video_annotations)} video annotations for global key {global_key}")
        
        # Calculate tracking statistics
        total_tracks = len(tracks)
        total_detections = len(video_annotations)
        total_frames = len(set(ann.frame for ann in video_annotations))
        
        # Calculate segment statistics per track
        track_segments_count = {}
        for track_id, track_detections in tracks.items():
            segments = []
            current_segment = []
            
            for i, detection in enumerate(track_detections):
                if i == 0:
                    current_segment = [detection]
                else:
                    prev_frame = track_detections[i-1]['frame']
                    curr_frame = detection['frame']
                    
                    if curr_frame - prev_frame == 1:
                        current_segment.append(detection)
                    else:
                        segments.append(current_segment)
                        current_segment = [detection]
                
                if i == len(track_detections) - 1:
                    segments.append(current_segment)
            
            track_segments_count[track_id] = len(segments)
        
        total_segments = sum(track_segments_count.values())
        tracks_with_gaps = sum(1 for count in track_segments_count.values() if count > 1)
        
        # Get class name from the first annotation for logging
        class_name = video_annotations[0].name if video_annotations else "Unknown"
        
        self.logger.info(f"  {class_name.title()}: {total_tracks} Sightline tracks across {total_frames} frames")
        self.logger.info(f"  Total keyframe annotations: {total_detections}")
        self.logger.info(f"  Continuous segments: {total_segments} ({tracks_with_gaps} tracks with temporal gaps)")
        self.logger.info(f"  Frame-local segment_index: 0,1,2... within each frame")
        self.logger.info(f"  Format: Continuous segments with keyframe interpolation + frame-local indexing")
        self.logger.info(f"  Note: Labelbox will interpolate between keyframes automatically!")
        
        return video_annotations
    
    def _convert_frame_annotations(
        self, 
        yolo_annotations: Dict[str, Any], 
        global_key: str
    ) -> List[lb_types.VideoObjectAnnotation]:
        """
        Convert original YOLO annotations (without tracking) to single-frame annotations
        """
        video_annotations = []
        global_segment_counter = 0
            
        for frame_num_str, frame_objects in yolo_annotations.get('frames', {}).items():
            frame_num = int(frame_num_str)
            
            for obj in frame_objects:
                class_name = obj['class_name']
                confidence = obj['confidence']
                bbox = obj['bounding_box']
                
                # Map class name to ontology tool name
                tool_name = self._map_class_to_tool_name(class_name)
                if not tool_name:
                    self.logger.warning(f"No ontology tool found for class: {class_name}")
                    continue
                
                # Convert YOLO bbox format to Labelbox format
                left = bbox['left']
                top = bbox['top']
                width_px = bbox['width']
                height_px = bbox['height']
                    
        # Create video object annotation with unique segment index
        video_annotation = lb_types.VideoObjectAnnotation(
            name=tool_name,
            keyframe=True,
            frame=frame_num + 1,  # Convert from 0-indexed to 1-indexed for Labelbox
            segment_index=global_segment_counter,
            value=lb_types.Rectangle(
                start=lb_types.Point(x=left, y=top),
                end=lb_types.Point(x=left + width_px, y=top + height_px)
            )
        )
        
        video_annotations.append(video_annotation)
        global_segment_counter += 1
            
        # Sort annotations by frame, then by segment_index
        video_annotations.sort(key=lambda x: (x.frame, x.segment_index))
            
        self.logger.info(f"Applied frame-based annotation processing")
        self.logger.info(f"Converted {len(video_annotations)} video annotations for global key {global_key}")
            
        # Get class name from the first annotation for logging
        class_name = video_annotations[0].name if video_annotations else "Unknown"
            
        self.logger.info(f"  {class_name.title()}: {len(video_annotations)} individual detections with unique segment indices")
            
        return video_annotations
    
    def _create_sightline_object_tracks(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Create object tracks from detections using Sightline tracking algorithm
        
        Args:
            detections: List of detection dictionaries with frame, tool_name, bbox, etc.
            
        Returns:
            Dictionary mapping track_id to track data
        """
        # Convert inference detections to format expected by Sightline tracker
        yolo_detections = []
        
        for detection in detections:
            # Convert back to pixel coordinates for ByteTrack processing
            bbox = detection['bbox']
            
            yolo_detection = {
                'frame_number': detection['frame'],
                'class_name': detection['tool_name'].lower(),  # Use original class name mapping
                'confidence': detection['confidence'],
                'bbox_pixel': [bbox['left'], bbox['top'], bbox['width'], bbox['height']],
                'bbox_normalized': [  # We'll calculate this if needed
                    (bbox['left'] + bbox['width']/2) / 1920,   # Assuming standard resolution
                    (bbox['top'] + bbox['height']/2) / 1080,
                    bbox['width'] / 1920,
                    bbox['height'] / 1080
                ]
            }
            yolo_detections.append(yolo_detection)
        
        # Apply Sightline tracking
        tracked_detections = self.sightline_tracker.process_inference_detections(yolo_detections)
        
        # Convert Sightline tracking results back to the format expected by Labelbox conversion
        tracks = {}
        
        for detection in tracked_detections:
            track_id = detection['track_id']
            
            if track_id not in tracks:
                tracks[track_id] = {
                    'tool_name': detection['class_name'].title(),  # Convert back to proper case
                    'detections': []
                }
            
            # Convert back to original detection format
            bbox_pixel = detection['bbox_pixel']
            original_detection = {
                'frame': detection['frame_number'],
                'tool_name': detection['class_name'].title(),
                'confidence': detection['confidence'],
                'bbox': {
                    'left': bbox_pixel[0],
                    'top': bbox_pixel[1],
                    'width': bbox_pixel[2],
                    'height': bbox_pixel[3]
                },
                'center_x': bbox_pixel[0] + bbox_pixel[2] / 2,
                'center_y': bbox_pixel[1] + bbox_pixel[3] / 2
            }
            
            tracks[track_id]['detections'].append(original_detection)
        
        self.logger.info(f"Sightline tracker created {len(tracks)} object tracks from {len(detections)} detections")
        
        return tracks
    
    def _map_class_to_tool_name(self, class_name: str) -> Optional[str]:
        """Map YOLO class name to Labelbox ontology tool name with proper case handling"""
        # If no ontology tools available, return the class name as-is
        if not self.tools:
            self.logger.warning(f"No ontology tools available, using class name as-is: {class_name}")
            return class_name
        
        # Try exact lowercase matching first and return the proper case name
        if class_name.lower() in self.tools:
            proper_case_name = self.tool_name_mapping[class_name.lower()]
            self.logger.debug(f"Exact match found: '{class_name}' -> '{proper_case_name}'")
            return proper_case_name
        
        # Try to find a close match (contains or is contained)
        for tool_key in self.tools.keys():
            if class_name.lower() in tool_key or tool_key in class_name.lower():
                proper_case_name = self.tool_name_mapping[tool_key]
                self.logger.debug(f"Partial match found: '{class_name}' -> '{proper_case_name}'")
                return proper_case_name
        
        # Try common mapping variations
        class_mappings = {
            'player': ['player', 'agent', 'character'],
            'agent': ['agent', 'player', 'character'],
            'weapon': ['weapon', 'gun', 'rifle'],
            'ability': ['ability', 'skill'],
            'ultimate': ['ultimate', 'ult'],
            'kill': ['kill', 'elimination'],
            'spike': ['spike', 'bomb'],
            'map': ['map', 'location'],
            'round_info': ['round info', 'round_info', 'roundinfo'],
            'ability_model': ['ability model', 'ability_model', 'abilitymodel'],
            'ultimate_model': ['ultimate model', 'ultimate_model', 'ultimatemodel']
        }
        
        if class_name.lower() in class_mappings:
            for potential_match in class_mappings[class_name.lower()]:
                if potential_match.lower() in self.tools:
                    proper_case_name = self.tool_name_mapping[potential_match.lower()]
                    self.logger.info(f"Mapped '{class_name}' -> '{proper_case_name}' via common mappings")
                    return proper_case_name
        
        # If still no match found, log available tools and skip this detection
        available_tools = [self.tool_name_mapping[key] for key in self.tools.keys()]
        self.logger.error(f"No ontology tool found for class '{class_name}'")
        self.logger.error(f"Available tools in ontology: {sorted(available_tools)}")
        self.logger.error(f"Skipping this detection to avoid import failure")
        return None
    
    def _get_global_key_for_data_row(self, data_row_id: str, annotations_file: Path) -> Optional[str]:
        """Get or create a global key for the data row using centralized logic"""
        try:
            # Try to get the data row and its existing global key
            data_row = self.client.get_data_row(data_row_id)
            if data_row.global_key:
                self.logger.info(f"Using existing global key: {data_row.global_key}")
                return data_row.global_key
            
            # Only create a global key if one doesn't exist (needed for import)
            self.logger.info(f"No global key found for data row {data_row_id}, creating one for import")
            
            # Get the filename from the data row's external_id (which contains the full filename)
            global_key = None
            if hasattr(data_row, 'external_id') and data_row.external_id:
                # Use the external_id as-is (this is the full filename)
                global_key = data_row.external_id
                self.logger.info(f"Using external_id as global key: {global_key}")
            else:
                # Fallback to using the data row ID if no external_id
                global_key = data_row_id
                self.logger.info(f"Using data_row_id as global key: {global_key}")
            
            # Set the global key on the DataRow in Labelbox
            if global_key:
                try:
                    data_row.update(global_key=global_key)
                    self.logger.info(f"Successfully set global key '{global_key}' for data row: {data_row_id}")
                except Exception as update_error:
                    self.logger.warning(f"Failed to update global key in Labelbox: {str(update_error)}")
                    # Continue anyway - we can still use the global key for import
            
            return global_key
            
        except Exception as e:
            self.logger.error(f"Error getting/creating global key for data row {data_row_id}: {str(e)}")
            return None
    
    def create_label_annotation(
        self, 
        data_row_id: str, 
        yolo_annotations: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create and import annotation to Labelbox as actual labels (not just predictions)
        
        Args:
            data_row_id: Data row ID for the video
            yolo_annotations: Annotations from YOLO inference
            
        Returns:
            Import job ID if successful, None otherwise
        """
        try:
            self.logger.info(f"Creating labels for data row: {data_row_id}")
            log_data_row_action(data_row_id, "LABEL_IMPORT_STARTED")
            
            # Get global key for the data row
            # Use temp directory for annotations file path since INFERENCE_DIR is now workflow-specific
            from core.config import TEMP_INFERENCE
            annotations_file = TEMP_INFERENCE / f"{data_row_id}_annotations.json"
            global_key = self._get_global_key_for_data_row(data_row_id, annotations_file)
            
            if not global_key:
                self.logger.error(f"Could not determine global key for data row: {data_row_id}")
                log_data_row_action(data_row_id, "LABEL_IMPORT_FAILED_NO_GLOBAL_KEY")
                return None
            
            # Extract video dimensions and store for bbox conversion
            video_info = yolo_annotations.get('video_info', {})
            self.video_width = video_info.get('width', 1920)
            self.video_height = video_info.get('height', 1080)
            
            # Convert YOLO annotations format to detections_data format
            detections_data = {}
            frames = yolo_annotations.get('frames', {})
            
            for frame_num_str, frame_objects in frames.items():
                frame_num = int(frame_num_str)
                detections_data[frame_num] = []
                
                for obj in frame_objects:
                    # The bounding box is already in pixel coordinates (left, top, width, height)
                    bbox_pixel = obj['bounding_box']  # Already in pixel format
                    
                    detection = {
                        'track_id': obj.get('track_id', 0),
                        'class_name': obj['class_name'],
                        'confidence': obj['confidence'],
                        'bbox': bbox_pixel  # Already in pixel coordinates
                    }
                    detections_data[frame_num].append(detection)
            
            # Convert to Labelbox format - now returns list of annotation arrays
            annotation_arrays = self.convert_yolo_to_labelbox_annotations(detections_data)
            
            if not annotation_arrays:
                self.logger.warning(f"No annotations to import for data row: {data_row_id}")
                log_data_row_action(data_row_id, "LABEL_IMPORT_NO_ANNOTATIONS")
                return None
            
            # Create label using proper lb_types format
            # Submit each annotation array (track) as a separate label
            labels = []
            for i, annotation_array in enumerate(annotation_arrays):
                if annotation_array:  # Make sure array is not empty
                    track_id = getattr(annotation_array[0], '_track_id', i) if annotation_array else i
                    
                    label = lb_types.Label(
                        data={"global_key": global_key},
                        annotations=annotation_array  # Each track as separate label
                    )
                    labels.append(label)
                    
            # Debug info for first few labels
            if i < 3:
                frames = [ann.frame for ann in annotation_array]
                segment_indices = [ann.segment_index for ann in annotation_array]
                print(f"  Label {i} (Track {track_id}): {len(annotation_array)} annotations")
                print(f"    Frames: {frames[:5]}{'...' if len(frames) > 5 else ''}")
                print(f"    Segment indices: {segment_indices[:5]}{'...' if len(segment_indices) > 5 else ''}")
            
            print(f"🚀 Submitting {len(labels)} separate labels (one per track) to Labelbox Label API")
            
            # Import as actual Labels (not just MAL predictions)
            # This creates real labels that can be moved to review queues
            upload_job = lb.LabelImport.create_from_objects(
                client=self.client,
                project_id=self.project.uid,
                name=f"label_import_{data_row_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                labels=labels  # Submit list of labels, one per track
            )
            
            # Wait for the job to complete
            upload_job.wait_till_done()
            
            # Check for errors
            if upload_job.errors:
                error_details = []
                for error in upload_job.errors:
                    error_details.append(str(error))
                error_msg = "; ".join(error_details)
                self.logger.error(f"Label import job failed with errors: {error_msg}")
                log_data_row_action(data_row_id, f"LABEL_IMPORT_FAILED: {error_msg}")
                return None
            
            # Check upload status
            if upload_job.statuses:
                for status in upload_job.statuses:
                    if hasattr(status, 'status') and status.status != 'SUCCESS':
                        self.logger.error(f"Label import status error: {status}")
                        log_data_row_action(data_row_id, f"LABEL_IMPORT_STATUS_ERROR: {status}")
                        return None
            
            self.logger.info(f"Successfully imported labels for data row: {data_row_id}")
            log_data_row_action(data_row_id, "LABEL_IMPORT_SUCCESS")
            
            # NEW: Automatically move datarow to "In review" status after successful import
            self.logger.info(f"Attempting to move datarow {data_row_id} to 'In review' status")
            review_success = self.move_datarow_to_review_status(data_row_id)
            
            if review_success:
                self.logger.info(f"Successfully moved datarow {data_row_id} to review status")
            else:
                self.logger.warning(f"Label import successful but failed to move to review status. Manual action needed for {data_row_id}")
            
            return upload_job.uid
            
        except Exception as e:
            self.logger.error(f"Error importing labels for {data_row_id}: {str(e)}")
            log_data_row_action(data_row_id, f"LABEL_IMPORT_ERROR: {str(e)}")
            return None
    
    def import_annotations_from_file(
        self, 
        annotations_file: Path, 
        data_row_id: str
    ) -> Optional[str]:
        """
        Import annotations from a YOLO inference file as labels
        
        Args:
            annotations_file: Path to the annotation file
            data_row_id: Data row ID for the video
            
        Returns:
            Import job ID if successful, None otherwise
        """
        try:
            if not self.client:
                self.logger.warning("Labelbox client not initialized - skipping label import")
                return None
            
            if not annotations_file.exists():
                self.logger.error(f"Annotation file not found: {annotations_file}")
                return None
            
            # Load annotations
            with open(annotations_file, 'r') as f:
                yolo_annotations = json.load(f)
            
            # Import to Labelbox as labels
            return self.create_label_annotation(
                data_row_id=data_row_id,
                yolo_annotations=yolo_annotations
            )
            
        except Exception as e:
            self.logger.error(f"Error importing from file {annotations_file}: {str(e)}")
            return None
    
    def bulk_import_annotations(
        self, 
        annotation_files: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Bulk import multiple annotation files as labels
        
        Args:
            annotation_files: List of dicts with 'file' and 'data_row_id' keys
            
        Returns:
            Dictionary mapping data_row_id to import job ID
        """
        results = {}
        
        for annotation_info in annotation_files:
            file_path = Path(annotation_info['file'])
            data_row_id = annotation_info['data_row_id']
            
            job_id = self.import_annotations_from_file(
                annotations_file=file_path,
                data_row_id=data_row_id
            )
            
            if job_id:
                results[data_row_id] = job_id
            
            # Add a small delay between imports to avoid rate limiting
            import time
            time.sleep(1)
        
        self.logger.info(f"Bulk label import completed: {len(results)}/{len(annotation_files)} successful")
        return results
    
    def validate_annotations(self, annotations: Dict[str, Any]) -> bool:
        """
        Validate annotation format
        
        Args:
            annotations: Annotation dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_keys = ['frames', 'video_info']
            
            for key in required_keys:
                if key not in annotations:
                    self.logger.error(f"Missing required key: {key}")
                    return False
            
            # Validate video info
            video_info = annotations['video_info']
            if 'width' not in video_info or 'height' not in video_info:
                self.logger.error("Missing width/height in video_info")
                return False
            
            # Validate frames
            frames = annotations['frames']
            if not isinstance(frames, dict):
                self.logger.error("Frames should be a dictionary")
                return False
            
            # Validate frame structure
            for frame_num, frame_objects in frames.items():
                if not isinstance(frame_objects, list):
                    self.logger.error(f"Frame {frame_num} objects should be a list")
                    return False
                
                for obj in frame_objects:
                    required_obj_keys = ['class_name', 'confidence', 'bounding_box']
                    for key in required_obj_keys:
                        if key not in obj:
                            self.logger.error(f"Missing key {key} in frame {frame_num} object")
                            return False
                    
                    bbox = obj['bounding_box']
                    bbox_keys = ['left', 'top', 'width', 'height']
                    for key in bbox_keys:
                        if key not in bbox:
                            self.logger.error(f"Missing bbox key {key} in frame {frame_num}")
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating annotations: {str(e)}")
            return False 

    def move_datarow_to_review_status(self, data_row_id: str) -> bool:
        """
        Move a datarow to "In review" status after successful annotation import
        
        Args:
            data_row_id: The ID of the data row to move to review
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client or not self.project:
                self.logger.warning("Labelbox client not initialized - cannot change task status")
                return False
                
            self.logger.info(f"Moving datarow {data_row_id} to 'In review' status")
            
            # Get the data row
            data_row = self.client.get_data_row(data_row_id)
            
            # Method 1: Try to move via task queue (most common approach)
            try:
                # Get project task queues
                task_queues = list(self.project.task_queues())
                review_queue = None
                
                # Find the review queue (common names: "Review", "To Review", etc.)
                for queue in task_queues:
                    queue_name = queue.name.lower()
                    if any(term in queue_name for term in ['review', 'to_review', 'in_review']):
                        review_queue = queue
                        break
                
                if review_queue:
                    # Move data row to review queue using correct API signature
                    try:
                        # Method 1: Try with DataRowIds class (this works!)
                        from labelbox.schema.identifiables import DataRowIds
                        identifier = DataRowIds([data_row_id])
                        self.project.move_data_rows_to_task_queue(
                            data_row_ids=identifier,
                            task_queue_id=review_queue.uid
                        )
                    except (ImportError, TypeError) as e1:
                        try:
                            # Method 2: Try with GlobalKeys as fallback
                            from labelbox.schema.identifiables import GlobalKeys
                            if data_row.global_key:
                                identifier = GlobalKeys([data_row.global_key])
                                self.project.move_data_rows_to_task_queue(
                                    data_row_ids=identifier,
                                    task_queue_id=review_queue.uid
                                )
                            else:
                                raise Exception("No global key found")
                        except Exception as e2:
                            raise Exception(f"Both methods failed: {e1}, {e2}")
                    self.logger.info(f"Successfully moved datarow {data_row_id} to review queue: {review_queue.name}")
                    log_data_row_action(data_row_id, "MOVED_TO_REVIEW", f"Queue: {review_queue.name}")
                    return True
                else:
                    self.logger.warning("No review queue found in project task queues")
                    
            except Exception as queue_error:
                self.logger.warning(f"Task queue approach failed: {str(queue_error)}")
            
            # Method 2: Try to create a label with review status (if queues don't work)
            try:
                # This creates a label that needs review rather than moving to queue
                # The exact implementation depends on your project's workflow setup
                
                self.logger.warning("Alternative method not yet implemented - manual review needed")
                return False
                
            except Exception as label_error:
                self.logger.warning(f"Label status approach failed: {str(label_error)}")
            
            self.logger.error(f"Failed to move datarow {data_row_id} to review status - no working method found")
            return False
            
        except Exception as e:
            self.logger.error(f"Error moving datarow {data_row_id} to review status: {str(e)}")
            log_data_row_action(data_row_id, "MOVE_TO_REVIEW_FAILED", str(e))
            return False 