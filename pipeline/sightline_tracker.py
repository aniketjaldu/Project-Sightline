"""
Sightline Custom Tracking Module

This module provides a custom tracking implementation inspired by ByteTrack
for both training data processing and inference, ensuring consistency across the entire pipeline.
Note: This is NOT the official ByteTrack implementation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math

from core.config import SIGHTLINE_TRACKER_CONFIG, CLASS_MAPPING
from utils.logger import setup_logger

class SimpleTracker:
    """Simplified custom tracker for Sightline project with bidirectional consistency"""
    
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.detections = []
        self.last_frame = -1
        self.last_center = (0, 0)
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        
    def update(self, detection: Dict, frame_num: int):
        """Update track with new detection"""
        self.detections.append(detection)
        self.last_frame = frame_num
        self.last_center = (detection['center_x'], detection['center_y'])
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
    
    def predict(self):
        """Predict next position (simplified)"""
        self.age += 1
        self.time_since_update += 1
        return self.last_center

class SightlineTracker:
    """
    Custom Sightline tracker for both training and inference workflows
    (formerly named ByteTrackProcessor - this is NOT the official ByteTrack)
    """
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = SIGHTLINE_TRACKER_CONFIG
        self.next_track_id = 0
        
    def process_training_annotations(self, labelbox_data: Dict) -> List[Dict]:
        """
        Process Labelbox training annotations using custom tracking to create consistent tracks
        
        Args:
            labelbox_data: Raw Labelbox annotation data
            
        Returns:
            List of detections with track IDs assigned
        """
        # Extract detections from Labelbox format
        detections = self._extract_labelbox_detections(labelbox_data)
        
        if not detections:
            return []
        
        # Get image dimensions for normalization
        img_width = labelbox_data.get('media_attributes', {}).get('width', 1920)
        img_height = labelbox_data.get('media_attributes', {}).get('height', 1080)
        
        # Apply tracking algorithm
        tracked_detections = self._apply_tracking(detections)
        
        # Add normalized bounding boxes
        for detection in tracked_detections:
            bbox_pixel = detection['bbox_pixel']
            x, y, w, h = bbox_pixel
            
            # Convert to normalized YOLO format (x_center, y_center, width, height)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height
            
            detection['bbox_normalized'] = [x_center, y_center, width, height]
        
        self.logger.info(f"Training SightlineTracker: {len(detections)} detections → {len(tracked_detections)} tracked")
        
        return tracked_detections
    
    def process_inference_detections(self, yolo_detections: List[Dict]) -> List[Dict]:
        """
        Process YOLO inference detections and apply custom tracking
        This ensures consistent track IDs between training and inference
        
        Args:
            yolo_detections: List of detections from YOLO inference
            
        Returns:
            List of tracked detections with consistent track IDs
        """
        try:
            # Reset for new inference run
            self.next_track_id = 0
            
            # Apply tracking to the detections
            tracked_detections = self._apply_tracking(yolo_detections)
            
            self.logger.info(f"Processed {len(yolo_detections)} detections into {len(tracked_detections)} tracked objects")
            
            # Log tracking summary
            track_counts = {}
            for detection in tracked_detections:
                track_id = detection['track_id']
                if track_id not in track_counts:
                    track_counts[track_id] = 0
                track_counts[track_id] += 1
            
            self.logger.info(f"Created {len(track_counts)} tracks: {list(track_counts.keys())}")
            
            return tracked_detections
            
        except Exception as e:
            self.logger.error(f"Error in inference detection processing: {str(e)}")
            return []
    
    def _extract_labelbox_detections(self, labelbox_data: Dict) -> List[Dict]:
        """Extract detections from Labelbox annotation format"""
        detections = []
        
        # Navigate Labelbox structure to find frame annotations
        projects = labelbox_data.get('projects', {})
        if not projects:
            return detections
        
        # Get first project's labels
        project_data = next(iter(projects.values()))
        labels = project_data.get('labels', [])
        
        for label in labels:
            if label.get('label_kind') != 'Video':
                continue
                
            # The correct path is: label -> annotations -> frames
            annotations = label.get('annotations', {})
            frames = annotations.get('frames', {})
            
            for frame_str, frame_data in frames.items():
                frame_num = int(frame_str)
                
                objects = frame_data.get('objects', {})
                for obj_id, obj in objects.items():
                    bbox = obj.get('bounding_box', {})
                    if not bbox:
                        continue
                    
                    detection = {
                        'frame_number': frame_num,
                        'class_name': obj.get('value', '').lower(),
                        'confidence': 1.0,  # Labelbox annotations are ground truth
                        'bbox_pixel': [
                            bbox['left'], bbox['top'], 
                            bbox['width'], bbox['height']
                        ],
                        'center_x': bbox['left'] + bbox['width'] / 2,
                        'center_y': bbox['top'] + bbox['height'] / 2
                    }
                    detections.append(detection)
        
        return detections
    
    def _apply_tracking(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply simplified custom tracking algorithm with Valorant optimizations
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of detections with track IDs
        """
        # Sort detections by frame number
        detections.sort(key=lambda x: x['frame_number'])
        
        # Group detections by frame
        frame_detections = {}
        for det in detections:
            frame = det['frame_number']
            if frame not in frame_detections:
                frame_detections[frame] = []
            frame_detections[frame].append(det)
        
        # Initialize tracking
        active_tracks = []
        tracked_detections = []
        
        # Process each frame
        for frame_num in sorted(frame_detections.keys()):
            frame_dets = frame_detections[frame_num]
            
            # Valorant optimization: Limit detections per frame to top 10 by confidence
            if self.config.get('valorant_optimized', False):
                max_dets_per_frame = self.config.get('max_detections_per_frame', 10)
                frame_dets = sorted(frame_dets, key=lambda x: x['confidence'], reverse=True)[:max_dets_per_frame]
            
            # Update track ages
            for track in active_tracks:
                track.predict()
            
            # Split detections by confidence (two-stage strategy)
            high_conf_dets = [d for d in frame_dets if d['confidence'] >= self.config['track_thresh']]
            low_conf_dets = [d for d in frame_dets if d['confidence'] < self.config['track_thresh'] 
                           and d['confidence'] >= self.config['track_low_thresh']]
            
            # First association: high confidence detections with active tracks
            matched_tracks, unmatched_dets = self._associate_detections_to_tracks(
                high_conf_dets, active_tracks
            )
            
            # Update matched tracks
            for det_idx, track in matched_tracks:
                detection = high_conf_dets[det_idx]
                track.update(detection, frame_num)
                tracked_detections.append({**detection, 'track_id': track.track_id})
            
            # Remove tracks that haven't been updated
            active_tracks = [track for track in active_tracks 
                           if track.time_since_update <= self.config['track_buffer']]
            
            # Create new tracks for unmatched high confidence detections
            # Valorant optimization: Limit total number of active tracks
            max_tracks = self.config.get('max_tracks', float('inf'))
            for det_idx in unmatched_dets:
                if len(active_tracks) >= max_tracks:
                    # Remove the oldest track if we're at the limit
                    if self.config.get('valorant_optimized', False):
                        oldest_track = min(active_tracks, key=lambda t: t.last_frame)
                        active_tracks.remove(oldest_track)
                        self.logger.debug(f"Removed oldest track {oldest_track.track_id} to make room for new track")
                    else:
                        break  # Don't create new tracks if we're at the limit
                
                detection = high_conf_dets[det_idx]
                new_track = SimpleTracker(self.next_track_id)
                self.next_track_id += 1
                new_track.update(detection, frame_num)
                active_tracks.append(new_track)
                tracked_detections.append({**detection, 'track_id': new_track.track_id})
            
            # Second association: low confidence detections with remaining tracks
            if low_conf_dets and self.config.get('valorant_optimized', False):
                remaining_tracks = [t for t in active_tracks if t.time_since_update == 0]
                if len(remaining_tracks) < max_tracks:
                    matched_low_conf, _ = self._associate_detections_to_tracks(
                        low_conf_dets, [t for t in active_tracks if t.time_since_update > 0]
                    )
                    
                    for det_idx, track in matched_low_conf:
                        detection = low_conf_dets[det_idx]
                        track.update(detection, frame_num)
                        tracked_detections.append({**detection, 'track_id': track.track_id})
        
        return tracked_detections
    
    def _associate_detections_to_tracks(self, detections: List[Dict], tracks: List[SimpleTracker]) -> Tuple[List[Tuple[int, SimpleTracker]], List[int]]:
        """
        Associate detections to tracks using distance threshold
        
        Returns:
            matched_pairs [(det_idx, track)], unmatched_detection_indices
        """
        if not detections or not tracks:
            return [], list(range(len(detections)))
        
        matched_pairs = []
        unmatched_dets = []
        used_tracks = set()
        
        for i, det in enumerate(detections):
            best_track = None
            best_distance = float('inf')
            
            # Find closest track
            for track in tracks:
                if track in used_tracks:
                    continue
                    
                distance = self._calculate_distance(
                    (det['center_x'], det['center_y']),
                    track.last_center
                )
                
                if distance < best_distance and distance < self.config.get('distance_threshold', 100):
                    best_distance = distance
                    best_track = track
            
            if best_track is not None:
                matched_pairs.append((i, best_track))
                used_tracks.add(best_track)
            else:
                unmatched_dets.append(i)
        
        return matched_pairs, unmatched_dets
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) 