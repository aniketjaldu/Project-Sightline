#!/usr/bin/env python3
"""
Coordinate Alignment Validator and Converter

This module handles alignment issues between YOLO and Labelbox coordinate systems.
Common issues include:
- Coordinate format differences (normalized vs pixel)
- Origin point differences (center vs top-left)
- Bounding box format differences (x,y,w,h vs x1,y1,x2,y2)
- Video resolution mismatches
- Frame indexing inconsistencies

Author: AI Assistant
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import labelbox.types as lb_types
from utils.logger import setup_logger

class CoordinateAlignmentValidator:
    """Validates and corrects coordinate system alignment between YOLO and Labelbox"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        
    def validate_coordinate_alignment(
        self, 
        video_path: Path,
        yolo_annotations: Dict[str, Any],
        labelbox_annotations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of coordinate alignment between YOLO and Labelbox.
        
        Args:
            video_path: Path to the video file
            yolo_annotations: YOLO inference annotations
            labelbox_annotations: Labelbox annotations for comparison (optional)
            
        Returns:
            Validation report with detected issues and suggested fixes
        """
        report = {
            'video_path': str(video_path),
            'issues': [],
            'suggestions': [],
            'coordinate_systems': {},
            'validation_passed': True
        }
        
        try:
            # Get video properties
            video_info = self._get_video_info(video_path)
            report['video_info'] = video_info
            
            # 1. Validate YOLO coordinate system
            yolo_issues = self._validate_yolo_coordinates(yolo_annotations, video_info)
            report['coordinate_systems']['yolo'] = yolo_issues
            
            # 2. Validate coordinate format consistency
            format_issues = self._validate_coordinate_formats(yolo_annotations, video_info)
            report['issues'].extend(format_issues)
            
            # 3. Validate frame indexing
            frame_issues = self._validate_frame_indexing(yolo_annotations, video_info)
            report['issues'].extend(frame_issues)
            
            # 4. Validate bounding box ranges
            bbox_issues = self._validate_bbox_ranges(yolo_annotations, video_info)
            report['issues'].extend(bbox_issues)
            
            # 5. Compare with Labelbox annotations if available
            if labelbox_annotations:
                comparison_issues = self._compare_coordinate_systems(
                    yolo_annotations, labelbox_annotations, video_info
                )
                report['issues'].extend(comparison_issues)
            
            # Generate suggestions based on found issues
            report['suggestions'] = self._generate_alignment_suggestions(report['issues'])
            
            # Overall validation status
            high_severity_issues = [issue for issue in report['issues'] if issue.get('severity') == 'high']
            report['validation_passed'] = len(high_severity_issues) == 0
            
            # Log summary
            self._log_validation_summary(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error during coordinate alignment validation: {str(e)}")
            report['issues'].append({
                'type': 'validation_error',
                'message': f"Validation failed: {str(e)}",
                'severity': 'high'
            })
            report['validation_passed'] = False
            return report
    
    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Extract video information using OpenCV"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'aspect_ratio': cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        }
        
        cap.release()
        return info
    
    def _validate_yolo_coordinates(self, annotations: Dict[str, Any], video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate YOLO coordinate system consistency"""
        issues = []
        
        frames = annotations.get('frames', {})
        if not frames:
            return {'issues': [{'type': 'no_annotations', 'message': 'No frame annotations found', 'severity': 'medium'}]}
        
        # Check coordinate ranges across all annotations
        all_coords = []
        all_bbox_formats = set()
        
        for frame_num, objects in frames.items():
            for obj in objects:
                bbox = obj.get('bounding_box', {})
                if bbox:
                    # Collect coordinate values
                    coords = [bbox.get('left', 0), bbox.get('top', 0), 
                             bbox.get('width', 0), bbox.get('height', 0)]
                    all_coords.extend(coords)
                    
                    # Detect bounding box format
                    if all(key in bbox for key in ['left', 'top', 'width', 'height']):
                        all_bbox_formats.add('xywh_pixel')
                    elif all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
                        all_bbox_formats.add('xyxy_pixel')
                    elif all(key in bbox for key in ['x_center', 'y_center', 'width', 'height']):
                        all_bbox_formats.add('xywh_normalized')
        
        # Analyze coordinate ranges
        if all_coords:
            min_coord = min(all_coords)
            max_coord = max(all_coords)
            
            # Check if coordinates are normalized (0-1) or pixel values
            if min_coord >= 0 and max_coord <= 1:
                coord_type = 'normalized'
            elif min_coord >= 0 and max_coord <= max(video_info['width'], video_info['height']):
                coord_type = 'pixel'
            else:
                coord_type = 'unknown'
                issues.append({
                    'type': 'coordinate_range_issue',
                    'message': f"Coordinates outside expected range: {min_coord} to {max_coord}",
                    'severity': 'high'
                })
        
        return {
            'coordinate_type': coord_type if all_coords else 'unknown',
            'bbox_formats': list(all_bbox_formats),
            'coordinate_range': (min_coord, max_coord) if all_coords else (0, 0),
            'issues': issues
        }
    
    def _validate_coordinate_formats(self, annotations: Dict[str, Any], video_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate coordinate format consistency"""
        issues = []
        
        frames = annotations.get('frames', {})
        bbox_formats = set()
        
        for frame_num, objects in frames.items():
            for obj in objects:
                bbox = obj.get('bounding_box', {})
                if bbox:
                    # Check for mixed coordinate formats
                    has_xywh = all(key in bbox for key in ['left', 'top', 'width', 'height'])
                    has_xyxy = all(key in bbox for key in ['x1', 'y1', 'x2', 'y2'])
                    has_center = all(key in bbox for key in ['x_center', 'y_center', 'width', 'height'])
                    
                    if sum([has_xywh, has_xyxy, has_center]) > 1:
                        issues.append({
                            'type': 'mixed_coordinate_formats',
                            'message': f"Mixed coordinate formats detected in frame {frame_num}",
                            'severity': 'high',
                            'frame': frame_num
                        })
                    elif not any([has_xywh, has_xyxy, has_center]):
                        issues.append({
                            'type': 'unknown_coordinate_format',
                            'message': f"Unknown coordinate format in frame {frame_num}",
                            'severity': 'high',
                            'frame': frame_num
                        })
        
        return issues
    
    def _validate_frame_indexing(self, annotations: Dict[str, Any], video_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate frame indexing consistency"""
        issues = []
        
        frames = annotations.get('frames', {})
        if not frames:
            return issues
        
        frame_numbers = [int(f) for f in frames.keys()]
        
        # Check frame number ranges
        min_frame = min(frame_numbers)
        max_frame = max(frame_numbers)
        
        if min_frame < 0:
            issues.append({
                'type': 'negative_frame_index',
                'message': f"Negative frame indices detected (min: {min_frame})",
                'severity': 'high'
            })
        
        if max_frame >= video_info['total_frames']:
            issues.append({
                'type': 'frame_index_out_of_range',
                'message': f"Frame index {max_frame} exceeds video length ({video_info['total_frames']} frames)",
                'severity': 'high'
            })
        
        # Check for frame gaps (might indicate tracking issues)
        if len(frame_numbers) > 1:
            sorted_frames = sorted(frame_numbers)
            gaps = []
            for i in range(1, len(sorted_frames)):
                if sorted_frames[i] - sorted_frames[i-1] > 1:
                    gaps.append((sorted_frames[i-1], sorted_frames[i]))
            
            if gaps and len(gaps) > len(frame_numbers) * 0.1:  # More than 10% gaps
                issues.append({
                    'type': 'excessive_frame_gaps',
                    'message': f"Many frame gaps detected ({len(gaps)} gaps), may indicate tracking issues",
                    'severity': 'medium',
                    'gaps': gaps[:5]  # Show first 5 gaps
                })
        
        return issues
    
    def _validate_bbox_ranges(self, annotations: Dict[str, Any], video_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate bounding box coordinate ranges"""
        issues = []
        
        frames = annotations.get('frames', {})
        
        for frame_num, objects in frames.items():
            for obj_idx, obj in enumerate(objects):
                bbox = obj.get('bounding_box', {})
                if not bbox:
                    continue
                
                # Check if bbox is within video bounds
                left = bbox.get('left', 0)
                top = bbox.get('top', 0)
                width = bbox.get('width', 0)
                height = bbox.get('height', 0)
                
                # Validate pixel coordinates
                if left < 0 or top < 0:
                    issues.append({
                        'type': 'negative_coordinates',
                        'message': f"Negative coordinates in frame {frame_num}, object {obj_idx}: left={left}, top={top}",
                        'severity': 'high',
                        'frame': frame_num,
                        'object_index': obj_idx
                    })
                
                if left + width > video_info['width'] or top + height > video_info['height']:
                    issues.append({
                        'type': 'bbox_out_of_bounds',
                        'message': f"Bounding box extends beyond video bounds in frame {frame_num}, object {obj_idx}",
                        'severity': 'high',
                        'frame': frame_num,
                        'object_index': obj_idx,
                        'bbox': bbox
                    })
                
                # Check for zero or negative dimensions
                if width <= 0 or height <= 0:
                    issues.append({
                        'type': 'invalid_bbox_dimensions',
                        'message': f"Invalid bbox dimensions in frame {frame_num}, object {obj_idx}: width={width}, height={height}",
                        'severity': 'high',
                        'frame': frame_num,
                        'object_index': obj_idx
                    })
        
        return issues
    
    def _compare_coordinate_systems(
        self, 
        yolo_annotations: Dict[str, Any], 
        labelbox_annotations: Dict[str, Any],
        video_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare YOLO and Labelbox coordinate systems for consistency"""
        issues = []
        
        # This would be implemented to compare actual coordinate values
        # For now, we'll focus on format differences
        
        yolo_frames = set(yolo_annotations.get('frames', {}).keys())
        labelbox_frames = set(labelbox_annotations.get('frames', {}).keys()) if labelbox_annotations else set()
        
        # Check frame coverage
        if labelbox_frames:
            missing_in_yolo = labelbox_frames - yolo_frames
            missing_in_labelbox = yolo_frames - labelbox_frames
            
            if missing_in_yolo:
                issues.append({
                    'type': 'frame_coverage_mismatch',
                    'message': f"Frames present in Labelbox but missing in YOLO: {sorted(list(missing_in_yolo))[:10]}",
                    'severity': 'medium'
                })
            
            if missing_in_labelbox:
                issues.append({
                    'type': 'frame_coverage_mismatch',
                    'message': f"Frames present in YOLO but missing in Labelbox: {sorted(list(missing_in_labelbox))[:10]}",
                    'severity': 'medium'
                })
        
        return issues
    
    def _generate_alignment_suggestions(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions based on detected issues"""
        suggestions = []
        
        issue_types = {issue['type'] for issue in issues}
        
        if 'coordinate_range_issue' in issue_types:
            suggestions.append("Consider normalizing coordinates to 0-1 range for consistency")
        
        if 'mixed_coordinate_formats' in issue_types:
            suggestions.append("Standardize bounding box format to left,top,width,height pixel coordinates")
        
        if 'bbox_out_of_bounds' in issue_types:
            suggestions.append("Clip bounding boxes to video dimensions before importing to Labelbox")
        
        if 'frame_index_out_of_range' in issue_types:
            suggestions.append("Validate frame indices against actual video length")
        
        if 'negative_coordinates' in issue_types:
            suggestions.append("Ensure all coordinates are non-negative before processing")
        
        if 'invalid_bbox_dimensions' in issue_types:
            suggestions.append("Filter out bounding boxes with zero or negative dimensions")
        
        if 'excessive_frame_gaps' in issue_types:
            suggestions.append("Review tracking algorithm - consider interpolation for missing frames")
        
        return suggestions
    
    def _log_validation_summary(self, report: Dict[str, Any]) -> None:
        """Log a summary of the validation results"""
        issues = report.get('issues', [])
        
        if report['validation_passed']:
            self.logger.info("✅ Coordinate alignment validation PASSED")
        else:
            self.logger.warning("⚠️  Coordinate alignment validation found issues")
        
        # Group issues by severity
        high_issues = [i for i in issues if i.get('severity') == 'high']
        medium_issues = [i for i in issues if i.get('severity') == 'medium']
        low_issues = [i for i in issues if i.get('severity') == 'low']
        
        if high_issues:
            self.logger.error(f"High severity issues: {len(high_issues)}")
            for issue in high_issues[:3]:  # Show first 3
                self.logger.error(f"  - {issue['type']}: {issue['message']}")
        
        if medium_issues:
            self.logger.warning(f"Medium severity issues: {len(medium_issues)}")
            for issue in medium_issues[:2]:  # Show first 2
                self.logger.warning(f"  - {issue['type']}: {issue['message']}")
        
        if low_issues:
            self.logger.info(f"Low severity issues: {len(low_issues)}")
        
        suggestions = report.get('suggestions', [])
        if suggestions:
            self.logger.info("Alignment suggestions:")
            for suggestion in suggestions[:3]:
                self.logger.info(f"  💡 {suggestion}")
    
    def fix_coordinate_alignment(
        self, 
        annotations: Dict[str, Any], 
        video_info: Dict[str, Any],
        target_format: str = 'labelbox_pixel'
    ) -> Dict[str, Any]:
        """
        Fix coordinate alignment issues in annotations.
        
        Args:
            annotations: YOLO annotations to fix
            video_info: Video information for coordinate conversion
            target_format: Target coordinate format ('labelbox_pixel', 'yolo_normalized', etc.)
            
        Returns:
            Fixed annotations
        """
        try:
            fixed_annotations = annotations.copy()
            fixes_applied = []
            
            frames = fixed_annotations.get('frames', {})
            
            for frame_num, objects in frames.items():
                for obj_idx, obj in enumerate(objects):
                    bbox = obj.get('bounding_box', {})
                    if not bbox:
                        continue
                    
                    original_bbox = bbox.copy()
                    
                    # Fix coordinate format and range issues
                    if target_format == 'labelbox_pixel':
                        fixed_bbox = self._convert_to_labelbox_pixel_format(
                            bbox, video_info
                        )
                        
                        # Clip to video bounds
                        fixed_bbox = self._clip_bbox_to_bounds(fixed_bbox, video_info)
                        
                        # Validate dimensions
                        if fixed_bbox['width'] > 0 and fixed_bbox['height'] > 0:
                            obj['bounding_box'] = fixed_bbox
                            if fixed_bbox != original_bbox:
                                fixes_applied.append({
                                    'frame': frame_num,
                                    'object': obj_idx,
                                    'fix_type': 'coordinate_conversion',
                                    'original': original_bbox,
                                    'fixed': fixed_bbox
                                })
                        else:
                            # Remove invalid bounding boxes
                            fixes_applied.append({
                                'frame': frame_num,
                                'object': obj_idx,
                                'fix_type': 'removed_invalid_bbox',
                                'original': original_bbox
                            })
            
            # Remove objects with invalid bounding boxes
            for frame_num, objects in frames.items():
                valid_objects = []
                for obj in objects:
                    bbox = obj.get('bounding_box', {})
                    if bbox and bbox.get('width', 0) > 0 and bbox.get('height', 0) > 0:
                        valid_objects.append(obj)
                frames[frame_num] = valid_objects
            
            self.logger.info(f"Applied {len(fixes_applied)} coordinate alignment fixes")
            
            return {
                'annotations': fixed_annotations,
                'fixes_applied': fixes_applied,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error fixing coordinate alignment: {str(e)}")
            return {
                'annotations': annotations,
                'fixes_applied': [],
                'success': False,
                'error': str(e)
            }
    
    def _convert_to_labelbox_pixel_format(self, bbox: Dict[str, Any], video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert bounding box to Labelbox pixel format (left, top, width, height)"""
        
        # If already in correct format, return as-is
        if all(key in bbox for key in ['left', 'top', 'width', 'height']):
            return bbox.copy()
        
        # Convert from normalized YOLO format (x_center, y_center, width, height)
        if all(key in bbox for key in ['x_center', 'y_center', 'width', 'height']):
            x_center = bbox['x_center'] * video_info['width']
            y_center = bbox['y_center'] * video_info['height']
            width = bbox['width'] * video_info['width']
            height = bbox['height'] * video_info['height']
            
            return {
                'left': int(x_center - width/2),
                'top': int(y_center - height/2),
                'width': int(width),
                'height': int(height)
            }
        
        # Convert from xyxy format
        if all(key in bbox for key in ['x1', 'y1', 'x2', 'y2']):
            return {
                'left': int(bbox['x1']),
                'top': int(bbox['y1']),
                'width': int(bbox['x2'] - bbox['x1']),
                'height': int(bbox['y2'] - bbox['y1'])
            }
        
        # If format is unknown, return original
        return bbox.copy()
    
    def _clip_bbox_to_bounds(self, bbox: Dict[str, Any], video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Clip bounding box to video bounds"""
        clipped = bbox.copy()
        
        # Ensure bbox is within video bounds
        clipped['left'] = max(0, min(clipped['left'], video_info['width'] - 1))
        clipped['top'] = max(0, min(clipped['top'], video_info['height'] - 1))
        
        # Adjust width and height to stay within bounds
        max_width = video_info['width'] - clipped['left']
        max_height = video_info['height'] - clipped['top']
        
        clipped['width'] = max(1, min(clipped['width'], max_width))
        clipped['height'] = max(1, min(clipped['height'], max_height))
        
        return clipped 