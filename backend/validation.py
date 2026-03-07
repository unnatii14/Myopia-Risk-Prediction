"""
Input validation schemas for API endpoints
"""
from typing import Optional, Literal

class ValidationError(Exception):
    """Raised when input validation fails"""
    pass


def validate_screening_data(data: dict) -> dict:
    """
    Validate and sanitize screening form data
    
    Args:
        data: Raw request body from frontend
        
    Returns:
        Validated and type-cast data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = []
    
    # ── Required fields ──
    required = ['age', 'sex', 'screenTime', 'nearWork', 'outdoorTime', 'sports']
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        raise ValidationError("; ".join(errors))
    
    validated = {}
    
    # ── Age: 5-18 years ──
    try:
        age = int(data['age'])
        if not 5 <= age <= 18:
            errors.append("Age must be between 5 and 18 years")
        validated['age'] = age
    except (ValueError, TypeError):
        errors.append("Age must be a valid integer")
    
    # ── Sex: male/female ──
    sex = str(data.get('sex', '')).lower()
    if sex not in ['male', 'female']:
        errors.append("Sex must be 'male' or 'female'")
    validated['sex'] = sex
    
    # ── Height: 80-200 cm (optional) ──
    height = data.get('height', 0)
    try:
        height = float(height) if height else 0
        if height != 0 and not 80 <= height <= 200:
            errors.append("Height must be between 80-200 cm or 0 if unknown")
        validated['height'] = height
    except (ValueError, TypeError):
        errors.append("Height must be a valid number")
    
    # ── Weight: 15-100 kg (optional) ──
    weight = data.get('weight', 0)
    try:
        weight = float(weight) if weight else 0
        if weight != 0 and not 15 <= weight <= 100:
            errors.append("Weight must be between 15-100 kg or 0 if unknown")
        validated['weight'] = weight
    except (ValueError, TypeError):
        errors.append("Weight must be a valid number")
    
    # ── Screen time: 0-24 hours/day ──
    try:
        screen_time = float(data['screenTime'])
        if not 0 <= screen_time <= 24:
            errors.append("Screen time must be between 0-24 hours/day")
        validated['screenTime'] = screen_time
    except (ValueError, TypeError):
        errors.append("Screen time must be a valid number")
    
    # ── Near work: 0-24 hours/day ──
    try:
        near_work = float(data['nearWork'])
        if not 0 <= near_work <= 24:
            errors.append("Near work must be between 0-24 hours/day")
        validated['nearWork'] = near_work
    except (ValueError, TypeError):
        errors.append("Near work must be a valid number")
    
    # ── Outdoor time: 0-24 hours/day ──
    try:
        outdoor_time = float(data['outdoorTime'])
        if not 0 <= outdoor_time <= 24:
            errors.append("Outdoor time must be between 0-24 hours/day")
        validated['outdoorTime'] = outdoor_time
    except (ValueError, TypeError):
        errors.append("Outdoor time must be a valid number")
    
    # ── Logical constraint: total time ≤ 24 hours ──
    if 'screenTime' in validated and 'nearWork' in validated and 'outdoorTime' in validated:
        total = validated['screenTime'] + validated['nearWork'] + validated['outdoorTime']
        if total > 24:
            errors.append(f"Total screen + near work + outdoor time ({total:.1f}h) cannot exceed 24 hours/day")
    
    # ── Sports: regular/occasional/rare ──
    sports = str(data.get('sports', '')).lower()
    if sports not in ['regular', 'occasional', 'rare']:
        errors.append("Sports must be 'regular', 'occasional', or 'rare'")
    validated['sports'] = sports
    
    # ── Family history: boolean or null ──
    family_history = data.get('familyHistory')
    if family_history is None or family_history in ['null', None]:
        validated['familyHistory'] = None
    elif family_history in [True, 'true']:
        validated['familyHistory'] = True
    elif family_history in [False, 'false']:
        validated['familyHistory'] = False
    else:
        errors.append("Family history must be true, false, or null")
    
    # ── Parents myopic: none/one/both ──
    parents = str(data.get('parentsMyopic', 'none')).lower()
    if parents not in ['none', 'one', 'both']:
        errors.append("Parents myopic must be 'none', 'one', or 'both'")
    validated['parentsMyopic'] = parents
    
    # ── Vitamin D: boolean or null ──
    vitamin_d = data.get('vitaminD')
    if vitamin_d is None or vitamin_d in ['null', None]:
        validated['vitaminD'] = None
    elif vitamin_d in [True, 'true']:
        validated['vitaminD'] = True
    elif vitamin_d in [False, 'false']:
        validated['vitaminD'] = False
    else:
        errors.append("Vitamin D must be true, false, or null")
    
    # ── State: valid Indian state (optional) ──
    valid_states = [
        'Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu',
        'Telangana', 'Maharashtra', 'Gujarat', 'Delhi'
    ]
    state = data.get('state', 'Andhra Pradesh')
    if state not in valid_states:
        # Default to Andhra Pradesh instead of erroring
        validated['state'] = 'Andhra Pradesh'
    else:
        validated['state'] = state
    
    if errors:
        raise ValidationError("; ".join(errors))
    
    return validated
