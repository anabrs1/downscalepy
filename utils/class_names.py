"""
Land Use Management (LUM) class name mappings.

Complete mapping of all 33 LUM classes with display names.
"""

from typing import Dict

# Complete LUM class name mapping
LUM_CLASS_NAMES: Dict[int, str] = {
    1: "Primary forest",
    2: "Close-to-nature management",
    3: "Combined objective forestry",
    4: "Intensive forestry",
    5: "Very intensive forestry",
    6: "Irrigated arable cropland",
    7: "Rainfed intensive arable cropland",
    8: "Rainfed extensive arable cropland",
    9: "Irrigated permanent cropland",
    10: "Rainfed intensive permanent cropland",
    11: "Rainfed extensive permanent cropland",
    12: "Intensive heterogeneous cropland classes",
    13: "Extensive heterogeneous cropland classes",
    14: "Agroforestry",
    15: "Very high density managed pasture system",
    16: "High density managed pasture system",
    17: "Moderate density managed pasture system",
    18: "Low density managed pasture system",
    19: "Very high density managed grassland",
    20: "High density managed grassland",
    21: "Moderate density managed grassland",
    22: "Low density managed grassland",
    23: "Rough grazing",
    24: "Silvo-pastoral agroforestry",
    25: "Managed semi-natural and natural grassland",
    26: "Unmanaged semi-natural and natural grassland",
    27: "High intensity urban (buildings)",
    28: "Medium intensity urban (buildings)",
    29: "Low intensity urban (buildings)",
    30: "Infrastructure",
    31: "Other urban",
    32: "Other land cover",
    33: "Unclassified",
}


def get_class_name(class_code: int) -> str:
    """
    Get display name for LUM class code.
    
    Parameters
    ----------
    class_code : int
        LUM class code (1-33)
    
    Returns
    -------
    str
        Class name or "Unknown class {code}" if not found
    """
    return LUM_CLASS_NAMES.get(class_code, f"Unknown class {class_code}")


def get_all_class_codes() -> list:
    """Get list of all valid LUM class codes."""
    return list(LUM_CLASS_NAMES.keys())


def format_class_list(class_codes: list) -> str:
    """
    Format list of class codes with names.
    
    Parameters
    ----------
    class_codes : list
        List of LUM class codes
    
    Returns
    -------
    str
        Formatted string with codes and names
    
    Examples
    --------
    >>> format_class_list([3, 4, 5])
    '3 (Combined objective forestry), 4 (Intensive forestry), 5 (Very intensive forestry)'
    """
    formatted = []
    for code in sorted(class_codes):
        name = get_class_name(code)
        formatted.append(f"{code} ({name})")
    return ", ".join(formatted)
