import os
from enum import Enum

standard_lead_order = ["I", 
    "II", 
    "III", 
    "aVR", 
    "aVL", 
    "aVF", 
    "V1", 
    "V2", 
    "V3", 
    "V4", 
    "V5", 
    "V6"
]
lead_to_idx = {lead: idx for idx, lead in enumerate(standard_lead_order)}


