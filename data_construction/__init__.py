"""
SaFeR-Steer Data Construction Module

Implements the three-stage data construction pipeline:
    - Stage 1: Intent Decomposition & Reconstruction
    - Stage 2: Synthetic Bootstrapping  
    - Stage 3: Tutor-in-the-loop Agentic RL (data used in verl/)

Key Components:
    - pipeline.py: Main data construction pipeline
    - prompts/: Attack pattern templates and generation prompts
"""

__version__ = "0.1.0"
