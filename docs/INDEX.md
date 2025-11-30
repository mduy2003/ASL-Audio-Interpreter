# Documentation Index

This folder contains all project documentation organized by topic.

---

## Quick Navigation

### Start Here
- **[project_context.md](project_context.md)** - Project overview, goals, and current status

### Results & Performance
- **[TRAINING_RESULTS.md](TRAINING_RESULTS.md)** - Complete training history and final results (99.74% accuracy)

### Implementation Details
- **[MODEL_IMPROVEMENTS_V2.md](MODEL_IMPROVEMENTS_V2.md)** - Detailed V2 improvements documentation (86.51% → 99.74%)

---

## Document Summaries

### project_context.md
**Purpose:** High-level project overview  
**Contains:**
- Project description and goals
- System architecture overview
- Tools and libraries used
- Current model performance (99.74%)
- Future expansion plans

**Audience:** New team members, stakeholders, general overview

---

### TRAINING_RESULTS.md
**Purpose:** Complete training results and model performance history  
**Contains:**
- Full performance comparison (V0 → V1 → V2)
- Per-class accuracy breakdown
- Training configuration summary
- Validation against overfitting
- Model usage examples
- Key insights and lessons learned

**Audience:** Technical team, ML engineers, for understanding model performance

---

### MODEL_IMPROVEMENTS_V2.md
**Purpose:** Detailed documentation of V2 improvements  
**Contains:**
- All implemented improvements with before/after comparisons
- Architecture enhancements (4 blocks, 19.25M parameters)
- Training parameter optimizations
- Class weight boosting implementation
- Files modified and code changes
- Complete improvement timeline

**Audience:** Developers working on model improvements, technical deep-dive

---

## Version History

| Version | Date | Test Accuracy | Documentation |
|---------|------|---------------|---------------|
| V0 | Nov 10, 2025 | 31.48% | Archived (broken model) |
| V1 | Nov 10, 2025 | 86.51% | Archived in TRAINING_RESULTS.md |
| **V2** | **Nov 29, 2025** | **99.74%** | **Current (all files)** |

---

## Related Files

### Root Directory
- **[../README.md](../README.md)** - Main project README with setup instructions

### Output Files
- **../output/training_history.png** - Training/validation curves visualization
- **../output/confusion_matrix.png** - Per-class classification matrix

### Model Files
- **../models/asl_model_v2_production.h5** - Current production model (99.74%)

### Source Code
- **../src/model.py** - Model architecture implementation
- **../src/train.py** - Training pipeline with all improvements
- **../src/data_loader.py** - Data loading and augmentation

---

## Updates

**Last Updated:** November 29, 2025  
**Status:** Documentation complete for V2 (99.74% accuracy)  
**Next Update:** When V3 improvements are implemented (if needed)

---

*For questions or clarifications, contact the project team.*
