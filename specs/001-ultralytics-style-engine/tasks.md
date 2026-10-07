# Tasks: Ultralytics-Inspired Decoupled Training Engine

**Feature**: `001-ultralytics-style-engine`  
**Plan Reference**: [`plan.md`](./plan.md)  
**Spec Reference**: [`spec.md`](./spec.md)  
**Created**: 2026-10-07  
**Status**: Completed  

---

## Phase 1: Setup & Project Infrastructure
- [x] T001 Setup unified package directory structure `src/tasdf/` and subpackages (`engine/`, `models/`, `losses/`, `data/`, `utils/`)
- [x] T002 Create root package definition `pyproject.toml` to support editable install (`pip install -e .`)

## Phase 2: Foundational Components (Core Math & Models)
- [x] T003 [P] Relocate and package core models (`unet_blocks.py`, `unet_backbone.py`, `dual_heads.py`, `itgm.py`, `final_model.py`) into `src/tasdf/models/`
- [x] T004 [P] Relocate and package core loss functions (`losses.py`: `MainLoss`, `sigmoid_rampup`) into `src/tasdf/losses/`
- [x] T005 [P] Create utility modules for config parsing and metrics (`src/tasdf/utils/config.py`, `src/tasdf/utils/metrics.py`)

## Phase 3: [US4] Dual-Stream Telemetry Engine
- [x] T006 [US4] Implement `DualLogger` in `src/tasdf/engine/logger.py` supporting formatted ASCII console/`train.log` and structured `metrics.json`
- [x] T007 [P] [US4] Write unit tests for `DualLogger` in `tests/test_logger.py` verifying text table and JSON stream outputs

## Phase 4: [US2] Dataset-Agnostic Abstraction Layer
- [x] T008 [US2] Implement `BaseEmbryoDataset` interface in `src/tasdf/data/base.py`
- [x] T009 [P] [US2] Implement `COCOEmbryoDataset` in `src/tasdf/data/coco.py` reading arbitrary paths via data config
- [x] T010 [P] [US2] Implement `FolderEmbryoDataset` in `src/tasdf/data/folder.py` reading image/mask directory pairs
- [x] T011 [US2] Implement unified dataset builder in `src/tasdf/data/builder.py` with dynamic format dispatch
- [x] T012 [P] [US2] Write unit tests for data builder in `tests/test_data_builder.py` using in-memory synthetic datasets

## Phase 5: [US3] High-Throughput & Fault-Tolerant Engine (Trainer)
- [x] T013 [US3] Implement `Trainer` class in `src/tasdf/engine/trainer.py` with AMP (FP16), Adam optimizer, loss logging, and dynamic ramp-up
- [x] T014 [US3] Implement full-state serialization (`save_checkpoint`) and seamless resumption (`load_checkpoint`, `resume=True`) in `Trainer`
- [x] T015 [P] [US3] Write unit tests for `Trainer` in `tests/test_trainer.py` verifying training step, AMP, checkpoint save and resume

## Phase 6: [US5] Evaluation & Prediction Engines (Validator & Predictor)
- [x] T016 [US5] Implement `Validator` class in `src/tasdf/engine/validator.py` computing per-class and mean Dice/IoU
- [x] T017 [US5] Implement `Predictor` class in `src/tasdf/engine/predictor.py` generating raw masks, color overlays, and 3-panel composites
- [x] T018 [P] [US5] Write unit tests for `Validator` and `Predictor` in `tests/test_engine.py`

## Phase 7: [US1] High-Level Python Facade API
- [x] T019 [US1] Implement central `TASDF` facade class in `src/tasdf/api.py` orchestrating `train()`, `val()`, and `predict()`
- [x] T020 [US1] Expose `TASDF` in `src/tasdf/__init__.py` for clean notebook imports (`from tasdf import TASDF`)
- [x] T021 [P] [US1] Write integration tests in `tests/test_api.py` validating 3-line Jupyter Notebook workflow

## Phase 8: Polish & Self-Contained Test Migration
- [x] T022 Rewrite `tests/conftest.py` with synthetic in-memory embryo fixtures, eliminating all external dataset paths
- [x] T023 Refactor root entrypoints (`train.py`, `eval.py`, `inference.py`) as thin CLI wrappers around `TASDF` API
- [x] T024 Remove redundant directories (`src/src/`, root `datasets/`), clean up dead code and run full `pytest` verification
