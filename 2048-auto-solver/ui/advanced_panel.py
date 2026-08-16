"""The collapsed-by-default Advanced panel: search depth, heuristic weights, confidence
threshold, settle-frame count, and the dry-run toggle.

Collapsed by default per the "no jargon where the user can see it" rule -- everything on this
panel assumes the reader already knows what a search depth or a confidence threshold is; the
main wizard never uses these terms.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from core.heuristics import HeuristicWeights


class AdvancedPanel(QWidget):
    """A collapsible section (QToolButton toggle + hidden content widget) editing AppConfig."""

    config_changed = Signal(object)  # emits an updated AppConfig

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._toggle = QToolButton(self)
        self._toggle.setText("Advanced")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle.toggled.connect(self._on_toggled)
        outer.addWidget(self._toggle)

        self._content = QWidget(self)
        self._content.setVisible(False)
        form = QFormLayout(self._content)

        self._depth_spin = QSpinBox(self._content)
        self._depth_spin.setRange(1, 8)
        self._depth_spin.setValue(self._config.base_search_depth)
        form.addRow("Search depth", self._depth_spin)

        self._confidence_spin = QDoubleSpinBox(self._content)
        self._confidence_spin.setRange(0.0, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setValue(self._config.confidence_threshold)
        form.addRow("Confidence threshold", self._confidence_spin)

        self._settle_frames_spin = QSpinBox(self._content)
        self._settle_frames_spin.setRange(1, 10)
        self._settle_frames_spin.setValue(self._config.settle_frames)
        form.addRow("Settle-frame count", self._settle_frames_spin)

        self._monotonicity_spin = self._weight_spin(form, "Weight: monotonicity", self._config.heuristic_weights.monotonicity)
        self._smoothness_spin = self._weight_spin(form, "Weight: smoothness", self._config.heuristic_weights.smoothness)
        self._empty_cells_spin = self._weight_spin(form, "Weight: empty cells", self._config.heuristic_weights.empty_cells)
        self._corner_max_spin = self._weight_spin(form, "Weight: corner max", self._config.heuristic_weights.corner_max)
        self._merge_potential_spin = self._weight_spin(form, "Weight: merge potential", self._config.heuristic_weights.merge_potential)

        self._dry_run_checkbox = QCheckBox("Dry run (decide but never press keys)", self._content)
        self._dry_run_checkbox.setChecked(self._config.dry_run)
        form.addRow(self._dry_run_checkbox)

        for widget in (
            self._depth_spin,
            self._confidence_spin,
            self._settle_frames_spin,
            self._monotonicity_spin,
            self._smoothness_spin,
            self._empty_cells_spin,
            self._corner_max_spin,
            self._merge_potential_spin,
        ):
            widget.valueChanged.connect(self._emit_config_changed)
        self._dry_run_checkbox.toggled.connect(self._emit_config_changed)

        outer.addWidget(self._content)

    def _weight_spin(self, form: QFormLayout, label: str, initial: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self._content)
        spin.setRange(-10.0, 10.0)
        spin.setSingleStep(0.1)
        spin.setValue(initial)
        form.addRow(label, spin)
        return spin

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def current_config(self) -> AppConfig:
        weights = HeuristicWeights(
            monotonicity=self._monotonicity_spin.value(),
            smoothness=self._smoothness_spin.value(),
            empty_cells=self._empty_cells_spin.value(),
            corner_max=self._corner_max_spin.value(),
            merge_potential=self._merge_potential_spin.value(),
        )
        return replace(
            self._config,
            base_search_depth=self._depth_spin.value(),
            confidence_threshold=self._confidence_spin.value(),
            settle_frames=self._settle_frames_spin.value(),
            dry_run=self._dry_run_checkbox.isChecked(),
            heuristic_weights=weights,
        )

    def _emit_config_changed(self, *_args) -> None:
        self._config = self.current_config()
        self.config_changed.emit(self._config)
