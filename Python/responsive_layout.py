"""Small reusable Qt layouts for width-adaptive desktop controls.

The acquisition pages are embedded both in their legacy standalone window and
in the unified studio.  A regular ``QHBoxLayout`` cannot reflow when either the
window becomes narrow or the system font becomes larger, so toolbars built
from it eventually elide their labels.  ``FlowLayout`` keeps every control in
the layout and moves whole controls to the next row instead.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets


class FlowLayout(QtWidgets.QLayout):
    """A wrapping layout based on Qt's flow-layout example.

    Widgets retain their normal size hint while there is room.  If one widget
    is wider than the available row (for example a word-wrapped status label),
    it is constrained to the viewport width and its height-for-width is
    honoured, preventing both horizontal clipping and geometry overlap.
    """

    def __init__(
        self,
        parent=None,
        *,
        margin: int = 0,
        horizontal_spacing: int = 8,
        vertical_spacing: int = 8,
    ):
        super().__init__(parent)
        self._items = []
        self._horizontal_spacing = int(horizontal_spacing)
        self._vertical_spacing = int(vertical_spacing)
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        while self.takeAt(0) is not None:
            pass

    def addItem(self, item):  # noqa: N802 - Qt virtual method name
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802 - Qt virtual method name
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):  # noqa: N802 - Qt virtual method name
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802 - Qt virtual method name
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self):  # noqa: N802 - Qt virtual method name
        return True

    def heightForWidth(self, width):  # noqa: N802 - Qt virtual method name
        return self._do_layout(QtCore.QRect(0, 0, max(0, int(width)), 0), True)

    def setGeometry(self, rect):  # noqa: N802 - Qt virtual method name
        super().setGeometry(rect)
        required_height = self._do_layout(rect, False)
        parent = self.parentWidget()
        if parent is not None and parent.minimumHeight() != required_height:
            # QWidget's generic size hint does not always propagate a custom
            # layout's height-for-width through another QVBoxLayout (notably
            # with QFrame + style-sheet borders on Qt 5).  Publishing the
            # current wrapped height prevents the parent layout from assigning
            # a one-row height after the items have already wrapped.
            parent.setMinimumHeight(required_height)

    def sizeHint(self):  # noqa: N802 - Qt virtual method name
        size = self.minimumSize()
        parent = self.parentWidget()
        if parent is not None and parent.width() > 0:
            size.setHeight(self.heightForWidth(parent.width()))
        return size

    def minimumSize(self):  # noqa: N802 - Qt virtual method name
        size = QtCore.QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QtCore.QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )
        return size

    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(),
            margins.top(),
            -margins.right(),
            -margins.bottom(),
        )
        available_width = max(1, effective.width())
        x = effective.x()
        y = effective.y()
        line_height = 0

        for item in self._items:
            if item.isEmpty():
                continue

            hint = item.sizeHint().expandedTo(item.minimumSize())
            item_width = min(max(1, hint.width()), available_width)
            next_x = x + item_width + self._horizontal_spacing
            if (
                x > effective.x()
                and next_x - self._horizontal_spacing > effective.right() + 1
            ):
                x = effective.x()
                y += line_height + self._vertical_spacing
                next_x = x + item_width + self._horizontal_spacing
                line_height = 0

            item_height = max(1, hint.height())
            if item.hasHeightForWidth():
                item_height = max(item_height, item.heightForWidth(item_width))
            if not test_only:
                item.setGeometry(
                    QtCore.QRect(x, y, item_width, item_height)
                )
            x = next_x
            line_height = max(line_height, item_height)

        return y + line_height - rect.y() + margins.bottom()


class ScrollContentWidget(QtWidgets.QWidget):
    """Scrollable content whose preferred size is its true layout minimum.

    Plot widgets and splitters often publish a very large preferred size left
    over from their standalone window.  Using that value inside a scroll area
    creates pages thousands of pixels tall.  This container instead follows
    the layout minimum, updating it after wrapping or a font/style change.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

    def sizeHint(self):  # noqa: N802 - Qt virtual method name
        layout = self.layout()
        if layout is not None:
            hint = layout.minimumSize()
            if hint.isValid():
                return hint
        return super().sizeHint()

    def event(self, event):
        result = super().event(event)
        if event.type() in (
            QtCore.QEvent.LayoutRequest,
            QtCore.QEvent.FontChange,
            QtCore.QEvent.ApplicationFontChange,
            QtCore.QEvent.StyleChange,
        ):
            QtCore.QTimer.singleShot(0, self._sync_layout_minimum)
        return result

    def _sync_layout_minimum(self):
        layout = self.layout()
        if layout is None:
            return
        hint = layout.minimumSize()
        if not hint.isValid():
            return
        if hint != self.minimumSize():
            self.setMinimumSize(hint)
        viewport = self.parentWidget()
        scroll_area = viewport.parentWidget() if viewport is not None else None
        sync = getattr(scroll_area, "sync_widget_size", None)
        if sync is not None:
            QtCore.QTimer.singleShot(0, sync)


class ResponsiveScrollArea(QtWidgets.QScrollArea):
    """A scroll area that expands content only to viewport-or-minimum size."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.viewport().installEventFilter(self)

    def setWidget(self, widget):  # noqa: N802 - Qt virtual method name
        super().setWidget(widget)
        QtCore.QTimer.singleShot(0, self.sync_widget_size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self.sync_widget_size)

    def eventFilter(self, watched, event):
        if watched is self.viewport() and event.type() == QtCore.QEvent.Resize:
            QtCore.QTimer.singleShot(0, self.sync_widget_size)
        return super().eventFilter(watched, event)

    def sync_widget_size(self):
        widget = self.widget()
        if widget is None:
            return
        minimum = widget.minimumSize().expandedTo(widget.minimumSizeHint())
        target = self.viewport().size().expandedTo(minimum)
        if widget.size() != target:
            widget.resize(target)
        layout = widget.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()


def compact_field(label_text: str, field: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Keep a short field label attached to its editor while a flow wraps."""

    container = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    label = QtWidgets.QLabel(label_text)
    label.setBuddy(field)
    row.addWidget(label)
    row.addWidget(field, 1)
    return container


def configure_form_layout(form: QtWidgets.QFormLayout) -> None:
    """Allow long form rows and fields to adapt to the viewport width."""

    form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    form.setFormAlignment(QtCore.Qt.AlignTop)
    form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)


__all__ = [
    "FlowLayout",
    "ResponsiveScrollArea",
    "ScrollContentWidget",
    "compact_field",
    "configure_form_layout",
]
