"""HDRAlgDemo 风格图形界面（纯 OpenCV 实现，无 Halcon 依赖）。

参数与 demo 一致：intensity / detail / border / gaussParam；
另开放了 demo 内部使用的三个常量：guided eps 与两次归一化的百分位截断。
"""

import os
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from hdr_opencv import hdr_process


def to8(img):
    """把任意灰度图转成 8 位用于显示（按 min/max 拉伸）。"""
    img = np.asarray(img, dtype=np.float64)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn) * 255.0
    else:
        img = np.zeros_like(img)
    return np.clip(img, 0, 255).astype(np.uint8)


def numpy_to_qimage(img8):
    h, w = img8.shape
    return QImage(img8.data, w, h, w, QImage.Format_Grayscale8).copy()


class ProcessThread(QThread):
    """后台执行 HDR 处理，避免界面卡顿。"""

    finished_ok = pyqtSignal(object, dict, float)
    failed = pyqtSignal(str)

    def __init__(self, src, params, parent=None):
        super().__init__(parent)
        self.src = src
        self.params = params

    def run(self):
        try:
            t0 = time.time()
            out = hdr_process(self.src, **self.params)
            self.finished_ok.emit(out, dict(self.params), time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class HdrWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDR 处理器")
        self.resize(1200, 760)

        self.src = None          # uint16 原图
        self.result = None       # uint16 结果
        self._thread = None
        self._pending = False
        self._closing = False
        self.job_id = 0

        self._build_ui()

        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(350)
        self.debounce.timeout.connect(self._on_process)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        # 顶部工具栏
        bar = QHBoxLayout()
        self.btn_open = QPushButton("打开图像")
        self.btn_process = QPushButton("处理")
        self.btn_save = QPushButton("保存结果")
        self.btn_save.setEnabled(False)
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(["显示结果", "显示原图", "并排对比"])
        self.chk_auto = QCheckBox("参数变化自动处理")
        self.chk_auto.setChecked(True)
        self.lb_info = QLabel("未加载图像")
        self.lb_info.setStyleSheet("color:#666;")
        for w in (self.btn_open, self.btn_process, self.btn_save, self.cmb_view,
                  self.chk_auto):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.lb_info)
        root.addLayout(bar)

        body = QHBoxLayout()

        # 图像显示区
        view_box = QVBoxLayout()
        self.view = QLabel("打开一张 16 位 TIFF 图像开始")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 480)
        self.view.setStyleSheet("background:#202020;color:#aaa;"
                                "border:1px solid #444;")
        self.lb_status = QLabel("就绪")
        self.lb_status.setStyleSheet("color:#0a0;")
        view_box.addWidget(self.view, 1)
        view_box.addWidget(self.lb_status)
        body.addLayout(view_box, 1)

        # 右侧参数面板
        panel = QVBoxLayout()
        grp = QGroupBox("参数设置（与 demo 一致）")
        form = QFormLayout(grp)
        self.sp_intensity = QSpinBox()
        self.sp_intensity.setRange(2, 300)
        self.sp_intensity.setValue(20)
        self.sp_detail = QSpinBox()
        self.sp_detail.setRange(1, 30)
        self.sp_detail.setValue(3)
        self.sp_border = QSpinBox()
        self.sp_border.setRange(1, 200)
        self.sp_border.setValue(2)
        self.sp_gauss = QSpinBox()
        self.sp_gauss.setRange(0, 30)
        self.sp_gauss.setValue(20)
        form.addRow("intensity（2~300）", self.sp_intensity)
        form.addRow("detail（1~30）", self.sp_detail)
        form.addRow("border（1~200）", self.sp_border)
        form.addRow("gaussParam（0~30）", self.sp_gauss)
        panel.addWidget(grp)

        adv = QGroupBox("高级参数（demo 内部常量，一般保持默认）")
        adv.setCheckable(True)
        adv.setChecked(False)
        aform = QFormLayout(adv)
        self.ds_eps = QDoubleSpinBox()
        self.ds_eps.setRange(0.0001, 0.1)
        self.ds_eps.setDecimals(4)
        self.ds_eps.setSingleStep(0.0005)
        self.ds_eps.setValue(0.001)
        self.ds_cut1 = QDoubleSpinBox()
        self.ds_cut1.setRange(0.0, 10.0)
        self.ds_cut1.setDecimals(2)
        self.ds_cut1.setSingleStep(0.1)
        self.ds_cut1.setValue(1.0)
        self.ds_cut2 = QDoubleSpinBox()
        self.ds_cut2.setRange(0.0, 10.0)
        self.ds_cut2.setDecimals(2)
        self.ds_cut2.setSingleStep(0.1)
        self.ds_cut2.setValue(2.0)
        aform.addRow("guided 滤波 eps", self.ds_eps)
        aform.addRow("首次归一化截断 %", self.ds_cut1)
        aform.addRow("二次归一化截断 %", self.ds_cut2)
        panel.addWidget(adv)
        panel.addStretch(1)
        body.addLayout(panel, 0)

        root.addLayout(body, 1)
        self.setCentralWidget(central)

        # 信号
        self.btn_open.clicked.connect(self.open_image)
        self.btn_process.clicked.connect(self._on_process)
        self.btn_save.clicked.connect(self.save_result)
        self.cmb_view.currentIndexChanged.connect(self._refresh_view)
        for w in (self.sp_intensity, self.sp_detail, self.sp_border, self.sp_gauss,
                  self.ds_eps, self.ds_cut1, self.ds_cut2):
            w.valueChanged.connect(self._on_param_changed)

    # --------------------------------------------------------------- actions
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "", "图像文件 (*.tif *.tiff *.png *.jpg *.bmp)")
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法读取图像：{path}")
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype == np.uint8:
            img = (img.astype(np.uint32) * 257).astype(np.uint16)
            QMessageBox.information(self, "提示", "8 位图像已按 16 位读取（×257）。")
        elif img.dtype != np.uint16:
            QMessageBox.warning(self, "错误",
                                f"不支持的图像类型：{img.dtype}")
            return
        self.src = np.ascontiguousarray(img)
        self.result = None
        self.btn_save.setEnabled(False)
        self.lb_info.setText(os.path.basename(path))
        self._refresh_view()
        self._on_process()

    def _params(self):
        return {
            "intensity": self.sp_intensity.value(),
            "detail": self.sp_detail.value(),
            "border": self.sp_border.value(),
            "gauss": self.sp_gauss.value(),
            "guided_eps": self.ds_eps.value(),
            "norm_eps1": self.ds_cut1.value() / 100.0,
            "norm_eps2": self.ds_cut2.value() / 100.0,
        }

    def _on_param_changed(self):
        if self.chk_auto.isChecked() and self.src is not None:
            self.debounce.start()

    def _on_process(self):
        if self.src is None or self._closing:
            return
        self._pending = True
        if self._thread is not None and self._thread.isRunning():
            self.lb_status.setText("处理中…（参数已更新，稍后重跑）")
            return
        self._start_job()

    def _start_job(self):
        if self._closing:
            return
        self._pending = False
        self.job_id += 1
        job = self.job_id
        params = self._params()
        self.lb_status.setText("处理中…")
        self.btn_process.setEnabled(False)
        t = ProcessThread(self.src, params)
        self._thread = t
        t.finished_ok.connect(
            lambda res, p, s, j=job: self._on_done(res, p, s, j))
        t.failed.connect(self._on_fail)
        t.finished.connect(self._on_thread_finished)
        t.start()

    def _on_thread_finished(self):
        if self._pending:
            QTimer.singleShot(0, self._start_job)

    def _on_done(self, result, params, seconds, job):
        if job != self.job_id:
            return                      # 过期结果，丢弃
        self.result = result
        self.btn_save.setEnabled(True)
        self.btn_process.setEnabled(True)
        self.lb_status.setText(
            f"完成（{seconds:.2f}s）  min={result.min()} max={result.max()}")
        self._refresh_view()

    def _on_fail(self, msg):
        self.btn_process.setEnabled(True)
        self.lb_status.setText("处理失败")
        QMessageBox.critical(self, "处理失败", msg)

    def save_result(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "hdr_out.tiff",
            "TIFF 16位 (*.tiff *.tif);;PNG 8位 (*.png);;BMP 8位 (*.bmp)")
        if not path:
            return
        if path.lower().endswith((".tif", ".tiff")):
            ok = cv2.imwrite(path, self.result)
        else:
            ok = cv2.imwrite(path, to8(self.result))
        if ok:
            self.lb_status.setText(f"已保存：{os.path.basename(path)}")
        else:
            QMessageBox.warning(self, "错误", "保存失败")

    def _refresh_view(self):
        mode = self.cmb_view.currentIndex()
        if self.src is None:
            self.view.setText("打开一张 16 位 TIFF 图像开始")
            return
        if mode == 0:        # 结果
            if self.result is None:
                self.view.setText("尚未处理，点击“处理”或等待自动处理")
                return
            disp = to8(self.result)
        elif mode == 1:      # 原图
            disp = to8(self.src)
        else:                # 并排
            left = to8(self.src)
            right = to8(self.result) if self.result is not None else np.zeros_like(left)
            disp = np.hstack([left, right])
        self._show_array(disp)

    def _show_array(self, arr8):
        arr8 = np.ascontiguousarray(arr8)
        h, w = arr8.shape
        max_w = max(self.view.width(), 1)
        max_h = max(self.view.height(), 1)
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            arr8 = cv2.resize(arr8, (max(int(w * scale), 1), max(int(h * scale), 1)),
                              interpolation=cv2.INTER_AREA)
        qimg = numpy_to_qimage(arr8)
        self.view.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self._closing = True
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(10000)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_view()


def main():
    app = QApplication(sys.argv)
    win = HdrWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
