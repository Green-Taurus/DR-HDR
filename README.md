# HDR_OpenCV —— 纯 OpenCV 版 HDR 处理（独立研究用）

本文件夹是从 HDRAlgDemo 项目中独立出来的 OpenCV 实现，
**不依赖 Halcon / halconcpp.dll**。算法结构与 demo 的 "hdr" 处理一致，
与 demo 输出平均差约 0.005%（肉眼无差）。

## 文件

- `hdr_opencv.py` —— 算法实现（核心，含 CLI）
- `hdr_gui.py` —— PyQt5 图形界面
- `requirements.txt` —— 依赖

## 快速运行

```bash
python hdr_gui.py                          # 图形界面
python hdr_opencv.py 输入.tiff -o 输出.tiff # 命令行
# 可选参数：--intensity 20 --detail 3 --border 2 --gauss 20
```

## 算法流程（对应 demo 的 hdr）

1. `median(circle, r=1)` 十字中值去噪
2. 归一化：范围 = `百分位截断最大值(Max1) - 全量最小值(Min2)`，
   最亮像素会超过满量程、被 uint16 截断（与 demo 一致）
3. 对数域处理：
   - `L = ln(img)`，`N = L / max(L)`
   - `base = guided_filter(N, N, 窗口=border, eps=0.001) * max(L)`
   - `detail = median_sep3x3(L - base)`
   - `k = ln(intensity) / (max(base) - min(base))`
   - `out = exp(k*(base - max(base)) + detail*detail)`
4. 再归一化 → ×65535 → uint16
5. 高斯：`sigma = gauss % 11`（取奇、≥3），再重复 `int(gauss/11)` 次 sigma=11

## 参数含义

| 参数 | 含义 | 默认 |
|---|---|---|
| intensity | 亮度层压缩强度（k 的来源） | 20 |
| detail | 细节层放大倍数 | 3 |
| border | guided filter 均值窗口（偶数自动取奇） | 2 |
| gaussParam | 高斯平滑强度 | 20 |
| guided eps / 归一化截断% | GUI 高级参数（demo 内部常量） | 0.001 / 1% / 2% |

## 与 demo 的一致性

- 默认参数、50/10/5、5/3/20 多组真实输出验证：平均差 ≤ 0.017%
- 剩余差异来自 Halcon `scale_image` 查找表与 `median_separate` 的内部数值实现
- 需要逐像素一致时用 DLL 版（`HDRAlgHalcon` 项目）

## 研究建议

- 想理解算法：从 `hdr_process()` 入口读，逐步看 `_a850` / `_d170` / `_guided_filter`
- 想改效果：先调 GUI 四个主参数，再看高级参数
- 想对比：`HDRAlgHalcon`（DLL 版）是逐像素一致的基准
