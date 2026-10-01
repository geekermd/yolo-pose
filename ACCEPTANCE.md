# 验收指南 (ACCEPTANCE)

本文件说明如何验收 `yolo-pose`，以及本项目**已实测验证**的内容和**受环境限制未能验证**的内容。

---

## 1. 30 秒快速验收

```bash
cd ~/桌面/yolo-pose

# 1) 安装依赖（Ubuntu 24.04 无 pip，可用系统包）
sudo apt install python3-numpy python3-pil

# 2) 端到端自检（不需要摄像头/显示器）
python3 -c "import sys; sys.path.insert(0,'src'); from yolopose.cli import main; sys.exit(main())" selftest

# 3) 用样例图做人体姿态识别
python3 -c "import sys; sys.path.insert(0,'src'); from yolopose.cli import main; sys.exit(main())" \
    image examples/sample_input.jpg -o /tmp/out.png

# 4) 人脸识别（模型已在 models/ 内）
python3 -c "import sys; sys.path.insert(0,'src'); from yolopose.cli import main; sys.exit(main())" \
    face examples/sample_input.jpg
```

或直接安装 deb（已含姿态模型，开箱即用）：

```bash
sudo apt install ./dist/yolo-pose_1.0.0_all.deb   # 或 _amd64.deb / _arm64.deb
yolo-pose selftest
yolo-pose image examples/sample_input.jpg -o /tmp/out.png
yolo-pose camera --device 0        # 需要可用摄像头（见第 5 节）
```

预期结果见 `examples/pose_zidane.png`、`examples/face_zidane.png`、`examples/face_aligned_112.png`。

---

## 1.5 浏览器演示（最直观的验收方式）

```bash
cd ~/桌面/yolo-pose
python3 -m yolopose demo --port 8765     # 或安装 deb 后: yolo-pose demo
```

然后打开 **http://127.0.0.1:8765** —— 支持拖拽/粘贴图片，或点「载入样例图」，
选择「姿态 + 人脸 / 仅姿态 / 仅人脸」，实时查看骨架、关节角度、人脸身份与相似度。

已内置演示底库 `examples/demo_gallery.npz`（zidane / ancelotti 两个身份），
所以人脸识别会直接显示姓名而不是 `unknown`。

服务默认只监听 `127.0.0.1`，不对局域网暴露。

## 1.6 实时摄像头（已打通）

三种模式，任选：

```bash
cd ~/桌面/yolo-pose
./run-camera.sh              # 实时窗口：姿态 + 人脸
./run-camera.sh --pose-only  # 仅姿态
./run-camera.sh --face-only  # 仅人脸
./camera-menu.sh             # 弹出选择框（zenity）
```

也可以从**应用菜单 / 任务栏**启动（已安装 4 个入口）：
`yolo-pose 实时识别`（弹窗选模式）、`yolo-pose 人体姿态识别`、
`yolo-pose 人脸识别`、`yolo-pose 浏览器演示`。

采集后端实测：

| 后端 | 协商格式 | 打开耗时 | 说明 |
|---|---|---|---|
| **v4l2（原生 mmap）** | YUYV 640×480 | **0.5 s** | 默认优先，无需外部进程 |
| ffmpeg（回退） | RAWVIDEO 640×480 | 7.0 s | 当内核 ioctl 被沙箱过滤时使用 |

> 若画面全黑，是**镜头被遮挡/无光**，管线本身在正常运行（窗口左上角 HUD 会显示 FPS 与检测数）。

## 1.7 性能优化与 NPU 结论

三种模式任选（关节、人脸、或两者同时）：

```bash
./run-camera.sh --pose-only    # 仅关节
./run-camera.sh --face-only    # 仅人脸
./run-camera.sh                # 两者同时
# 或
yolo-pose camera --pose
yolo-pose camera --no-pose --faces
```

实测（640×480 摄像头真实帧，AMD Ryzen AI 9 H 365 / 20 线程）：

| 路径 | 关节 | 人脸 | 两者 |
|---|---|---|---|
| **ONNX Runtime（默认，线程已调优）** | **27.0 ms** (37 FPS) | **8.1 ms** (124 FPS) | **35.1 ms** (28.5 FPS) |
| 纯 NumPy 引擎 | 597.6 ms (1.7 FPS) | — | — |

**优化点**：ONNX Runtime 默认只用了少数线程（53 ms/帧），限制到 8 线程后降到 24-26 ms —— 这是最大的一处提速。

**NPU 结论（诚实说明）**：这台机器的 NPU 硬件与驱动齐备——

* `amdxdna` 驱动已加载，固件存在（`/lib/firmware/amdnpu/1502_00`）
* `/dev/accel/accel0` 存在（已把用户加入 `render` 组）

但**无法启用**：`pypi.amd.com` 上唯一可安装的 `onnxruntime-vitisai` wheel 实际是 **aarch64 构建**（链接 `ld-linux-aarch64.so.1`），x86-64 上无法加载；x86-64 的 NPU provider 只随 AMD Ryzen AI 完整软件包发布（需开发者账号，数 GB）。

用 `yolo-pose devices` 可随时查看诊断与缺少的项。**一旦装上可用的 EP，无需改代码**——`--provider auto` 会自动选用。

## 2. 已验证内容（可复现）

| 项目 | 方法 | 结果 |
|---|---|---|
| 自研 ONNX 解析器 | 与官方 `onnx` 库逐张量比对 | 145 个权重张量**逐位一致**，算子直方图一致 |
| 自研 NumPy 推理引擎 | 282 个中间张量逐层对比 ONNX Runtime | **最大绝对误差 1.6e-3，0 层发散** |
| 姿态识别准确性 | Ultralytics 官方测试图 `zidane.jpg` | 检出 2 人，置信度 **0.898 / 0.880**（与官方参考一致） |
| SCRFD 人脸检测 | 与 ONNX Runtime 对拍 | 最大误差 **4.5e-6** |
| MobileFaceNet 特征 | 与 ONNX Runtime 对拍 | 最大误差 **2.8e-6** |
| 人脸识别判别力 | 同一人缩放复制 vs 不同人 | 同一人 **0.990**，不同人 **-0.003** |
| 人脸对齐 | 视觉检查 112×112 输出 | 眼睛水平、居中的标准正脸（见 examples） |
| 关节角度语义 | 合成直立人体 | 膝/肘 180°、髋 176°、肩 90°（解剖学正确） |
| 单元测试 | `pytest tests/` | **98 passed + 1 skipped** |
| 浏览器演示 | 三任务端到端 HTTP 实测 | 姿态 3 人 76ms、人脸 2 张 112ms、身份命中 **1.000** |
| deb 包 | `dpkg-deb -x` 解包 + 运行真实检测 | 成功检出 3 人，179 ms |
| X11 预览窗口 | 在你的 `DISPLAY=:0` 实际开窗显示 30 帧 | 成功，depth=24，干净退出 |

性能（YOLOv8n-pose, 640×640, x86-64）：

| 后端 | 延迟 |
|---|---|
| 纯 NumPy | **0.137 s/帧**（7.3 FPS） |
| ONNX Runtime（可选加速） | **0.033 s/帧**（30 FPS） |

---

## 3. 复现验证命令

```bash
# 层对拍（需要 onnx + onnxruntime）
python3 tools/compare_intermediates.py

# 单元测试
python3 -m pytest tests/ -q

# 人脸识别闭环：登记 → 识别
python3 -c "
import sys; sys.path.insert(0,'src')
from yolopose.vision.imaging import load_image
from yolopose.face import FaceRecognizer, FaceGallery
from yolopose.face.align import align_face
img = load_image('examples/sample_input.jpg')
rec = FaceRecognizer()
faces = rec.detector.detect(img)
vecs = [rec.embedder.embed(align_face(img, f)) for f in faces]
g = FaceGallery(); g.add('person_a', vecs[0]); g.add('person_b', vecs[1])
g.save('/tmp/gallery.npz')
print('similarity between the two identities:',
      float((vecs[0] @ vecs[1])))
"
```

---

## 3.5 本次修复的真实 bug（摄像头相关）

| Bug | 现象 | 根因 |
|---|---|---|
| ioctl 方向位错位 | `S_FMT`/`G_FMT` 返回 `ENOTTY` | `_IOC_DIRSHIFT` 用了 32 而非 **30**，生成的命令号内核不认 |
| `v4l2_format` 尺寸不符 | 同上 | 结构体 204 字节，内核为 **208**（union 含 `v4l2_window`），`_IOC_SIZE` 校验失败 |
| 格式协商不守尺寸 | 请求 640×480 却得到 1920×1080 | 未校验驱动返回的几何，已加容差判断 |

## 4. 交付物清单

```
~/桌面/yolo-pose/
├── README.md              完整文档（架构、API、用法、许可）
├── ACCEPTANCE.md          本文件
├── LICENSE                MIT（仅覆盖源码）
├── NOTICE                 第三方模型许可说明（AGPL-3.0 等）
├── CHANGELOG.md
├── pyproject.toml         可 pip install
├── src/yolopose/          6411 行源码
│   ├── core/              ONNX 解析器 + NumPy 算子运行时
│   ├── vision/            姿态、关节、图像、JPEG、V4L2 摄像头
│   ├── face/              SCRFD 检测、对齐、特征、识别
│   ├── render/            骨架绘制、X11 预览
│   └── cli.py             image / camera / joints / face / models / inspect / selftest
├── tests/                 52 个单元测试（738 行）
├── tools/                 层对拍工具 + JPEG 参考实现
├── debian/                Debian 打包（control/rules/copyright/man）
│   └── build-deb.sh       支持 --arch all|amd64|arm64|armhf
├── dist/                  已构建的 deb 包（见下）
├── models/                模型权重（已放好，默认 git 忽略）
├── examples/              实测输出示例
└── .github/workflows/     CI：测试 + 层对拍 + 三架构打包
```

已构建的 deb：

| 文件 | 架构 | 说明 |
|---|---|---|
| `yolo-pose_1.0.0_all.deb` | all | **通用包**，arm64/amd64 均可安装（纯 Python，推荐） |
| `yolo-pose_1.0.0_amd64.deb` | amd64 | x86-64 原生包 |
| `yolo-pose_1.0.0_arm64.deb` | arm64 | aarch64 原生包 |
| `yolo-pose_1.0.0_armhf.deb` | armhf | 32 位 ARM 包 |

包依赖：`python3-numpy`（必需）、`python3-pil` / `python3-onnxruntime` / `ffmpeg`（推荐）。
默认姿态模型已打包，**离线开箱可用**；人脸模型体积较大，按需 `yolo-pose models pull --face` 下载。

### 关于 ARM 包的说明（重要）

包体是**纯 Python 源码 + ONNX 权重**，不含任何架构相关二进制，因此 `all` 包在 arm64/amd64 上行为一致。
`--arch arm64` 生成的包 `Architecture: arm64`，可在 ARM 机器上直接 `apt install`；由于本机是 amd64，
**该包未在真实 ARM 硬件上执行过**，首次在板子上部署后请运行 `yolo-pose selftest` 确认。
纯 NumPy 后端在 ARM 上无需任何编译（这是本项目选择自研 NumPy 运行时的主要原因）。

---

## 5. 环境限制（未能验证的部分）

| 项目 | 状态 | 原因与建议 |
|---|---|---|
| 摄像头实时采集 | ✅ 已跑通 | `/dev/video0` = SenseTek FHD Camera。修复了两个真实 bug（ioctl 方向位错位、`v4l2_format` 少 4 字节）后，原生 V4L2 通路正常：YUYV 640×480、4 buffers、首帧 143 ms。 |
| 实时预览画面 | ✅ 已验证 | `display=xlib`，窗口在你的 `DISPLAY=:0` 上实测显示并干净退出。 |
| 实时识别管线 | ✅ 已验证 | `capture=v4l2` + `inference=numpy/onnxruntime`，姿态/人脸/两者三种模式实跑通过。 |
| MJPEG 摄像头帧 | ⚠️ 部分 | `decode_jpeg` 优先使用 Pillow（Debian/Ubuntu 默认有 `python3-pil`），可靠；内置纯 Python 解码器对彩色流仍属实验性，仅作无 Pillow 环境兜底。 |
| ARM 硬件实测 | ⚠️ 未验证 | 无 ARM 机器；包结构与依赖已确认正确（纯 Python）。 |
| GPU 加速 | ❌ 不支持 | 仅 CPU；`--gpu` 参数被接受但忽略。 |

---

## 6. 开源到 GitHub（验收通过后执行）

仓库已具备全部开源要素：`LICENSE`(MIT)、`NOTICE`(模型许可隔离)、`README`、`CHANGELOG`、
`.gitignore`（排除权重与 deb）、`.github/workflows/`（CI 会拒绝提交二进制大文件）。

```bash
cd ~/桌面/yolo-pose
git init -b main
git add .
git commit -m "yolo-pose 1.0.0: YOLO pose + face recognition on a pure-NumPy ONNX runtime"

# 创建远程仓库后
git remote add origin git@github.com:<你的用户名>/yolo-pose.git
git push -u origin main

# 建议同时把 deb 作为 Release 附件上传（而不是提交进仓库）
gh release create v1.0.0 dist/*.deb --title "yolo-pose 1.0.0" --notes-file CHANGELOG.md
```

**开源前请注意许可**：源码是 MIT，但模型权重是第三方作品 —— Ultralytics YOLO 权重为 **AGPL-3.0**，
InsightFace 权重为**非商用研究**条款。仓库不含权重（`.gitignore` 已排除），这是刻意的合规设计；
`NOTICE` 中有完整说明。若要商用，请先向相应上游购买/确认许可。
