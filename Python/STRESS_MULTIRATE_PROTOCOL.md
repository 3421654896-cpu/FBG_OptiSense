# CH1 九光栅多速率应力扫描协议

本功能是显式启用、断电不保留的实验协议。未成功协商 v3/v4 的客户端仍得到真实 MAP45；固件不会把缓存行伪装成本帧采样。多速率会话固定为 CH1 独占。当前主机请求 v4，显式请求 v3 的旧客户端不会收到 profile=3。

## v4 固件扫描策略（兼容 v3）

每帧均按原顺序完整回放 45 行标定 DAC，包括 `order2` 隐藏前驱和 5 ms 保持。调度器只决定哪些行执行 CH1 ADC 转换：

- `MAP45` (`profile=0`)：45 行全部新鲜，用于首帧缓存/基线初始化和完整谱形复核。
- `SURVEY9` (`profile=1`)：无接触时，每个光栅只采一个认证高斜率哨兵，用于全空间接触发现；它不具备双斜坡中心位移估计条件。
- `TRACK13` (`profile=2`)：两个最强 ROI 各采 local offset 1/2/3，共 6 点；其余 7 个光栅各采 1 个哨兵，共 13 点。因此活动期每帧仍能看到 9 个光栅，新接触可替换任一 ROI。
- `TRACK11_SINGLE` (`profile=3`，仅 v4)：主 ROI 三点加其余 8 个光栅的共同哨兵。secondary 必须为 `0xFF`，严格检查 11 点 bitmap；不是 11 个完整峰中心。

首次触发必须进入 TRACK13_WIDE。仅连续 8 个宽域帧确认同一紧凑响应区后，允许 TRACK11_SINGLE：主响应占比至少 60%、次响应小于主响应 35%、有效响应个数不超过 1.8，且无其他哨兵达到原始 ADC 触发阈值。主 ROI 本帧三点还必须未饱和、有可分辨的内部隆起。空间排序只使用九个共同哨兵，按认证斜率的倒数归一化（Q8 权重 `[136,79,82,205,128,134,182,71,177]`）；额外 ROI 点不能偏置空间排名。v3 保留原始 ADC 差值排序。

第二响应出现、主响应变化、缺失/越界哨兵或形状不合格，下一帧立即回 TRACK13。连续 4 帧安静后释放到 SURVEY9。这些门槛是保守工程启发式，不是已训练的位置/面积模型；不含噪声模型或实际接触面积标定。

哨兵 local offset 为 `[3,3,3,3,1,3,1,1,1]`，来源是认证报告 `ch1_measured_template_selection_v4_certified_20260905.json`（SHA-256 `2330F3D5091744329F0170B8823BDE89D3BDDCB5897DACE67C64A7113C8E9C3E`）：每峰在可靠权重合格的 local 1/3 中选 `abs(selected_template_slope_v_per_nm)` 较大者。

默认 `map_period_frames=0`，表示连续带宽模式：会话首帧仍强制真实 MAP45，之后不自动插入 MAP。无接触时跑 SURVEY9，检出活动后连续跑 TRACK13；连续 4 帧低于 release 阈值后才解除活动状态。长期哨兵漂移由安静基线跟踪，完整峰形复核作为显式维护操作。

`map_period_frames=3..100` 保留周期复核语义：例行 MAP 只在“已释放且连续安静 4 帧”时执行；活动期会推迟，不得每 30 帧打断动态通道。非零周期模式为防长期接触时谱形无限漂移，在 120 帧调度边界强制 MAP45。这次 MAP 必然造成大于 33.333 ms 的同峰新鲜间隔，所以 v3 必须置位 `bandwidth_discontinuity`，不伪装连续。显式 ForceMap 若发生在活动期也使用同一标志。

主机必须丢弃已标记 MAP 的动态间隔及其后第一个 TRACK 过渡，等到两个连续、未标记的 TRACK13 后再恢复导数/带宽判定。

基线仅在未锁定接触且连续 4 帧低于 release 阈值后，对本帧真正新鲜的行做 `alpha=1/64` 的 Q6 EMA。接触帧和缓存行不会被学入基线。

## B5 4D v1 扩展字节布局

| 偏移 | 长度 | 含义 |
|---:|---:|---|
| 0..27 | 28 | 旧版 board status，保持不变 |
| 28..31 | 4 | 帧序号，大端 `uint32` |
| 32..35 | 4 | 帧采集总时长 us，大端 `uint32` |
| 36 | 1 | 调度扩展版本：协商的 `3` 或 `4` |
| 37 | 1 | profile：MAP=0 / SURVEY=1 / TRACK13_WIDE=2 / TRACK11_SINGLE=3 |
| 38 | 1 | primary ROI，非 TRACK 为 `0xFF` |
| 39 | 1 | secondary ROI，仅 TRACK13 有效，其余为 `0xFF` |
| 40 | 1 | fresh count：45 / 9 / 13，v4 profile=3 必须为 11 |
| 41 | 1 | bit0..6：距上次 MAP 的帧数（封顶 127）；bit7：活动期强制 MAP 带宽中断 |
| 42..47 | 6 | 45-bit fresh bitmap，LSB 对应点索引，最后 3 bit 必须为 0 |
| 48..51 | 4 | `frame_start_ms`，MCU tick 大端 `uint32`，自然回绕 |
| 52..141 | 90 | 45 个大端 `uint16` 采样结束时间，相对帧开始，单位 2 us |

v3 总长度是 142 字节。新鲜点时间必须非 `0xFFFF`、按 45 行扫描顺序严格递增，且不超过帧时长；缓存/未采点必须是 `0xFFFF`。`0xFFFE` 对应可表示上限 131068 us，超限时固件取消多速率并回退真实 MAP45。`bandwidth_discontinuity` 只允许出现在 MAP45。

兼容规则：

- 28/36 字节旧扩展不含调度，视为真实全图。
- 40 字节非多速率全图把 `frame_start_ms` 放在 36..39。
- 52 字节 v1 为 SURVEY18/TRACK6，可用 bitmap 过滤新鲜点，但无逐点时间，不得用于 15 Hz 动态拟合。
- 142 字节 v2 为 SURVEY18/TRACK13，有逐点时间，byte41 全字节仍是 map age，没有带宽中断位。
- 142 字节 v3 为 SURVEY9/TRACK13，严格校验认证哨兵 bitmap 和带宽中断位。
- 142 字节 v4 增加 TRACK11_SINGLE；其余字节和 v3 一致。数据集保存调度版本、profile 编号/名称、逐点新鲜性和时间戳，训练不得把缓存当新采样。

## 安全启用与停用顺序

启用：

1. 发送现有模式命令进入 `EXTRA` (`FF FF 01 02`, byte8=2)，并观察到模式已切换。
2. 发送 808 字节 arm (`FF FF 01 08`)：byte4=1，byte5=map period，6..7=activate，8..9=release，**byte10=4**（显式兼容客户端可请求 3）。
3. 等待 20 字节 `FF FF 02 05` ACK，必须同时满足 byte10=`0x21`、byte11 等于请求版本，且回显配置与请求一致。旧固件的 byte11=0 只可诊断，不得继续启动减采样。
4. 发送模式命令进入 `STRESS` (byte8=0)。
5. 只有收到第一个合法且版本匹配的 MAP45 后才接受 SURVEY/TRACK 数据。

停用：先进入并确认 `EXTRA`，再发送 byte4=0 的同一 arm 命令并等待成功 ACK。断链、急停、OTA 入口和应用退出均要清缓存并回到未启用 MAP45；能通信时走上述停用顺序，不能通信时依赖固件的本地清理。

## 离线时序上限与上机验收

当前保守模型使用每帧 177 次 DAC 写、每次 90 us，每个 CH1 新鲜点 675 us，隐藏前驱 5000 us，固定开销 1000 us：

| profile | 新鲜点 | 估算周期 | 估算速率 |
|---|---:|---:|---:|
| MAP45 | 45 | 52.305 ms | 19.12 Hz |
| SURVEY9 | 9 | 28.005 ms | 35.71 Hz |
| TRACK13 | 13 | 30.705 ms | 32.57 Hz |
| TRACK11_SINGLE | 11 | 29.355 ms | 34.07 Hz |

这些值只是离线预算，且 TRACK13 在 33.333 ms Nyquist 上限下只剩约 2.63 ms 理论余量。SURVEY9 是接触发现器，不是完整峰形通道。实机验收必须连续记录至少 300 帧，逐 profile 统计周期，并对每个光栅使用逐点时间验证最大新鲜间隔（不只是平均或 p99）、丢帧、序号跳变和 gap 标志。每峰哨兵最大间隔小于 33.333 ms 仅满足 15 Hz 的采样时序必要条件，不能证明峰中心、激光器、模拟前端、机械结构及最终位置/真实面积估计具备 15 Hz 幅相响应。还需已知输入的动态按压与独立位置/面积真值验证；已标记强制 MAP 的区段必须报告为不连续。

## 1.0.26 配对瞬态诊断（不改变正常多速率协议）

`FF FF 03 01` 保持 808 字节请求；byte4..5 是前驱行、6..7 是目标行；byte8=3 显式使用已安装的 45 行认证表；byte9..12 为非零请求标签。索引限制 0..44，不接受任意电流。目标行 2 保留隐藏前驱与 5 ms 保持，其余目标行使用正常应力写入顺序。源行保持 100 ms，但未测窗口稳定性，不能仅凭等待时间声称源行已稳定。

响应是 560 字节：`D5 5D 02 03`、两个 u16 索引、u32 写入路径耗时、33/6 两个维度字节、u32 标签、6 个 u16 基线、33 个 `(u32 时间, 6*u16 ADC)`、尾 `5D D5`。整数大端。通道顺序 CH0/CH1/CH2/CH3/PDT/PDR。时间是自首个目标路径 DAC 写入开始的**六通道读取组开始时间**，不是每通道独立转换时间；读取一组实际约 55 us，早期预约点可能因转换未完成而延后。主机必须使用实测时戳，不可用预约时间替代。

诊断不重置共享 DWT 时钟，ADC 失败不发送成功响应。主机匹配标签、索引、维度、单调时间、ADC 范围与包尾，超时后执行停用及 SOA 关光并核验精确 ACK。

`capture_certified_ch1_transients.py --stable-teacher` 在每条曲线结束后再等待至少 200 ms，丢弃旧监测帧，在**不重写 DAC**的同一目标状态下采稳定单值窗口（总目标保持至少 300 ms）。保存原始窗口、噪声、漂移与是否通过判定，未通过的教师点不得训练。`analyze_certified_ch1_transients.py` 只使用截止时刻之前已完成读取的 ADC 组，并按整个光栅留出验证；多采样曲线模型的正则化在训练光栅内再次验证，不能借用测试光栅标签选参数。按每峰幅度及跨阻归一化比较，不以低跨阻电压较小冒充误差改善。该静态配对实验不证明连续扫描或 15 Hz 按压、定位、真实接触面积精度。

## 1.0.27 保留实际 profile 的直接 ADC 诊断

旧 `FF FF 03 02` 的 byte4=0 请求不变，仍强制 MAP45 并回复 `D6 6D 01`。新请求 byte4=1、byte5..8=非零 u32 标签，仅在已启用多速率的应力会话中接受。它只记录下一完整帧，不调用 ForceMap、不改变 ADC 时序/次数或 DAC 序列。待处理与正在捕获的标签分开，模式切换、停用、上一帧中断均不能把旧标签带入下一帧。

新响应 `D6 6D 02` 总长度 627 字节：

| 偏移 | 字段 |
|---|---|
| 0..5 | 标记、版本、u16 点数45、活动通道位图 |
| 6..9 | u32 请求标签 |
| 10..13 | u32 当前调度 planSequence（不是传输帧序号） |
| 14..17 | u32 frameStartMs |
| 18..21 | u32 采集耗时 µs |
| 22..27 | 协商版本、profile、primary、secondary、freshCount、mapAge/gap |
| 28..33 | 45 位 fresh bitmap |
| 34..35 | CH0/CH1 实际跨阻选择码 |
| 36..39 | u32 运行固件版本 |
| 40..624 | 45 条记录，每条13字节 |
| 625..626 | `6D D6` 包尾 |

每条记录为：首次直接 ADC u16、第二次直接 ADC u16、慢复读 ADC u16、RC 估计 u16、复读标志 u8、采样结束偏移 µs u32。未采行四个值和标志为零、时间为 `0xFFFFFFFF`，主机必须以 fresh bitmap 识别，不能把零当成实际转换。慢复读标志为零时，其数值也不可用作样本。全部整数大端。

`capture_preserved_ch1_raw.py` 默认先验证 v4 首帧 MAP45，等待100帧，再逐次捕获；两次诊断之间等待8个正常帧。核验标签、运行版本、跨阻、完整 profile/ROI/fresh bitmap、单调时间和ADC范围。诊断替代一个 USB 输出帧，故用于对比真实采样路径，不用来证明传输带宽。仍须最后精确停用及确认 SOA 关光。

## 1.0.28 真实扫描前缀与保持目标诊断

`FF FF 03 03` 为新的 808 字节 USB 请求，byte4..5 为安装表索引 0..44，byte6..9 为非零 u32 标签。只接受已协商 v4 的应力多速率会话，不接受任意 DAC 电流；与原始帧诊断互斥。请求在下一帧生效，不强制 MAP，也不在目标点之前添加 ADC 或改写 DAC。到目标点完成正常 DAC 写入后，停止前进，记录保持该目标的响应。目标点也允许是当前计划未采的候选点，但不会冒充正常 fresh 样本。

响应 `D7 7D 01` 共 1206 字节，整数大端：

| 字节 | 内容 |
|---|---|
| 0..3 | 标记、版本、状态：0 才表示全部采样及关光成功 |
| 4..13 | 标签 u32、固件版本 u32、目标索引 u16 |
| 14..19 | 协商版本、原 profile、primary、secondary、计划 freshCount、mapAge/gap |
| 20..31 | planSequence、frameStartMs、构造包头时距帧开始的微秒偏移，均 u32 |
| 32..37 | CH0/CH1 跨阻选择码、33 个瞬态组、25 个教师组、6 通道、显式诊断中断标志1 |
| 38..43 | 原计划 bitmap；不是整段诊断的新鲜采样 bitmap |
| 44..1203 | 58组 `(u32 ADC组开始µs, u32 ADC组结束µs, 六个u16 ADC)` |
| 1204..1205 | `7D D7` |

组内通道顺序 CH0/CH1/CH2/CH3/PDT/PDR。组时刻从目标 DAC 写入完成后进入诊断函数时算起；不用预约时间替代实际时间。早期密集采样使部分预约点顺延，此后采样行为不同于生产模式的两次 CH1 读取，不能将两者混作等效时序。包头 byte28..31 在构造包头时读取，并非精确 DAC 写完时刻；早期 JSON 中旧字段 `target_write_end_frame_offset_us` 应按此含义解读，新主机改名 `header_build_frame_offset_us`。因果分析只使用该组结束时间，不使用这个近似帧偏移。

33 组瞬态沿用 0..100 ms 时间表；25 组教师在保持同一 DAC 的 300..780 ms 期间每20ms采一次。长等待期间保留温控/保护服务，检测到主机输入、USB关闭、保护关光或 ADC 错误则失败退出。成功和失败均停止当前会话并关光，不发正常光谱帧、不更新模型，不将诊断后的不同光学历史继续送入快速估计。

`capture_sparse_held_trajectory.py` 每次重新确认跨阻、协商会话、验证初始MAP、预热100帧再请求；每次诊断结束都需要显式重启下一会话。教师窗口复用单值模式的噪声/漂移判据，保存全部直接ADC及是否合格。最后再次要求精确停用和关光 ACK。

`analyze_sparse_held_trajectory.py` 对每个截止时刻只取已完成的读取组；不做未来插值或模型训练。图和报告中的“采样后缀进入容差”只指已观测的瞬态样本，描述性容差取 `max(3码, 教师值的10%)`，不能解释为连续稳定时间、15 Hz 压力带宽、定位/真实面积准确率。
# v1.0.29 / explicitly negotiated schedule v5: G8-right experiment

Schedule v4 remains the default in the GUI and command builders. Version 5 is
an opt-in optical-path experiment, not a declaration of 15 Hz pressure accuracy.
It uses the same 45 installed DAC rows, row order, hidden predecessor, current
limits, ADC-pair implementation, and MAP/SURVEY/WIDE/SINGLE policies as v4.
Only the G8 sentinel changes from local offset 1 (row 36) to local offset 3
(row 38). The full sentinel tuple is `(3,3,3,3,1,3,1,3,1)`; an active G8 ROI
still refreshes its three inner points. Every grating remains observable.

This changes ADC dwell history before later DAC rows. Qualifying row 38 with
the old v4 prefix alone is insufficient. Tests must explicitly arm v5, require
its exact 02/05 ACK and first MAP45, warm the real v5 loop, and inspect its
preserved ADC pairs and held-target teacher. Changing a v4 bitmap without
changing the negotiated version is forbidden. The v3/v4 contracts are unchanged.

The wire layouts remain unchanged: arm byte 10 and ACK byte 11 carry 5; normal
142-byte board extension byte 36 carries 5; D66D02 preserved-raw byte 22 and
D77D01 held-target byte 14 carry 5. Older decoders must reject the unknown
schedule, not infer cached rows are fresh. New diagnostics require the exact
requested schedule and firmware >=1.0.29 for v5. Capture files record the
schedule, and the held-trajectory auditor groups v4 and v5 separately.

G8's static ranking weight changes from 71 to 165, preserving the original
relative scale: round(71 * abs(3.4974732128486252 / -1.5070150869701238)).
Slopes come from the existing certified template's left/right rows. This
weight is neither transient correction nor a pressure-trained model.

CLI `--schedule-version 5` is supported by the realtime benchmark and both
sparse diagnostics; omission preserves v4. Safe cleanup uses an explicitly
acknowledged disarm and SOA shutter. No experiment command moves the CNC.

## 1.0.30：同一次生产双读与保持目标教师

USB请求 `FF FF 03 04`，总长808，byte4..5目标安装表索引0..44，byte6..9非零u32标签。
仅接受显式协商v4/v5、CH1多速率会话；与03/02、03/03诊断互斥。在下一帧的正常DAC路径
到达目标后，先执行原有 `sampleVoltageStable` 两读及条件慢复读、计算RC估计，随后不重写
DAC、不进入下一个波长，采集教师。目标必须本就在该帧计划fresh位图内，否则失败退出，
不强制MAP也不把缓存变为新鲜值。仍保留原有45行顺序、电流上限和隐藏前驱。

响应 `D8 8D 01`，固定604字节，全部整数大端：

| 字节 | 字段 |
|---|---|
| 0..3 | D8 8D 01、状态；0才表示全部读取及关光成功 |
| 4..13 | 请求标签u32、运行固件版本u32、目标索引u16 |
| 14..19 | 协商版本、profile、primary、secondary、计划freshCount、mapAge/gap |
| 20..31 | planSequence、frameStartMs、构造包头时的帧偏移µs，均u32 |
| 32..37 | CH0/CH1实际跨阻选择码、生产读有效位图3或7、25教师组、6通道、诊断中断标志1 |
| 38..43 | 原正常帧计划fresh位图（45位） |
| 44..51 | wave_time µs u32、固件RC估计u16、生产通道掩码2、同次保持无DAC重写标志1 |
| 52..81 | 首读、次读、可选慢复读，每条为开始µs u32、结束µs u32、CH1 ADC u16 |
| 82..101 | 生产双读/慢复读之后额外读取一次六通道光学状态 |
| 102..601 | 25组教师，保持目标后的300..780ms期间每20ms一组 |
| 602..603 | 8D D8 |

六通道组每组20字节：开始µs u32、结束µs u32、六个u16 ADC（CH0/CH1/CH2/CH3/PDT/PDR）。
未发生慢复读时第三条10字节必须全零；不得用其零值当作样本。所有采样时间相对目标DAC
完成后设置的时间原点，记录的是ADC API前后时刻，不冒充精确转换瞬间。新增条件判断和
计时有少量插桩开销，须用普通帧另测时序；byte28..31也不是精确的DAC完成时刻。

额外六通道状态严格在全部生产读取及RC计算之后，不能当作当前普通帧已有的模型特征。
教师复用单值模式25点噪声/漂移检查。成功与失败均关光、停用会话、清除前驱缓存，返回
诊断包而非普通动态帧。长等待继续服务温控和保护，主机输入、断连、保护关光或ADC失败
会终止试验。没有机床指令，也没有训练或部署模型。

`capture_production_pair_teacher.py` 每条记录重新确认跨阻、精确协商版本、检查首次MAP，
预热100帧；版本/增益条件交错采集。保存原始生产读、额外光学状态和全部教师，不将不同
试验的双读与教师拼接。`analyze_production_pair_teacher.py` 重新核对教师统计和因果时序，
按版本/跨阻/光栅汇总首读、次读、最后生产读、RC、额外直接读的误差；不合格教师保留但
排除出汇总。max(3码,10%教师)仅为诊断容差，不是定位或压力精度标准。

若用最后两次生产读a、b和教师t反推alpha=(b-t)/(a-t)，不在[0,1)的结果与精确单极点
单调RC收敛不相容；这只是基于未来标签的诊断，未计量化/噪声容差，不能部署为预测器，
也不能单独区分激光瞬态、输入变化和其它原因。所有记录缺少独立压力/面积真值；稳定教师
不证明早期与教师时段外部压力相同。这类中断扫描试验不证明15Hz物理带宽。
