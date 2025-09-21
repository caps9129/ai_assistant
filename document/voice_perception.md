# Perception Pipeline 技術文件

## 1. 總覽

`PerceptionPipeline` 是一個專為即時語音互動系統設計的音訊處理管線。其核心功能是**偵測使用者何時說話（尤其是在系統正在播放音訊時），並對其進行智慧判斷，以決定是否要中斷系統、處理使用者指令**。

這個管線整合了以下幾個關鍵技術：

1.  **語音活動偵測 (VAD)**：使用 WebRTC VAD 演算法來初步判斷音訊幀中是否包含人聲。
2.  **狀態機 (FSM)**：一個精巧的狀態機，用來過濾短暫的噪音（如咳嗽、雜音），並確認使用者是否真的打算說話。
3.  **前置音訊回補 (Pre-roll)**：一種緩衝機制，用來避免因 VAD 判斷延遲而遺失語句開頭的第一個字。
4.  **二階段式決策 (ASR Fine-screening)**：在 VAD 確認語音後，會將音訊片段交給 ASR（自動語音辨識）引擎進行最終確認。只有當 ASR 辨識出有效文字時，才會觸發真正的「插話」行為。
5.  **多執行緒架構**：將 VAD 處理和 ASR 處理分離到不同的執行緒中，確保即時音訊輸入不會被較慢的 ASR 模型所阻塞。

這個設計的主要目標是實現流暢、準確的「**Barge-in**」（使用者插話）體驗。

---

## 2. 核心架構與工作流程

此管線採用了兩個主要的執行緒來分工合作：

* **VAD 處理執行緒 (`_process_loop`)**: 負責即時處理音訊輸入、執行 VAD 判斷、管理狀態機，並決定何時將音訊片段送去辨識。
* **ASR 工作執行緒 (`_asr_loop`)**: 接收 VAD 執行緒送來的音訊片段，呼叫 ASR 引擎進行辨識，並根據辨識結果執行最終的決策回呼 (Callbacks)。



**資料流程如下：**

1.  外部音訊來源（如麥克風）透過 `process_audio()` 方法，將音訊區塊 (`audio_chunk`) 送入 `audio_queue` 佇列。
2.  `_process_loop` 執行緒從 `audio_queue` 中取出音訊，進行 VAD 偵測和狀態判斷。
3.  當 VAD 狀態機確認一段完整的語音（committed utterance）結束後，會將累積的音訊緩衝 (`audio_buffer`) 和時間戳記放入 `asr_queue`。
4.  `_asr_loop` 執行緒從 `asr_queue` 中取出音訊，呼叫 `asr_router.transcribe()` 進行辨識。
5.  **決策分流**:
    * **辨識成功 (有文字)**: 判定為有效插話。觸發 `on_stop_current_utterance`、`on_resume_flush` 等回呼，中斷系統當前行為，並將辨識結果傳遞給 `on_result_callback`。
    * **辨識失敗 (無文字)**: 判定為噪音。觸發 `on_resume_continue`、`on_speech_cancel` 等回呼，讓系統繼續先前的行為，忽略這次的聲音。

---

## 3. 關鍵機制詳解

### 3.1. VAD 狀態機 (Finite State Machine)

這是管線的核心，用來區分無意義的短促噪音和真正的使用者意圖。它包含幾個關鍵狀態與門檻：

* **平滑化 (N-out-of-M Smoothing)**:
    * 為了避免 VAD 的瞬間抖動，系統採用了一個滑動視窗 (`_speech_window`)。只有當視窗內 (`window_frames`, M) 的語音幀數達到一定數量 (`window_min_speech`, N) 時，才會將當前狀態判定為「有語音 (`smooth_speech`)」。這大大增加了穩定性。

* **Onset (語音起始)**:
    * 當 `smooth_speech` 從 `False` 變為 `True` 時，代表語音開始。此時狀態機進入 `in_speech` 狀態。

* **Commit (意圖確認)**:
    * 只有當連續的語音長度超過 `commit_min_ms` 毫秒時，狀態機才會進入 `committed` 狀態。這個門檻是為了過濾掉使用者無意的短音（如清喉嚨）。
    * 在 `commit` 之前，如果語音就中斷了，會被視為一次「取消 (`cancel`)」，系統狀態會重置，不會送交 ASR。

* **Onset Duck Delay (延遲降低背景音)**:
    * 為了避免系統音量因極短的噪音而頻繁變化，只有當語音持續超過 `onset_duck_delay_ms` 後，才會觸發 `on_speech_onset` 回呼（通常用來稍微降低系統背景音量，即 "ducking"）。

* **Finalization (語句結束)**:
    * 當處於 `in_speech` 狀態時，如果偵測到持續的靜音超過 `min_silence_duration_ms`，則認為一句話結束。此時，會將累積的音訊送往 ASR 處理。

### 3.2. Pre-roll (前置音訊回補)

這是一個非常重要的機制，用來解決「**吃字**」問題。

* **問題**: 當 VAD 狀態機（經過平滑化後）確認語音開始時，實際上使用者已經說出了幾個音節。如果從這一刻才開始錄音，會遺失語句的開頭。
* **解決方案**:
    1.  系統維護一個固定容量的環形緩衝區 `_pre_roll`。
    2.  只要**原始 VAD** 偵測到任何語音幀 (`raw_is_speech=True`)，就立即將其存入 `_pre_roll`。
    3.  當**平滑化後的 VAD** 觸發 `Onset` 事件時，立刻將 `_pre_roll` 中緩存的所有音訊幀，一次性地加到主要音訊緩衝區 (`audio_buffer`) 的最前端。
    4.  這樣就能確保即使 VAD 決策有延遲，語句的開頭部分也能被完整保留。

* **Pre-roll 模式**:
    * `static`: `_pre_roll` 緩衝區的大小固定為 `pre_roll_ms`。
    * `auto`: 緩衝區大小根據平滑化視窗 (`window_min_speech`) 自動計算，並加上一個安全邊際。
    * `adaptive`: 緩衝區大小會根據**上一次實際回補的長度**，透過 EWMA (指數加權移動平均) 演算法動態調整，使其能自適應不同說話者的語速習慣。

### 3.3. ASR Fine-Screening (ASR 精細過濾)

這是管線的第二道防線，用來確認 VAD 捕捉到的聲音是否真的有意義。

* **動機**: VAD 只能判斷有無人聲，但無法區分有意義的語言和無意義的聲音（如咳嗽、關門聲、驚呼）。如果僅憑 VAD 就中斷系統，會導致頻繁的誤觸發。
* **流程**:
    1.  VAD 執行緒將它認為是完整語句的音訊片段送到 `_asr_loop`。
    2.  `_asr_loop` 呼叫 ASR 引擎進行辨識。
    3.  **如果辨識結果為空字串**，代表這段音訊很可能只是噪音。管線會觸發「繼續播放」和「取消 ducking」的相關回呼，使用者幾乎不會察覺到系統有任何反應。
    4.  **如果辨識結果包含有效文字**，管線才會執行真正的中斷邏輯，停止系統播放並處理使用者指令。

---

## 4. 類別介面 (API)

### `PerceptionPipeline`

#### 初始化參數 (`__init__`)

##### **核心回呼 (Callbacks)**

* `asr_router: AsrRouter`: ASR 辨識器的實例。
* `on_result_callback: Callable`: 當 ASR 辨識出有效結果時的回呼函式。
* `on_speech_onset: Optional[Callable]`: 語音持續超過 `onset_duck_delay_ms` 時觸發，建議用於**降低系統音量 (ducking)**。
* `on_speech_commit: Optional[Callable]`: 語音持續超過 `commit_min_ms` 時觸發，建議用於**暫停系統音訊輸出**。
* `on_speech_cancel: Optional[Callable]`: 語音在 `commit` 前中斷或 ASR 辨識為空時觸發，建議用於**恢復系統音量 (unducking)**。
* `on_stop_current_utterance: Optional[Callable]`: ASR 辨識成功時觸發，建議用於**徹底停止當前系統話語**。
* `on_resume_continue: Optional[Callable]`: ASR 辨識為空時觸發，建議用於**從暫停處恢復系統輸出 (不清除佇列)**。
* `on_resume_flush: Optional[Callable]`: ASR 辨識成功時觸發，建議用於**恢復系統輸出並清除待播放佇列**。

##### **靈敏度參數**

* `commit_min_ms: int`: 確認使用者意圖所需的最短語音時長 (毫秒)。
* `onset_duck_delay_ms: int`: 觸發 `on_speech_onset` 的延遲時間 (毫秒)。
* `window_frames: int`: VAD 平滑化視窗的大小 (M)。
* `window_min_speech: int`: 視窗內被視為語音所需的最少幀數 (N)。
* `vad_frame_ms: int`: VAD 處理的音訊幀長度 (毫秒)。

##### **Pre-roll 參數**

* `pre_roll_mode: str`: 模式，可選 `"static"`, `"auto"`, `"adaptive"`。
* `pre_roll_ms: int`: 在 `static` 模式下，設定固定的回補時長。
* `pre_roll_alpha: float`: 在 `adaptive` 模式下，EWMA 的平滑係數。

#### 公開方法 (Public Methods)

* `start()`: 啟動 VAD 和 ASR 執行緒，開始處理管線。
* `stop()`: 停止所有執行緒，安全地關閉管線。
* `pause()`: 暫停 VAD 處理，管線將忽略所有傳入的音訊。
* `resume()`: 恢復 VAD 處理。
* `process_audio(audio_chunk: np.ndarray)`: 向管線餵送即時的音訊資料。
* `last_activity_ts() -> float`: (執行緒安全) 獲取最後一次偵測到原始 VAD 活動的時間戳。
* `is_processing: bool`: (屬性) 回傳管線當前是否正在處理一段語音（從 Onset 到 ASR 結果出爐）。