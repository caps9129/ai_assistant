# VoiceProcessor 技術文件

## 1. 總覽

`VoiceProcessor` 是整個音訊處理流程的**起點與狀態控制器**。它的主要職責是從麥克風即時擷取音訊，並根據當前的系統狀態，決定如何處理這些音訊。

您可以將 `VoiceProcessor` 想像成一個**總機或交換機**。它的核心功能是管理兩種主要模式：

1.  **待機模式 (`IDLE`)**: 在此模式下，它只專注於一件事：聆聽**喚醒詞 (Wake-Word)**。
2.  **對話模式 (`DIALOG`)**: 當喚醒詞被偵測到後，它會切換到此模式，並將所有後續的音訊**轉交**給更專業的 `PerceptionPipeline` 進行深入處理（例如語音偵測、打斷判斷、語音轉文字等）。

這個類別的設計體現了**職責分離 (Separation of Concerns)** 的原則：`VoiceProcessor` 負責簡單的狀態切換和喚醒詞偵測，而 `PerceptionPipeline` 則負責複雜的對話理解和打斷邏輯。

---

## 2. 核心邏輯：狀態管理

`VoiceProcessor` 的行為完全由其內部狀態 `self.state` 決定。

### 2.1. `IDLE` 狀態 (待機/喚醒模式)

這是系統的預設狀態。

* **目標**: 高效率地等待使用者說出喚醒詞 (e.g., "Hey, Jarvis")。
* **音訊處理**:
    * 所有從麥克風收到的音訊都會被送入 `openwakeword` 模型進行預測。
    * 在此狀態下，音訊**不會**被送往 `PerceptionPipeline`，從而節省了不必要的計算資源。
* **狀態轉換**:
    * 一旦 `openwakeword` 模型的預測分數超過設定的門檻 (`self.wakeword_threshold`)，`VoiceProcessor` 會立即呼叫 `self.on_wakeword()` 回呼函式。
    * 通常，這個回呼函式會由外部邏輯（例如主應用程式）接收，然後外部邏輯會呼叫 `set_state("DIALOG")` 來命令 `VoiceProcessor` 切換到對話模式。

### 2.2. `DIALOG` 狀態 (對話/指令模式)

當系統準備好接收使用者的指令時，會進入此狀態。

* **目標**: 完整地捕捉使用者的語音指令，並交由 `PerceptionPipeline` 進行處理。
* **音訊處理**:
    * `VoiceProcessor` 不再進行喚醒詞偵測。
    * 所有從麥克風收到的音訊，會被直接、不間斷地傳遞給 `self.perception_pipeline.process_audio()` 方法。
* **狀態轉換**:
    * `VoiceProcessor` 本身不會自動從 `DIALOG` 切換回 `IDLE`。
    * 這個轉換通常由外部邏輯控制。例如，當 `PerceptionPipeline` 完成一次完整的語音辨識、或對話超時後，外部主應用程式會呼叫 `set_state("IDLE")`，讓系統回到待機模式。

---

## 3. 工作流程

1.  **啟動**: 外部程式呼叫 `voice_processor.start()`。
2.  **音訊串流**: `sounddevice` 套件開始從麥克風擷取音訊，並以固定的時間間隔（`frame_duration_ms`）呼叫 `_audio_callback` 方法。
3.  **狀態判斷**: 在 `_audio_callback` 中，程式首先檢查當前的 `self.state`。
    * **若為 `IDLE`**: 音訊被送入 `wakeword_model`。如果偵測到喚醒詞，則觸發 `on_wakeword` 回呼。
    * **若為 `DIALOG`**: 音訊被直接轉發給 `perception_pipeline`。
4.  **狀態切換**: 外部主程式邏輯根據應用流程（如聽到喚醒詞、對話結束等）呼叫 `voice_processor.set_state()` 來改變 `VoiceProcessor` 的行為模式。
5.  **停止**: 外部程式呼叫 `voice_processor.stop()`，安全地關閉音訊串流。

---

## 4. 類別介面 (API)

### `VoiceProcessor`

#### 初始化參數 (`__init__`)

* `on_wakeword`: **(必要)** 一個回呼函式。當偵測到喚醒詞時會被呼叫。
* `perception_pipeline`: **(必要)** 一個 `PerceptionPipeline` 的實例。在 `DIALOG` 狀態下，所有音訊都將被傳遞給它。

#### 公開方法 (Public Methods)

* `set_state(new_state: str)`:
    * 線程安全地設定處理器的狀態 (`"IDLE"` 或 `"DIALOG"`)。
    * 這是控制 `VoiceProcessor` 行為的核心方法。
    * 切換狀態時，會自動重置喚醒詞模型的內部緩衝區，以避免狀態間的干擾。

* `start()`:
    * 初始化並啟動 `sounddevice` 的音訊輸入串流。
    * 讓 `VoiceProcessor` 開始接收並處理麥克風音訊。

* `stop()`:
    * 停止並關閉音訊串流，釋放音訊設備。
    * 讓 `VoiceProcessor` 停止運作。

#### 私有方法 (`_audio_callback`)

* 這是由 `sounddevice` 內部呼叫的回呼函式，是所有即時音訊處理的入口點。它根據當前狀態將音訊路由到正確的處理路徑。