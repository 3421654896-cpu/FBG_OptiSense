#include "main.h"
#include "wifi_transport.h"
#include "stress_table.h"
#include "ota_layout.h"
#include "ota_update.h"

#include <stdarg.h>
#include <stdlib.h>

extern PCD_HandleTypeDef hpcd_USB_OTG_FS;

#define WIFI_CONFIG_FLASH_ADDRESS       OTA_CONFIG_ADDRESS
#define WIFI_CONFIG_LEGACY_ADDRESS      0x08060000UL
#define WIFI_CONFIG_STORE_MAGIC         0x57464331UL /* WFC1 */
#define WIFI_CONFIG_STORE_VERSION       1U

#define WIFI_RX_RING_SIZE               4096U
#define WIFI_RX_RING_MASK               (WIFI_RX_RING_SIZE - 1U)
#define WIFI_LINE_SIZE                  512U
#define WIFI_AT_COMMAND_SIZE            384U

#define WIFI_SSID_MAX                   32U
#define WIFI_WIFI_PASSWORD_MAX          64U
#define WIFI_HOST_MAX                   128U
#define WIFI_CLIENT_ID_MAX              64U
#define WIFI_USERNAME_MAX               64U
#define WIFI_MQTT_PASSWORD_MAX          64U
#define WIFI_TOPIC_PREFIX_MAX           96U
#define WIFI_DEVICE_ID_MAX              32U
#define WIFI_TOPIC_SIZE                 192U

/* HUAWEI-10ES6Z advertises the same SSID on 2.4 GHz and 5 GHz.  The fitted
 * IPEX antenna is 2.4-GHz-only, so select the router's 2.4-GHz radio by BSSID
 * instead of leaving the dual-band choice to the BW20 roaming algorithm. */
#define WIFI_FORCED_24G_BSSID            "14:77:40:4d:96:70"

#define WIFI_TELEMETRY_HEADER_SIZE      56U
#define WIFI_SMALL_PAYLOAD_SIZE         128U
#define WIFI_ACK_QUEUE_DEPTH             8U

/* A same-WiFi viewer opens an outbound TCP connection to port 45670 and sends
 * FBGL1|HELLO once per second.  TCP avoids Windows public-network inbound UDP
 * rules while preserving the raw FBG1/FBGM/FACK1 payloads byte-for-byte.
 * USB > LAN > MQTT is the data-path priority, so one raw scan is never sent on
 * LAN and WAN at the same time. */
#define WIFI_LAN_SERVER_PORT          45670U
#define WIFI_LAN_SOCKET_ID                9
#define WIFI_LAN_CLIENT_TIMEOUT_MS     3500U
#define WIFI_LAN_RETRY_MS              2000U
#define WIFI_LAN_STALE_DELETE_GRACE_MS  750U
#define WIFI_LAN_STARTUP_CLEANUP_MAX      16U
#define WIFI_LAN_SERVER_DELETE_VERIFY_MAX 16U
#define WIFI_LAN_HELLO                 "FBGL1|HELLO"
#define WIFI_LAN_RAW_HELLO             "FBGL1|RAW"
#define WIFI_LAN_RAW_COMPACT_CAP       "FBGL1|CAPS|CS1"
#define WIFI_LAN_RAW_COMMAND_PREFIX     "FUSB1|"
#define WIFI_LAN_RAW_MAX_PAYLOAD        2048U
#define WIFI_LAN_RAW_PACKET_OVERHEAD      16U
#define WIFI_LAN_RAW_KIND_DATA              0U
#define WIFI_LAN_RAW_KIND_COMMAND_ACK       1U
#define WIFI_LAN_RAW_KIND_READY             2U
#define WIFI_LAN_RAW_KIND_COMPACT_SCAN      3U
#define WIFI_LAN_RAW_KIND_COMPACT_F45       4U
#define WIFI_LAN_RAW_KIND_COMPACT_F45_BATCH 5U
#define WIFI_LAN_RAW_COMPACT_VERSION        1U
#define WIFI_LAN_RAW_COMPACT_HEADER_SIZE    6U
#define WIFI_LAN_RAW_COMPACT_F45_SIZE     128U
#define WIFI_LAN_RAW_NATIVE_F45_SIZE       453U
#define WIFI_LAN_RAW_NATIVE_F45_ROWS        45U
#define WIFI_LAN_RAW_F45_BATCH_MAX           4U
#define WIFI_LAN_RAW_PACKET_MAX_SIZE    \
    (WIFI_LAN_RAW_MAX_PAYLOAD + WIFI_LAN_RAW_PACKET_OVERHEAD)
#define WIFI_LAN_RAW_SMALL_PACKET_SIZE    36U
#define WIFI_LAN_RAW_PRIORITY_DEPTH        8U
#define WIFI_OTA_RESPONSE_DRAIN_MS      1000U

#define WIFI_LAN_DELETE_SERVER             0U
#define WIFI_LAN_DELETE_OWNER              1U
#define WIFI_LAN_DELETE_REJECTED           2U

#define WIFI_STRESS_POINT_COUNT         STRESS_TABLE_POINT_COUNT
#define WIFI_TEMPERATURE_START_ROW       0U
#define WIFI_TEMPERATURE_POINT_COUNT     TEMPERATURE_TABLE_POINT_COUNT
#define WIFI_METADATA_MAX_POINT_COUNT   \
    ((WIFI_STRESS_POINT_COUNT > WIFI_TEMPERATURE_POINT_COUNT) ? \
        WIFI_STRESS_POINT_COUNT : WIFI_TEMPERATURE_POINT_COUNT)
#define WIFI_METADATA_MAX_SIZE          \
    (12U + WIFI_METADATA_MAX_POINT_COUNT * 4U + 4U)
#define WIFI_STRESS_PAYLOAD_MAX         (WIFI_STRESS_POINT_COUNT * 4U * 2U)
#define WIFI_TEMPERATURE_PAYLOAD_MAX    (WIFI_TEMPERATURE_POINT_COUNT * 2U * 2U)
#define WIFI_TELEMETRY_PAYLOAD_MAX      \
    ((WIFI_STRESS_PAYLOAD_MAX > WIFI_TEMPERATURE_PAYLOAD_MAX) ? \
        WIFI_STRESS_PAYLOAD_MAX : WIFI_TEMPERATURE_PAYLOAD_MAX)
#define WIFI_TELEMETRY_MAX_SIZE         \
    (WIFI_TELEMETRY_HEADER_SIZE + WIFI_TELEMETRY_PAYLOAD_MAX + 4U)

#define WIFI_CONFIG_FLAG_ENABLE         0x01U
#define WIFI_CONFIG_FLAG_PERSIST        0x02U

#define WIFI_PACKET_TYPE_TELEMETRY      1U
#define WIFI_SAMPLE_FORMAT_U16_BE       1U

#define WIFI_ERROR_NONE                 0U
#define WIFI_ERROR_CONFIG_HEADER        1U
#define WIFI_ERROR_CONFIG_CRC           2U
#define WIFI_ERROR_CONFIG_FIELD         3U
#define WIFI_ERROR_FLASH_ERASE          4U
#define WIFI_ERROR_FLASH_PROGRAM        5U
#define WIFI_ERROR_UART                 10U
#define WIFI_ERROR_AT_TIMEOUT           11U
#define WIFI_ERROR_AT_RESPONSE          12U
#define WIFI_ERROR_WIFI_TIMEOUT         13U
#define WIFI_ERROR_MQTT_TIMEOUT         14U
#define WIFI_ERROR_PUBLISH_TIMEOUT      15U
#define WIFI_ERROR_FRAME_FORMAT         16U
#define WIFI_ERROR_RX_OVERFLOW          17U
#define WIFI_ERROR_ACK_OVERFLOW         18U
#define WIFI_ERROR_LAN_SOCKET           19U

typedef enum
{
    WIFI_STATE_DISABLED = 0,
    WIFI_STATE_BACKOFF,
    WIFI_STATE_WAIT_AT,
    WIFI_STATE_WAIT_ATE0,
    WIFI_STATE_WAIT_MQTT_DISCONNECT,
    WIFI_STATE_WAIT_WARM_WIFI_QUERY,
    WIFI_STATE_WAIT_WMODE,
    WIFI_STATE_WAIT_WJAP,
    WIFI_STATE_WAIT_WIFI_IP,
    WIFI_STATE_WAIT_WIFI_QUERY,
    WIFI_STATE_WAIT_LAN_RECV_CFG,
    WIFI_STATE_WAIT_LAN_DELETE_STALE,
    WIFI_STATE_WAIT_LAN_CREATE,
    WIFI_STATE_WAIT_MQTT_HOST,
    WIFI_STATE_WAIT_MQTT_PORT,
    WIFI_STATE_WAIT_MQTT_SCHEME,
    WIFI_STATE_WAIT_MQTT_CLIENT,
    WIFI_STATE_WAIT_MQTT_USER,
    WIFI_STATE_WAIT_MQTT_PASSWORD,
    WIFI_STATE_WAIT_MQTT_VERSION,
    WIFI_STATE_WAIT_MQTT_BUFFER,
    WIFI_STATE_WAIT_MQTT_KEEPALIVE,
    WIFI_STATE_WAIT_MQTT_CERT,
    WIFI_STATE_WAIT_MQTT_LWT,
    WIFI_STATE_WAIT_MQTT_START,
    WIFI_STATE_WAIT_MQTT_CONNECT,
    WIFI_STATE_WAIT_MQTT_QUERY,
    WIFI_STATE_WAIT_SUBSCRIBE,
    WIFI_STATE_WAIT_SUBSCRIBE_SETTLE,
    WIFI_STATE_WAIT_MQTT_RECONNECT,
    WIFI_STATE_ONLINE,
    WIFI_STATE_PUBLISH_WAIT_PROMPT,
    WIFI_STATE_PUBLISH_WAIT_RESULT,
    WIFI_STATE_LAN_READ,
    WIFI_STATE_LAN_DELETE,
    WIFI_STATE_LAN_RECREATE,
    WIFI_STATE_LAN_SEND_WAIT_PROMPT,
    WIFI_STATE_LAN_SEND_WAIT_RESULT,
    WIFI_STATE_WAIT_LAN_QUERY_STALE,
    WIFI_STATE_WAIT_LAN_DELETE_CHILD,
    WIFI_STATE_WAIT_LAN_VERIFY_SERVER_GONE,
    WIFI_STATE_WAIT_LAN_SERVER_DELETE_GRACE,
    WIFI_STATE_WAIT_LAN_CHILD_DELETE_GRACE,
    WIFI_STATE_WAIT_LAN_VERIFY_CREATED,
    WIFI_STATE_WAIT_LAN_CREATE_GRACE
} WifiState;

typedef enum
{
    WIFI_PUBLISH_NONE = 0,
    WIFI_PUBLISH_STATUS,
    WIFI_PUBLISH_METADATA_STRESS,
    WIFI_PUBLISH_METADATA_TEMPERATURE,
    WIFI_PUBLISH_ACK,
    WIFI_PUBLISH_OTA,
    WIFI_PUBLISH_LAN_RAW,
    WIFI_PUBLISH_TELEMETRY
} WifiPublishKind;

typedef struct
{
    uint8_t flags;
    uint8_t scheme;
    uint16_t port;
    char ssid[WIFI_SSID_MAX + 1U];
    char wifi_password[WIFI_WIFI_PASSWORD_MAX + 1U];
    char host[WIFI_HOST_MAX + 1U];
    char client_id[WIFI_CLIENT_ID_MAX + 1U];
    char username[WIFI_USERNAME_MAX + 1U];
    char mqtt_password[WIFI_MQTT_PASSWORD_MAX + 1U];
    char topic_prefix[WIFI_TOPIC_PREFIX_MAX + 1U];
    char device_id[WIFI_DEVICE_ID_MAX + 1U];
} WifiConfig;

typedef struct
{
    uint32_t magic;
    uint16_t version;
    uint16_t size;
    uint32_t config_crc;
    WifiConfig config;
    uint32_t record_crc;
} WifiStoredConfig;

/* CommitMetadata() must erase flash sector 1 before publishing new OTA
 * metadata.  It preserves OTA_CONFIG_PRESERVE_SIZE bytes starting at the
 * Wi-Fi record, so make any future credential/schema growth fail the build
 * instead of silently truncating the persisted configuration. */
typedef char WifiStoredConfigMustFitOtaPreserve[
    (sizeof(WifiStoredConfig) <= OTA_CONFIG_PRESERVE_SIZE) ? 1 : -1];

typedef struct
{
    uint8_t data[WIFI_TELEMETRY_MAX_SIZE];
    uint16_t length;
    uint8_t ready;
    uint8_t mode;
    uint32_t sequence;
} WifiSnapshot;

typedef struct
{
    uint8_t payload[WIFI_SMALL_PAYLOAD_SIZE];
    uint16_t length;
} WifiAckEntry;

typedef struct
{
    uint8_t data[WIFI_LAN_RAW_SMALL_PACKET_SIZE];
    uint16_t length;
} WifiLanRawSmallPacket;

static WifiConfig wifiConfig;
static WifiConfig pendingConfig;
static uint32_t wifiConfigCrc = 0U;
static uint32_t pendingConfigCrc = 0U;
static uint8_t wifiConfigValid = 0U;
static uint8_t pendingConfigValid = 0U;
static uint8_t pendingConfigPersist = 0U;
static uint8_t flashPersistPending = 0U;
static uint8_t flashFrameBoundaryPermit = 0U;
static uint8_t usbConfigAckPending = 0U;
static uint8_t usbConfigAckStatus = 0U;
static uint8_t usbAckCommand = 0x20U;
static uint8_t usbConfigAckBoundaryPermit = 0U;
static uint8_t restartRequested = 0U;

static char telemetryTopic[WIFI_TOPIC_SIZE];
static char commandTopic[WIFI_TOPIC_SIZE];
static char ackTopic[WIFI_TOPIC_SIZE];
static char statusTopic[WIFI_TOPIC_SIZE];
static char metadataStressTopic[WIFI_TOPIC_SIZE];
static char metadataTemperatureTopic[WIFI_TOPIC_SIZE];

static volatile uint8_t uartTxBusy = 0U;
static volatile uint8_t uartErrorPending = 0U;
static volatile uint32_t uartLastError = 0U;
static volatile uint16_t uartTxCompleteCount = 0U;
static volatile uint16_t uartRxByteCount = 0U;
static uint8_t atCommandBuffer[WIFI_AT_COMMAND_SIZE];

static volatile uint16_t rxHead = 0U;
static volatile uint16_t rxTail = 0U;
static volatile uint8_t rxOverflow = 0U;
static uint8_t rxRing[WIFI_RX_RING_SIZE];
static char lineBuffer[WIFI_LINE_SIZE];
static uint16_t lineLength = 0U;
static uint8_t lineDiscarding = 0U;

static volatile uint8_t responseOk = 0U;
static volatile uint8_t responseError = 0U;
static volatile uint8_t responsePrompt = 0U;
static volatile uint8_t responsePublishBusy = 0U;
static uint8_t wifiConnected = 0U;
static uint8_t mqttConnected = 0U;
static uint8_t warmWifiStatusSeen = 0U;
static uint8_t warmWifiMatchesConfig = 0U;

static WifiState wifiState = WIFI_STATE_DISABLED;
static uint32_t stateDeadlineMs = 0U;
static uint32_t connectDeadlineMs = 0U;
static uint32_t nextPollMs = 0U;
static uint32_t nextPublishMs = 0U;
static uint32_t nextLocalHeartbeatMs = 0U;
static uint8_t retryExponent = 0U;
static uint16_t wifiLastError = WIFI_ERROR_NONE;

static uint32_t clockLastCycles = 0U;
static uint32_t clockRemainderCycles = 0U;
static uint32_t clockMilliseconds = 0U;
static uint32_t clockCyclesPerMs = 1U;

static uint32_t bootId = 0U;
static uint32_t telemetrySequence = 0U;
static uint32_t stressTableCrc = 0U;
static uint32_t temperatureTableCrc = 0U;

static WifiSnapshot snapshots[2];
static int8_t latestSnapshot = -1;
static int8_t activeSnapshot = -1;
static volatile int8_t dmaReservedSnapshot = -1;
static uint8_t activeTelemetryDiscarded = 0U;
static uint8_t queueOverrunLatched = 0U;

static uint8_t metadataFrames[2][WIFI_METADATA_MAX_SIZE];
static uint16_t metadataLengths[2] = {0U, 0U};
static uint8_t metadataPendingMask = 0U;
static uint8_t statusPending = 0U;
static uint8_t localControlActive = 0U;
/* Combo-AT documents ConID as u32 and real BW20 sessions exceed 127 after
 * repeated reconnects.  Signed 8-bit storage silently ignored those seeds
 * and eventually made every LAN client loop in accept/reset. */
static int32_t lanSocketId = -1;
static int32_t lanCreatedSocketId = -1;
static int32_t lanCleanupSocketId = -1;
static int32_t lanDeleteTargetId = -1;
static uint8_t lanCleanupCanPromote = 0U;
static uint8_t lanCleanupOverflow = 0U;
static uint8_t lanReadPending = 0U;
static uint8_t lanClientActive = 0U;
static uint8_t lanRawMode = 0U;
static uint8_t lanRawCompactScan = 0U;
static uint8_t lanSocketResetPending = 0U;
static uint8_t lanDeleteKind = WIFI_LAN_DELETE_SERVER;
static int32_t lanStartupStaleChildId = -1;
static volatile int8_t lanStartupServerStatus = -1;
static uint8_t lanStartupCleanupAttempts = 0U;
static uint8_t lanServerDeleteVerifyAttempts = 0U;
static uint8_t lanServerDeleteVerifyStartup = 0U;
static uint8_t lanListenerCreateVerifyAttempts = 0U;
static uint8_t lanListenerCreateStartup = 0U;
static uint8_t lanMetadataPendingMask = 0U;
static uint32_t lanClientDeadlineMs = 0U;
static uint32_t nextLanRetryMs = 0U;
static WifiState lanResumeState = WIFI_STATE_ONLINE;
static uint32_t lanResumeDeadlineMs = 0U;
static WifiAckEntry ackQueue[WIFI_ACK_QUEUE_DEPTH];
static uint8_t ackQueueHead = 0U;
static uint8_t ackQueueCount = 0U;
static WifiLanRawSmallPacket lanRawPriority[WIFI_LAN_RAW_PRIORITY_DEPTH];
static uint8_t lanRawPriorityHead = 0U;
static uint8_t lanRawPriorityCount = 0U;
static WifiLanRawSmallPacket lanRawMonitor[2];
static uint8_t lanRawMonitorReady[2] = {0U, 0U};
static uint8_t lanRawScan[WIFI_LAN_RAW_PACKET_MAX_SIZE];
static uint16_t lanRawScanLength = 0U;
static uint8_t lanRawScanReady = 0U;
static uint8_t lanRawF45Batch[WIFI_LAN_RAW_COMPACT_F45_SIZE *
                              WIFI_LAN_RAW_F45_BATCH_MAX];
static uint8_t lanRawF45BatchCount = 0U;
static uint8_t lanRawPublish[WIFI_LAN_RAW_PACKET_MAX_SIZE];
static uint8_t lanRawActivePriority = 0U;
static uint32_t lanRawSequence = 0U;
static uint8_t lanRawCommand[USART_RX_SIZE];
static uint32_t lanRawCommandId = 0U;
static uint16_t lanRawCommandExpected = 0U;
static uint16_t lanRawCommandTotal = 0U;

static WifiPublishKind publishKind = WIFI_PUBLISH_NONE;
static const uint8_t *publishPayload = NULL;
static uint16_t publishPayloadLength = 0U;
static uint8_t publishQos = 0U;
static uint8_t publishRetained = 0U;
static uint8_t publishOverLan = 0U;
static uint8_t publishSmallPayload[WIFI_SMALL_PAYLOAD_SIZE];
static char publishTopic[WIFI_TOPIC_SIZE];
static uint8_t otaResponse[WIFI_SMALL_PAYLOAD_SIZE];
static uint16_t otaResponseLength = 0U;
static uint8_t otaResponseReady = 0U;
static uint8_t otaRebootAfterResponse = 0U;
static uint8_t otaResetAfterLanDelete = 0U;

static uint8_t pendingModeValid = 0U;
static uint8_t pendingMode = 0U;
static uint32_t pendingModeRequestId = 0U;
static uint8_t modeAckWaitFirstFrame = 0U;
static uint8_t modeAckTargetMode = 0U;
static uint32_t modeAckRequestId = 0U;
static uint32_t lastAppliedRequestId = 0U;
static uint8_t lastAppliedMode = 0xFFU;
static uint32_t lastAppliedSequence = 0U;
static uint8_t appliedFrameMustPublish = 0U;
static uint32_t appliedFrameSequence = 0U;

static uint8_t IsScanMode(uint8_t mode);
static uint8_t StoredConfigSave(void);

static uint32_t WifiCrc32(const uint8_t *data, uint32_t length)
{
    uint32_t crc = 0xFFFFFFFFUL;
    for(uint32_t index = 0U; index < length; index++)
    {
        crc ^= data[index];
        for(uint8_t bit = 0U; bit < 8U; bit++)
        {
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0U);
        }
    }
    return crc ^ 0xFFFFFFFFUL;
}

static void PutU16Be(uint8_t *destination, uint16_t value)
{
    destination[0] = (uint8_t)(value >> 8);
    destination[1] = (uint8_t)value;
}

static void PutU32Be(uint8_t *destination, uint32_t value)
{
    destination[0] = (uint8_t)(value >> 24);
    destination[1] = (uint8_t)(value >> 16);
    destination[2] = (uint8_t)(value >> 8);
    destination[3] = (uint8_t)value;
}

static uint32_t GetU32Be(const uint8_t *source)
{
    return ((uint32_t)source[0] << 24) |
           ((uint32_t)source[1] << 16) |
           ((uint32_t)source[2] << 8) |
           (uint32_t)source[3];
}

static uint16_t BuildLanRawPacket(uint8_t *destination,
                                  uint16_t capacity,
                                  uint8_t kind,
                                  uint32_t sequence,
                                  const uint8_t *payload,
                                  uint16_t payloadLength)
{
    uint16_t position = 12U;
    uint16_t total = (uint16_t)(WIFI_LAN_RAW_PACKET_OVERHEAD + payloadLength);
    if(destination == NULL || total > capacity ||
       payloadLength > WIFI_LAN_RAW_MAX_PAYLOAD) return 0U;
    destination[0] = 'F'; destination[1] = 'B';
    destination[2] = 'R'; destination[3] = '1';
    destination[4] = 1U;
    destination[5] = kind;
    PutU32Be(&destination[6], sequence);
    PutU16Be(&destination[10], payloadLength);
    if(payloadLength > 0U && payload != NULL)
    {
        memcpy(&destination[position], payload, payloadLength);
        position += payloadLength;
    }
    PutU32Be(&destination[position], WifiCrc32(destination, position));
    return total;
}

static uint8_t QueueLanRawPriority(uint8_t kind,
                                   uint32_t sequence,
                                   const uint8_t *payload,
                                   uint16_t payloadLength)
{
    uint8_t tail;
    WifiLanRawSmallPacket *slot;
    if(!lanClientActive || !lanRawMode ||
       payloadLength + WIFI_LAN_RAW_PACKET_OVERHEAD >
           WIFI_LAN_RAW_SMALL_PACKET_SIZE ||
       lanRawPriorityCount >= WIFI_LAN_RAW_PRIORITY_DEPTH) return 0U;
    tail = (uint8_t)((lanRawPriorityHead + lanRawPriorityCount) %
                     WIFI_LAN_RAW_PRIORITY_DEPTH);
    slot = &lanRawPriority[tail];
    slot->length = BuildLanRawPacket(slot->data, sizeof(slot->data), kind,
                                     sequence, payload, payloadLength);
    if(slot->length == 0U) return 0U;
    lanRawPriorityCount++;
    return 1U;
}

static void ResetLanRawState(void)
{
    lanRawMode = 0U;
    lanRawCompactScan = 0U;
    lanRawPriorityHead = 0U;
    lanRawPriorityCount = 0U;
    lanRawMonitorReady[0] = 0U;
    lanRawMonitorReady[1] = 0U;
    lanRawScanReady = 0U;
    lanRawScanLength = 0U;
    lanRawF45BatchCount = 0U;
    lanRawActivePriority = 0U;
    lanRawCommandId = 0U;
    lanRawCommandExpected = 0U;
    lanRawCommandTotal = 0U;
}

static uint8_t TimeReached(uint32_t now, uint32_t deadline)
{
    return ((int32_t)(now - deadline) >= 0) ? 1U : 0U;
}

static uint32_t WifiNowMs(void)
{
    uint32_t currentCycles = DWT->CYCCNT;
    uint32_t elapsedCycles = currentCycles - clockLastCycles;
    clockLastCycles = currentCycles;

    if(elapsedCycles >= clockCyclesPerMs)
    {
        clockMilliseconds += elapsedCycles / clockCyclesPerMs;
        elapsedCycles %= clockCyclesPerMs;
    }
    clockRemainderCycles += elapsedCycles;
    if(clockRemainderCycles >= clockCyclesPerMs)
    {
        clockMilliseconds++;
        clockRemainderCycles -= clockCyclesPerMs;
    }
    return clockMilliseconds;
}

static void ClearResponseEvents(void)
{
    responseOk = 0U;
    responseError = 0U;
    responsePrompt = 0U;
    responsePublishBusy = 0U;
}

static void ResetRxParser(void)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    rxTail = rxHead;
    rxOverflow = 0U;
    if(primask == 0U) __enable_irq();
    lineLength = 0U;
    lineDiscarding = 0U;
}

static uint8_t WifiUartSend(const uint8_t *data, uint16_t length)
{
    if(uartTxBusy || length == 0U) return 0U;
    uartTxBusy = 1U;
    if(HAL_UART_Transmit_DMA(&huart6, (uint8_t *)data, length) != HAL_OK)
    {
        uartTxBusy = 0U;
        return 0U;
    }
    return 1U;
}

static uint8_t SendAtCommand(WifiState waitState,
                             uint32_t timeoutMs,
                             const char *format,
                             ...)
{
    va_list arguments;
    int length;

    if(uartTxBusy) return 0U;
    va_start(arguments, format);
    length = vsnprintf((char *)atCommandBuffer,
                       sizeof(atCommandBuffer),
                       format,
                       arguments);
    va_end(arguments);
    if(length <= 0 || length >= (int)sizeof(atCommandBuffer))
    {
        wifiLastError = WIFI_ERROR_CONFIG_FIELD;
        return 0U;
    }

    ClearResponseEvents();
    wifiState = waitState;
    stateDeadlineMs = WifiNowMs() + timeoutMs;
    if(!WifiUartSend(atCommandBuffer, (uint16_t)length))
    {
        wifiLastError = WIFI_ERROR_UART;
        return 0U;
    }
    return 1U;
}

static uint8_t ConfigStringSafe(const char *value, uint16_t maxLength,
                                uint8_t disallowComma)
{
    uint16_t length = 0U;
    while(value[length] != '\0')
    {
        char character = value[length];
        if(length >= maxLength || character == '\r' || character == '\n' ||
           character == '"' || (disallowComma && character == ',')) return 0U;
        length++;
    }
    return 1U;
}

static uint8_t TextFieldEquals(const char *start, const char *end,
                               const char *expected, uint8_t ignoreCase)
{
    size_t expectedLength;

    if(start == NULL || end == NULL || expected == NULL || end < start) return 0U;
    expectedLength = strlen(expected);
    if((size_t)(end - start) != expectedLength) return 0U;
    for(size_t index = 0U; index < expectedLength; index++)
    {
        char actualCharacter = start[index];
        char expectedCharacter = expected[index];
        if(ignoreCase)
        {
            if(actualCharacter >= 'A' && actualCharacter <= 'Z')
                actualCharacter = (char)(actualCharacter - 'A' + 'a');
            if(expectedCharacter >= 'A' && expectedCharacter <= 'Z')
                expectedCharacter = (char)(expectedCharacter - 'A' + 'a');
        }
        if(actualCharacter != expectedCharacter) return 0U;
    }
    return 1U;
}

static void ParseWifiJoinStatus(const char *line)
{
    int status = -1;
    int payloadOffset = 0;
    const char *ssidStart;
    const char *ssidEnd;
    const char *passwordEnd;
    const char *bssidStart;
    const char *bssidEnd;

    if(sscanf(line, "+WJAP:%d,%n", &status, &payloadOffset) != 1 ||
       status < 0 || status > 4) return;

    /* Combo-AT defines status 3 as associated with an IP address.  Other
     * statuses must not preserve a stale STM32-side connected flag. */
    wifiConnected = (status == 3) ? 1U : 0U;
    if(wifiState != WIFI_STATE_WAIT_WARM_WIFI_QUERY) return;

    warmWifiStatusSeen = 1U;
    warmWifiMatchesConfig = 0U;
    if(status != 3 || payloadOffset <= 0) return;

    /* Query response format is
     * +WJAP:status,ssid,pwd,bssid,security,mac,ch,ip,gateway.
     * SSID/password are already rejected if they contain commas.  Require the
     * current SSID and the fitted 2.4-GHz radio's BSSID before treating a warm
     * module as reusable; a cold module associated with an old AP therefore
     * falls back to the normal configured join. */
    ssidStart = line + payloadOffset;
    ssidEnd = strchr(ssidStart, ',');
    if(ssidEnd == NULL ||
       !TextFieldEquals(ssidStart, ssidEnd, wifiConfig.ssid, 0U)) return;
    passwordEnd = strchr(ssidEnd + 1, ',');
    if(passwordEnd == NULL) return;
    bssidStart = passwordEnd + 1;
    bssidEnd = strchr(bssidStart, ',');
    if(bssidEnd == NULL ||
       !TextFieldEquals(bssidStart, bssidEnd, WIFI_FORCED_24G_BSSID, 1U)) return;
    warmWifiMatchesConfig = 1U;
}

static uint8_t StartWarmWifiProbe(void)
{
    warmWifiStatusSeen = 0U;
    warmWifiMatchesConfig = 0U;
    wifiConnected = 0U;
    return SendAtCommand(WIFI_STATE_WAIT_WARM_WIFI_QUERY, 1500U,
                         "AT+WJAP?\r\n");
}

static uint8_t ContinueAfterMqttDisconnect(void)
{
    uint8_t reuseAssociation =
        (warmWifiStatusSeen && warmWifiMatchesConfig && wifiConnected) ? 1U : 0U;

    mqttConnected = 0U;
    warmWifiStatusSeen = 0U;
    warmWifiMatchesConfig = 0U;
    if(reuseAssociation)
    {
        /* The WJAP? result was captured before touching the retained MQTT
         * task.  MQTTDISCONN may return OK, idempotent ERROR, or time out on
         * different Combo-AT builds; none of those outcomes invalidates a
         * separately verified AP association unless a disconnect URC did. */
        return SendAtCommand(WIFI_STATE_WAIT_LAN_RECV_CFG, 1500U,
                             "AT+SOCKETRECVCFG=1\r\n");
    }

    wifiConnected = 0U;
    return SendAtCommand(WIFI_STATE_WAIT_WMODE, 1000U,
                         "AT+WMODE=1,0\r\n");
}

static uint8_t DeviceIdIsCanonical(const char *value)
{
    uint16_t index;

    /* The BW20 command subscription must be a flat topic.  Requiring one
     * canonical spelling prevents IDs such as "board-1" and "board_1" from
     * being silently normalised onto the same command topic. */
    if(strncmp(value, "fbg-", 4U) != 0 || value[4] == '\0') return 0U;
    for(index = 4U; value[index] != '\0'; index++)
    {
        char character;
        if(index >= WIFI_DEVICE_ID_MAX) return 0U;
        character = value[index];
        if(!((character >= 'A' && character <= 'Z') ||
             (character >= 'a' && character <= 'z') ||
             (character >= '0' && character <= '9'))) return 0U;
    }
    return 1U;
}

static uint8_t ConfigIsValid(const WifiConfig *config)
{
    if(config->port == 0U || (config->scheme != 1U && config->scheme != 2U)) return 0U;
    if(config->ssid[0] == '\0' || config->host[0] == '\0' ||
       config->topic_prefix[0] == '\0' || config->device_id[0] == '\0') return 0U;
    /* Combo-AT uses comma-separated, unquoted values for these commands. */
    if(!ConfigStringSafe(config->ssid, WIFI_SSID_MAX, 1U) ||
       !ConfigStringSafe(config->wifi_password, WIFI_WIFI_PASSWORD_MAX, 1U) ||
       !ConfigStringSafe(config->host, WIFI_HOST_MAX, 1U) ||
       !ConfigStringSafe(config->client_id, WIFI_CLIENT_ID_MAX, 1U) ||
       !ConfigStringSafe(config->username, WIFI_USERNAME_MAX, 1U) ||
       !ConfigStringSafe(config->mqtt_password, WIFI_MQTT_PASSWORD_MAX, 1U) ||
       !ConfigStringSafe(config->topic_prefix, WIFI_TOPIC_PREFIX_MAX, 1U) ||
       !ConfigStringSafe(config->device_id, WIFI_DEVICE_ID_MAX, 1U)) return 0U;
    if(!DeviceIdIsCanonical(config->device_id)) return 0U;
    /* '>' is the raw-publish prompt.  The parser now recognizes it only at an
     * empty line in the prompt state, but excluding it from MQTT topic paths
     * also keeps command URCs unambiguous across Combo-AT firmware versions. */
    if(strchr(config->topic_prefix, '>') != NULL ||
       strchr(config->device_id, '>') != NULL) return 0U;
    return 1U;
}

static void BuildTopics(void)
{
    const char *client = wifiConfig.client_id[0] ? wifiConfig.client_id : wifiConfig.device_id;
    if(wifiConfig.client_id[0] == '\0')
    {
        strncpy(wifiConfig.client_id, client, WIFI_CLIENT_ID_MAX);
        wifiConfig.client_id[WIFI_CLIENT_ID_MAX] = '\0';
    }
    snprintf(telemetryTopic, sizeof(telemetryTopic), "%s/%s/telemetry",
             wifiConfig.topic_prefix, wifiConfig.device_id);
    /* BW20 Combo-AT P1.0.22 disconnects and reports error 198 when its
     * MQTTSUB topic contains '/'.  Keep only this device-to-board command
     * topic flat; all board-to-cloud topics retain the structured namespace. */
    snprintf(commandTopic, sizeof(commandTopic), "fbgcmd%s",
             &wifiConfig.device_id[4]);
    snprintf(ackTopic, sizeof(ackTopic), "%s/%s/ack",
             wifiConfig.topic_prefix, wifiConfig.device_id);
    snprintf(statusTopic, sizeof(statusTopic), "%s/%s/status",
             wifiConfig.topic_prefix, wifiConfig.device_id);
    snprintf(metadataStressTopic, sizeof(metadataStressTopic), "%s/%s/metadata/stress",
             wifiConfig.topic_prefix, wifiConfig.device_id);
    snprintf(metadataTemperatureTopic, sizeof(metadataTemperatureTopic),
             "%s/%s/metadata/temperature",
             wifiConfig.topic_prefix, wifiConfig.device_id);
}

static uint8_t StoredConfigLoadAt(uint32_t address)
{
    const WifiStoredConfig *stored = (const WifiStoredConfig *)address;
    WifiStoredConfig copy;
    uint32_t expected;

    memcpy(&copy, stored, sizeof(copy));
    if(copy.magic != WIFI_CONFIG_STORE_MAGIC ||
       copy.version != WIFI_CONFIG_STORE_VERSION ||
       copy.size != sizeof(copy)) return 0U;
    expected = WifiCrc32((const uint8_t *)&copy,
                         (uint32_t)sizeof(copy) - sizeof(copy.record_crc));
    if(copy.record_crc != expected || !ConfigIsValid(&copy.config)) return 0U;
    wifiConfig = copy.config;
    wifiConfigCrc = copy.config_crc;
    return 1U;
}

static uint8_t StoredConfigLoad(void)
{
    /* Sector 7 became the OTA receive area.  A board updated once by cable
     * migrates its old record into sector 1 before any OTA erase can touch it. */
    if(StoredConfigLoadAt(WIFI_CONFIG_FLASH_ADDRESS)) return 1U;
    if(StoredConfigLoadAt(WIFI_CONFIG_LEGACY_ADDRESS))
    {
        (void)StoredConfigSave();
        return 1U;
    }
    return 0U;
}

static uint8_t StoredConfigSave(void)
{
    WifiStoredConfig record;
    FLASH_EraseInitTypeDef erase;
    uint32_t sectorError = 0U;
    uint32_t address = WIFI_CONFIG_FLASH_ADDRESS;

    memset(&record, 0, sizeof(record));
    record.magic = WIFI_CONFIG_STORE_MAGIC;
    record.version = WIFI_CONFIG_STORE_VERSION;
    record.size = sizeof(record);
    record.config_crc = wifiConfigCrc;
    record.config = wifiConfig;
    record.record_crc = WifiCrc32((const uint8_t *)&record,
                                  (uint32_t)sizeof(record) - sizeof(record.record_crc));

    if(HAL_FLASH_Unlock() != HAL_OK) return 0U;
    memset(&erase, 0, sizeof(erase));
    erase.TypeErase = FLASH_TYPEERASE_SECTORS;
    erase.VoltageRange = FLASH_VOLTAGE_RANGE_3;
    erase.Sector = FLASH_SECTOR_1;
    erase.NbSectors = 1U;
    if(HAL_FLASHEx_Erase(&erase, &sectorError) != HAL_OK)
    {
        HAL_FLASH_Lock();
        wifiLastError = WIFI_ERROR_FLASH_ERASE;
        return 0U;
    }

    for(uint32_t offset = 0U; offset < sizeof(record); offset += 4U)
    {
        uint32_t word = 0xFFFFFFFFUL;
        uint32_t remaining = (uint32_t)sizeof(record) - offset;
        memcpy(&word, ((const uint8_t *)&record) + offset,
               remaining >= 4U ? 4U : remaining);
        if(HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD, address + offset, word) != HAL_OK)
        {
            HAL_FLASH_Lock();
            wifiLastError = WIFI_ERROR_FLASH_PROGRAM;
            return 0U;
        }
    }
    HAL_FLASH_Lock();
    if(memcmp((const void *)WIFI_CONFIG_FLASH_ADDRESS, &record, sizeof(record)) != 0)
    {
        wifiLastError = WIFI_ERROR_FLASH_PROGRAM;
        return 0U;
    }
    return 1U;
}

static uint8_t ExternalState(void)
{
    if(!wifiConfigValid || !(wifiConfig.flags & WIFI_CONFIG_FLAG_ENABLE)) return 0U;
    if(wifiState == WIFI_STATE_ONLINE ||
       wifiState == WIFI_STATE_PUBLISH_WAIT_PROMPT ||
       wifiState == WIFI_STATE_PUBLISH_WAIT_RESULT) return 4U;
    if(wifiState == WIFI_STATE_LAN_READ ||
       wifiState == WIFI_STATE_LAN_DELETE ||
       wifiState == WIFI_STATE_LAN_RECREATE ||
       wifiState == WIFI_STATE_LAN_SEND_WAIT_PROMPT ||
       wifiState == WIFI_STATE_LAN_SEND_WAIT_RESULT)
        return (lanResumeState == WIFI_STATE_ONLINE) ? 4U : 3U;
    if(wifiState == WIFI_STATE_WAIT_LAN_VERIFY_SERVER_GONE ||
       wifiState == WIFI_STATE_WAIT_LAN_SERVER_DELETE_GRACE)
        return lanServerDeleteVerifyStartup ? 2U :
               ((lanResumeState == WIFI_STATE_ONLINE) ? 4U : 3U);
    if(wifiState == WIFI_STATE_WAIT_LAN_CHILD_DELETE_GRACE) return 2U;
    if(wifiState == WIFI_STATE_WAIT_LAN_VERIFY_CREATED) return 2U;
    if(wifiState == WIFI_STATE_WAIT_LAN_CREATE_GRACE)
        return lanListenerCreateStartup ? 2U :
               ((lanResumeState == WIFI_STATE_ONLINE) ? 4U : 3U);
    if(wifiState >= WIFI_STATE_WAIT_MQTT_HOST &&
       wifiState <= WIFI_STATE_WAIT_MQTT_RECONNECT) return 3U;
    if(wifiState >= WIFI_STATE_WAIT_WARM_WIFI_QUERY &&
       wifiState <= WIFI_STATE_WAIT_LAN_CREATE) return 2U;
    if(wifiState == WIFI_STATE_WAIT_LAN_QUERY_STALE ||
       wifiState == WIFI_STATE_WAIT_LAN_DELETE_CHILD) return 2U;
    if(wifiState == WIFI_STATE_BACKOFF) return 5U;
    return 1U;
}

static void SendUsbConfigAck(uint8_t status)
{
    usbAckCommand = 0x20U;
    usbConfigAckStatus = status;
    usbConfigAckPending = 1U;
}

static void SendUsbStatusAck(void)
{
    usbAckCommand = 0x21U;
    usbConfigAckStatus = 0U;
    usbConfigAckPending = 1U;
}

static void FlushUsbConfigAckIfSafe(void)
{
    uint8_t response[USART_TX_SIZE] = {0};
    if(!usbConfigAckPending) return;
    if(IsScanMode(workState) && !usbConfigAckBoundaryPermit) return;

    response[0] = 0xFFU;
    response[1] = 0xFFU;
    response[2] = workState;
    response[3] = usbAckCommand;
    if(usbAckCommand == 0x21U)
    {
        uint16_t ringUsed = (uint16_t)((rxHead - rxTail) & WIFI_RX_RING_MASK);
        uint8_t flags = (wifiConnected ? 0x01U : 0U)
                | (mqttConnected ? 0x02U : 0U)
                | (uartTxBusy ? 0x04U : 0U)
                | (rxOverflow ? 0x08U : 0U)
                | (uartErrorPending ? 0x10U : 0U)
                | (lanClientActive ? 0x20U : 0U)
                | (OtaUpdate_IsActive() ? 0x40U : 0U);
        response[4] = 1U;
        response[5] = ExternalState();
        response[6] = wifiConfigValid;
        response[7] = (wifiConfigValid &&
                (wifiConfig.flags & WIFI_CONFIG_FLAG_ENABLE)) ? 1U : 0U;
        PutU16Be(&response[8], wifiLastError);
        response[10] = (uint8_t)wifiState;
        response[11] = retryExponent;
        response[12] = flags;
        response[13] = (ringUsed > 255U) ? 255U : (uint8_t)ringUsed;
        PutU16Be(&response[14], uartTxCompleteCount);
        PutU16Be(&response[16], uartRxByteCount);
    }
    else
    {
        response[4] = usbConfigAckStatus;
        response[5] = ExternalState();
        response[6] = wifiConfigValid;
        response[7] = (wifiConfigValid && (wifiConfig.flags & WIFI_CONFIG_FLAG_ENABLE)) ? 1U : 0U;
        PutU32Be(&response[8], wifiConfigCrc);
        PutU16Be(&response[12], wifiLastError);
        response[14] = wifiConfigValid ? wifiConfig.flags : 0U;
        response[15] = wifiConfigValid ? wifiConfig.scheme : 0U;
        PutU16Be(&response[16], wifiConfigValid ? wifiConfig.port : 0U);
    }
    response[18] = 0xFFU;
    response[19] = 0xEFU;
    usbConfigAckPending = 0U;
    usbConfigAckBoundaryPermit = 0U;
    USB_Queue_Send(response, sizeof(response));
}

static void BuildMetadataFrame(uint8_t metadataIndex,
                               uint8_t mode,
                               uint16_t startRow,
                               uint16_t pointCount)
{
    uint8_t *frame = metadataFrames[metadataIndex];
    uint8_t wavelengthBytes[WIFI_METADATA_MAX_POINT_COUNT * 4U];
    uint16_t position = 12U;
    uint16_t wavelengthLength = pointCount * 4U;
    uint32_t tableCrc;

    for(uint16_t point = 0U; point < pointCount; point++)
    {
        uint16_t row = startRow + point;
        uint32_t wavelengthPm;
        if(mode == TABLE_STATE)
        {
            wavelengthPm = (uint32_t)Stress_Wave_DATA[point][0] * 1000UL +
                           (uint32_t)Stress_Wave_DATA[point][1];
        }
        else
        {
            wavelengthPm = (uint32_t)Wave_DATA[row][0] * 1000UL +
                           (uint32_t)Wave_DATA[row][1];
        }
        PutU32Be(&wavelengthBytes[point * 4U], wavelengthPm);
    }
    tableCrc = WifiCrc32(wavelengthBytes, wavelengthLength);

    frame[0] = 'F'; frame[1] = 'B'; frame[2] = 'G'; frame[3] = 'M';
    frame[4] = 1U;
    frame[5] = mode;
    PutU16Be(&frame[6], pointCount);
    PutU32Be(&frame[8], tableCrc);
    memcpy(&frame[position], wavelengthBytes, wavelengthLength);
    position += wavelengthLength;
    PutU32Be(&frame[position], WifiCrc32(frame, position));
    position += 4U;
    metadataLengths[metadataIndex] = position;
    if(metadataIndex == 0U) stressTableCrc = tableCrc;
    else temperatureTableCrc = tableCrc;
}

static uint8_t QueueModeAck(uint32_t requestId,
                            const char *status,
                            uint8_t mode,
                            uint32_t sequence)
{
    uint8_t formatted[WIFI_SMALL_PAYLOAD_SIZE];
    const char *modeName = (mode == TABLE_STATE) ? "STRESS" : "TEMPERATURE";
    int length = snprintf((char *)formatted, sizeof(formatted),
                           "FACK1|%08lX|%s|%s|%08lX",
                           (unsigned long)requestId, status, modeName,
                           (unsigned long)sequence);
    if(length <= 0 || length >= (int)sizeof(formatted)) return 0U;

    /* QoS-1 may redeliver a command while its ACK is still queued or in
     * flight.  Keep a single copy so duplicates cannot exhaust the bounded
     * queue and starve the terminal APPLIED/REJECTED acknowledgement. */
    for(uint8_t index = 0U; index < ackQueueCount; index++)
    {
        uint8_t slot = (uint8_t)((ackQueueHead + index) % WIFI_ACK_QUEUE_DEPTH);
        if(ackQueue[slot].length == (uint16_t)length &&
           memcmp(ackQueue[slot].payload, formatted, (size_t)length) == 0)
            return 1U;
    }

    if(ackQueueCount >= WIFI_ACK_QUEUE_DEPTH)
    {
        wifiLastError = WIFI_ERROR_ACK_OVERFLOW;
        return 0U;
    }

    {
        uint8_t tail = (uint8_t)((ackQueueHead + ackQueueCount) % WIFI_ACK_QUEUE_DEPTH);
        memcpy(ackQueue[tail].payload, formatted, (size_t)length);
        ackQueue[tail].length = (uint16_t)length;
        ackQueueCount++;
    }
    return 1U;
}

static void DiscardQueuedTelemetry(void)
{
    appliedFrameMustPublish = 0U;
    latestSnapshot = -1;
    for(uint8_t index = 0U; index < 2U; index++) snapshots[index].ready = 0U;
    if(activeSnapshot >= 0 && publishKind == WIFI_PUBLISH_TELEMETRY)
    {
        /* DMA cannot be cancelled safely here.  The active old-mode frame may
         * finish at the module, but it is never retried and every queued frame
         * is already discarded. */
        snapshots[(uint8_t)activeSnapshot].ready = 0U;
        activeTelemetryDiscarded = 1U;
    }
}

static void RestartBackoff(uint16_t errorCode)
{
    uint32_t delayMs;
    uint32_t now = WifiNowMs();
    uint8_t hadLanOwner = (lanClientActive || publishOverLan) ? 1U : 0U;
    uint8_t rawSessionWasActive = lanRawMode;

    /* A RAW client can directly leave the PI11210 in source mode.  UART/Wi-Fi
     * recovery must therefore fail dark before ownership and RAW state are
     * discarded, even when no final desktop shutter command can arrive. */
    if(hadLanOwner && rawSessionWasActive)
        (void)PI11210_SetSOAShutter(1U);

    if(errorCode != WIFI_ERROR_NONE) wifiLastError = errorCode;
    wifiConnected = 0U;
    mqttConnected = 0U;
    warmWifiStatusSeen = 0U;
    warmWifiMatchesConfig = 0U;
    lanSocketId = -1;
    lanCreatedSocketId = -1;
    lanCleanupSocketId = -1;
    lanDeleteTargetId = -1;
    lanCleanupCanPromote = 0U;
    lanCleanupOverflow = 0U;
    lanReadPending = 0U;
    lanClientActive = 0U;
    ResetLanRawState();
    lanSocketResetPending = 0U;
    lanDeleteKind = WIFI_LAN_DELETE_SERVER;
    lanStartupStaleChildId = -1;
    lanStartupServerStatus = -1;
    lanStartupCleanupAttempts = 0U;
    lanServerDeleteVerifyAttempts = 0U;
    lanServerDeleteVerifyStartup = 0U;
    lanListenerCreateVerifyAttempts = 0U;
    lanListenerCreateStartup = 0U;
    lanMetadataPendingMask = 0U;
    lanClientDeadlineMs = 0U;
    nextLanRetryMs = 0U;
    publishOverLan = 0U;
    otaResetAfterLanDelete = 0U;
    responseOk = responseError = responsePrompt = responsePublishBusy = 0U;

    if(publishKind == WIFI_PUBLISH_STATUS) statusPending = 1U;
    else if(publishKind == WIFI_PUBLISH_METADATA_STRESS) metadataPendingMask |= 0x01U;
    else if(publishKind == WIFI_PUBLISH_METADATA_TEMPERATURE) metadataPendingMask |= 0x02U;
    /* ACKs remain at ackQueueHead until PublishSuccess(), so reconnecting
     * automatically retries the exact in-flight acknowledgement. */
    else if(publishKind == WIFI_PUBLISH_TELEMETRY && activeSnapshot >= 0)
    {
        uint8_t active = (uint8_t)activeSnapshot;
        if(uartTxBusy)
        {
            /* USART6 DMA still owns these bytes.  Do not also expose this slot
             * as latestSnapshot: the next completed scan would overwrite it
             * before the DMA callback releases the reservation.  Latest-only
             * telemetry deliberately drops this interrupted retry in favour
             * of the next complete frame. */
            snapshots[active].ready = 0U;
            dmaReservedSnapshot = activeSnapshot;
        }
        else if(!activeTelemetryDiscarded && latestSnapshot < 0)
        {
            snapshots[active].ready = 1U;
            latestSnapshot = activeSnapshot;
        }
        else snapshots[active].ready = 0U;
    }
    activeSnapshot = -1;
    activeTelemetryDiscarded = 0U;
    publishKind = WIFI_PUBLISH_NONE;
    publishPayload = NULL;
    if(hadLanOwner)
    {
        pendingModeValid = 0U;
        modeAckWaitFirstFrame = 0U;
        lastAppliedRequestId = 0U;
        lastAppliedMode = 0xFFU;
        lastAppliedSequence = 0U;
        ackQueueHead = 0U;
        ackQueueCount = 0U;
    }

    if(!wifiConfigValid || !(wifiConfig.flags & WIFI_CONFIG_FLAG_ENABLE))
    {
        wifiState = WIFI_STATE_DISABLED;
        return;
    }

    delayMs = 1000UL << (retryExponent > 5U ? 5U : retryExponent);
    if(retryExponent < 5U) retryExponent++;
    wifiState = WIFI_STATE_BACKOFF;
    stateDeadlineMs = now + delayMs;
}

static uint8_t ParseHexRequestId(const char *text, uint32_t *value)
{
    uint32_t parsed = 0U;
    for(uint8_t index = 0U; index < 8U; index++)
    {
        char character = text[index];
        uint8_t nibble;
        if(character >= '0' && character <= '9') nibble = (uint8_t)(character - '0');
        else if(character >= 'A' && character <= 'F') nibble = (uint8_t)(character - 'A' + 10);
        else if(character >= 'a' && character <= 'f') nibble = (uint8_t)(character - 'a' + 10);
        else return 0U;
        parsed = (parsed << 4) | nibble;
    }
    *value = parsed;
    return 1U;
}

static uint8_t HexNibble(char character, uint8_t *value)
{
    if(character >= '0' && character <= '9')
        *value = (uint8_t)(character - '0');
    else if(character >= 'A' && character <= 'F')
        *value = (uint8_t)(character - 'A' + 10);
    else if(character >= 'a' && character <= 'f')
        *value = (uint8_t)(character - 'a' + 10);
    else return 0U;
    return 1U;
}

static uint8_t ParseHexU16(const char *text, uint16_t *value)
{
    uint16_t parsed = 0U;
    for(uint8_t index = 0U; index < 4U; index++)
    {
        uint8_t nibble;
        if(!HexNibble(text[index], &nibble)) return 0U;
        parsed = (uint16_t)((parsed << 4) | nibble);
    }
    *value = parsed;
    return 1U;
}

static void ResetModeTransactions(void)
{
    pendingModeValid = 0U;
    modeAckWaitFirstFrame = 0U;
    lastAppliedRequestId = 0U;
    lastAppliedMode = 0xFFU;
    lastAppliedSequence = 0U;
    ackQueueHead = 0U;
    ackQueueCount = 0U;
}

static void SetLanClientActive(uint8_t active)
{
    uint32_t now = WifiNowMs();
    uint8_t rawSessionWasActive = lanRawMode;
    active = active ? 1U : 0U;
    if(active == lanClientActive)
    {
        if(active) lanClientDeadlineMs = now + WIFI_LAN_CLIENT_TIMEOUT_MS;
        return;
    }

    lanClientActive = active;
    DiscardQueuedTelemetry();
    ResetModeTransactions();
    if(active)
    {
        lanClientDeadlineMs = now + WIFI_LAN_CLIENT_TIMEOUT_MS;
        lanMetadataPendingMask = 0x03U;
        metadataPendingMask = 0U;
        nextLocalHeartbeatMs = now + 20000U;
    }
    else
    {
        /* Closing, timing out or replacing a byte-stream control session must
         * never preserve its last optical-output command.  This is a board-
         * side safety action and does not depend on a working TCP reply. */
        if(rawSessionWasActive)
            (void)PI11210_SetSOAShutter(1U);
        ResetLanRawState();
        OtaUpdate_OnClientDisconnected();
        lanClientDeadlineMs = 0U;
        lanMetadataPendingMask = 0U;
        if(!localControlActive) metadataPendingMask = 0x03U;
        nextLocalHeartbeatMs = 0U;
    }
    statusPending = 1U;
    nextPublishMs = now + 20U;
}

static void ParseLanRawCommand(const char *command, uint32_t length)
{
    uint32_t requestId;
    uint16_t offset;
    uint16_t total;
    uint16_t chunkLength;
    uint16_t nextOffset;
    uint8_t ackPayload[2];

    /* FUSB1|IIIIIIII|OOOO|TTTT|HH... where the hex payload is one
     * sequential chunk of a zero-padded 808-byte native USB command. */
    if(length < 25U || strncmp(command, WIFI_LAN_RAW_COMMAND_PREFIX, 6U) != 0 ||
       command[14] != '|' || command[19] != '|' || command[24] != '|') return;
    if(!ParseHexRequestId(&command[6], &requestId) ||
       !ParseHexU16(&command[15], &offset) ||
       !ParseHexU16(&command[20], &total)) return;
    if(total == 0U || total > USART_RX_SIZE || length < 25U ||
       ((length - 25U) & 1U) != 0U) return;
    chunkLength = (uint16_t)((length - 25U) / 2U);
    if(chunkLength == 0U || offset > total || chunkLength > total - offset) return;

    if(offset == 0U &&
       (requestId != lanRawCommandId || total != lanRawCommandTotal ||
        lanRawCommandExpected >= lanRawCommandTotal))
    {
        memset(lanRawCommand, 0, sizeof(lanRawCommand));
        lanRawCommandId = requestId;
        lanRawCommandExpected = 0U;
        lanRawCommandTotal = total;
    }
    if(requestId != lanRawCommandId || total != lanRawCommandTotal) return;

    if(offset == lanRawCommandExpected)
    {
        for(uint16_t index = 0U; index < chunkLength; index++)
        {
            uint8_t high;
            uint8_t low;
            if(!HexNibble(command[25U + index * 2U], &high) ||
               !HexNibble(command[26U + index * 2U], &low)) return;
            lanRawCommand[offset + index] = (uint8_t)((high << 4) | low);
        }
        lanRawCommandExpected = (uint16_t)(offset + chunkLength);
    }
    else if(offset > lanRawCommandExpected)
    {
        return;
    }

    /* Only acknowledge bytes that are durably present in the sequential
     * assembly buffer.  A retry therefore resumes from the returned offset. */
    nextOffset = lanRawCommandExpected;
    ackPayload[0] = (uint8_t)(nextOffset >> 8);
    ackPayload[1] = (uint8_t)nextOffset;
    (void)QueueLanRawPriority(WIFI_LAN_RAW_KIND_COMMAND_ACK,
                              requestId, ackPayload, sizeof(ackPayload));
    nextPublishMs = WifiNowMs();

    if(lanRawCommandExpected == lanRawCommandTotal && ReceEndFlag == 0U)
    {
        memcpy(aRxBuffer, lanRawCommand, sizeof(aRxBuffer));
        ReceEndFlag = 1U;
    }
}

static void ParseModeCommandPayload(const char *payload,
                                    uint32_t declaredLength,
                                    uint8_t fromLan)
{
    uint32_t requestId;
    uint8_t requestedMode;

    if(localControlActive) return;
    if(declaredLength < 26U || strncmp(payload, "FCMD1|", 6U) != 0 ||
       payload[14] != '|' || strncmp(&payload[15], "MODE|", 5U) != 0) return;
    if(!ParseHexRequestId(&payload[6], &requestId)) return;

    if(declaredLength == 26U && strncmp(&payload[20], "STRESS", 6U) == 0)
        requestedMode = TABLE_STATE;
    else if(declaredLength == 31U &&
            strncmp(&payload[20], "TEMPERATURE", 11U) == 0)
        requestedMode = PRECISION_TABLE_STATE;
    else return;

    /* A socket connection or malformed look-alike command must not steal
     * ownership from MQTT.  Claim LAN only after the complete command has
     * passed its length, grammar, request-id and mode checks. */
    if(fromLan) SetLanClientActive(1U);
    else if(lanClientActive) return;

    /* P1.0.22 reports publish error 198 if a RAW publish is launched inside
     * its MQTT subscription callback.  The delay is harmless for LAN and
     * keeps the shared acknowledgement path deterministic. */
    nextPublishMs = WifiNowMs() + (fromLan ? 20U : 1000U);

    if(requestId == lastAppliedRequestId && requestedMode == lastAppliedMode &&
       workState == requestedMode)
    {
        QueueModeAck(requestId, "APPLIED", requestedMode,
                     lastAppliedSequence);
        return;
    }
    if(pendingModeValid)
    {
        if(requestId == pendingModeRequestId && requestedMode == pendingMode)
        {
            if(ackQueueCount + 1U < WIFI_ACK_QUEUE_DEPTH)
                QueueModeAck(requestId, "ACCEPTED", requestedMode, 0U);
        }
        else if(ackQueueCount + 1U < WIFI_ACK_QUEUE_DEPTH)
        {
            QueueModeAck(requestId, "REJECTED", requestedMode, 0U);
        }
        return;
    }
    if(modeAckWaitFirstFrame)
    {
        if(requestId == modeAckRequestId && requestedMode == modeAckTargetMode)
        {
            if(ackQueueCount + 1U < WIFI_ACK_QUEUE_DEPTH)
                QueueModeAck(requestId, "ACCEPTED", requestedMode, 0U);
        }
        else if(ackQueueCount + 1U < WIFI_ACK_QUEUE_DEPTH)
        {
            QueueModeAck(requestId, "REJECTED", requestedMode, 0U);
        }
        return;
    }

    if(ackQueueCount + 1U < WIFI_ACK_QUEUE_DEPTH &&
       QueueModeAck(requestId, "ACCEPTED", requestedMode, 0U))
    {
        pendingMode = requestedMode;
        pendingModeRequestId = requestId;
        pendingModeValid = 1U;
    }
}

static void ParseRemoteCommand(const char *line)
{
    const char prefix[] = "+EVENT:MQTT_SUB,";
    const char *topicStart;
    const char *topicEnd;
    const char *lengthEnd;
    const char *payload;
    uint32_t declaredLength = 0U;

    /* An open local CDC session owns the board.  Do not acknowledge or stage
     * public commands; local debug is deliberately silent on both network
     * data paths. */
    if(localControlActive) return;

    if(strncmp(line, prefix, sizeof(prefix) - 1U) != 0) return;
    topicStart = line + sizeof(prefix) - 1U;
    topicEnd = strchr(topicStart, ',');
    if(topicEnd == NULL ||
       (uint16_t)(topicEnd - topicStart) != strlen(commandTopic) ||
       strncmp(topicStart, commandTopic, (size_t)(topicEnd - topicStart)) != 0) return;
    lengthEnd = strchr(topicEnd + 1, ',');
    if(lengthEnd == NULL) return;
    for(const char *cursor = topicEnd + 1; cursor < lengthEnd; cursor++)
    {
        if(*cursor < '0' || *cursor > '9') return;
        declaredLength = declaredLength * 10U + (uint32_t)(*cursor - '0');
        if(declaredLength >= WIFI_LINE_SIZE) return;
    }
    payload = lengthEnd + 1;
    if(strlen(payload) != declaredLength) return;
    ParseModeCommandPayload(payload, declaredLength, 0U);
}

static uint8_t IsLanSessionHello(const char *payload, uint32_t length)
{
    if(payload == NULL) return 0U;
    if(length == (sizeof(WIFI_LAN_HELLO) - 1U) &&
       memcmp(payload, WIFI_LAN_HELLO, sizeof(WIFI_LAN_HELLO) - 1U) == 0)
        return 1U;
    if(length == (sizeof(WIFI_LAN_RAW_HELLO) - 1U) &&
       memcmp(payload, WIFI_LAN_RAW_HELLO,
              sizeof(WIFI_LAN_RAW_HELLO) - 1U) == 0)
        return 1U;
    return 0U;
}

static void QueueLanSocketCleanup(int connectionId, uint8_t canPromote)
{
    if(connectionId < 0 ||
       connectionId == WIFI_LAN_SOCKET_ID ||
       connectionId == lanSocketId) return;

    /* A BW20 listener which lost an OTA owner has repeatedly demonstrated
     * that deleting only its child can leave the server accepting connections
     * which it immediately resets.  Do not promote any contender until the
     * listener itself has gone through the verified recycle path. */
    if(OtaUpdate_IsActive() || lanCleanupOverflow) canPromote = 0U;

    /* Keep the first waiting/rejected seed until it is promoted or its
     * AT+SOCKETDEL has completed.  A later URC cannot change the target of an
     * in-flight delete.  Further contenders never become owner. */
    if(lanCleanupSocketId < 0)
    {
        lanCleanupSocketId = (int32_t)connectionId;
        lanCleanupCanPromote = canPromote ? 1U : 0U;
        if(!lanSocketResetPending) nextLanRetryMs = WifiNowMs();
    }
    else if(lanCleanupSocketId == connectionId && !canPromote)
    {
        /* A Disconnect URC makes a previously waiting connection ineligible
         * for promotion, but it still needs manual deletion. */
        lanCleanupCanPromote = 0U;
    }
    else if(lanCleanupSocketId != connectionId)
    {
        /* A single scalar cannot retain every ConID from a connection burst.
         * Once the selected and queued seeds are gone, recycle the listener
         * through the verified-delete path so untracked seeds cannot leak. */
        lanCleanupOverflow = 1U;
    }
}

static void RejectSelectedLanSocket(void)
{
    lanReadPending = 0U;
    SetLanClientActive(0U);
    lanClientDeadlineMs = 0U;
    lanSocketResetPending = 1U;
    /* BW20 P1.0.22 can leave the TCPServer accepting-and-resetting every
     * subsequent client after any selected child disappears.  This also
     * happens when the post-OTA RAW READY probe closes, after the OTA session
     * has already been released.  Always delete the exact child first and
     * then recycle/verify the parent listener; never promote a contender into
     * the known-bad parent generation. */
    lanCleanupOverflow = 1U;
    lanCleanupCanPromote = 0U;
    /* AutoDel can follow Disconnect late, and a timed-out ConID may be reused
     * before the host sees that URC.  Keep output fail-dark immediately, but
     * defer the destructive AT+SOCKETDEL briefly so a replacement Seed/HELLO
     * can prove that the same numeric ConID now names a new connection. */
    nextLanRetryMs = WifiNowMs() + WIFI_LAN_STALE_DELETE_GRACE_MS;
}

static uint8_t ReuseSelectedLanSocketCandidate(int connectionId)
{
    if(localControlActive || connectionId != lanSocketId ||
       !lanSocketResetPending || lanCleanupOverflow) return 0U;
    if(wifiState == WIFI_STATE_LAN_DELETE &&
       lanDeleteKind == WIFI_LAN_DELETE_OWNER &&
       connectionId == lanDeleteTargetId)
    {
        /* AT+SOCKETDEL already owns this numeric target.  It cannot be
         * cancelled safely; the desktop will reconnect after it completes. */
        return 0U;
    }

    lanSocketResetPending = 0U;
    lanReadPending = 0U;
    lanClientDeadlineMs = WifiNowMs() + WIFI_LAN_CLIENT_TIMEOUT_MS;
    return 1U;
}

static void PromoteWaitingLanSocket(void)
{
    if(localControlActive || lanCleanupSocketId < 0 ||
       !lanCleanupCanPromote) return;

    lanSocketId = lanCleanupSocketId;
    lanCleanupSocketId = -1;
    lanCleanupCanPromote = 0U;
    lanReadPending = 0U;
    /* Promotion grants only a bounded candidate slot.  The connection still
     * owns no controls until a subsequent exact HELLO is received. */
    lanClientDeadlineMs = WifiNowMs() + WIFI_LAN_CLIENT_TIMEOUT_MS;
}

static void ParseLanPayload(const char *payload, uint32_t declaredLength)
{
    const char *command;
    uint32_t commandLength;

    uint8_t rebootAfterResponse = 0U;

    if(strlen(payload) != declaredLength) return;
    if(declaredLength == (sizeof(WIFI_LAN_HELLO) - 1U) &&
       memcmp(payload, WIFI_LAN_HELLO,
              sizeof(WIFI_LAN_HELLO) - 1U) == 0)
    {
        if(!localControlActive)
        {
            SetLanClientActive(1U);
            if(lanRawMode)
            {
                /* A conventional LAN viewer may take over immediately after
                 * the byte-stream client closes.  Its HELLO explicitly exits
                 * RAW mode even if the modem has not reported SocketDown yet. */
                ResetLanRawState();
                lanMetadataPendingMask = 0x03U;
                statusPending = 1U;
                nextPublishMs = WifiNowMs();
            }
        }
        return;
    }

    if(declaredLength == (sizeof(WIFI_LAN_RAW_HELLO) - 1U) &&
       memcmp(payload, WIFI_LAN_RAW_HELLO,
              sizeof(WIFI_LAN_RAW_HELLO) - 1U) == 0)
    {
        if(!localControlActive)
        {
            uint8_t compactAlreadyNegotiated = lanRawCompactScan;
            SetLanClientActive(1U);
            lanRawMode = 1U;
            lanMetadataPendingMask = 0U;
            DiscardQueuedTelemetry();
            statusPending = 0U;
            /* Kind 2 is the explicit RAW-ready handshake.  The desktop does
             * not expose the link as connected until this frame arrives, so
             * its first native DAC/ADC command cannot race the mode switch. */
            /* The READY sequence is the current boot token.  It remains
             * transparent to normal LAN clients but lets the OTA uploader
             * distinguish the freshly installed application from the old
             * half-open TCP connection left by a BW20 software reset. */
            /* Once a new client has acknowledged READY by negotiating CS1,
             * later lease-refresh HELLOs must not inject a priority packet
             * into the live scan stream.  Until then READY is repeated, so a
             * lost initial handshake remains self-healing and old clients
             * retain the legacy behaviour. */
            if(!compactAlreadyNegotiated)
                (void)QueueLanRawPriority(WIFI_LAN_RAW_KIND_READY,
                                          bootId, NULL, 0U);
            nextPublishMs = WifiNowMs();
        }
        return;
    }

    if(declaredLength == (sizeof(WIFI_LAN_RAW_COMPACT_CAP) - 1U) &&
       memcmp(payload, WIFI_LAN_RAW_COMPACT_CAP,
              sizeof(WIFI_LAN_RAW_COMPACT_CAP) - 1U) == 0)
    {
        if(!localControlActive && lanRawMode)
        {
            /* Capability negotiation is intentionally one-way and opt-in.
             * An old desktop never sends it and therefore continues to get
             * byte-for-byte native USB frames. */
            SetLanClientActive(1U);
            lanRawCompactScan = 1U;
        }
        return;
    }

    if(!localControlActive && lanRawMode &&
       declaredLength >= 25U &&
       strncmp(payload, WIFI_LAN_RAW_COMMAND_PREFIX, 6U) == 0)
    {
        SetLanClientActive(1U);
        ParseLanRawCommand(payload, declaredLength);
        return;
    }

    if(!localControlActive &&
       OtaUpdate_HandleCommand(payload, declaredLength,
                               wifiConfig.wifi_password,
                               (char *)otaResponse,
                               sizeof(otaResponse),
                               &rebootAfterResponse))
    {
        SetLanClientActive(1U);
        DiscardQueuedTelemetry();
        otaResponseLength = (uint16_t)strlen((const char *)otaResponse);
        otaResponseReady = (otaResponseLength > 0U) ? 1U : 0U;
        otaRebootAfterResponse = rebootAfterResponse;
        nextPublishMs = WifiNowMs();
        return;
    }

    command = strstr(payload, "FCMD1|");
    if(command == NULL) return;
    commandLength = (uint32_t)strlen(command);
    if(commandLength == 26U || commandLength == 31U)
        ParseModeCommandPayload(command, commandLength, 1U);
}

static void ParseLanSocketTableRow(const char *line)
{
    int connectionId;
    int type;
    int status;
    int remotePort;
    int localPort;
    int serverConnectionId;
    int remoteHostOffset = 0;
    const char *remoteHostEnd;

    if(wifiState != WIFI_STATE_WAIT_LAN_QUERY_STALE &&
       wifiState != WIFI_STATE_WAIT_LAN_VERIFY_SERVER_GONE &&
       wifiState != WIFI_STATE_WAIT_LAN_VERIFY_CREATED) return;
    if(sscanf(line, "%d,%d,%d,%n", &connectionId, &type, &status,
              &remoteHostOffset) != 3 || remoteHostOffset <= 0) return;
    remoteHostEnd = strchr(line + remoteHostOffset, ',');
    if(remoteHostEnd == NULL ||
       sscanf(remoteHostEnd + 1, "%d,%d,%d", &remotePort, &localPort,
              &serverConnectionId) != 3) return;
    (void)remotePort;

    if(connectionId == WIFI_LAN_SOCKET_ID && type == 3 &&
       localPort == (int)WIFI_LAN_SERVER_PORT)
    {
        lanStartupServerStatus =
            (status >= -1 && status <= 127) ? (int8_t)status : -1;
    }
    else if(wifiState == WIFI_STATE_WAIT_LAN_QUERY_STALE &&
            type == 5 && serverConnectionId == WIFI_LAN_SOCKET_ID &&
            connectionId >= 0 &&
            connectionId != WIFI_LAN_SOCKET_ID &&
            lanStartupStaleChildId < 0)
    {
        /* Combo-AT requires every disconnected TCPServer seed to be deleted
         * explicitly.  At MCU startup no desktop owns a session yet, so every
         * seed inherited from the previous application generation is stale. */
        lanStartupStaleChildId = (int32_t)connectionId;
    }
}

static void ParseLanSocketLine(const char *line)
{
    int connectionId;
    int serverConnectionId;
    unsigned long declaredLength;
    int payloadOffset = 0;
    const char *payload;
    const char *idMarker;

    ParseLanSocketTableRow(line);

    idMarker = strstr(line, "ConID=");
    if(idMarker != NULL &&
       (wifiState == WIFI_STATE_WAIT_LAN_CREATE ||
        wifiState == WIFI_STATE_LAN_RECREATE))
    {
        long parsedId = strtol(idMarker + 6, NULL, 10);
        if(parsedId >= 0)
            lanCreatedSocketId = (int32_t)parsedId;
    }

    if(sscanf(line, "+Seed:%d,%d", &connectionId,
              &serverConnectionId) == 2 ||
       sscanf(line, "+EVENT:SocketSeed,%d,%d", &connectionId,
              &serverConnectionId) == 2)
    {
        if(serverConnectionId == WIFI_LAN_SOCKET_ID &&
           connectionId >= 0)
        {
            if(connectionId == lanSocketId)
            {
                /* Normally this is a duplicate URC.  While the previous
                 * generation is awaiting cleanup, however, the module may
                 * already have recycled the same numeric ConID. */
                (void)ReuseSelectedLanSocketCandidate(connectionId);
            }
            else if(localControlActive || lanClientActive ||
                    lanSocketId >= 0 || lanSocketResetPending)
            {
                /* An active owner is immutable until it disconnects or times
                 * out.  In particular, a new Seed must not overwrite the id
                 * used by SEND/READ and inherit the old RAW/OTA session. */
                QueueLanSocketCleanup(connectionId,
                                      localControlActive ? 0U : 1U);
            }
            else
            {
                /* This is only a candidate.  Ownership starts after its exact
                 * FBGL1 HELLO payload has been received and validated. */
                lanSocketId = (int32_t)connectionId;
                lanClientDeadlineMs = WifiNowMs() +
                                      WIFI_LAN_CLIENT_TIMEOUT_MS;
            }
        }
        return;
    }

    if(sscanf(line, "+EVENT:SocketAutoDel,%d", &connectionId) == 1)
    {
        if(connectionId == lanCleanupSocketId &&
            !(wifiState == WIFI_STATE_LAN_DELETE &&
             lanDeleteKind == WIFI_LAN_DELETE_REJECTED &&
             connectionId == lanDeleteTargetId))
        {
            lanCleanupSocketId = -1;
            lanCleanupCanPromote = 0U;
        }
        if(connectionId == lanSocketId)
        {
            /* AutoDel may arrive without a preceding Disconnect URC.  Apply
             * the same deterministic parent recycle as RejectSelectedLanSocket. */
            lanCleanupOverflow = 1U;
            lanCleanupCanPromote = 0U;
            SetLanClientActive(0U);
            lanReadPending = 0U;
            lanSocketId = -1;
            lanClientDeadlineMs = 0U;
            if(!(wifiState == WIFI_STATE_LAN_DELETE &&
                 lanDeleteKind == WIFI_LAN_DELETE_OWNER &&
                 connectionId == lanDeleteTargetId))
                lanSocketResetPending = 0U;
            if(!lanSocketResetPending) PromoteWaitingLanSocket();
        }
        return;
    }

    if(sscanf(line, "+EVENT:SocketDisconnect,%d", &connectionId) == 1 ||
       sscanf(line, "+EVENT:SocketDissconnect,%d", &connectionId) == 1)
    {
        if(connectionId == lanSocketId)
        {
            /* P1.0.22 changes a disconnected TCP seed to state 127 but does
             * not release its ConID.  Delete that child explicitly; otherwise
             * repeated viewer sessions eventually exhaust all socket slots. */
            RejectSelectedLanSocket();
        }
        else if(connectionId != WIFI_LAN_SOCKET_ID)
            QueueLanSocketCleanup(connectionId, 0U);
        return;
    }

    payloadOffset = 0;
    if(sscanf(line, "+EVENT:SocketDown,%d,%lu,%n", &connectionId,
              &declaredLength, &payloadOffset) == 2)
    {
        uint8_t validChild = (connectionId >= 0 &&
                              connectionId != WIFI_LAN_SOCKET_ID) ? 1U : 0U;
        if(!validChild || declaredLength == 0U) return;

        if(localControlActive ||
           (lanClientActive && connectionId != lanSocketId) ||
           (lanSocketResetPending && connectionId != lanSocketId))
        {
            QueueLanSocketCleanup(connectionId,
                                  localControlActive ? 0U : 1U);
            return;
        }

        /* Some BW20 P1.0.22 builds omit +Seed.  Re-learn an id only as an
         * unowned candidate; its first payload must still be an exact HELLO. */
        if(!lanClientActive && lanSocketId < 0 && !lanSocketResetPending)
        {
            lanSocketId = (int32_t)connectionId;
            lanClientDeadlineMs = WifiNowMs() +
                                  WIFI_LAN_CLIENT_TIMEOUT_MS;
        }
        if(connectionId != lanSocketId)
        {
            QueueLanSocketCleanup(connectionId, 1U);
            return;
        }
        if(lanSocketResetPending)
        {
            if(payloadOffset <= 0)
            {
                /* Read before deleting: an exact HELLO may prove that the
                 * module has already reused this ConID for a live socket. */
                lanReadPending = 1U;
                return;
            }
            payload = line + payloadOffset;
            if(strlen(payload) != declaredLength ||
               !IsLanSessionHello(payload, (uint32_t)declaredLength) ||
               !ReuseSelectedLanSocketCandidate(connectionId)) return;
            ParseLanPayload(payload, (uint32_t)declaredLength);
            return;
        }

        if(payloadOffset > 0)
        {
            payload = line + payloadOffset;
            if(strlen(payload) != declaredLength) return;
            if(!lanClientActive &&
               !IsLanSessionHello(payload, (uint32_t)declaredLength))
            {
                RejectSelectedLanSocket();
                return;
            }
            ParseLanPayload(payload, (uint32_t)declaredLength);
        }
        else
            lanReadPending = 1U;
        return;
    }

    if(sscanf(line, "+SOCKETREAD,%d,%lu,%n", &connectionId,
              &declaredLength, &payloadOffset) != 2 || payloadOffset <= 0 ||
       connectionId != lanSocketId || declaredLength >= WIFI_LINE_SIZE) return;
    payload = line + payloadOffset;
    if(strlen(payload) != declaredLength) return;

    if(!lanClientActive &&
       !IsLanSessionHello(payload, (uint32_t)declaredLength))
    {
        RejectSelectedLanSocket();
        return;
    }

    if(lanSocketResetPending &&
       !ReuseSelectedLanSocketCandidate(connectionId)) return;

    ParseLanPayload(payload, (uint32_t)declaredLength);
}

static void ParseAtLine(const char *line)
{
    if(strcmp(line, "ready") == 0)
    {
        /* The module rebooted independently (for example after a brownout or
         * an old AT-firmware fault).  Its MQTT session and RAM configuration
         * are gone even if the STM32-side flags still said ONLINE. */
        RestartBackoff(WIFI_ERROR_NONE);
        return;
    }
    if(strcmp(line, "OK") == 0 || strcmp(line, "+OK") == 0) responseOk = 1U;
    else if(strncmp(line, "+MQTTPUBRAW:198", 16U) == 0)
        responsePublishBusy = 1U;
    else if(strncmp(line, "ERROR", 5U) == 0 || strcmp(line, "FAIL") == 0 ||
            strstr(line, "Unknown cmd:") != NULL)
    {
        if(!responsePublishBusy) responseError = 1U;
    }

    ParseWifiJoinStatus(line);
    if(strstr(line, "+EVENT:WIFI_GOT_IP") != NULL) wifiConnected = 1U;
    if(strstr(line, "+EVENT:WIFI_DISCONNECT") != NULL)
    {
        wifiConnected = 0U;
        mqttConnected = 0U;
        warmWifiMatchesConfig = 0U;
    }
    if(strstr(line, "+EVENT:MQTT_CONNECT") != NULL ||
       strstr(line, "+MQTT:3,") != NULL) mqttConnected = 1U;
    if(strstr(line, "+EVENT:MQTT_DISCONNECT") != NULL ||
       strstr(line, "+EVENT:MQTT_MALLOC_ERROR") != NULL) mqttConnected = 0U;
    ParseLanSocketLine(line);
    ParseRemoteCommand(line);
}

static void DrainRx(void)
{
    uint16_t budget = 1024U;
    while(rxTail != rxHead && budget--)
    {
        char character = (char)rxRing[rxTail];
        rxTail = (uint16_t)((rxTail + 1U) & WIFI_RX_RING_MASK);

        if(lineDiscarding)
        {
            if(character == '\n') lineDiscarding = 0U;
            continue;
        }

        if(character == '>' && lineLength == 0U &&
           (wifiState == WIFI_STATE_PUBLISH_WAIT_PROMPT ||
            wifiState == WIFI_STATE_LAN_SEND_WAIT_PROMPT ||
            wifiState == WIFI_STATE_WAIT_MQTT_CERT))
        {
            if(wifiState == WIFI_STATE_PUBLISH_WAIT_PROMPT ||
               wifiState == WIFI_STATE_LAN_SEND_WAIT_PROMPT)
                responsePrompt = 1U;
            continue;
        }
        if(character == '\r') continue;
        if(character == '\n')
        {
            if(lineLength > 0U)
            {
                lineBuffer[lineLength] = '\0';
                ParseAtLine(lineBuffer);
                lineLength = 0U;
            }
        }
        else if(lineLength + 1U < WIFI_LINE_SIZE)
        {
            lineBuffer[lineLength++] = character;
        }
        else
        {
            lineLength = 0U;
            lineDiscarding = 1U;
            rxOverflow = 1U;
        }
    }
}

static void MarkOnline(void)
{
    wifiState = WIFI_STATE_ONLINE;
    retryExponent = 0U;
    wifiLastError = WIFI_ERROR_NONE;
    statusPending = 1U;
    metadataPendingMask = (localControlActive || lanClientActive) ? 0U : 0x03U;
    nextPublishMs = WifiNowMs() + 500U;
}

static void StartLanStartupSocketQuery(void)
{
    lanStartupStaleChildId = -1;
    lanStartupServerStatus = -1;
    (void)SendAtCommand(WIFI_STATE_WAIT_LAN_QUERY_STALE, 1500U,
                        "AT+SOCKET?\r\n");
}

static void ContinueAfterLanListenerReady(void)
{
    /* AT+SOCKET starts accepting before the following table verification is
     * complete.  Preserve any Seed/HELLO that arrived in that interval;
     * clearing it here strands the desktop on an open connection which the
     * STM32 has forgotten.  Startup cleanup already reset all owner fields
     * before the listener was created. */
    lanCreatedSocketId = -1;
    lanDeleteTargetId = -1;
    lanDeleteKind = WIFI_LAN_DELETE_SERVER;
    lanStartupStaleChildId = -1;
    lanStartupCleanupAttempts = 0U;
    lanServerDeleteVerifyAttempts = 0U;
    lanServerDeleteVerifyStartup = 0U;
    lanListenerCreateVerifyAttempts = 0U;
    (void)SendAtCommand(WIFI_STATE_WAIT_MQTT_HOST, 1500U,
                        "AT+MQTT=1,%s\r\n", wifiConfig.host);
}

static void StartLanListenerCreate(uint8_t startup)
{
    WifiState createState = startup ? WIFI_STATE_WAIT_LAN_CREATE :
                                      WIFI_STATE_LAN_RECREATE;
    lanCreatedSocketId = WIFI_LAN_SOCKET_ID;
    if(!SendAtCommand(createState, 2500U,
                      "AT+SOCKET=3,%u,0,%d\r\n",
                      WIFI_LAN_SERVER_PORT, WIFI_LAN_SOCKET_ID))
        RestartBackoff(WIFI_ERROR_UART);
}

static void ScheduleLanListenerCreate(uint8_t startup)
{
    /* P1.0.22 can report the old TCPServer absent before its internal seed
     * storage is reusable.  A real quiet interval prevents an immediately
     * recreated status-3 listener from accepting and resetting every client. */
    lanListenerCreateStartup = startup ? 1U : 0U;
    ClearResponseEvents();
    wifiState = WIFI_STATE_WAIT_LAN_CREATE_GRACE;
    stateDeadlineMs = WifiNowMs() + 3000U;
}

static void VerifyCreatedLanListener(void)
{
    lanStartupServerStatus = -1;
    if(!SendAtCommand(WIFI_STATE_WAIT_LAN_VERIFY_CREATED, 1500U,
                      "AT+SOCKET?\r\n"))
        RestartBackoff(WIFI_ERROR_UART);
}

static void ContinueCreatedLanListenerVerification(void)
{
    if(lanStartupServerStatus >= 0)
    {
        ContinueAfterLanListenerReady();
        return;
    }
    if(lanListenerCreateVerifyAttempts >=
       WIFI_LAN_SERVER_DELETE_VERIFY_MAX)
    {
        RestartBackoff(WIFI_ERROR_LAN_SOCKET);
        return;
    }
    lanListenerCreateVerifyAttempts++;
    StartLanListenerCreate(1U);
}

static void StartLanServerGoneVerification(uint8_t startup)
{
    lanServerDeleteVerifyStartup = startup ? 1U : 0U;
    lanServerDeleteVerifyAttempts = 0U;
    lanStartupServerStatus = -1;
    if(!SendAtCommand(WIFI_STATE_WAIT_LAN_VERIFY_SERVER_GONE, 1500U,
                      "AT+SOCKET?\r\n"))
        RestartBackoff(WIFI_ERROR_UART);
}

static void ScheduleLanServerGoneRequery(void)
{
    if(lanServerDeleteVerifyAttempts >=
       WIFI_LAN_SERVER_DELETE_VERIFY_MAX)
    {
        RestartBackoff(WIFI_ERROR_LAN_SOCKET);
        return;
    }
    lanServerDeleteVerifyAttempts++;
    ClearResponseEvents();
    wifiState = WIFI_STATE_WAIT_LAN_SERVER_DELETE_GRACE;
    stateDeadlineMs = WifiNowMs() + WIFI_LAN_STALE_DELETE_GRACE_MS;
}

static void ScheduleLanStartupChildRequery(void)
{
    lanDeleteTargetId = -1;
    ClearResponseEvents();
    wifiState = WIFI_STATE_WAIT_LAN_CHILD_DELETE_GRACE;
    stateDeadlineMs = WifiNowMs() + WIFI_LAN_STALE_DELETE_GRACE_MS;
}

static void ContinueLanServerGoneVerification(void)
{
    if(lanStartupServerStatus < 0)
    {
        ScheduleLanListenerCreate(lanServerDeleteVerifyStartup);
        return;
    }
    ScheduleLanServerGoneRequery();
}

static void ContinueLanStartupSocketCleanup(void)
{
    if(lanStartupStaleChildId >= 0 &&
       lanStartupCleanupAttempts < WIFI_LAN_STARTUP_CLEANUP_MAX)
    {
        lanDeleteTargetId = lanStartupStaleChildId;
        lanStartupStaleChildId = -1;
        lanStartupCleanupAttempts++;
        (void)SendAtCommand(WIFI_STATE_WAIT_LAN_DELETE_CHILD, 1500U,
                            "AT+SOCKETDEL=%d\r\n",
                            (int)lanDeleteTargetId);
        return;
    }

    if(lanStartupStaleChildId >= 0)
    {
        RestartBackoff(WIFI_ERROR_LAN_SOCKET);
        return;
    }

    lanStartupStaleChildId = -1;
    if(lanStartupServerStatus < 0)
    {
        ScheduleLanListenerCreate(1U);
        return;
    }

    /* BW20 P1.0.22 can report a retained TCPServer as healthy status 3 while
     * every accepted child is immediately reset.  Child cleanup and the
     * status value cannot prove the parent reusable, so every MCU generation
     * deletes the retained listener, verifies that it is absent, and creates
     * a new server.  This deliberately favours deterministic recovery over
     * the faster but unsafe retained-listener optimisation. */
    lanDeleteTargetId = WIFI_LAN_SOCKET_ID;
    (void)SendAtCommand(WIFI_STATE_WAIT_LAN_DELETE_STALE, 1500U,
                        "AT+SOCKETDEL=%d\r\n", WIFI_LAN_SOCKET_ID);
}

static void DrainBw20RxFor(uint32_t durationMs)
{
    uint32_t deadline = WifiNowMs() + durationMs;
    do
    {
        DrainRx();
        HAL_Delay(1U);
    } while(!TimeReached(WifiNowMs(), deadline));
    DrainRx();
    ClearResponseEvents();
}

static void ResetMcuAfterUsbDetach(void)
{
    /* SYSRESETREQ does not guarantee a host-visible USB disconnect interval
     * on this board.  Explicitly remove the FS pull-up first so Windows drops
     * the old CDC instance instead of retaining an unusable Code-31 device
     * while the bootloader installs and starts the new application. */
    (void)HAL_PCD_DevDisconnect(&hpcd_USB_OTG_FS);
    HAL_Delay(100U);
    NVIC_SystemReset();
}

static void PublishSuccess(void)
{
    WifiPublishKind completedKind = publishKind;
    uint8_t completedOverLan = publishOverLan;
    WifiState completedResumeState = lanResumeState;

    if(publishKind == WIFI_PUBLISH_TELEMETRY && activeSnapshot >= 0)
    {
        if(appliedFrameMustPublish &&
           snapshots[(uint8_t)activeSnapshot].sequence == appliedFrameSequence)
            appliedFrameMustPublish = 0U;
        snapshots[(uint8_t)activeSnapshot].ready = 0U;
        activeSnapshot = -1;
    }
    else if(publishKind == WIFI_PUBLISH_ACK && ackQueueCount > 0U)
    {
        ackQueueHead = (uint8_t)((ackQueueHead + 1U) % WIFI_ACK_QUEUE_DEPTH);
        ackQueueCount--;
    }
    else if(publishKind == WIFI_PUBLISH_OTA)
    {
        uint8_t reboot = otaRebootAfterResponse;
        otaResponseReady = 0U;
        otaResponseLength = 0U;
        otaRebootAfterResponse = 0U;
        if(reboot)
        {
            /* The authenticated image is already CRC-checked and committed.
             * Give the accepted REBOOT payload and the client's TCP FIN time
             * to leave the module.  Combo-AT does not automatically free a
             * TCPServer seed, so delete the exact OTA child before resetting
             * the STM32; otherwise the independently powered BW20 can retain
             * a seed which resets every client accepted by the recreated
             * listener.  Do not issue AT+RST here: this PCB connects BW20
             * PB21/SWD_DAT to STM32 NRST, and two real cycles left both the
             * radio silent and the MCU at its reset entry. */
            DrainBw20RxFor(WIFI_OTA_RESPONSE_DRAIN_MS);
            if(lanSocketId >= 0)
            {
                otaResetAfterLanDelete = 1U;
                lanDeleteKind = WIFI_LAN_DELETE_OWNER;
                lanDeleteTargetId = lanSocketId;
                lanResumeState = WIFI_STATE_ONLINE;
                lanResumeDeadlineMs = 0U;
                if(SendAtCommand(WIFI_STATE_LAN_DELETE, 1500U,
                                 "AT+SOCKETDEL=%d\r\n",
                                 (int)lanDeleteTargetId)) return;
                otaResetAfterLanDelete = 0U;
            }
            ResetMcuAfterUsbDetach();
        }
    }
    else if(publishKind == WIFI_PUBLISH_LAN_RAW && lanRawActivePriority &&
            lanRawPriorityCount > 0U)
    {
        lanRawPriorityHead = (uint8_t)((lanRawPriorityHead + 1U) %
                                      WIFI_LAN_RAW_PRIORITY_DEPTH);
        lanRawPriorityCount--;
    }
    lanRawActivePriority = 0U;
    activeTelemetryDiscarded = 0U;
    publishKind = WIFI_PUBLISH_NONE;
    publishPayload = NULL;
    publishPayloadLength = 0U;
    publishOverLan = 0U;
    wifiState = completedOverLan ? completedResumeState : WIFI_STATE_ONLINE;
    if(completedOverLan) stateDeadlineMs = lanResumeDeadlineMs;
    /* V4.18_P1.0.9 can reboot when several retained QoS-1 RAW publishes are
     * launched back-to-back.  Space boot/status metadata while keeping live
     * telemetry latency essentially unchanged. */
    if(completedOverLan)
        nextPublishMs = WifiNowMs() +
            ((completedKind == WIFI_PUBLISH_LAN_RAW) ? 1U :
             (completedKind == WIFI_PUBLISH_TELEMETRY) ? 20U : 10U);
    else if(completedKind == WIFI_PUBLISH_STATUS ||
       completedKind == WIFI_PUBLISH_METADATA_STRESS ||
       completedKind == WIFI_PUBLISH_METADATA_TEMPERATURE)
        nextPublishMs = WifiNowMs() + 750U;
    else if(completedKind == WIFI_PUBLISH_ACK)
        nextPublishMs = WifiNowMs() + 250U;
    else if(completedKind == WIFI_PUBLISH_TELEMETRY)
        nextPublishMs = WifiNowMs() +
            ((workState == PRECISION_TABLE_STATE) ? 750U : 100U);
    else
        nextPublishMs = WifiNowMs() + 20U;
    ClearResponseEvents();
}

static void DeferLanTransfer(void)
{
    WifiState resumeState = lanResumeState;
    uint8_t otaWasActive = OtaUpdate_IsActive();

    /* A failed UDP write means the learned peer or socket is stale.  Drop the
     * latest-only frame and relearn a viewer instead of ever leaking this LAN
     * command/ACK into the public MQTT path. */
    SetLanClientActive(0U);
    if(activeSnapshot >= 0)
    {
        snapshots[(uint8_t)activeSnapshot].ready = 0U;
        activeSnapshot = -1;
    }
    activeTelemetryDiscarded = 0U;
    publishKind = WIFI_PUBLISH_NONE;
    publishPayload = NULL;
    publishPayloadLength = 0U;
    publishOverLan = 0U;
    wifiState = resumeState;
    stateDeadlineMs = lanResumeDeadlineMs;
    lanSocketResetPending = 1U;
    if(otaWasActive)
    {
        lanCleanupOverflow = 1U;
        lanCleanupCanPromote = 0U;
    }
    nextLanRetryMs = WifiNowMs() + WIFI_LAN_RETRY_MS;
    wifiLastError = WIFI_ERROR_LAN_SOCKET;
    ClearResponseEvents();
}

static void DeferBusyPublish(void)
{
    if(publishKind == WIFI_PUBLISH_STATUS) statusPending = 1U;
    else if(publishKind == WIFI_PUBLISH_METADATA_STRESS) metadataPendingMask |= 0x01U;
    else if(publishKind == WIFI_PUBLISH_METADATA_TEMPERATURE) metadataPendingMask |= 0x02U;
    else if(publishKind == WIFI_PUBLISH_TELEMETRY && activeSnapshot >= 0)
    {
        uint8_t active = (uint8_t)activeSnapshot;
        if(uartTxBusy)
        {
            snapshots[active].ready = 0U;
            dmaReservedSnapshot = activeSnapshot;
        }
        else if(!activeTelemetryDiscarded && latestSnapshot < 0)
        {
            snapshots[active].ready = 1U;
            latestSnapshot = activeSnapshot;
        }
        else snapshots[active].ready = 0U;
    }

    /* ACK entries are removed only after PublishSuccess(), so the queue head
     * already retains the exact message for a retry. */
    activeSnapshot = -1;
    activeTelemetryDiscarded = 0U;
    publishKind = WIFI_PUBLISH_NONE;
    publishPayload = NULL;
    publishPayloadLength = 0U;
    wifiState = WIFI_STATE_ONLINE;
    nextPublishMs = WifiNowMs() + 1000U;
    ClearResponseEvents();
}

static const char *ModeName(uint8_t mode)
{
    if(mode == TABLE_STATE) return "STRESS";
    if(mode == PRECISION_TABLE_STATE) return "TEMPERATURE";
    return "IDLE";
}

static uint8_t StartNextLanTransfer(WifiState resumeState)
{
    const uint8_t *payload = NULL;
    uint16_t length = 0U;

    if(localControlActive || !lanClientActive || lanSocketId < 0) return 0U;

    lanRawActivePriority = 0U;
    if(lanRawMode)
    {
        if(lanRawPriorityCount > 0U)
        {
            WifiLanRawSmallPacket *slot = &lanRawPriority[lanRawPriorityHead];
            memcpy(lanRawPublish, slot->data, slot->length);
            length = slot->length;
            lanRawActivePriority = 1U;
        }
        else if(lanRawScanReady)
        {
            memcpy(lanRawPublish, lanRawScan, lanRawScanLength);
            length = lanRawScanLength;
            lanRawScanReady = 0U;
        }
        else if(lanRawMonitorReady[0])
        {
            memcpy(lanRawPublish, lanRawMonitor[0].data,
                   lanRawMonitor[0].length);
            length = lanRawMonitor[0].length;
            lanRawMonitorReady[0] = 0U;
        }
        else if(lanRawMonitorReady[1])
        {
            memcpy(lanRawPublish, lanRawMonitor[1].data,
                   lanRawMonitor[1].length);
            length = lanRawMonitor[1].length;
            lanRawMonitorReady[1] = 0U;
        }
        else return 0U;
        payload = lanRawPublish;
        publishKind = WIFI_PUBLISH_LAN_RAW;
    }

    else if(otaResponseReady)
    {
        payload = otaResponse;
        length = otaResponseLength;
        publishKind = WIFI_PUBLISH_OTA;
    }
    else if(ackQueueCount > 0U)
    {
        memcpy(publishSmallPayload, ackQueue[ackQueueHead].payload,
               ackQueue[ackQueueHead].length);
        payload = publishSmallPayload;
        length = ackQueue[ackQueueHead].length;
        publishKind = WIFI_PUBLISH_ACK;
    }
    else if(lanMetadataPendingMask & 0x01U)
    {
        lanMetadataPendingMask &= (uint8_t)~0x01U;
        payload = metadataFrames[0];
        length = metadataLengths[0];
        publishKind = WIFI_PUBLISH_METADATA_STRESS;
    }
    else if(lanMetadataPendingMask & 0x02U)
    {
        lanMetadataPendingMask &= (uint8_t)~0x02U;
        payload = metadataFrames[1];
        length = metadataLengths[1];
        publishKind = WIFI_PUBLISH_METADATA_TEMPERATURE;
    }
    else if(latestSnapshot >= 0)
    {
        activeSnapshot = latestSnapshot;
        activeTelemetryDiscarded = 0U;
        latestSnapshot = -1;
        snapshots[(uint8_t)activeSnapshot].ready = 0U;
        payload = snapshots[(uint8_t)activeSnapshot].data;
        length = snapshots[(uint8_t)activeSnapshot].length;
        publishKind = WIFI_PUBLISH_TELEMETRY;
    }
    else return 0U;

    publishPayload = payload;
    publishPayloadLength = length;
    publishOverLan = 1U;
    lanResumeState = resumeState;
    lanResumeDeadlineMs = stateDeadlineMs;
    if(!SendAtCommand(WIFI_STATE_LAN_SEND_WAIT_PROMPT, 3000U,
                      "AT+SOCKETSEND=%d,%u\r\n",
                      (int)lanSocketId, publishPayloadLength))
    {
        DeferLanTransfer();
        return 0U;
    }
    return 1U;
}

static uint8_t StartNextPublish(void)
{
    const uint8_t *payload = NULL;
    uint16_t length = 0U;
    const char *topic = NULL;
    uint8_t qos = 0U;
    uint8_t retained = 0U;

    /* Local USB debug is completely silent: no LAN frames, MQTT spectra,
     * status packets or application-level heartbeats are uploaded.  The MQTT
     * socket may remain established internally and is revalidated on resume. */
    if(localControlActive) return 0U;

    if(otaResponseReady && lanClientActive)
        return StartNextLanTransfer(WIFI_STATE_ONLINE);

    /* LAN control suppresses public raw spectra.  MQTT remains connected and
     * publishes only a retained LAN ownership status. */
    if(lanClientActive && !statusPending)
    {
        return StartNextLanTransfer(WIFI_STATE_ONLINE);
    }

    /* During a normal remote reconnect, publish both retained wavelength
     * tables before announcing ONLINE.  The remote viewer enables its command
     * buttons from that retained status; making it the final bootstrap packet
     * prevents a command from colliding with BW20 P1.0.22's RAW metadata
     * publishing window.  LOCAL has no metadata and remains immediate. */
    if(statusPending &&
       (lanClientActive || metadataPendingMask == 0U))
    {
        int statusLength;
        if(lanClientActive)
            statusLength = snprintf((char *)publishSmallPayload,
                                    sizeof(publishSmallPayload),
                                    "ONLINE|LAN|%08lX", (unsigned long)bootId);
        else
            statusLength = snprintf((char *)publishSmallPayload,
                                    sizeof(publishSmallPayload),
                                    "ONLINE|%s|%08lX", ModeName(workState),
                                    (unsigned long)bootId);
        if(statusLength <= 0 || statusLength >= (int)sizeof(publishSmallPayload)) return 0U;
        statusPending = 0U;
        payload = publishSmallPayload;
        length = (uint16_t)statusLength;
        topic = statusTopic;
        qos = 0U;
        retained = 1U;
        publishKind = WIFI_PUBLISH_STATUS;
    }
    else if(lanClientActive)
    {
        return StartNextLanTransfer(WIFI_STATE_ONLINE);
    }
    else if(metadataPendingMask & 0x01U)
    {
        metadataPendingMask &= (uint8_t)~0x01U;
        payload = metadataFrames[0];
        length = metadataLengths[0];
        topic = metadataStressTopic;
        qos = 0U;
        retained = 1U;
        publishKind = WIFI_PUBLISH_METADATA_STRESS;
    }
    else if(metadataPendingMask & 0x02U)
    {
        metadataPendingMask &= (uint8_t)~0x02U;
        payload = metadataFrames[1];
        length = metadataLengths[1];
        topic = metadataTemperatureTopic;
        qos = 0U;
        retained = 1U;
        publishKind = WIFI_PUBLISH_METADATA_TEMPERATURE;
    }
    else if(ackQueueCount > 0U)
    {
        memcpy(publishSmallPayload, ackQueue[ackQueueHead].payload,
               ackQueue[ackQueueHead].length);
        payload = publishSmallPayload;
        length = ackQueue[ackQueueHead].length;
        topic = ackTopic;
        qos = 1U;
        retained = 0U;
        publishKind = WIFI_PUBLISH_ACK;
    }
    else if(latestSnapshot >= 0)
    {
        activeSnapshot = latestSnapshot;
        activeTelemetryDiscarded = 0U;
        latestSnapshot = -1;
        snapshots[(uint8_t)activeSnapshot].ready = 0U;
        payload = snapshots[(uint8_t)activeSnapshot].data;
        length = snapshots[(uint8_t)activeSnapshot].length;
        topic = telemetryTopic;
        qos = 0U;
        retained = 0U;
        publishKind = WIFI_PUBLISH_TELEMETRY;
    }
    else return 0U;

    strncpy(publishTopic, topic, sizeof(publishTopic) - 1U);
    publishTopic[sizeof(publishTopic) - 1U] = '\0';
    publishPayload = payload;
    publishPayloadLength = length;
    publishQos = qos;
    publishRetained = retained;
    publishOverLan = 0U;

    if(!SendAtCommand(WIFI_STATE_PUBLISH_WAIT_PROMPT, 3000U,
                      "AT+MQTTPUBRAW=%s,%u,%u,%u\r\n",
                      publishTopic, publishQos, publishRetained,
                      publishPayloadLength))
    {
        RestartBackoff(WIFI_ERROR_UART);
        return 0U;
    }
    return 1U;
}

static uint8_t LanOverlayAllowed(WifiState state)
{
    return (state == WIFI_STATE_WAIT_MQTT_CONNECT ||
            state == WIFI_STATE_WAIT_SUBSCRIBE_SETTLE ||
            state == WIFI_STATE_WAIT_MQTT_RECONNECT ||
            state == WIFI_STATE_ONLINE) ? 1U : 0U;
}

static uint8_t StartLanOverlayIfNeeded(uint32_t now)
{
    WifiState resumeState = wifiState;

    if(!LanOverlayAllowed(resumeState) || uartTxBusy) return 0U;

    if(lanCleanupOverflow && !lanClientActive && lanSocketId < 0 &&
       lanCleanupSocketId < 0 && !lanSocketResetPending)
    {
        /* At least one burst seed could not fit the bounded cleanup scalar.
         * Recycle the listener only after all known owners are gone. */
        lanCleanupOverflow = 0U;
        lanSocketResetPending = 1U;
        nextLanRetryMs = now;
    }

    /* Always drain passive receive buffers, including while COM6 owns the
     * instrument, so harmless discovery heartbeats cannot consume BW20 RAM. */
    if(lanReadPending && lanSocketId >= 0)
    {
        lanReadPending = 0U;
        lanResumeState = resumeState;
        lanResumeDeadlineMs = stateDeadlineMs;
        if(!SendAtCommand(WIFI_STATE_LAN_READ, 1500U,
                          "AT+SOCKETREAD=%d\r\n", (int)lanSocketId))
        {
            wifiState = resumeState;
            wifiLastError = WIFI_ERROR_UART;
        }
        return 1U;
    }

    if(lanSocketResetPending && TimeReached(now, nextLanRetryMs))
    {
        int deleteId = (lanSocketId >= 0) ? (int)lanSocketId
                                          : WIFI_LAN_SOCKET_ID;
        lanDeleteKind = (deleteId != WIFI_LAN_SOCKET_ID) ?
                        WIFI_LAN_DELETE_OWNER : WIFI_LAN_DELETE_SERVER;
        lanDeleteTargetId = (int32_t)deleteId;
        lanResumeState = resumeState;
        lanResumeDeadlineMs = stateDeadlineMs;
        if(!SendAtCommand(WIFI_STATE_LAN_DELETE, 1500U,
                          "AT+SOCKETDEL=%d\r\n", deleteId))
        {
            lanDeleteKind = WIFI_LAN_DELETE_SERVER;
            lanDeleteTargetId = -1;
            wifiState = resumeState;
            nextLanRetryMs = now + WIFI_LAN_RETRY_MS;
        }
        return 1U;
    }

    if(lanCleanupSocketId >= 0 && TimeReached(now, nextLanRetryMs))
    {
        /* Delete a true contender only after a stale selected owner has been
         * reaped.  If owner cleanup is pending, that contender may instead
         * be promoted as the next HELLO-validated candidate. */
        lanDeleteKind = WIFI_LAN_DELETE_REJECTED;
        lanDeleteTargetId = lanCleanupSocketId;
        lanResumeState = resumeState;
        lanResumeDeadlineMs = stateDeadlineMs;
        if(!SendAtCommand(WIFI_STATE_LAN_DELETE, 1500U,
                          "AT+SOCKETDEL=%d\r\n", (int)lanDeleteTargetId))
        {
            lanDeleteKind = WIFI_LAN_DELETE_SERVER;
            lanDeleteTargetId = -1;
            wifiState = resumeState;
            nextLanRetryMs = now + WIFI_LAN_RETRY_MS;
        }
        return 1U;
    }

    if(resumeState != WIFI_STATE_ONLINE && lanClientActive &&
       TimeReached(now, nextPublishMs))
        return StartNextLanTransfer(resumeState);
    return 0U;
}

static void CompleteLanChildDelete(void)
{
    if(lanDeleteKind == WIFI_LAN_DELETE_REJECTED)
    {
        if(lanCleanupSocketId == lanDeleteTargetId)
        {
            lanCleanupSocketId = -1;
            lanCleanupCanPromote = 0U;
        }
    }
    else if(lanDeleteKind == WIFI_LAN_DELETE_OWNER)
    {
        if(lanSocketId == lanDeleteTargetId)
            lanSocketId = -1;
        lanReadPending = 0U;
        lanSocketResetPending = 0U;
        lanClientDeadlineMs = 0U;
        PromoteWaitingLanSocket();
    }

    lanDeleteKind = WIFI_LAN_DELETE_SERVER;
    lanDeleteTargetId = -1;
    ClearResponseEvents();
    if(otaResetAfterLanDelete)
    {
        otaResetAfterLanDelete = 0U;
        DrainBw20RxFor(50U);
        ResetMcuAfterUsbDetach();
        return;
    }
    wifiState = lanResumeState;
    stateDeadlineMs = lanResumeDeadlineMs;
}

static uint8_t IsScanMode(uint8_t mode)
{
    return (mode == TABLE_STATE || mode == PRECISION_TABLE_STATE) ? 1U : 0U;
}

static uint8_t TransferStateBusy(WifiState state)
{
    return (state == WIFI_STATE_PUBLISH_WAIT_PROMPT ||
            state == WIFI_STATE_PUBLISH_WAIT_RESULT ||
            state == WIFI_STATE_LAN_SEND_WAIT_PROMPT ||
            state == WIFI_STATE_LAN_SEND_WAIT_RESULT) ? 1U : 0U;
}

static void ApplyPendingConfigIfSafe(void)
{
    if(!pendingConfigValid || uartTxBusy || TransferStateBusy(wifiState)) return;

    wifiConfig = pendingConfig;
    wifiConfigCrc = pendingConfigCrc;
    wifiConfigValid = 1U;
    flashPersistPending = pendingConfigPersist;
    pendingConfigValid = 0U;
    BuildTopics();
    DiscardQueuedTelemetry();
    metadataPendingMask = 0x03U;
    statusPending = 1U;
    restartRequested = 1U;
    wifiLastError = WIFI_ERROR_NONE;
    /* A persisted configuration is acknowledged as successful only after the
     * reserved sector-1 write and read-back verification complete. */
    if(!flashPersistPending) SendUsbConfigAck(0U);
}

void WifiTransport_Init(void)
{
    uint32_t uid0 = *(const uint32_t *)0x1FFF7A10UL;
    uint32_t uid1 = *(const uint32_t *)0x1FFF7A14UL;
    uint32_t uid2 = *(const uint32_t *)0x1FFF7A18UL;
    uint32_t otaInstallToken =
        ((const OtaMetadata *)OTA_METADATA_ADDRESS)->reserved[0];

    memset(&wifiConfig, 0, sizeof(wifiConfig));
    memset(&pendingConfig, 0, sizeof(pendingConfig));
    memset(snapshots, 0, sizeof(snapshots));
    memset(lanRawPriority, 0, sizeof(lanRawPriority));
    memset(lanRawMonitor, 0, sizeof(lanRawMonitor));
    ResetLanRawState();

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    clockCyclesPerMs = SystemCoreClock / 1000U;
    if(clockCyclesPerMs == 0U) clockCyclesPerMs = 1U;
    clockLastCycles = DWT->CYCCNT;

    bootId = uid0 ^ uid1 ^ uid2 ^ SysTick->VAL ^ DWT->CYCCNT;
    if(otaInstallToken != 0U && otaInstallToken != 0xFFFFFFFFUL)
        bootId = otaInstallToken;
    if(bootId == 0U) bootId = 1U;
    OtaUpdate_Init(bootId);
    BuildMetadataFrame(0U, TABLE_STATE, 0U, WIFI_STRESS_POINT_COUNT);
    BuildMetadataFrame(1U, PRECISION_TABLE_STATE,
                       WIFI_TEMPERATURE_START_ROW,
                       WIFI_TEMPERATURE_POINT_COUNT);

    wifiConfigValid = StoredConfigLoad();
    if(wifiConfigValid)
    {
        BuildTopics();
        if(wifiConfig.flags & WIFI_CONFIG_FLAG_ENABLE)
        {
            wifiState = WIFI_STATE_BACKOFF;
            stateDeadlineMs = WifiNowMs() + 1500U;
        }
    }
}

uint32_t WifiTransport_GetBootId(void)
{
    return bootId;
}

uint32_t WifiTransport_GetTableCrc(uint8_t mode)
{
    if(mode == TABLE_STATE) return stressTableCrc;
    if(mode == PRECISION_TABLE_STATE) return temperatureTableCrc;
    return 0U;
}

void WifiTransport_OnUartRx(const uint8_t *data, uint16_t length)
{
    uartRxByteCount = (uint16_t)(uartRxByteCount + length);
    for(uint16_t index = 0U; index < length; index++)
    {
        uint16_t next = (uint16_t)((rxHead + 1U) & WIFI_RX_RING_MASK);
        if(next == rxTail)
        {
            rxOverflow = 1U;
            break;
        }
        rxRing[rxHead] = data[index];
        rxHead = next;
    }
}

void WifiTransport_OnUartTxComplete(void)
{
    uartTxCompleteCount++;
    uartTxBusy = 0U;
    dmaReservedSnapshot = -1;
}

void WifiTransport_OnUartError(uint32_t error_code)
{
    uartLastError = error_code;
    uartErrorPending = 1U;
    uartTxBusy = 0U;
    dmaReservedSnapshot = -1;
}

uint8_t WifiTransport_HandleUsbStatusQuery(const uint8_t *frame, uint16_t length)
{
    if(length != WIFI_USB_CONFIG_COMMAND_SIZE || frame[0] != 0xFFU ||
       frame[1] != 0xFFU || frame[2] != 0x01U || frame[3] != 0x21U) return 0U;
    SendUsbStatusAck();
    return 1U;
}

uint8_t WifiTransport_HandleUsbConfig(const uint8_t *frame, uint16_t length)
{
    static const uint8_t maximumLengths[8] = {
        WIFI_SSID_MAX, WIFI_WIFI_PASSWORD_MAX, WIFI_HOST_MAX,
        WIFI_CLIENT_ID_MAX, WIFI_USERNAME_MAX, WIFI_MQTT_PASSWORD_MAX,
        WIFI_TOPIC_PREFIX_MAX, WIFI_DEVICE_ID_MAX
    };
    char *destinations[8];
    uint16_t offset = 18U;
    uint32_t expectedCrc;
    uint32_t actualCrc;

    if(length != WIFI_USB_CONFIG_COMMAND_SIZE || frame[0] != 0xFFU ||
       frame[1] != 0xFFU || frame[2] != 0x01U || frame[3] != 0x20U) return 0U;

    if(frame[4] != 1U || frame[17] != 0U)
    {
        wifiLastError = WIFI_ERROR_CONFIG_HEADER;
        SendUsbConfigAck(1U);
        return 1U;
    }
    if((frame[5] & (uint8_t)~(WIFI_CONFIG_FLAG_ENABLE |
                              WIFI_CONFIG_FLAG_PERSIST)) != 0U)
    {
        wifiLastError = WIFI_ERROR_CONFIG_HEADER;
        SendUsbConfigAck(1U);
        return 1U;
    }
    expectedCrc = GetU32Be(&frame[804]);
    actualCrc = WifiCrc32(frame, 804U);
    if(expectedCrc != actualCrc)
    {
        wifiLastError = WIFI_ERROR_CONFIG_CRC;
        SendUsbConfigAck(2U);
        return 1U;
    }

    memset(&pendingConfig, 0, sizeof(pendingConfig));
    pendingConfig.flags = frame[5] & (WIFI_CONFIG_FLAG_ENABLE | WIFI_CONFIG_FLAG_PERSIST);
    pendingConfig.port = ((uint16_t)frame[14] << 8) | frame[15];
    pendingConfig.scheme = frame[16];
    destinations[0] = pendingConfig.ssid;
    destinations[1] = pendingConfig.wifi_password;
    destinations[2] = pendingConfig.host;
    destinations[3] = pendingConfig.client_id;
    destinations[4] = pendingConfig.username;
    destinations[5] = pendingConfig.mqtt_password;
    destinations[6] = pendingConfig.topic_prefix;
    destinations[7] = pendingConfig.device_id;

    for(uint8_t field = 0U; field < 8U; field++)
    {
        uint16_t fieldLength = frame[6U + field];
        if(fieldLength > maximumLengths[field] || offset + fieldLength > 804U)
        {
            wifiLastError = WIFI_ERROR_CONFIG_FIELD;
            SendUsbConfigAck(3U);
            return 1U;
        }
        if(memchr(&frame[offset], 0, fieldLength) != NULL)
        {
            wifiLastError = WIFI_ERROR_CONFIG_FIELD;
            SendUsbConfigAck(3U);
            return 1U;
        }
        memcpy(destinations[field], &frame[offset], fieldLength);
        destinations[field][fieldLength] = '\0';
        offset += fieldLength;
    }
    for(uint16_t index = offset; index < 804U; index++)
    {
        if(frame[index] != 0U)
        {
            wifiLastError = WIFI_ERROR_CONFIG_FIELD;
            SendUsbConfigAck(3U);
            return 1U;
        }
    }
    while(strlen(pendingConfig.topic_prefix) > 1U &&
          pendingConfig.topic_prefix[strlen(pendingConfig.topic_prefix) - 1U] == '/')
    {
        pendingConfig.topic_prefix[strlen(pendingConfig.topic_prefix) - 1U] = '\0';
    }
    if(!ConfigIsValid(&pendingConfig))
    {
        wifiLastError = WIFI_ERROR_CONFIG_FIELD;
        SendUsbConfigAck(3U);
        return 1U;
    }

    pendingConfigCrc = expectedCrc;
    pendingConfigPersist = (pendingConfig.flags & WIFI_CONFIG_FLAG_PERSIST) ? 1U : 0U;
    pendingConfigValid = 1U;
    return 1U;
}

uint8_t WifiTransport_QueueRawFrame(const uint8_t *data, uint16_t length)
{
    uint32_t sequence;
    uint16_t packetLength;
    if(data == NULL || length == 0U || length > WIFI_LAN_RAW_MAX_PAYLOAD ||
       localControlActive || !lanClientActive || !lanRawMode ||
       OtaUpdate_IsActive()) return 0U;

    sequence = lanRawSequence++;
    if(length >= 4U && length <= USART_TX_SIZE &&
       data[0] == 0xFFU && data[1] == 0xFFU &&
       (data[3] == 0x01U || data[3] == 0x02U))
    {
        uint8_t monitor = (data[3] == 0x01U) ? 0U : 1U;
        packetLength = BuildLanRawPacket(lanRawMonitor[monitor].data,
                sizeof(lanRawMonitor[monitor].data), WIFI_LAN_RAW_KIND_DATA,
                sequence, data, length);
        if(packetLength == 0U) return 0U;
        lanRawMonitor[monitor].length = packetLength;
        lanRawMonitorReady[monitor] = 1U;
    }
    else if(length <= USART_TX_SIZE)
    {
        if(!QueueLanRawPriority(WIFI_LAN_RAW_KIND_DATA,
                                sequence, data, length)) return 0U;
    }
    else
    {
        /* The scan slot is a single latest-value mailbox for ordinary live
         * spectra, but temporary-route frames are lossless and sequence
         * checked by the host.  Report back-pressure instead of overwriting
         * an unsent large RAW frame; runFastFlankStream() then services Wi-Fi
         * and retries this exact frame before advancing the route sequence. */
        if(lanRawScanReady) return 0U;
        packetLength = BuildLanRawPacket(lanRawScan, sizeof(lanRawScan),
                WIFI_LAN_RAW_KIND_DATA, sequence, data, length);
        if(packetLength == 0U) return 0U;
        lanRawScanLength = packetLength;
        lanRawScanReady = 1U;
    }
    nextPublishMs = WifiNowMs();
    /* A long temporary route can legitimately exceed the HELLO lease while
     * this single-threaded MCU is continuously producing and successfully
     * queuing RAW frames.  Outbound progress proves the selected TCP owner is
     * still usable; refresh its lease so the terminal frame can be followed
     * by the host's explicit disarm/shutter confirmation.  A real disconnect
     * is still detected by the SEND failure path and fails dark. */
    lanClientDeadlineMs = nextPublishMs + WIFI_LAN_CLIENT_TIMEOUT_MS;
    return 1U;
}

uint8_t WifiTransport_IsLanRawClientActive(void)
{
    return (!localControlActive && lanClientActive && lanRawMode &&
            lanSocketId >= 0 && !OtaUpdate_IsActive()) ? 1U : 0U;
}

uint8_t WifiTransport_IsLanRawQueueIdle(void)
{
    return (lanRawPriorityCount == 0U && !lanRawScanReady &&
            lanRawF45BatchCount == 0U &&
            publishKind != WIFI_PUBLISH_LAN_RAW) ? 1U : 0U;
}

static uint8_t CommitLanRawF45Batch(void)
{
    uint16_t payload = 12U;
    uint16_t payloadLength;
    uint32_t sequence;

    if(lanRawF45BatchCount == 0U) return 1U;
    if(lanRawScanReady || !WifiTransport_IsLanRawClientActive()) return 0U;
    payloadLength = (uint16_t)(2U +
            (uint16_t)lanRawF45BatchCount * WIFI_LAN_RAW_COMPACT_F45_SIZE);
    sequence = lanRawSequence++;
    lanRawScan[0] = 'F'; lanRawScan[1] = 'B';
    lanRawScan[2] = 'R'; lanRawScan[3] = '1';
    lanRawScan[4] = 1U;
    lanRawScan[5] = WIFI_LAN_RAW_KIND_COMPACT_F45_BATCH;
    PutU32Be(&lanRawScan[6], sequence);
    PutU16Be(&lanRawScan[10], payloadLength);
    lanRawScan[payload++] = WIFI_LAN_RAW_COMPACT_VERSION;
    lanRawScan[payload++] = lanRawF45BatchCount;
    memcpy(&lanRawScan[payload], lanRawF45Batch,
           (uint16_t)lanRawF45BatchCount * WIFI_LAN_RAW_COMPACT_F45_SIZE);
    payload = (uint16_t)(payload +
            (uint16_t)lanRawF45BatchCount * WIFI_LAN_RAW_COMPACT_F45_SIZE);
    PutU32Be(&lanRawScan[payload], WifiCrc32(lanRawScan, payload));
    payload += 4U;
    lanRawScanLength = payload;
    lanRawScanReady = 1U;
    lanRawF45BatchCount = 0U;
    nextPublishMs = WifiNowMs();
    return 1U;
}

uint8_t WifiTransport_FlushFastFullMapFrames(void)
{
    return CommitLanRawF45Batch();
}

uint8_t WifiTransport_QueueFastFullMapFrame(const uint8_t *data,
                                            uint16_t length)
{
    uint16_t payload;

    if(data == NULL || length != WIFI_LAN_RAW_NATIVE_F45_SIZE ||
       data[0] != 0xD9U || data[1] != 0x9DU || data[2] != 0x03U ||
       !WifiTransport_IsLanRawClientActive()) return 0U;

    /* Older clients receive the original frame.  The current desktop
     * explicitly negotiates CS1 before it is marked connected. */
    if(!lanRawCompactScan) return WifiTransport_QueueRawFrame(data, length);
    if(lanRawF45BatchCount >= WIFI_LAN_RAW_F45_BATCH_MAX) return 0U;
    if(lanRawF45BatchCount + 1U == WIFI_LAN_RAW_F45_BATCH_MAX &&
       lanRawScanReady) return 0U;

    for(uint8_t row = 0U; row < WIFI_LAN_RAW_NATIVE_F45_ROWS; row++)
        if(data[40U + row] != row) return 0U;

    payload = (uint16_t)lanRawF45BatchCount * WIFI_LAN_RAW_COMPACT_F45_SIZE;
    lanRawF45Batch[payload++] = WIFI_LAN_RAW_COMPACT_VERSION;
    lanRawF45Batch[payload++] = data[3];
    memcpy(&lanRawF45Batch[payload], &data[4], 36U);
    payload += 36U;
    for(uint8_t row = 0U; row < WIFI_LAN_RAW_NATIVE_F45_ROWS; row++)
    {
        uint16_t source = (uint16_t)(85U + (uint16_t)row * 8U);
        lanRawF45Batch[payload++] = data[source];
        lanRawF45Batch[payload++] = data[source + 1U];
    }
    lanRawF45BatchCount++;
    if(lanRawF45BatchCount == WIFI_LAN_RAW_F45_BATCH_MAX)
        return CommitLanRawF45Batch();
    return 1U;
}

uint8_t WifiTransport_QueueRawScanFrame(const uint8_t *data,
                                        uint16_t length,
                                        uint16_t point_count,
                                        uint8_t active_channel_mask)
{
    uint8_t channelCount = 0U;
    uint32_t nativeSampleEnd;
    uint16_t trailerLength;
    uint32_t compactPayloadLength;
    uint16_t position;
    uint32_t sequence;

    /* CS1 is strictly negotiated.  Any legacy/malformed/non-beneficial case
     * uses the existing native frame path without changing its sequence or
     * queue semantics. */
    if(!lanRawCompactScan || data == NULL || point_count == 0U ||
       (active_channel_mask & 0xF0U) != 0U || active_channel_mask == 0U)
        return WifiTransport_QueueRawFrame(data, length);

    nativeSampleEnd = 4UL + (uint32_t)point_count * 8UL;
    if(length > WIFI_LAN_RAW_MAX_PAYLOAD || nativeSampleEnd >= length ||
       data[0] != 0xEEU || data[1] != 0xEEU ||
       (((uint16_t)data[2] << 8) | data[3]) != point_count ||
       data[nativeSampleEnd] != 0xABU ||
       length < 2U || data[length - 2U] != 0xFFU ||
       data[length - 1U] != 0xEFU)
        return WifiTransport_QueueRawFrame(data, length);

    for(uint8_t channel = 0U; channel < 4U; channel++)
        if((active_channel_mask & (uint8_t)(1U << channel)) != 0U)
            channelCount++;

    trailerLength = (uint16_t)(length - nativeSampleEnd);
    compactPayloadLength = WIFI_LAN_RAW_COMPACT_HEADER_SIZE +
            (uint32_t)point_count * (uint32_t)channelCount * 2UL +
            trailerLength;
    if(channelCount >= 4U || compactPayloadLength >= length ||
       compactPayloadLength > WIFI_LAN_RAW_MAX_PAYLOAD ||
       compactPayloadLength + WIFI_LAN_RAW_PACKET_OVERHEAD >
           sizeof(lanRawScan) ||
       localControlActive || !lanClientActive || !lanRawMode ||
       OtaUpdate_IsActive())
        return WifiTransport_QueueRawFrame(data, length);

    sequence = lanRawSequence++;
    lanRawScan[0] = 'F'; lanRawScan[1] = 'B';
    lanRawScan[2] = 'R'; lanRawScan[3] = '1';
    lanRawScan[4] = 1U;
    lanRawScan[5] = WIFI_LAN_RAW_KIND_COMPACT_SCAN;
    PutU32Be(&lanRawScan[6], sequence);
    PutU16Be(&lanRawScan[10], (uint16_t)compactPayloadLength);
    position = 12U;
    lanRawScan[position++] = WIFI_LAN_RAW_COMPACT_VERSION;
    lanRawScan[position++] = active_channel_mask;
    PutU16Be(&lanRawScan[position], point_count);
    position += 2U;
    PutU16Be(&lanRawScan[position], trailerLength);
    position += 2U;

    for(uint16_t point = 0U; point < point_count; point++)
    {
        uint32_t pointOffset = 4UL + (uint32_t)point * 8UL;
        for(uint8_t channel = 0U; channel < 4U; channel++)
        {
            if((active_channel_mask & (uint8_t)(1U << channel)) == 0U)
                continue;
            lanRawScan[position++] = data[pointOffset + channel * 2U];
            lanRawScan[position++] = data[pointOffset + channel * 2U + 1U];
        }
    }
    memcpy(&lanRawScan[position], &data[nativeSampleEnd], trailerLength);
    position = (uint16_t)(position + trailerLength);
    PutU32Be(&lanRawScan[position], WifiCrc32(lanRawScan, position));
    position += 4U;
    lanRawScanLength = position;
    lanRawScanReady = 1U;
    nextPublishMs = WifiNowMs();
    return 1U;
}

void WifiTransport_QueueScanFrame(const uint8_t *usb_scan_frame,
                                  uint16_t point_count,
                                  uint8_t mode,
                                  int32_t temperature_mC,
                                  uint16_t gain20k_mask_ch0,
                                  uint16_t gain20k_mask_ch1,
                                  uint8_t active_channel_mask)
{
    uint8_t channelMask;
    uint8_t channels;
    uint16_t payloadLength;
    uint16_t frameLength;
    uint8_t slotIndex;
    int8_t reservedSnapshot;
    WifiSnapshot *slot;
    uint8_t *frame;
    uint16_t position = WIFI_TELEMETRY_HEADER_SIZE;
    uint32_t tableCrc;
    uint8_t flags = 0U;
    uint32_t frameSequence;
    CH224Q_Status pd;
    uint8_t pdFlags;

    if(localControlActive || OtaUpdate_IsActive() || lanRawMode) return;

    if(mode == TABLE_STATE && point_count == WIFI_STRESS_POINT_COUNT)
    {
        channelMask = active_channel_mask & 0x0FU;
        /* Protocol v2 requires at least one selected channel.  The all-zero
         * discovery result is kept as a four-channel zero frame so an older
         * viewer remains online and can report that no waveform was found. */
        if(channelMask == 0U) channelMask = 0x0FU;
        channels = 0U;
        for(uint8_t channel = 0U; channel < 4U; channel++)
            if(channelMask & (uint8_t)(1U << channel)) channels++;
        tableCrc = stressTableCrc;
    }
    else if(mode == PRECISION_TABLE_STATE && point_count == WIFI_TEMPERATURE_POINT_COUNT)
    {
        channelMask = 0x03U;
        channels = 2U;
        tableCrc = temperatureTableCrc;
    }
    else
    {
        wifiLastError = WIFI_ERROR_FRAME_FORMAT;
        return;
    }
    payloadLength = point_count * channels * 2U;
    frameLength = WIFI_TELEMETRY_HEADER_SIZE + payloadLength + 4U;
    if(frameLength > WIFI_TELEMETRY_MAX_SIZE) return;

    /* Read the ISR-owned release flag once.  Treating a just-released slot as
     * reserved for one extra frame is harmless; observing it inconsistently
     * across the selection branches is not. */
    reservedSnapshot = dmaReservedSnapshot;
    if(latestSnapshot >= 0 && appliedFrameMustPublish &&
       snapshots[(uint8_t)latestSnapshot].sequence == appliedFrameSequence)
    {
        /* Keep the exact frame named by APPLIED until it has actually left
         * the module.  Newer scans are still produced locally but are dropped
         * here rather than making the acknowledgement refer to an unseen seq. */
        queueOverrunLatched = 1U;
        return;
    }
    if(latestSnapshot >= 0 && latestSnapshot != reservedSnapshot)
    {
        slotIndex = (uint8_t)latestSnapshot;
        queueOverrunLatched = 1U;
    }
    else
    {
        if(latestSnapshot >= 0)
        {
            /* A DMA-reserved slot is never also a writable retry slot. */
            snapshots[(uint8_t)latestSnapshot].ready = 0U;
            latestSnapshot = -1;
            queueOverrunLatched = 1U;
        }
        if(activeSnapshot != 0 && reservedSnapshot != 0) slotIndex = 0U;
        else if(activeSnapshot != 1 && reservedSnapshot != 1) slotIndex = 1U;
        else
        {
            /* Both fixed snapshots are momentarily owned by DMA/publishing.
             * The scanner remains non-blocking and drops only this frame. */
            queueOverrunLatched = 1U;
            return;
        }
    }
    slot = &snapshots[slotIndex];
    frame = slot->data;

    if(wifiConfigValid && (wifiConfig.flags & WIFI_CONFIG_FLAG_ENABLE)) flags |= 0x01U;
    if(wifiConnected) flags |= 0x02U;
    if(mqttConnected) flags |= 0x04U;
    if(queueOverrunLatched) flags |= 0x08U;
    if(gain20k_mask_ch0 || gain20k_mask_ch1) flags |= 0x10U;
    if(metadataPendingMask) flags |= 0x20U;
    if(lanClientActive) flags |= 0x40U;

    frame[0] = 'F'; frame[1] = 'B'; frame[2] = 'G'; frame[3] = '1';
    frame[4] = 2U;
    frame[5] = WIFI_PACKET_TYPE_TELEMETRY;
    frame[6] = mode;
    frame[7] = flags;
    PutU16Be(&frame[8], WIFI_TELEMETRY_HEADER_SIZE);
    PutU16Be(&frame[10], payloadLength);
    PutU16Be(&frame[12], point_count);
    frame[14] = channelMask;
    frame[15] = WIFI_SAMPLE_FORMAT_U16_BE;
    PutU32Be(&frame[16], bootId);
    frameSequence = telemetrySequence++;
    PutU32Be(&frame[20], frameSequence);
    PutU32Be(&frame[24], WifiNowMs());
    PutU32Be(&frame[28], (uint32_t)temperature_mC);
    PutU32Be(&frame[32], tableCrc);
    PutU16Be(&frame[36], gain20k_mask_ch0);
    PutU16Be(&frame[38], gain20k_mask_ch1);
    pd = CH224Q_GetStatus();
    pdFlags = (pd.online ? 0x01U : 0U)
            | (pd.current_valid ? 0x02U : 0U)
            | ((pd.i2c_address == 0x23U) ? 0x04U : 0U);
    PutU16Be(&frame[40], pd.requested_voltage_mV);
    PutU16Be(&frame[42], pd.available_current_mA);
    PutU32Be(&frame[44], pd.power_limit_mW);
    PutU16Be(&frame[48], ThermalControl_GetFanRpm());
    PutU16Be(&frame[50], ThermalControl_GetFanDutyPermille());
    PutU16Be(&frame[52], ThermalControl_GetStatusFlags());
    frame[54] = pdFlags;
    frame[55] = pd.protocol_status;

    for(uint16_t point = 0U; point < point_count; point++)
    {
        const uint8_t *source = &usb_scan_frame[4U + point * 8U];
        uint8_t channelLimit = (mode == PRECISION_TABLE_STATE) ? 2U : 4U;
        for(uint8_t channel = 0U; channel < channelLimit; channel++)
        {
            if((channelMask & (uint8_t)(1U << channel)) == 0U) continue;
            frame[position++] = source[channel * 2U];
            frame[position++] = source[channel * 2U + 1U];
        }
    }
    PutU32Be(&frame[position], WifiCrc32(frame, position));
    position += 4U;
    slot->length = position;
    slot->mode = mode;
    slot->sequence = frameSequence;
    slot->ready = 1U;
    latestSnapshot = (int8_t)slotIndex;
    queueOverrunLatched = 0U;

    if(modeAckWaitFirstFrame && mode == modeAckTargetMode)
    {
        if(QueueModeAck(modeAckRequestId, "APPLIED", mode, frameSequence))
        {
            lastAppliedRequestId = modeAckRequestId;
            lastAppliedMode = mode;
            lastAppliedSequence = frameSequence;
            appliedFrameMustPublish = 1U;
            appliedFrameSequence = frameSequence;
            modeAckWaitFirstFrame = 0U;
        }
    }
}

void WifiTransport_NotifyFrameBoundary(void)
{
    if(flashPersistPending) flashFrameBoundaryPermit = 1U;
    if(usbConfigAckPending) usbConfigAckBoundaryPermit = 1U;
}

void WifiTransport_SetLocalControlActive(uint8_t active)
{
    if(OtaUpdate_IsActive()) return;
    active = active ? 1U : 0U;
    if(active == localControlActive) return;

    if(active)
    {
        if(lanClientActive) SetLanClientActive(0U);
        /* Opening local USB is also an explicit opportunity to reap a child
         * whose disconnect URC was lost before it could claim LAN ownership. */
        if(lanSocketId >= 0)
        {
            lanSocketResetPending = 1U;
            nextLanRetryMs = WifiNowMs();
        }
    }
    localControlActive = active;
    DiscardQueuedTelemetry();

    /* Cancel any remote transaction that has not completed.  Local mode does
     * not upload a terminal ACK or ownership status. */
    ResetModeTransactions();

    if(localControlActive)
    {
        metadataPendingMask = 0U;
        lanMetadataPendingMask = 0U;
        nextLocalHeartbeatMs = 0U;
        statusPending = 0U;
    }
    else
    {
        metadataPendingMask = 0x03U;
        nextLocalHeartbeatMs = 0U;
        statusPending = 1U;
    }
    nextPublishMs = WifiNowMs() + 20U;
}

uint8_t WifiTransport_TakePendingMode(uint8_t current_mode,
                                      uint8_t at_frame_boundary,
                                      uint8_t *requested_mode,
                                      uint32_t *request_id)
{
    if(localControlActive) return 0U;
    if(!pendingModeValid) return 0U;
    if(IsScanMode(current_mode) && !at_frame_boundary) return 0U;
    *requested_mode = pendingMode;
    *request_id = pendingModeRequestId;
    pendingModeValid = 0U;
    return 1U;
}

void WifiTransport_OnRemoteModeApplied(uint8_t mode, uint32_t request_id)
{
    DiscardQueuedTelemetry();
    modeAckWaitFirstFrame = 1U;
    modeAckTargetMode = mode;
    modeAckRequestId = request_id;
    statusPending = 1U;
}

void WifiTransport_OnLocalModeChanged(uint8_t mode)
{
    (void)mode;
    if(pendingModeValid)
        QueueModeAck(pendingModeRequestId, "REJECTED", pendingMode, 0U);
    if(modeAckWaitFirstFrame)
        QueueModeAck(modeAckRequestId, "REJECTED", modeAckTargetMode, 0U);
    DiscardQueuedTelemetry();
    pendingModeValid = 0U;
    modeAckWaitFirstFrame = 0U;
    lastAppliedRequestId = 0U;
    lastAppliedMode = 0xFFU;
    lastAppliedSequence = 0U;
    statusPending = 1U;
}

uint8_t WifiTransport_GetState(void)
{
    return ExternalState();
}

uint16_t WifiTransport_GetLastError(void)
{
    return wifiLastError;
}

uint8_t WifiTransport_IsOtaActive(void)
{
    return OtaUpdate_IsActive();
}

void WifiTransport_Process(void)
{
    uint32_t now = WifiNowMs();
    uint8_t timedOut;

    /* A lost UART byte invalidates both the current AT response and any raw
     * command line.  Flush before parsing so a suffix cannot be mistaken for
     * a complete MQTT/LAN/OTA command. */
    if(rxOverflow)
    {
        ResetRxParser();
        RestartBackoff(WIFI_ERROR_RX_OVERFLOW);
        return;
    }
    if(uartErrorPending)
    {
        uartErrorPending = 0U;
        (void)uartLastError;
        ResetRxParser();
        RestartBackoff(WIFI_ERROR_UART);
        return;
    }
    DrainRx();
    if(lanSocketId >= 0 && !lanSocketResetPending &&
       lanClientDeadlineMs != 0U &&
       TimeReached(now, lanClientDeadlineMs))
    {
        /* Apply the same lease to an active owner and to a newly promoted
         * candidate, including during a resumable OTA session.  Every valid
         * BEGIN/DATA command refreshes the lease via SetLanClientActive(); if
         * BW20 loses the Disconnect URC, suppressing this timeout while OTA
         * is active would strand the old ConID forever and delete every
         * reconnect seed as a contender. */
        RejectSelectedLanSocket();
    }
    if(rxOverflow)
    {
        wifiLastError = WIFI_ERROR_RX_OVERFLOW;
        return;
    }

    ApplyPendingConfigIfSafe();
    if(flashPersistPending && dma_transfer_complete && !uartTxBusy &&
       !TransferStateBusy(wifiState) &&
       (!IsScanMode(workState) || flashFrameBoundaryPermit))
    {
        uint8_t restoreLaserSource =
            (PI11210_GetStatus().soaMode == PI11210_SOA_SOURCE) ? 1U : 0U;
        flashPersistPending = 0U;
        flashFrameBoundaryPermit = 0U;
        /* Sector erase/program stalls this single-bank MCU.  Gate optical
         * output around the blocking operation, then restore the verified
         * source state if it was active before provisioning. */
        (void)PI11210_SetSOAShutter(1U);
        if(!StoredConfigSave()) SendUsbConfigAck(4U);
        else SendUsbConfigAck(0U);
        if(restoreLaserSource) (void)PI11210_SetSOAShutter(0U);
        usbConfigAckBoundaryPermit = 1U;
    }
    FlushUsbConfigAckIfSafe();
    if(restartRequested && !uartTxBusy && !TransferStateBusy(wifiState))
    {
        restartRequested = 0U;
        retryExponent = 0U;
        RestartBackoff(WIFI_ERROR_NONE);
        wifiState = (wifiConfigValid && (wifiConfig.flags & WIFI_CONFIG_FLAG_ENABLE))
                  ? WIFI_STATE_BACKOFF : WIFI_STATE_DISABLED;
        stateDeadlineMs = now + 500U;
    }

    if(wifiState == WIFI_STATE_DISABLED) return;
    if((wifiState >= WIFI_STATE_WAIT_MQTT_HOST) &&
       (wifiState <= WIFI_STATE_WAIT_MQTT_RECONNECT) && !wifiConnected)
    {
        RestartBackoff(WIFI_ERROR_WIFI_TIMEOUT);
        return;
    }
    if(wifiState == WIFI_STATE_ONLINE && !mqttConnected)
    {
        /* Combo-AT V4.18 has its own MQTT reconnect and subscription restore.
         * Preserve a healthy Wi-Fi association and give that path time to
         * work instead of immediately tearing down/rejoining the AP. */
        ClearResponseEvents();
        wifiState = WIFI_STATE_WAIT_MQTT_RECONNECT;
        stateDeadlineMs = now + 30000U;
        return;
    }
    if((wifiState == WIFI_STATE_PUBLISH_WAIT_PROMPT ||
        wifiState == WIFI_STATE_PUBLISH_WAIT_RESULT) && !mqttConnected)
    {
        RestartBackoff(WIFI_ERROR_MQTT_TIMEOUT);
        return;
    }

    timedOut = (stateDeadlineMs != 0U && TimeReached(now, stateDeadlineMs));
    if(responsePublishBusy &&
       (wifiState == WIFI_STATE_PUBLISH_WAIT_PROMPT ||
        wifiState == WIFI_STATE_PUBLISH_WAIT_RESULT))
    {
        DeferBusyPublish();
        return;
    }
    if(responseError)
    {
        if(wifiState == WIFI_STATE_WAIT_WARM_WIFI_QUERY && !uartTxBusy)
        {
            /* Older/variant Combo-AT builds may not implement WJAP?.  The
             * optimisation is optional.  Preserve the original MQTT cleanup,
             * then return to the configured join sequence. */
            responseError = 0U;
            wifiConnected = 0U;
            warmWifiStatusSeen = 0U;
            warmWifiMatchesConfig = 0U;
            SendAtCommand(WIFI_STATE_WAIT_MQTT_DISCONNECT, 3000U,
                          "AT+MQTTDISCONN\r\n");
            return;
        }
        if(wifiState == WIFI_STATE_WAIT_LAN_RECV_CFG && !uartTxBusy)
        {
            responseError = 0U;
            StartLanStartupSocketQuery();
            return;
        }
        if(wifiState == WIFI_STATE_WAIT_LAN_QUERY_STALE && !uartTxBusy)
        {
            responseError = 0U;
            ContinueLanStartupSocketCleanup();
            return;
        }
        if(wifiState == WIFI_STATE_WAIT_LAN_DELETE_CHILD && !uartTxBusy)
        {
            responseError = 0U;
            ScheduleLanStartupChildRequery();
            return;
        }
        if(wifiState == WIFI_STATE_WAIT_LAN_DELETE_STALE && !uartTxBusy)
        {
            responseError = 0U;
            StartLanServerGoneVerification(1U);
            return;
        }
        if(wifiState == WIFI_STATE_WAIT_LAN_VERIFY_SERVER_GONE && !uartTxBusy)
        {
            responseError = 0U;
            ScheduleLanServerGoneRequery();
            return;
        }
        if(wifiState == WIFI_STATE_WAIT_LAN_VERIFY_CREATED && !uartTxBusy)
        {
            responseError = 0U;
            ContinueCreatedLanListenerVerification();
            return;
        }
        if(wifiState == WIFI_STATE_WAIT_LAN_CREATE && !uartTxBusy)
        {
            responseError = 0U;
            lanSocketId = -1;
            lanCleanupSocketId = -1;
            lanCleanupCanPromote = 0U;
            lanReadPending = 0U;
            lanSocketResetPending = 1U;
            nextLanRetryMs = now + WIFI_LAN_RETRY_MS;
            SendAtCommand(WIFI_STATE_WAIT_MQTT_HOST, 1500U,
                          "AT+MQTT=1,%s\r\n", wifiConfig.host);
            return;
        }
        if(wifiState == WIFI_STATE_LAN_READ && !uartTxBusy)
        {
            responseError = 0U;
            wifiState = lanResumeState;
            stateDeadlineMs = lanResumeDeadlineMs;
            return;
        }
        if(wifiState == WIFI_STATE_LAN_DELETE && !uartTxBusy)
        {
            responseError = 0U;
            if(lanDeleteKind == WIFI_LAN_DELETE_OWNER ||
               lanDeleteKind == WIFI_LAN_DELETE_REJECTED)
            {
                /* ERROR normally means that an AutoDel already removed the
                 * child.  Either way, never apply this result to another id. */
                CompleteLanChildDelete();
            }
            else
            {
                lanSocketId = -1;
                lanCleanupSocketId = -1;
                lanCleanupCanPromote = 0U;
                lanDeleteTargetId = -1;
                lanReadPending = 0U;
                StartLanServerGoneVerification(0U);
            }
            return;
        }
        if(wifiState == WIFI_STATE_LAN_RECREATE && !uartTxBusy)
        {
            responseError = 0U;
            lanSocketId = -1;
            lanCleanupSocketId = -1;
            lanCleanupCanPromote = 0U;
            lanDeleteTargetId = -1;
            lanDeleteKind = WIFI_LAN_DELETE_SERVER;
            lanReadPending = 0U;
            lanSocketResetPending = 1U;
            nextLanRetryMs = now + WIFI_LAN_RETRY_MS;
            wifiState = lanResumeState;
            stateDeadlineMs = lanResumeDeadlineMs;
            return;
        }
        if(wifiState == WIFI_STATE_LAN_SEND_WAIT_PROMPT ||
           wifiState == WIFI_STATE_LAN_SEND_WAIT_RESULT)
        {
            responseError = 0U;
            DeferLanTransfer();
            return;
        }
        /* MQTTDISCONN is deliberately idempotent.  A freshly powered module
         * may report ERROR simply because no old MQTT task exists; that is
         * already the desired state, so continue with a clean configuration.
         * All other command errors still enter the normal recovery path. */
        if(wifiState == WIFI_STATE_WAIT_MQTT_DISCONNECT && !uartTxBusy)
        {
            responseError = 0U;
            ContinueAfterMqttDisconnect();
            return;
        }
        if(lanClientActive && wifiConnected &&
           wifiState >= WIFI_STATE_WAIT_MQTT_HOST &&
           wifiState <= WIFI_STATE_WAIT_MQTT_RECONNECT)
        {
            /* LAN remains fully usable when the Internet/broker is down.
             * Once the LAN viewer releases control the normal backoff path
             * resumes and rebuilds MQTT from a clean state. */
            responseError = 0U;
            mqttConnected = 0U;
            wifiState = WIFI_STATE_WAIT_MQTT_RECONNECT;
            stateDeadlineMs = now + 30000U;
            return;
        }
        responseError = 0U;
        RestartBackoff(WIFI_ERROR_AT_RESPONSE);
        return;
    }

    if(StartLanOverlayIfNeeded(now)) return;

    switch(wifiState)
    {
        case WIFI_STATE_BACKOFF:
            if(TimeReached(now, stateDeadlineMs) && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_AT, 1000U, "AT\r\n");
            break;

        case WIFI_STATE_WAIT_AT:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_ATE0, 1000U, "ATE0\r\n");
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_ATE0:
            if(responseOk && !uartTxBusy)
            {
                /* Combo-AT retains its MQTT task across STM32 resets and can
                 * also auto-reconnect before the host has rewritten all TLS
                 * parameters.  Query the independently powered Wi-Fi state
                 * before touching that task: some builds return a slow/error
                 * MQTTDISCONN even though the AP lease is still healthy. */
                StartWarmWifiProbe();
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_DISCONNECT:
            if(responseOk && !uartTxBusy)
                ContinueAfterMqttDisconnect();
            else if(timedOut && !uartTxBusy)
                ContinueAfterMqttDisconnect();
            break;

        case WIFI_STATE_WAIT_WARM_WIFI_QUERY:
            if(responseOk && !uartTxBusy)
            {
                /* A complete query is kept across the following idempotent
                 * MQTT cleanup; ContinueAfterMqttDisconnect consumes it once. */
                if(!warmWifiStatusSeen || !warmWifiMatchesConfig)
                    wifiConnected = 0U;
                SendAtCommand(WIFI_STATE_WAIT_MQTT_DISCONNECT, 3000U,
                              "AT+MQTTDISCONN\r\n");
            }
            else if(timedOut && !uartTxBusy)
            {
                wifiConnected = 0U;
                warmWifiStatusSeen = 0U;
                warmWifiMatchesConfig = 0U;
                SendAtCommand(WIFI_STATE_WAIT_MQTT_DISCONNECT, 3000U,
                              "AT+MQTTDISCONN\r\n");
            }
            break;

        case WIFI_STATE_WAIT_WMODE:
            if(responseOk && !uartTxBusy)
            {
                if(wifiConfig.wifi_password[0] != '\0')
                    SendAtCommand(WIFI_STATE_WAIT_WJAP, 30000U,
                                  "AT+WJAP=%s,%s,%s\r\n",
                                  wifiConfig.ssid, wifiConfig.wifi_password,
                                  WIFI_FORCED_24G_BSSID);
                else
                    SendAtCommand(WIFI_STATE_WAIT_WJAP, 30000U,
                                  "AT+WJAP=%s,\"\",%s\r\n", wifiConfig.ssid,
                                  WIFI_FORCED_24G_BSSID);
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_WJAP:
            if(responseOk)
            {
                ClearResponseEvents();
                wifiState = WIFI_STATE_WAIT_WIFI_IP;
                connectDeadlineMs = now + 30000U;
                nextPollMs = now + 1000U;
                stateDeadlineMs = 0U;
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_WIFI_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_WIFI_IP:
            if(wifiConnected && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_LAN_RECV_CFG, 1500U,
                              "AT+SOCKETRECVCFG=1\r\n");
            else if(TimeReached(now, connectDeadlineMs))
                RestartBackoff(WIFI_ERROR_WIFI_TIMEOUT);
            else if(TimeReached(now, nextPollMs) && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_WIFI_QUERY, 1500U, "AT+WJAP?\r\n");
            break;

        case WIFI_STATE_WAIT_WIFI_QUERY:
            if(responseOk)
            {
                ClearResponseEvents();
                wifiState = WIFI_STATE_WAIT_WIFI_IP;
                nextPollMs = now + 1500U;
                stateDeadlineMs = 0U;
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_WIFI_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_LAN_RECV_CFG:
            if(responseOk && !uartTxBusy)
                StartLanStartupSocketQuery();
            else if(timedOut)
            {
                StartLanStartupSocketQuery();
            }
            break;

        case WIFI_STATE_WAIT_LAN_QUERY_STALE:
            if((responseOk || timedOut) && !uartTxBusy)
                ContinueLanStartupSocketCleanup();
            break;

        case WIFI_STATE_WAIT_LAN_DELETE_CHILD:
            if((responseOk || timedOut) && !uartTxBusy)
                ScheduleLanStartupChildRequery();
            break;

        case WIFI_STATE_WAIT_LAN_CHILD_DELETE_GRACE:
            if(timedOut && !uartTxBusy)
                StartLanStartupSocketQuery();
            break;

        case WIFI_STATE_WAIT_LAN_DELETE_STALE:
            if((responseOk || timedOut) && !uartTxBusy)
                StartLanServerGoneVerification(1U);
            break;

        case WIFI_STATE_WAIT_LAN_VERIFY_SERVER_GONE:
            if(responseOk && !uartTxBusy)
                ContinueLanServerGoneVerification();
            else if(timedOut && !uartTxBusy)
                ScheduleLanServerGoneRequery();
            break;

        case WIFI_STATE_WAIT_LAN_SERVER_DELETE_GRACE:
            if(timedOut && !uartTxBusy)
            {
                lanStartupServerStatus = -1;
                if(!SendAtCommand(WIFI_STATE_WAIT_LAN_VERIFY_SERVER_GONE,
                                  1500U, "AT+SOCKET?\r\n"))
                    RestartBackoff(WIFI_ERROR_UART);
            }
            break;

        case WIFI_STATE_WAIT_LAN_CREATE:
            if(responseOk && !uartTxBusy)
            {
                /* Combo-AT may acknowledge create while its retained socket
                 * table still has no listener.  Verify ConID 9 exists before
                 * trusting it; otherwise the desktop sees connection refused
                 * even though the application reports ONLINE. */
                VerifyCreatedLanListener();
            }
            else if(timedOut)
            {
                lanSocketId = -1;
                lanCleanupSocketId = -1;
                lanCleanupCanPromote = 0U;
                lanDeleteTargetId = -1;
                lanDeleteKind = WIFI_LAN_DELETE_SERVER;
                lanReadPending = 0U;
                lanSocketResetPending = 1U;
                lanStartupCleanupAttempts = 0U;
                nextLanRetryMs = now + WIFI_LAN_RETRY_MS;
                SendAtCommand(WIFI_STATE_WAIT_MQTT_HOST, 1500U,
                              "AT+MQTT=1,%s\r\n", wifiConfig.host);
            }
            break;

        case WIFI_STATE_WAIT_LAN_VERIFY_CREATED:
            if(responseOk && !uartTxBusy)
                ContinueCreatedLanListenerVerification();
            else if(timedOut && !uartTxBusy)
                ContinueCreatedLanListenerVerification();
            break;

        case WIFI_STATE_WAIT_LAN_CREATE_GRACE:
            if(timedOut && !uartTxBusy)
                StartLanListenerCreate(lanListenerCreateStartup);
            break;

        case WIFI_STATE_WAIT_MQTT_HOST:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_MQTT_PORT, 1500U,
                              "AT+MQTT=2,%u\r\n", wifiConfig.port);
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_PORT:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_MQTT_SCHEME, 1500U,
                              "AT+MQTT=3,%u\r\n", wifiConfig.scheme);
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_SCHEME:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_MQTT_CLIENT, 1500U,
                              "AT+MQTT=4,%s\r\n", wifiConfig.client_id);
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_CLIENT:
            if(responseOk && !uartTxBusy)
            {
                if(wifiConfig.username[0] != '\0')
                    SendAtCommand(WIFI_STATE_WAIT_MQTT_USER, 1500U,
                                  "AT+MQTT=5,%s\r\n", wifiConfig.username);
                else
                    SendAtCommand(WIFI_STATE_WAIT_MQTT_USER, 1500U,
                                  "AT+MQTT=5,\"\"\r\n");
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_USER:
            if(responseOk && !uartTxBusy)
            {
                if(wifiConfig.mqtt_password[0] != '\0')
                    SendAtCommand(WIFI_STATE_WAIT_MQTT_PASSWORD, 1500U,
                                  "AT+MQTT=6,%s\r\n", wifiConfig.mqtt_password);
                else
                    SendAtCommand(WIFI_STATE_WAIT_MQTT_PASSWORD, 1500U,
                                  "AT+MQTT=6,\"\"\r\n");
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_PASSWORD:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_MQTT_VERSION, 1500U,
                              "AT+MQTTVER=4\r\n");
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_VERSION:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_MQTT_BUFFER, 1500U,
                              "AT+MQTTBUF=2048,2048\r\n");
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_BUFFER:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_MQTT_KEEPALIVE, 1500U,
                              "AT+MQTTKEEPALIVE=60,10\r\n");
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_KEEPALIVE:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_MQTT_LWT, 1500U,
                              "AT+MQTT=7,\"%s\",1,1,\"OFFLINE\"\r\n",
                              statusTopic);
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_CERT:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_MQTT_LWT, 1500U,
                              "AT+MQTT=7,\"%s\",1,1,\"OFFLINE\"\r\n",
                              statusTopic);
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_LWT:
            if(responseOk && !uartTxBusy)
                SendAtCommand(WIFI_STATE_WAIT_MQTT_START, 3000U, "AT+MQTT\r\n");
            else if(timedOut) RestartBackoff(WIFI_ERROR_AT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_START:
            if(responseOk)
            {
                ClearResponseEvents();
                wifiState = WIFI_STATE_WAIT_MQTT_CONNECT;
                connectDeadlineMs = now + 30000U;
                nextPollMs = now + 1500U;
                stateDeadlineMs = 0U;
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_MQTT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_MQTT_CONNECT:
            if(mqttConnected && !uartTxBusy)
            {
                /* The CONNECT event precedes the point at which P1.0.22 can
                 * safely accept MQTTSUB. */
                if(stateDeadlineMs == 0U) stateDeadlineMs = now + 2000U;
                else if(timedOut)
                    SendAtCommand(WIFI_STATE_WAIT_SUBSCRIBE, 10000U,
                                  "AT+MQTTSUB=%s,0\r\n", commandTopic);
            }
            else if(TimeReached(now, connectDeadlineMs))
            {
                if(lanClientActive) connectDeadlineMs = now + 30000U;
                else RestartBackoff(WIFI_ERROR_MQTT_TIMEOUT);
            }
            break;

        case WIFI_STATE_WAIT_MQTT_QUERY:
            if(responseOk)
            {
                ClearResponseEvents();
                wifiState = WIFI_STATE_WAIT_MQTT_CONNECT;
                nextPollMs = now + 1500U;
                stateDeadlineMs = 0U;
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_MQTT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_SUBSCRIBE:
            if(responseOk && !uartTxBusy)
            {
                /* Successful subscription briefly restarts the module's MQTT
                 * connection.  Wait for the reconnect before publishing. */
                ClearResponseEvents();
                wifiState = WIFI_STATE_WAIT_SUBSCRIBE_SETTLE;
                stateDeadlineMs = now + 3000U;
                connectDeadlineMs = now + 30000U;
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_MQTT_TIMEOUT);
            break;

        case WIFI_STATE_WAIT_SUBSCRIBE_SETTLE:
            if(timedOut && mqttConnected && !uartTxBusy) MarkOnline();
            else if(TimeReached(now, connectDeadlineMs))
            {
                if(lanClientActive) connectDeadlineMs = now + 30000U;
                else RestartBackoff(WIFI_ERROR_MQTT_TIMEOUT);
            }
            break;

        case WIFI_STATE_WAIT_MQTT_RECONNECT:
            if(mqttConnected && !uartTxBusy)
            {
                /* Existing subscriptions are restored by the module after a
                 * reconnect (MQTTSUB status 2).  Re-advertise metadata/status
                 * only after a short settle window so the retained ONLINE
                 * packet cannot race the reconnect callback. */
                ClearResponseEvents();
                wifiState = WIFI_STATE_WAIT_SUBSCRIBE_SETTLE;
                stateDeadlineMs = now + 2000U;
                connectDeadlineMs = now + 10000U;
            }
            else if(timedOut)
            {
                if(lanClientActive) stateDeadlineMs = now + 30000U;
                else RestartBackoff(WIFI_ERROR_MQTT_TIMEOUT);
            }
            break;

        case WIFI_STATE_ONLINE:
            if(lanClientActive && !lanRawMode &&
               nextLocalHeartbeatMs != 0U &&
               TimeReached(now, nextLocalHeartbeatMs))
            {
                /* BW20 P1.0.22 does not reliably emit MQTT PINGREQ while the
                 * LAN owner suppresses public spectra.  A retained LAN status
                 * heartbeat keeps the broker session alive without exposing
                 * any ADC measurement data. */
                statusPending = 1U;
                nextLocalHeartbeatMs = now + 20000U;
                nextPublishMs = now;
            }
            if(!uartTxBusy && TimeReached(now, nextPublishMs)) StartNextPublish();
            break;

        case WIFI_STATE_PUBLISH_WAIT_PROMPT:
            if(responsePrompt && !uartTxBusy)
            {
                responsePrompt = 0U;
                responseOk = responseError = 0U;
                wifiState = WIFI_STATE_PUBLISH_WAIT_RESULT;
                stateDeadlineMs = now + 10000U;
                if(!WifiUartSend(publishPayload, publishPayloadLength))
                    RestartBackoff(WIFI_ERROR_UART);
            }
            else if(timedOut) RestartBackoff(WIFI_ERROR_PUBLISH_TIMEOUT);
            break;

        case WIFI_STATE_PUBLISH_WAIT_RESULT:
            if(responseOk && !uartTxBusy) PublishSuccess();
            else if(timedOut) RestartBackoff(WIFI_ERROR_PUBLISH_TIMEOUT);
            break;

        case WIFI_STATE_LAN_READ:
            if(responseOk && !uartTxBusy)
            {
                ClearResponseEvents();
                wifiState = lanResumeState;
                stateDeadlineMs = lanResumeDeadlineMs;
            }
            else if(timedOut)
            {
                ClearResponseEvents();
                wifiState = lanResumeState;
                stateDeadlineMs = lanResumeDeadlineMs;
            }
            break;

        case WIFI_STATE_LAN_DELETE:
            if((responseOk || timedOut) && !uartTxBusy)
            {
                if(lanDeleteKind == WIFI_LAN_DELETE_OWNER ||
                   lanDeleteKind == WIFI_LAN_DELETE_REJECTED)
                {
                    CompleteLanChildDelete();
                }
                else
                {
                    lanSocketId = -1;
                    lanCleanupSocketId = -1;
                    lanCleanupCanPromote = 0U;
                    lanDeleteTargetId = -1;
                    lanReadPending = 0U;
                    StartLanServerGoneVerification(0U);
                }
            }
            break;

        case WIFI_STATE_LAN_RECREATE:
            if(responseOk && !uartTxBusy)
            {
                lanSocketId = -1;
                lanCleanupSocketId = -1;
                lanCleanupCanPromote = 0U;
                lanCleanupOverflow = 0U;
                lanDeleteTargetId = -1;
                lanDeleteKind = WIFI_LAN_DELETE_SERVER;
                lanReadPending = 0U;
                lanSocketResetPending = 0U;
                wifiLastError = WIFI_ERROR_NONE;
                ClearResponseEvents();
                wifiState = lanResumeState;
                stateDeadlineMs = lanResumeDeadlineMs;
            }
            else if(timedOut)
            {
                lanSocketId = -1;
                lanCleanupSocketId = -1;
                lanCleanupCanPromote = 0U;
                lanDeleteTargetId = -1;
                lanDeleteKind = WIFI_LAN_DELETE_SERVER;
                lanReadPending = 0U;
                lanSocketResetPending = 1U;
                nextLanRetryMs = now + WIFI_LAN_RETRY_MS;
                ClearResponseEvents();
                wifiState = lanResumeState;
                stateDeadlineMs = lanResumeDeadlineMs;
            }
            break;

        case WIFI_STATE_LAN_SEND_WAIT_PROMPT:
            if(responsePrompt && !uartTxBusy)
            {
                responsePrompt = 0U;
                responseOk = responseError = 0U;
                wifiState = WIFI_STATE_LAN_SEND_WAIT_RESULT;
                stateDeadlineMs = now + 5000U;
                if(!WifiUartSend(publishPayload, publishPayloadLength))
                    DeferLanTransfer();
            }
            else if(timedOut) DeferLanTransfer();
            break;

        case WIFI_STATE_LAN_SEND_WAIT_RESULT:
            if(responseOk && !uartTxBusy) PublishSuccess();
            else if(timedOut) DeferLanTransfer();
            break;

        default:
            RestartBackoff(WIFI_ERROR_AT_RESPONSE);
            break;
    }
}
