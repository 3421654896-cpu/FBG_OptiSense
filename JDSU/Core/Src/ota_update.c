#include "main.h"
#include "ota_layout.h"
#include "ota_update.h"
#include "sha256.h"
#include "firmware_version.h"
#include "usbd_cdc_if.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OTA_COMMAND_MAX       500U
#define OTA_DATA_MAX          180U
#define OTA_NONCE_LENGTH       16U
#define OTA_TRANSPORT_USB       3U
#define OTA_USB_FRAME_SIZE    808U

extern USBD_HandleTypeDef hUsbDeviceFS;
static uint8_t usbUpdateFrame[OTA_USB_FRAME_SIZE];
static volatile uint8_t usbUpdatePending;
static char usbUpdateResponse[192];
static uint16_t usbUpdateResponseLength;
static uint8_t usbUpdateResponseInFlight;
static uint8_t usbUpdateReboot;

typedef struct
{
    uint8_t active;
    uint32_t session;
    uint32_t image_version;
    uint32_t image_size;
    uint32_t image_crc32;
    uint32_t offset;
    uint8_t transport_kind;
    uint8_t committed;
    char nonce[OTA_NONCE_LENGTH + 1U];
} OtaSession;

static OtaSession otaSession;
static uint8_t otaPendingCommitted;
static uint8_t configPreserve[OTA_CONFIG_PRESERVE_SIZE];
static uint32_t otaRuntimeBootId;

static uint32_t OtaCrc32(const uint8_t *data, uint32_t length)
{
    uint32_t crc = 0xFFFFFFFFUL;
    uint32_t index;
    uint8_t bit;
    for(index = 0U; index < length; index++)
    {
        crc ^= data[index];
        for(bit = 0U; bit < 8U; bit++)
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0U);
    }
    return crc ^ 0xFFFFFFFFUL;
}

static int HexValue(char character)
{
    if(character >= '0' && character <= '9') return character - '0';
    if(character >= 'A' && character <= 'F') return character - 'A' + 10;
    if(character >= 'a' && character <= 'f') return character - 'a' + 10;
    return -1;
}

static uint8_t ParseHexU32Exact(const char *text, uint32_t *value)
{
    uint32_t parsed = 0U;
    uint32_t index;
    if(text == NULL || value == NULL || strlen(text) != 8U) return 0U;
    for(index = 0U; index < 8U; index++)
    {
        int digit = HexValue(text[index]);
        if(digit < 0) return 0U;
        parsed = (parsed << 4) | (uint32_t)digit;
    }
    *value = parsed;
    return 1U;
}

static uint8_t DecodeHex(const char *text, uint8_t *output, uint32_t outputLength)
{
    uint32_t index;
    if(strlen(text) != outputLength * 2U) return 0U;
    for(index = 0U; index < outputLength; index++)
    {
        int high = HexValue(text[index * 2U]);
        int low = HexValue(text[index * 2U + 1U]);
        if(high < 0 || low < 0) return 0U;
        output[index] = (uint8_t)((high << 4) | low);
    }
    return 1U;
}

static int Base64Value(char character)
{
    if(character >= 'A' && character <= 'Z') return character - 'A';
    if(character >= 'a' && character <= 'z') return character - 'a' + 26;
    if(character >= '0' && character <= '9') return character - '0' + 52;
    if(character == '+') return 62;
    if(character == '/') return 63;
    if(character == '=') return -2;
    return -1;
}

static uint8_t DecodeBase64(const char *input, uint8_t *output,
                            uint32_t capacity, uint32_t *outputLength)
{
    uint32_t inputLength = (uint32_t)strlen(input);
    uint32_t inputIndex;
    uint32_t outputIndex = 0U;
    if(inputLength == 0U || (inputLength & 3U) != 0U) return 0U;
    for(inputIndex = 0U; inputIndex < inputLength; inputIndex += 4U)
    {
        int a=Base64Value(input[inputIndex]);
        int b=Base64Value(input[inputIndex+1U]);
        int c=Base64Value(input[inputIndex+2U]);
        int d=Base64Value(input[inputIndex+3U]);
        uint32_t packed;
        if(a < 0 || b < 0 || c == -1 || d == -1) return 0U;
        if((c == -2 && d != -2) || (inputIndex + 4U < inputLength && (c == -2 || d == -2))) return 0U;
        packed = ((uint32_t)a << 18) | ((uint32_t)b << 12) |
                 ((uint32_t)((c < 0) ? 0 : c) << 6) |
                 (uint32_t)((d < 0) ? 0 : d);
        if(outputIndex >= capacity) return 0U;
        output[outputIndex++] = (uint8_t)(packed >> 16);
        if(c != -2)
        {
            if(outputIndex >= capacity) return 0U;
            output[outputIndex++] = (uint8_t)(packed >> 8);
        }
        if(d != -2)
        {
            if(outputIndex >= capacity) return 0U;
            output[outputIndex++] = (uint8_t)packed;
        }
    }
    *outputLength = outputIndex;
    return 1U;
}

static uint8_t FlashErase(uint32_t sector, uint32_t count)
{
    FLASH_EraseInitTypeDef erase;
    uint32_t sectorError = 0U;
    memset(&erase, 0, sizeof(erase));
    erase.TypeErase = FLASH_TYPEERASE_SECTORS;
    erase.VoltageRange = FLASH_VOLTAGE_RANGE_3;
    erase.Sector = sector;
    erase.NbSectors = count;
    return (HAL_FLASHEx_Erase(&erase, &sectorError) == HAL_OK) ? 1U : 0U;
}

static uint8_t FlashProgram(uint32_t address, const uint8_t *data, uint32_t length)
{
    uint32_t offset;
    for(offset = 0U; offset < length; offset += 4U)
    {
        uint32_t word = 0xFFFFFFFFUL;
        uint32_t remaining = length - offset;
        memcpy(&word, data + offset, (remaining >= 4U) ? 4U : remaining);
        if(HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD, address + offset, word) != HAL_OK)
            return 0U;
    }
    return (memcmp((const void *)address, data, length) == 0) ? 1U : 0U;
}

static uint8_t EraseStaging(void)
{
    uint8_t result;
    if(HAL_FLASH_Unlock() != HAL_OK) return 0U;
    result = FlashErase(FLASH_SECTOR_6, 2U);
    HAL_FLASH_Lock();
    return result;
}

static uint8_t ProgramStaging(uint32_t offset, const uint8_t *data, uint32_t length)
{
    uint8_t result;
    if(data == NULL || length == 0U || (offset & 3U) != 0U ||
       offset > OTA_STAGING_MAX_SIZE || length > OTA_STAGING_MAX_SIZE - offset)
        return 0U;
    if(HAL_FLASH_Unlock() != HAL_OK) return 0U;
    result = FlashProgram(OTA_STAGING_ADDRESS + offset, data, length);
    HAL_FLASH_Lock();
    return result;
}

static uint8_t ImageVectorValid(void)
{
    uint32_t initialSp = *(const uint32_t *)OTA_STAGING_ADDRESS;
    uint32_t resetVector = *(const uint32_t *)(OTA_STAGING_ADDRESS + 4U);
    uint32_t resetAddress = resetVector & ~1UL;
    return (initialSp >= 0x20000000UL && initialSp <= 0x20020000UL &&
            (initialSp & 3U) == 0U && (resetVector & 1U) != 0U &&
            resetAddress >= OTA_APPLICATION_ADDRESS &&
            resetAddress < OTA_APPLICATION_END) ? 1U : 0U;
}

static uint8_t CommitMetadata(void)
{
    OtaMetadata metadata;
    uint32_t offset;
    uint8_t result = 1U;
    memcpy(configPreserve, (const void *)OTA_CONFIG_ADDRESS, sizeof(configPreserve));
    memset(&metadata, 0xFF, sizeof(metadata));
    metadata.magic = OTA_METADATA_MAGIC_PENDING;
    metadata.format_version = OTA_METADATA_FORMAT_VERSION;
    metadata.target_id = OTA_TARGET_ID;
    metadata.image_version = otaSession.image_version;
    metadata.image_size = otaSession.image_size;
    metadata.image_crc32 = otaSession.image_crc32;
    metadata.link_address = OTA_APPLICATION_ADDRESS;
    metadata.staging_address = OTA_STAGING_ADDRESS;
    metadata.reserved[0] = otaSession.session;
    metadata.metadata_crc32 = OtaCrc32(((const uint8_t *)&metadata) + 4U,
                                      sizeof(metadata) - 8U);

    if(HAL_FLASH_Unlock() != HAL_OK) return 0U;
    if(!FlashErase(FLASH_SECTOR_1, 1U)) result = 0U;
    if(result && !FlashProgram(OTA_CONFIG_ADDRESS, configPreserve,
                               sizeof(configPreserve))) result = 0U;
    if(result && !FlashProgram(OTA_METADATA_ADDRESS,
                               (const uint8_t *)&metadata,
                               sizeof(metadata))) result = 0U;
    HAL_FLASH_Lock();
    for(offset = 0U; offset < sizeof(configPreserve); offset++)
    {
        if(((const uint8_t *)OTA_CONFIG_ADDRESS)[offset] != configPreserve[offset])
            return 0U;
    }
    return result;
}

static uint8_t SecureEqual(const uint8_t *left, const uint8_t *right, uint32_t length)
{
    uint8_t difference = 0U;
    uint32_t index;
    for(index = 0U; index < length; index++) difference |= left[index] ^ right[index];
    return difference == 0U ? 1U : 0U;
}

static void ErrorResponse(char *response, uint16_t capacity,
                          const char *code, const char *detail)
{
    snprintf(response, capacity, "FOTA1|ERROR|%s|%s\n", code, detail);
}

static uint8_t HandleBegin(char *command, const char *authorizationKey,
                           uint8_t transportKind,
                           char *response, uint16_t capacity)
{
    char original[OTA_COMMAND_MAX];
    char *signatureSeparator;
    char *token;
    char *fields[8];
    uint32_t fieldCount = 0U;
    uint8_t supplied[32];
    uint8_t expected[32];
    uint32_t version, size, crc;
    strncpy(original, command, sizeof(original) - 1U);
    original[sizeof(original) - 1U] = '\0';
    signatureSeparator = strrchr(original, '|');
    if(signatureSeparator == NULL)
    {
        ErrorResponse(response, capacity, "FORMAT", "missing-signature");
        return 1U;
    }
    *signatureSeparator = '\0';
    HmacSha256((const uint8_t *)authorizationKey,
               (uint32_t)strlen(authorizationKey),
               (const uint8_t *)original, (uint32_t)strlen(original), expected);

    token = strtok(command, "|");
    while(token != NULL && fieldCount < 8U)
    {
        fields[fieldCount++] = token;
        token = strtok(NULL, "|");
    }
    if(fieldCount != 8U || token != NULL || strcmp(fields[0], "FOTA1") != 0 ||
       strcmp(fields[1], "BEGIN") != 0 || strcmp(fields[2], "F205RE") != 0 ||
       strlen(fields[6]) != OTA_NONCE_LENGTH ||
       !DecodeHex(fields[7], supplied, sizeof(supplied)))
    {
        ErrorResponse(response, capacity, "FORMAT", "invalid-begin");
        return 1U;
    }
    if(!SecureEqual(supplied, expected, sizeof(expected)))
    {
        ErrorResponse(response, capacity, "AUTH", "authentication-failed");
        return 1U;
    }
    if(!ParseHexU32Exact(fields[3], &version) ||
       !ParseHexU32Exact(fields[4], &size) ||
       !ParseHexU32Exact(fields[5], &crc))
    {
        ErrorResponse(response, capacity, "FORMAT", "invalid-begin-number");
        return 1U;
    }
    if(size < 8U || size > OTA_APPLICATION_MAX_SIZE)
    {
        ErrorResponse(response, capacity, "SIZE", "image-out-of-range");
        return 1U;
    }
    /* Metadata already points at this staging area. Erasing it before the
     * coordinated reset could invalidate a fully committed update. */
    if(otaPendingCommitted)
    {
        ErrorResponse(response, capacity, "SESSION", "reboot-pending");
        return 1U;
    }
    if(otaSession.active && otaSession.transport_kind != transportKind)
    {
        ErrorResponse(response, capacity, "SESSION", "different-transport-owner");
        return 1U;
    }
    if(otaSession.active && otaSession.image_version == version &&
       otaSession.image_size == size && otaSession.image_crc32 == crc &&
       strcmp(otaSession.nonce, fields[6]) == 0)
    {
        snprintf(response, capacity, "FOTA1|READY|%08lX|%08lX|%08lX\n",
                 (unsigned long)otaSession.session,
                 (unsigned long)otaSession.offset,
                 (unsigned long)otaRuntimeBootId);
        return 1U;
    }
    /* BEGIN erases two large flash sectors synchronously.  Close the optical
     * path before that first blocking flash operation; the main loop only gets
     * a chance to observe otaSession.active after this function returns. */
    if(PI11210_SetSOAShutter(1U) != HAL_OK)
    {
        ErrorResponse(response, capacity, "SAFETY", "shutter-not-confirmed");
        return 1U;
    }
    if(!EraseStaging())
    {
        ErrorResponse(response, capacity, "FLASH", "staging-erase-failed");
        return 1U;
    }
    memset(&otaSession, 0, sizeof(otaSession));
    otaSession.active = 1U;
    otaSession.transport_kind = transportKind;
    otaSession.image_version = version;
    otaSession.image_size = size;
    otaSession.image_crc32 = crc;
    memcpy(otaSession.nonce, fields[6], OTA_NONCE_LENGTH);
    otaSession.nonce[OTA_NONCE_LENGTH] = '\0';
    otaSession.session = (*(const uint32_t *)0x1FFF7A10UL) ^
                         (*(const uint32_t *)0x1FFF7A14UL) ^
                         DWT->CYCCNT ^ version ^ crc;
    if(otaSession.session == 0U) otaSession.session = 1U;
    snprintf(response, capacity, "FOTA1|READY|%08lX|00000000|%08lX\n",
             (unsigned long)otaSession.session,
             (unsigned long)otaRuntimeBootId);
    return 1U;
}



static uint8_t HandleData(char *command, uint8_t transportKind,
                          char *response, uint16_t capacity)
{
    char *fields[6];
    char *token;
    uint32_t fieldCount = 0U;
    uint32_t session, offset, expectedCrc, decodedLength = 0U;
    uint8_t decoded[OTA_DATA_MAX];
    token = strtok(command, "|");
    while(token != NULL && fieldCount < 6U)
    {
        fields[fieldCount++] = token;
        token = strtok(NULL, "|");
    }
    if(fieldCount != 6U || token != NULL || strcmp(fields[0], "FOTA1") != 0 ||
       strcmp(fields[1], "DATA") != 0)
    {
        ErrorResponse(response, capacity, "FORMAT", "invalid-data");
        return 1U;
    }
    if(!ParseHexU32Exact(fields[2], &session) ||
       !ParseHexU32Exact(fields[3], &offset) ||
       !ParseHexU32Exact(fields[4], &expectedCrc))
    {
        ErrorResponse(response, capacity, "FORMAT", "invalid-data-number");
        return 1U;
    }
    if(!otaSession.active || session != otaSession.session ||
       otaSession.transport_kind != transportKind)
    {
        ErrorResponse(response, capacity, "SESSION", "session-not-active");
        return 1U;
    }
    if(!DecodeBase64(fields[5], decoded, sizeof(decoded), &decodedLength) ||
       decodedLength == 0U || offset > otaSession.image_size ||
       decodedLength > otaSession.image_size - offset ||
       (offset & 3U) != 0U ||
       ((offset + decodedLength) < otaSession.image_size &&
        (decodedLength & 3U) != 0U) ||
       OtaCrc32(decoded, decodedLength) != expectedCrc)
    {
        ErrorResponse(response, capacity, "CRC", "invalid-data-block");
        return 1U;
    }
    /* QoS-1 may redeliver any previously committed block after later blocks.
     * Confirm the actual flash bytes before returning the current offset. */
    if(offset != otaSession.offset)
    {
        if(offset < otaSession.offset &&
           decodedLength <= otaSession.offset - offset &&
           memcmp((const void *)(OTA_STAGING_ADDRESS + offset),
                  decoded, decodedLength) == 0)
        {
            snprintf(response, capacity, "FOTA1|ACK|%08lX|%08lX\n",
                     (unsigned long)otaSession.session,
                     (unsigned long)otaSession.offset);
            return 1U;
        }
        ErrorResponse(response, capacity, "OFFSET", "restart-begin-to-resume");
        return 1U;
    }
    if(!ProgramStaging(offset, decoded, decodedLength))
    {
        ErrorResponse(response, capacity, "FLASH", "staging-write-failed");
        return 1U;
    }
    otaSession.offset += decodedLength;
    snprintf(response, capacity, "FOTA1|ACK|%08lX|%08lX\n",
             (unsigned long)otaSession.session,
             (unsigned long)otaSession.offset);
    return 1U;
}

static uint8_t HandleEnd(char *command,
                         uint8_t transportKind,
                         char *response, uint16_t capacity,
                         uint8_t *rebootAfterResponse)
{
    char *fields[6];
    char *token;
    uint32_t fieldCount = 0U;
    uint32_t session, size, crc;
    token = strtok(command, "|");
    while(token != NULL && fieldCount < 6U)
    {
        fields[fieldCount++] = token;
        token = strtok(NULL, "|");
    }
    if(fieldCount != 5U ||
       token != NULL || strcmp(fields[0], "FOTA1") != 0 ||
       strcmp(fields[1], "END") != 0)
    {
        ErrorResponse(response, capacity, "FORMAT", "invalid-end");
        return 1U;
    }
    if(!ParseHexU32Exact(fields[2], &session) ||
       !ParseHexU32Exact(fields[3], &size) ||
       !ParseHexU32Exact(fields[4], &crc))
    {
        ErrorResponse(response, capacity, "FORMAT", "invalid-end-number");
        return 1U;
    }
    if(!otaSession.active || session != otaSession.session ||
       otaSession.transport_kind != transportKind ||
       size != otaSession.image_size || crc != otaSession.image_crc32 ||
       otaSession.offset != otaSession.image_size)
    {
        ErrorResponse(response, capacity, "SESSION", "incomplete-image");
        return 1U;
    }
    if(otaSession.committed)
    {
        snprintf(response, capacity, "FOTA1|REBOOT|%08lX\n",
                 (unsigned long)otaSession.image_version);
        *rebootAfterResponse = 1U;
        return 1U;
    }
    if(OtaCrc32((const uint8_t *)OTA_STAGING_ADDRESS, size) != crc ||
       !ImageVectorValid())
    {
        ErrorResponse(response, capacity, "VERIFY", "whole-image-invalid");
        return 1U;
    }
    if(!CommitMetadata())
    {
        ErrorResponse(response, capacity, "FLASH", "metadata-commit-failed");
        return 1U;
    }
    otaSession.committed = 1U;
    otaPendingCommitted = 1U;
    snprintf(response, capacity, "FOTA1|REBOOT|%08lX\n",
             (unsigned long)otaSession.image_version);
    *rebootAfterResponse = 1U;
    return 1U;
}

void OtaUpdate_Init(uint32_t runtimeBootId)
{
    memset(&otaSession, 0, sizeof(otaSession));
    otaPendingCommitted = 0U;
    otaRuntimeBootId = runtimeBootId;
}

static uint8_t OtaUpdate_HandleCommand(const char *payload, uint32_t payloadLength,
                                const char *authorizationKey,
                                uint8_t transportKind,
                                char *response, uint16_t responseCapacity,
                                uint8_t *rebootAfterResponse)
{
    static const char prefix[] = "FOTA1|";
    const char *start = NULL;
    char command[OTA_COMMAND_MAX];
    uint32_t length;
    uint32_t index;
    if(payload == NULL || response == NULL || responseCapacity == 0U ||
       transportKind != OTA_TRANSPORT_USB ||
       rebootAfterResponse == NULL) return 0U;
    for(index = 0U; index + sizeof(prefix) - 1U <= payloadLength; index++)
    {
        if(memcmp(payload + index, prefix, sizeof(prefix) - 1U) == 0)
        {
            start = payload + index;
            break;
        }
    }
    if(start == NULL) return 0U;
    length = payloadLength - (uint32_t)(start - payload);
    while(length > 0U && (start[length - 1U] == '\r' || start[length - 1U] == '\n')) length--;
    if(length == 0U || length >= sizeof(command) || memchr(start, '\0', length) != NULL)
    {
        ErrorResponse(response, responseCapacity, "FORMAT", "command-too-long");
        return 1U;
    }
    memcpy(command, start, length);
    command[length] = '\0';
    *rebootAfterResponse = 0U;
    if(strncmp(command, "FOTA1|BEGIN|", 12U) == 0)
    {
        if(authorizationKey == NULL || authorizationKey[0] == '\0')
        {
            ErrorResponse(response, responseCapacity, "AUTH", "key-not-configured");
            return 1U;
        }
        return HandleBegin(command, authorizationKey, transportKind,
                           response, responseCapacity);
    }
    if(strncmp(command, "FOTA1|DATA|", 11U) == 0)
        return HandleData(command, transportKind, response, responseCapacity);
    if(strncmp(command, "FOTA1|END|", 10U) == 0)
        return HandleEnd(command, transportKind,
                         response, responseCapacity, rebootAfterResponse);
    ErrorResponse(response, responseCapacity, "FORMAT", "unknown-command");
    return 1U;
}

uint8_t OtaUpdate_IsActive(void)
{
    return otaSession.active;
}



void OtaUpdate_AbortUncommittedSession(void)
{
    if(otaSession.active && !otaPendingCommitted)
        memset(&otaSession, 0, sizeof(otaSession));
}




uint8_t OtaUpdate_QueueUsbFrame(const uint8_t *frame, uint16_t length)
{
    if(frame == NULL || length != OTA_USB_FRAME_SIZE ||
       frame[0] != 0xFFU || frame[1] != 0xFFU ||
       frame[2] != EXTRA_STATE || frame[3] != 0x22U) return 0U;
    if(!usbUpdatePending)
    {
        memcpy(usbUpdateFrame, frame, OTA_USB_FRAME_SIZE);
        __DMB();
        usbUpdatePending = 1U;
    }
    return 1U;
}

void OtaUpdate_ProcessUsb(void)
{
    USBD_CDC_HandleTypeDef *cdc =
        (USBD_CDC_HandleTypeDef *)hUsbDeviceFS.pClassData;
    uint16_t length;
    uint32_t expectedCrc;
    static char command[OTA_COMMAND_MAX];
    char serial[13], key[20];
    uint32_t uid0, uid1, uid2;
    uint8_t reboot = 0U;
    if(usbUpdateResponseInFlight)
    {
        if(cdc == NULL || cdc->TxState != 0U) return;
        usbUpdateResponseInFlight = 0U;
        /* Metadata is committed only after complete image verification.
         * Reset only after the REBOOT reply has actually left the endpoint. */
        if(usbUpdateReboot) NVIC_SystemReset();
    }
    if(usbUpdateResponseLength)
    {
        if(CDC_Transmit_FS((uint8_t *)usbUpdateResponse,
                           usbUpdateResponseLength) == USBD_OK)
        {
            usbUpdateResponseLength = 0U;
            usbUpdateResponseInFlight = 1U;
        }
        return;
    }
    if(!usbUpdatePending) return;
    if(workState != EXTRA_STATE) ApplyWorkState(EXTRA_STATE);
    /* INFO also pauses acquisition. EXTRA alone does not gate optical output:
     * explicitly close the shutter even for a rejected USB wrapper. */
    (void)PI11210_SetSOAShutter(1U);
    ReceEndFlag = 0U;
    USB_ResetQueue();
    length = ((uint16_t)usbUpdateFrame[6] << 8) | usbUpdateFrame[7];
    expectedCrc = ((uint32_t)usbUpdateFrame[804] << 24) |
                  ((uint32_t)usbUpdateFrame[805] << 16) |
                  ((uint32_t)usbUpdateFrame[806] << 8) | usbUpdateFrame[807];
    if(usbUpdateFrame[4] != 1U || usbUpdateFrame[5] != 0U ||
       length == 0U || length >= OTA_COMMAND_MAX ||
       OtaCrc32(usbUpdateFrame, 804U) != expectedCrc ||
       memchr(&usbUpdateFrame[8], '\0', length) != NULL)
    {
        ErrorResponse(usbUpdateResponse, sizeof(usbUpdateResponse),
                      "USB_FRAME", "invalid-wrapper");
    }
    else
    {
        memcpy(command, &usbUpdateFrame[8], length);
        command[length] = '\0';
        uid0 = *(const uint32_t *)0x1FFF7A10UL;
        uid1 = *(const uint32_t *)0x1FFF7A14UL;
        uid2 = *(const uint32_t *)0x1FFF7A18UL;
        snprintf(serial, sizeof(serial), "%08lX%04lX",
                 (unsigned long)(uid0 + uid2), (unsigned long)(uid1 >> 16));
        /* This is an identity binding, NOT a secret. Physical USB access is
         * the only update trust boundary; network firmware updates are removed. */
        snprintf(key, sizeof(key), "USB:%s", serial);
        if(strcmp(command, "FUSB1|INFO") == 0)
        {
            uint32_t sp = *(const uint32_t *)OTA_BOOT_ADDRESS;
            uint32_t reset = *(const uint32_t *)(OTA_BOOT_ADDRESS + 4U);
            uint8_t bootValid = (sp >= 0x20000000UL && sp <= 0x20020000UL &&
                !(sp & 3U) && (reset & 1U) &&
                (reset & ~1UL) >= OTA_BOOT_ADDRESS &&
                (reset & ~1UL) < OTA_CONFIG_ADDRESS &&
                SCB->VTOR == OTA_APPLICATION_ADDRESS) ? 1U : 0U;
            snprintf(usbUpdateResponse, sizeof(usbUpdateResponse),
                "FUSB1|INFO|%s|%08lX|%08lX|%08lX|%08lX\n", serial,
                (unsigned long)APP_FIRMWARE_VERSION,
                (unsigned long)otaRuntimeBootId,
                (unsigned long)OTA_APPLICATION_MAX_SIZE, (unsigned long)bootValid);
        }
        else if(strcmp(command, "FUSB1|ABORT") == 0)
        {
            if(otaPendingCommitted || (otaSession.active &&
               otaSession.transport_kind != OTA_TRANSPORT_USB))
                ErrorResponse(usbUpdateResponse, sizeof(usbUpdateResponse),
                              "SESSION", "abort-not-permitted");
            else
            {
                OtaUpdate_AbortUncommittedSession();
                (void)PI11210_SetSOAShutter(1U);
                snprintf(usbUpdateResponse, sizeof(usbUpdateResponse),
                         "FUSB1|ABORTED\n");
            }
        }
        else if(!OtaUpdate_HandleCommand(command, length, key, OTA_TRANSPORT_USB,
                    usbUpdateResponse, sizeof(usbUpdateResponse), &reboot))
            ErrorResponse(usbUpdateResponse, sizeof(usbUpdateResponse),
                          "FORMAT", "unknown-usb-command");
    }
    usbUpdateReboot = reboot;
    usbUpdateResponseLength = (uint16_t)strlen(usbUpdateResponse);
    __DMB();
    usbUpdatePending = 0U;
}
