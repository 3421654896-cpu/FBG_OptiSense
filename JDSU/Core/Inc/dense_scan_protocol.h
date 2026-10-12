#ifndef DENSE_SCAN_PROTOCOL_H
#define DENSE_SCAN_PROTOCOL_H
#include <stdint.h>
#include <stddef.h>

/* Finite dense reacquisition; RAM-only, not a new work mode or installed
 * table. 808-byte request: header 24, up to 48 records of 16 bytes, zero
 * padding, CRC32 at 804. A record is index + five DAC codes + dwell_us.
 * Bit 15 of index identifies an existing calibrated transition guard:
 * it is held but never sampled or counted as a spectrum point. */
#define DENSE_SCAN_MAX_OPS 48U
#define DENSE_SCAN_CRC_OFFSET 804U
#define DENSE_SCAN_MAX_DWELL_US 100000UL
#define DENSE_SCAN_MAX_BLOCK_WAIT_US 10000000UL
#define DENSE_SCAN_GUARD_FLAG 0x8000U
typedef struct {
    uint32_t tag, crc, settleUs;
    uint16_t sequence, total;
    uint8_t count, flags, selectors[2];
    uint16_t indices[DENSE_SCAN_MAX_OPS];
    uint16_t codes[DENSE_SCAN_MAX_OPS][5];
    uint32_t waits[DENSE_SCAN_MAX_OPS];
} DenseScanRequest;

static uint16_t DenseScan_U16(const uint8_t *p)
{ return (uint16_t)(((uint16_t)p[0] << 8) | p[1]); }
static uint32_t DenseScan_U32(const uint8_t *p)
{ return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; }
static uint32_t DenseScan_Crc(const uint8_t *p, size_t count)
{
    uint32_t crc = 0xFFFFFFFFUL;
    for(size_t i = 0U; i < count; i++) {
        crc ^= p[i];
        for(uint8_t bit = 0U; bit < 8U; bit++)
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0U);
    }
    return crc ^ 0xFFFFFFFFUL;
}
static uint8_t DenseScan_Parse(const uint8_t *p, size_t length, DenseScanRequest *out)
{
    static const uint16_t limits[5] = {63351U,63351U,32767U,24575U,24575U};
    if(!p || !out || length != 808U || p[0] != 255U || p[1] != 255U ||
       p[2] != 3U || p[3] != 13U || p[4] != 'D' || p[5] != 'N' ||
       p[6] != 'S' || p[7] != '1') return 0U;
    out->tag = DenseScan_U32(p + 8);
    out->sequence = DenseScan_U16(p + 12);
    out->count = p[14]; out->flags = p[15];
    out->settleUs = DenseScan_U32(p + 16);
    out->selectors[0] = p[20]; out->selectors[1] = p[21];
    out->total = DenseScan_U16(p + 22);
    out->crc = DenseScan_U32(p + DENSE_SCAN_CRC_OFFSET);
    if(!out->tag || !out->count || out->count > DENSE_SCAN_MAX_OPS ||
       out->flags > 3U || out->settleUs > DENSE_SCAN_MAX_DWELL_US ||
       out->selectors[0] > 3U || out->selectors[1] > 3U ||
       !out->total || out->total > 2001U ||
       out->crc != DenseScan_Crc(p, DENSE_SCAN_CRC_OFFSET)) return 0U;
    uint32_t waitSum = 0U;
    uint16_t lastNormal = 0U;
    uint8_t haveNormal = 0U;
    for(uint8_t i = 0U; i < out->count; i++) {
        const uint8_t *record = p + 24U + (uint16_t)i * 16U;
        uint16_t index = DenseScan_U16(record);
        uint16_t physical = index & 0x7FFFU;
        uint8_t guard = (index & DENSE_SCAN_GUARD_FLAG) != 0U;
        out->indices[i] = index;
        out->waits[i] = DenseScan_U32(record + 12);
        if(physical > 2000U || (haveNormal && physical <= lastNormal)) return 0U;
        if(guard) {
            if(!out->waits[i] || out->waits[i] > 2000000UL) return 0U;
        } else {
            if(out->waits[i] != out->settleUs) return 0U;
            haveNormal = 1U; lastNormal = physical;
        }
        waitSum += out->waits[i];
        if(waitSum > DENSE_SCAN_MAX_BLOCK_WAIT_US) return 0U;
        for(uint8_t c = 0U; c < 5U; c++) {
            out->codes[i][c] = DenseScan_U16(record + 2U + c * 2U);
            if(out->codes[i][c] > limits[c]) return 0U;
        }
    }
    for(size_t i = 24U + out->count * 16U; i < DENSE_SCAN_CRC_OFFSET; i++)
        if(p[i]) return 0U;
    return 1U;
}
#endif
