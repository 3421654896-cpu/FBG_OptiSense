#ifndef BOARD_THERMAL_PROTOCOL_H
#define BOARD_THERMAL_PROTOCOL_H
#include <stdint.h>

/* Backward-compatible 20-byte FF FF state/01 temperature monitor.
 * Bytes 4..7 keep signed integer + signed fraction (1e-4 C). TH1 occupies
 * previously unused bytes: FG RPM, PWM permille, validity/fault flags,
 * 'TH', version 1, saturated sensor age / 10 ms, CRC16-CCITT-FALSE.
 * This is independent of optical/dense/finger payloads and never queries
 * hardware or changes DACs, ownership, temperature targets or protection. */
static uint16_t BoardThermal_Crc(const uint8_t *data)
{
    uint16_t crc = 0xFFFFU;
    for(uint8_t i = 0U; i < 18U; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for(uint8_t bit = 0U; bit < 8U; bit++)
            crc = (uint16_t)((crc << 1) ^ ((crc & 0x8000U) ? 0x1021U : 0U));
    }
    return crc;
}
static void BoardThermal_Encode(uint8_t *out, uint8_t state, float temperatureC,
                               uint16_t rpm, uint16_t duty, uint16_t flags,
                               uint16_t ageMs)
{
    int16_t integer = (int16_t)temperatureC;
    int16_t fraction = (int16_t)((temperatureC - integer) * 10000.0f);
    out[0] = 255U; out[1] = 255U; out[2] = state; out[3] = 1U;
    uint16_t fields[5] = {(uint16_t)integer, (uint16_t)fraction, rpm, duty, flags};
    for(uint8_t i = 0U; i < 5U; i++) {
        out[4U + i * 2U] = (uint8_t)(fields[i] >> 8);
        out[5U + i * 2U] = (uint8_t)fields[i];
    }
    out[14] = 'T'; out[15] = 'H'; out[16] = 1U;
    out[17] = (uint8_t)(ageMs >= 2550U ? 255U : ageMs / 10U);
    uint16_t crc = BoardThermal_Crc(out);
    out[18] = (uint8_t)(crc >> 8); out[19] = (uint8_t)crc;
}
#endif
