#ifndef CANDIDATE_ROUTE_PROTOCOL_H
#define CANDIDATE_ROUTE_PROTOCOL_H
#include <stdint.h>
#include <stddef.h>

/* USB-only diagnostic. Never writes installed tables or flash.
 * Common header 0..23; 27 records (fullband index + five DAC codes), 12 bytes
 * each. Protocol v1 keeps its CRC at 348..351 and zero tail from 352.
 * Protocol v2 keeps 348..351 zero, adds one bounded hidden guard record at
 * 352..363, moves the packet CRC to 364..367, and keeps a zero tail from 368.
 * Protocol v3 uses 45 strictly increasing selected rows, T45! identity.
 * Protocol v4 keeps the same bounded packet but uses TVR! and 5..45 rows in
 * multiples of five, so a three-peak spectrum scans only its 15 real points.
 * Protocol v5 uses TVC!, the same variable row count, and packs the selected
 * ADC channel (high nibble) plus its feedback selector (low two bits) in byte
 * 19. CH2/CH3 require selector zero because their 2 kOhm feedback is fixed.
 * optional PWT1 per-point waits, and separate response frame types. Automatic
 * routes may be equally spaced; operator-edited routes need not be. It never
 * inserts hidden writes.
 * No caller-controlled SOA mode or arbitrary channel mask is accepted.
 */
#define CANDIDATE_ROUTE_COUNT 27U
#define TEMPORARY_ROUTE_COUNT 45U
#define TEMPORARY_ROUTE_MIN_COUNT 5U
#define CANDIDATE_ROUTE_V1_CRC_OFFSET 348U
#define CANDIDATE_ROUTE_V2_GUARD_OFFSET 352U
#define CANDIDATE_ROUTE_V2_CRC_OFFSET 364U
typedef struct {
    uint16_t indices[TEMPORARY_ROUTE_COUNT];
    uint16_t codes[TEMPORARY_ROUTE_COUNT][5];
    uint16_t firstDelayUs;
    uint16_t spacingUs;
    uint16_t boundaryExtraUs;
    uint16_t pointDelayUs[TEMPORARY_ROUTE_COUNT];
    uint8_t pointCount;
    uint32_t totalWaitUs;
    uint32_t tableCrc;
    uint8_t protocolVersion;
    uint8_t adcChannel;
    uint8_t feedbackSelector;
    uint8_t guardBeforeLocal;
    uint16_t guardHoldUs;
    uint16_t guardIndex;
    uint16_t guardCodes[5];
} CandidateRoute;

static uint16_t CandidateRoute_U16(const uint8_t *p)
{
    return (uint16_t)(((uint16_t)p[0] << 8) | p[1]);
}
static uint32_t CandidateRoute_Crc(const uint8_t *p, size_t count)
{
    uint32_t crc = 0xFFFFFFFFUL;
    for(size_t i = 0U; i < count; i++) {
        crc ^= p[i];
        for(uint8_t bit = 0U; bit < 8U; bit++)
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0U);
    }
    return crc ^ 0xFFFFFFFFUL;
}
static uint8_t CandidateRoute_Parse(const uint8_t *packet, size_t length, CandidateRoute *out)
{
    static const uint16_t legacyLimits[5] = {58981U, 58981U, 32767U, 24575U, 24575U};
    static const uint16_t temporaryLimits[5] = {63351U, 63351U, 32767U, 24575U, 24575U};
    if(!packet || !out || length != 808U) return 0U;
    uint8_t version = packet[4];
    /* v3/v4: temporary selected rows, no hidden writes. Separate magic,
     * CRC and frame types keep the existing v1/v2 wire contract unchanged. */
    if(version == 3U || version == 4U || version == 5U) {
        uint8_t pointCount = packet[5];
        uint8_t countValid = version == 3U
                ? pointCount == TEMPORARY_ROUTE_COUNT
                : (pointCount >= TEMPORARY_ROUTE_MIN_COUNT &&
                   pointCount <= TEMPORARY_ROUTE_COUNT && pointCount % 5U == 0U);
        uint8_t magicValid = version == 3U
                ? (packet[12] == 'T' && packet[13] == '4' &&
                   packet[14] == '5' && packet[15] == '!')
                : (version == 4U
                   ? (packet[12] == 'T' && packet[13] == 'V' &&
                      packet[14] == 'R' && packet[15] == '!')
                   : (packet[12] == 'T' && packet[13] == 'V' &&
                      packet[14] == 'C' && packet[15] == '!'));
        uint8_t packedChannel = packet[19];
        uint8_t adcChannel = version == 5U ? (uint8_t)(packedChannel >> 4) : 1U;
        uint8_t feedbackSelector = version == 5U
                ? (uint8_t)(packedChannel & 0x03U)
                : (packedChannel ? (uint8_t)(packedChannel - 1U) : 2U);
        uint8_t channelValid = version == 5U
                ? (adcChannel <= 3U && (packedChannel & 0x0CU) == 0U &&
                   (adcChannel < 2U || feedbackSelector == 0U))
                : packedChannel <= 4U;
        if(packet[0] != 0xFFU || packet[1] != 0xFFU || packet[2] != 3U ||
           packet[3] != 10U || !countValid || !magicValid ||
           packet[18] != 16U || !channelValid ||
           !(packet[8] | packet[9] | packet[10] | packet[11])) return 0U;
        uint16_t cycles = CandidateRoute_U16(packet + 16U);
        uint16_t first = CandidateRoute_U16(packet + 6U);
        uint16_t spacing = CandidateRoute_U16(packet + 20U);
        uint16_t boundary = CandidateRoute_U16(packet + 22U);
        if(boundary > 3000U || boundary % 25U) return 0U;
        if((cycles != 0U && (cycles < 32U || cycles > 512U)) ||
           first < 50U || first > 850U || spacing < 50U || spacing > 600U ||
           first % 25U || spacing % 25U || (uint32_t)first + spacing > 900U) return 0U;
        uint32_t supplied = ((uint32_t)packet[564] << 24) |
            ((uint32_t)packet[565] << 16) | ((uint32_t)packet[566] << 8) | packet[567];
        if(CandidateRoute_Crc(packet, 564U) != supplied) return 0U;
        /* Optional PWT1 extension: absolute first-read waits, no boundary
         * addition. CRC binds header, route and all waits; legacy stays exact. */
        uint8_t perPoint = packet[568] == 'P' && packet[569] == 'W' &&
                           packet[570] == 'T' && packet[571] == '1';
        if(perPoint) {
            if(boundary != 0U) return 0U;
            for(uint16_t row = 0U; row < pointCount; row++) {
                uint16_t wait = CandidateRoute_U16(packet + 572U + 2U * row);
                if(wait < 50U || wait > 15000U || wait % 25U) return 0U;
            }
            for(uint16_t row = pointCount; row < TEMPORARY_ROUTE_COUNT; row++)
                if(CandidateRoute_U16(packet + 572U + 2U * row)) return 0U;
            uint32_t extensionCrc = ((uint32_t)packet[662] << 24) |
                ((uint32_t)packet[663] << 16) | ((uint32_t)packet[664] << 8) | packet[665];
            if(CandidateRoute_Crc(packet, 662U) != extensionCrc) return 0U;
        }
        for(size_t i = perPoint ? 666U : 568U; i < length; i++)
            if(packet[i]) return 0U;
        for(size_t i = 24U + (size_t)pointCount * 12U; i < 564U; i++)
            if(packet[i]) return 0U;
        for(uint16_t row = 0U; row < pointCount; row++) {
            size_t at = 24U + row * 12U;
            uint16_t index = CandidateRoute_U16(packet + at);
            if(index > 2000U || (row && index <= CandidateRoute_U16(packet + at - 12U)))
                return 0U;
            for(uint16_t channel = 0U; channel < 5U; channel++)
                if(CandidateRoute_U16(packet + at + 2U + 2U * channel) > temporaryLimits[channel])
                    return 0U;
        }
        out->totalWaitUs = (uint32_t)pointCount * spacing;
        for(uint16_t row = 0U; row < pointCount; row++) {
            size_t at = 24U + row * 12U;
            out->indices[row] = CandidateRoute_U16(packet + at);
            out->pointDelayUs[row] = perPoint
                ? CandidateRoute_U16(packet + 572U + row * 2U)
                : (uint16_t)(first + ((row % 5U == 0U) ? boundary : 0U));
            out->totalWaitUs += out->pointDelayUs[row];
            for(uint16_t channel = 0U; channel < 5U; channel++)
                out->codes[row][channel] = CandidateRoute_U16(
                    packet + at + 2U + 2U * channel);
        }
        out->protocolVersion = version;
        out->adcChannel = adcChannel;
        out->feedbackSelector = feedbackSelector;
        out->pointCount = pointCount;
        out->firstDelayUs = first;
        out->spacingUs = spacing;
        out->boundaryExtraUs = boundary;
        out->tableCrc = perPoint ? CandidateRoute_Crc(packet, 662U)
                                : CandidateRoute_Crc(packet + 24U, 540U);
        out->guardBeforeLocal = 0U;
        out->guardHoldUs = 0U;
        out->guardIndex = 0U;
        for(uint16_t channel = 0U; channel < 5U; channel++) out->guardCodes[channel] = 0U;
        return 1U;
    }
    if(packet[0] != 0xFFU || packet[1] != 0xFFU || packet[2] != 3U || packet[3] != 10U ||
       (version != 1U && version != 2U) || packet[5] != 27U || packet[12] != 'C' ||
       packet[13] != '2' || packet[14] != '7' || packet[15] != '!' ||
       packet[18] != 16U || packet[19] != 0U) return 0U;
    if(!(packet[8] | packet[9] | packet[10] | packet[11])) return 0U;
    uint16_t cycles = CandidateRoute_U16(packet + 16U);
    uint16_t first = CandidateRoute_U16(packet + 6U);
    uint16_t spacing = CandidateRoute_U16(packet + 20U);
    if(cycles < 32U || cycles > 1024U || first < 50U || first > 600U ||
       spacing < 50U || spacing > 600U || first % 25U || spacing % 25U ||
       (uint32_t)first + spacing > 650U) return 0U;
    size_t crcOffset = version == 1U ? CANDIDATE_ROUTE_V1_CRC_OFFSET
                                    : CANDIDATE_ROUTE_V2_CRC_OFFSET;
    size_t tailOffset = crcOffset + 4U;
    if(version == 1U && (packet[22] != 0U || packet[23] != 0U)) return 0U;
    if(version == 2U &&
       (packet[22] < 1U || packet[22] >= CANDIDATE_ROUTE_COUNT ||
        packet[23] < 20U || packet[23] > 200U ||
        packet[348] || packet[349] || packet[350] || packet[351])) return 0U;
    uint32_t supplied = ((uint32_t)packet[crcOffset] << 24) |
                        ((uint32_t)packet[crcOffset + 1U] << 16) |
                        ((uint32_t)packet[crcOffset + 2U] << 8) |
                        packet[crcOffset + 3U];
    if(CandidateRoute_Crc(packet, crcOffset) != supplied) return 0U;
    for(size_t i = tailOffset; i < length; i++) if(packet[i]) return 0U;
    uint16_t previous = 0U;
    for(uint16_t row = 0U; row < 27U; row++) {
        size_t at = 24U + row * 12U;
        uint16_t index = CandidateRoute_U16(packet + at);
        if(index > 2000U || (row && index <= previous)) return 0U;
        previous = index;
        for(uint16_t channel = 0U; channel < 5U; channel++)
            if(CandidateRoute_U16(packet + at + 2U + channel * 2U) > legacyLimits[channel]) return 0U;
    }
    if(version == 2U) {
        uint8_t before = packet[22];
        uint16_t guardIndex = CandidateRoute_U16(packet + CANDIDATE_ROUTE_V2_GUARD_OFFSET);
        uint16_t lowerIndex = CandidateRoute_U16(packet + 24U + (before - 1U) * 12U);
        uint16_t upperIndex = CandidateRoute_U16(packet + 24U + before * 12U);
        if(guardIndex <= lowerIndex || guardIndex >= upperIndex || guardIndex > 2000U) return 0U;
        for(uint16_t channel = 0U; channel < 5U; channel++)
            if(CandidateRoute_U16(packet + CANDIDATE_ROUTE_V2_GUARD_OFFSET + 2U + channel * 2U) >
               legacyLimits[channel]) return 0U;
    }
    /* Publish only after every byte and bound has been checked. */
    for(uint16_t row = 0U; row < 27U; row++) {
        size_t at = 24U + row * 12U;
        out->indices[row] = CandidateRoute_U16(packet + at);
        for(uint16_t channel = 0U; channel < 5U; channel++)
            out->codes[row][channel] = CandidateRoute_U16(packet + at + 2U + channel * 2U);
    }
    out->firstDelayUs = first;
    out->spacingUs = spacing;
    out->boundaryExtraUs = 0U;
    out->totalWaitUs = 0U;
    for(uint16_t row = 0U; row < TEMPORARY_ROUTE_COUNT; row++) out->pointDelayUs[row] = 0U;
    out->protocolVersion = version;
    out->adcChannel = 1U;
    out->feedbackSelector = 2U;
    out->pointCount = CANDIDATE_ROUTE_COUNT;
    if(version == 2U) {
        out->guardBeforeLocal = packet[22];
        out->guardHoldUs = (uint16_t)packet[23] * 25U;
        out->guardIndex = CandidateRoute_U16(packet + CANDIDATE_ROUTE_V2_GUARD_OFFSET);
        for(uint16_t channel = 0U; channel < 5U; channel++)
            out->guardCodes[channel] = CandidateRoute_U16(
                packet + CANDIDATE_ROUTE_V2_GUARD_OFFSET + 2U + channel * 2U);
        out->tableCrc = CandidateRoute_Crc(packet + 24U, 340U);
    } else {
        out->guardBeforeLocal = 0U;
        out->guardHoldUs = 0U;
        out->guardIndex = 0U;
        for(uint16_t channel = 0U; channel < 5U; channel++) out->guardCodes[channel] = 0U;
        out->tableCrc = CandidateRoute_Crc(packet + 24U, 324U);
    }
    return 1U;
}
#endif
