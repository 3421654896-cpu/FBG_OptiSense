#ifndef PEAK_DETECTION_H
#define PEAK_DETECTION_H

#include "stm32f2xx.h"
#include "datastruct.h"

#define INTERVAL 30
#define PEAKS_COUNT 15
#define PEAK_THRESHOLD 164

extern uint16_t item;

extern uint16_t adc1[Number];
extern uint16_t adc2[Number];
extern uint16_t adc3[Number];
extern uint16_t adc4[Number];

extern int adc_norvec[Number];

extern float peaks1[PEAKS_COUNT];
extern float peaks2[PEAKS_COUNT];
extern float peaks3[PEAKS_COUNT];
extern float peaks4[PEAKS_COUNT];

extern int int_part;
extern int decimal_part;

uint8_t Find_Peaks(uint16_t *adc_vec, float *peaks_vec, uint16_t adc_length);

void Find_Initial(uint16_t *adc_norvec, uint16_t *initials, uint16_t adc_length);

void FillPeaks(uint8_t p1, uint8_t p2, uint8_t p3, uint8_t p4);

#endif
