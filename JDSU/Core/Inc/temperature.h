#ifndef __TEMPERATURE_H
#define __TEMPERATURE_H

#include "stm32f2xx.h"
#include <stdint.h>

#define TEMP_PORT                         GPIOB
#define TEMP_PIN                          GPIO_PIN_4
#define FAN_PORT                          GPIOC
#define FAN_PIN                           GPIO_PIN_2
#define FAN_FG_PORT                       GPIOC
#define FAN_FG_PIN                        GPIO_PIN_3

#define SKIP_ROM                          0xCCU
#define CONVERT_T                         0x44U
#define READ_SCRATCHPAD                   0xBEU

/* TIM1 update DMA emits one complete 25 kHz four-wire fan PWM cycle. */
#define ARR_1                             100U

/* Thermal status flags sent to the local and remote user interfaces. */
#define THERMAL_FLAG_SENSOR_VALID         0x0001U
#define THERMAL_FLAG_ABOVE_TARGET         0x0002U
#define THERMAL_FLAG_SENSOR_FAULT         0x0004U
#define THERMAL_FLAG_FAN_TACH_FAULT       0x0008U
#define THERMAL_FLAG_PID_ACTIVE           0x0010U
#define THERMAL_FLAG_FAN_PWM_FAULT        0x0020U

extern uint16_t temperature;
extern uint32_t pwm_buffer[ARR_1];

void DQ_IN(void);
void DQ_OUT(void);

uint8_t M1820Z_Reset(void);
void M1820Z_WriteBit(uint8_t bit);
uint8_t M1820Z_ReadBit(void);
void M1820Z_WriteByte(uint8_t data);
uint8_t M1820Z_ReadByte(void);
float M1820Z_GetTmp(void);

/* cooling_percent is the physical fan command: 0 = minimum/off request,
 * 100 = full cooling.  The board's N-MOS inverts PC2 before J16. */
void Set_Soft_PWM_Duty(uint8_t cooling_percent);

void ThermalControl_Init(void);
/* Constant-time safety housekeeping for latency-sensitive wavelength scans.
 * It updates fan tach/stale-sensor fail-safe state without a 1-Wire slot.
 * ThermalControl_Process() advances at most one cooperative 1-Wire primitive
 * per call; its nominal worst case is the 980 us reset sequence. */
void ThermalControl_SafetyTick(void);
void ThermalControl_Process(void);
void ThermalControl_OnFanFgEdge(void);
void ThermalControl_OnFanPwmFault(void);
float ThermalControl_GetTemperatureC(void);
uint8_t ThermalControl_IsTemperatureValid(void);
uint16_t ThermalControl_GetFanDutyPermille(void);
uint16_t ThermalControl_GetFanRpm(void);
uint16_t ThermalControl_GetStatusFlags(void);
uint16_t ThermalControl_GetSensorAgeMs(void);

#endif
