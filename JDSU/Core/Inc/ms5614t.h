/* App/ms5614t.h */
#ifndef __MS5614T_H
#define __MS5614T_H

#include "stm32f2xx.h"

#ifdef __cplusplus
extern "C" {
#endif

#if 0  /* Legacy MS5614T voltage DAC definitions retained for reference. */
typedef enum
{
    MS5614T_DAC_A = 0x00,
    MS5614T_DAC_B = 0x01,
    MS5614T_DAC_C = 0x02,
    MS5614T_DAC_D = 0x03
} MS5614T_Channel_t;

typedef enum
{
    MS5614T_SPEED_SLOW = 0,
    MS5614T_SPEED_FAST = 1
} MS5614T_Speed_t;

typedef enum
{
    MS5614T_NORMAL    = 0,
    MS5614T_POWERDOWN = 1
} MS5614T_Power_t;
#endif

typedef enum
{
		IDAC1 = 0x11,
		IDAC4 = 0x14,
		IDAC5 = 0x15,
		IDAC6 = 0x16,
		IDAC7 = 0x17
} PI11210_Channeld_t;

typedef enum
{
		PI11210_SOA_SOURCE = 0,
		PI11210_SOA_SHUTTER = 1
} PI11210_SOAMode_t;

typedef struct
{
		uint16_t rawStatus;
		uint16_t i2cErrorCount;
		uint8_t online;
		uint8_t partValid;
		uint8_t initialized;
		uint8_t busRecovered;
		uint8_t soaMode;
} PI11210_Status_t;

#if 0  /* Legacy MS5614T voltage DAC GPIO interface retained for reference. */
/* ===================== DAC1(SPI3)?? ===================== */
#define DAC1_CS_PORT      GPIOB
#define DAC1_CS_PIN       GPIO_PIN_12
#define DAC1_FS_PORT      GPIOB
#define DAC1_FS_PIN       GPIO_PIN_14
#define DAC1_LDAC_PORT    GPIOC
#define DAC1_LDAC_PIN     GPIO_PIN_9
#define DAC1_PD_PORT      GPIOA
#define DAC1_PD_PIN       GPIO_PIN_8

#define DAC1_CS_LOW()    	HAL_GPIO_WritePin(DAC1_CS_PORT, DAC1_CS_PIN, GPIO_PIN_RESET)
#define DAC1_CS_HIGH()   	HAL_GPIO_WritePin(DAC1_CS_PORT, DAC1_CS_PIN, GPIO_PIN_SET)
#define DAC1_FS_LOW()    	HAL_GPIO_WritePin(DAC1_FS_PORT, DAC1_FS_PIN, GPIO_PIN_RESET)
#define DAC1_FS_HIGH()   	HAL_GPIO_WritePin(DAC1_FS_PORT, DAC1_FS_PIN, GPIO_PIN_SET)
#define DAC1_LDAC_LOW()  	HAL_GPIO_WritePin(DAC1_LDAC_PORT, DAC1_LDAC_PIN, GPIO_PIN_RESET)
#define DAC1_PD_LOW()    	HAL_GPIO_WritePin(DAC1_PD_PORT, DAC1_PD_PIN, GPIO_PIN_RESET)
#define DAC1_PD_HIGH()   	HAL_GPIO_WritePin(DAC1_PD_PORT, DAC1_PD_PIN, GPIO_PIN_SET)

/* ===================== DAC2(SPI2)?? ===================== */
#define DAC2_CS_PORT      GPIOA
#define DAC2_CS_PIN       GPIO_PIN_15
#define DAC2_FS_PORT      GPIOC
#define DAC2_FS_PIN       GPIO_PIN_11
#define DAC2_LDAC_PORT    GPIOD
#define DAC2_LDAC_PIN     GPIO_PIN_2
#define DAC2_PD_PORT      GPIOB
#define DAC2_PD_PIN       GPIO_PIN_3

#define DAC2_CS_LOW()    	HAL_GPIO_WritePin(DAC2_CS_PORT, DAC2_CS_PIN, GPIO_PIN_RESET)
#define DAC2_CS_HIGH()   	HAL_GPIO_WritePin(DAC2_CS_PORT, DAC2_CS_PIN, GPIO_PIN_SET)
#define DAC2_FS_LOW()    	HAL_GPIO_WritePin(DAC2_FS_PORT, DAC2_FS_PIN, GPIO_PIN_RESET)
#define DAC2_FS_HIGH()   	HAL_GPIO_WritePin(DAC2_FS_PORT, DAC2_FS_PIN, GPIO_PIN_SET)
#define DAC2_LDAC_LOW()  	HAL_GPIO_WritePin(DAC2_LDAC_PORT, DAC2_LDAC_PIN, GPIO_PIN_RESET)
#define DAC2_PD_LOW()    	HAL_GPIO_WritePin(DAC2_PD_PORT, DAC2_PD_PIN, GPIO_PIN_RESET)
#define DAC2_PD_HIGH()   	HAL_GPIO_WritePin(DAC2_PD_PORT, DAC2_PD_PIN, GPIO_PIN_SET)

#endif

#define IDAC_7BIT_ADDR 		0x58

/* PI11210 CLR is active high and is routed to STM32 PB5. */
#define PI11210_CLR_PORT       GPIOB
#define PI11210_CLR_PIN        GPIO_PIN_5

#define GAIN 3357
#define SOA 3357

#if 0  /* Legacy MS5614T frame storage. */
extern uint16_t frame;
#endif
extern uint16_t adcData;

extern uint8_t codeBuf[2];
extern uint8_t readBuf[2];
extern uint16_t IDACData[5];
extern uint16_t prevDAC[5];
extern uint16_t uADCOriginvalues[4];

extern float tempData;
extern uint16_t tempInt;
extern uint16_t tempDec;

extern uint16_t start_wave;
extern uint16_t end_wave;

extern HAL_StatusTypeDef dacRet;

#if 0  /* Legacy MS5614T write APIs retained for reference. */
/* ?1? DAC */
void MS5614T_SetCode(MS5614T_Channel_t channel, uint16_t code,
                     MS5614T_Speed_t speed, MS5614T_Power_t power);

/* ?2? DAC */
void MS5614T2_SetCode(MS5614T_Channel_t channel, uint16_t code,
                      MS5614T_Speed_t speed, MS5614T_Power_t power);
#endif

HAL_StatusTypeDef PI11210_Init(void);
void PI11210_Process(void);
void PI11210_EmergencyShutter(void);
PI11210_Status_t PI11210_GetStatus(void);
HAL_StatusTypeDef PI11210_SetCode(PI11210_Channeld_t channel, uint16_t code);
HAL_StatusTypeDef PI11210_SetSOAShutter(uint8_t shutterEnabled);

void modify_table_loop(void);
void ApplyWorkState(uint8_t newState);
void checkTemp(uint8_t mode);
void ClearRxBuff(void);

void write_ms5614t_table(void);
void sampleVoltage(void);
void sampleVoltageStable(uint16_t i);
void sampleVoltagePrecision(uint16_t i);
void sampleTemperature(void);
uint8_t sendTxBuffer(int dac_size, int p1, int p2, int p3, int p4);
void getFilterDiff(void);
void getWaveRange(void);

void write_ms5614t_manual(void);
void ClearTxBuff(void);
void scanWave(void);
#if 0  /* Legacy MS5614T scan path retained for reference. */
void scanWave_U(void);
#endif
void scanWave_I(void);

void write_ms5614t_extra(void);
void singleValue(void);
#if 0  /* Legacy MS5614T single-value path retained for reference. */
void singleValue_U(void);
#endif
void singleValue_I(void);
void checkRT(void);
void runLaserSwitchTest(void);

void delay_us(__IO uint32_t us);
void delay_ms(__IO uint32_t ms);
void delay_s(__IO uint32_t s);
void short_delay(volatile uint32_t n);

#ifdef __cplusplus
}
#endif

#endif
