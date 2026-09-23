#ifndef __ADC_H
#define __ADC_H

/* Includes ------------------------------------------------------------------*/
#include "stm32f2xx.h"
#include "datastruct.h"
#include <stdio.h>

#define ADC_CS_PORT					GPIOA
#define ADC_CS_PIN					GPIO_PIN_4
#define ADC_GPIO_PORT 			GPIOC
#define ADC_GPIO_PIN 				GPIO_PIN_4

#define ADC_CS_LOW()				HAL_GPIO_WritePin(ADC_CS_PORT, ADC_CS_PIN, GPIO_PIN_RESET)
#define ADC_CS_HIGH()				HAL_GPIO_WritePin(ADC_CS_PORT, ADC_CS_PIN, GPIO_PIN_SET)

#define ADC_RESET_FRAME			(0x2 << 12) | (0x1 << 11) | (0x1 << 10)
#define ADC_ENTER_FRAME			(0x2 << 12) | (0x1 << 11)
#define ADC_LOOP_FRAME			(0x2 << 12)

#define STABLECOUNT					1
#define STABLERANGE					4
#define WINDOW_SIZE 				25
#define QUEUE_SIZE 				  400

extern uint16_t adcSPI;
extern uint32_t adcCount;
extern uint16_t adcStable;
extern uint16_t adcUnstable;
extern uint16_t adcQueue[WINDOW_SIZE];
extern uint16_t adcUnstableList[Number];

void Reset_ADC_Queue(void);
void ADC_LOOP_SPI_Init(void);
void ADC_MANUAL_SPI_Init(void);
uint16_t ADC_SPI_Cmd(uint16_t cmdF);
uint16_t ADC_Write_Read(uint8_t ch);
void ADC_ReadTwo(uint16_t values[2]);
void ADC_ReadFour(uint16_t values[4]);
void ADC_ReadMask(uint8_t channelMask, uint16_t values[4]);
void ADC_ReadSix(uint16_t values[6]);
uint16_t ADC_Write_Read_Stable(uint8_t ch, uint8_t* unstable, uint8_t multi);
void ADC_Loop_Start(void);
uint16_t ADC_Write_Loop(void);
uint8_t ADC_LastTransferOk(void);
uint32_t ADC_GetSpiErrorCount(void);

#endif /* __ADC_H */
