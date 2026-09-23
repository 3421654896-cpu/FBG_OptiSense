/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f2xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stm32f2xx.h"
#include "datastruct.h"
#include "ms5614t.h"
#include "peak_detection.h"
#include "stm32f2xx_it.h"
#include "adc.h"
#include "com.h"
#include "temperature.h"
#include "ch224q.h"

/* Set to 1 only for a temporary COM6 <-> BW20 AT diagnostic build. */
#define BW20_RECOVERY_BRIDGE 0
#include "stdlib.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */
typedef enum
{
    TABLE_STATE = 0,
    MANUAL_STATE = 1,
		EXTRA_STATE = 2,
		PRECISION_TABLE_STATE = 3,
}MAIN_LOOP_STATE;

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

#define   Para_Addr               0x08020000
#define   Wave_Addr               0x08030000
#define   Pwr_Addr								0x08040000

#define LED_MANUAL_PORT    				GPIOC
#define LED_MANUAL_PIN						GPIO_PIN_0
#define LED_TABLE_PORT						GPIOC
#define LED_TABLE_PIN							GPIO_PIN_1
#define LED_MANUAL_LOW()    			HAL_GPIO_WritePin(LED_MANUAL_PORT, LED_MANUAL_PIN, GPIO_PIN_RESET)
#define LED_MANUAL_HIGH()   			HAL_GPIO_WritePin(LED_MANUAL_PORT, LED_MANUAL_PIN, GPIO_PIN_SET)
#define LED_TABLE_LOW()						HAL_GPIO_WritePin(LED_TABLE_PORT, LED_TABLE_PIN, GPIO_PIN_RESET)
#define LED_TABLE_HIGH()					HAL_GPIO_WritePin(LED_TABLE_PORT, LED_TABLE_PIN, GPIO_PIN_SET)

#define UART_USB_SWAP_PORT				GPIOC
#define UART_USB_SWAP_PIN					GPIO_PIN_8
#define UART_ENABLE								HAL_GPIO_WritePin(UART_USB_SWAP_PORT, UART_USB_SWAP_PIN, GPIO_PIN_SET)
#define USB_ENABLE								HAL_GPIO_WritePin(UART_USB_SWAP_PORT, UART_USB_SWAP_PIN, GPIO_PIN_RESET)

#define CHOISE_0_A_PORT 					GPIOA
#define CHOISE_0_A_PIN 					  GPIO_PIN_3
#define CHOISE_0_B_PORT						GPIOB
#define CHOISE_0_B_PIN 					  GPIO_PIN_2
#define CHOISE_1_A_PORT 					GPIOC
#define CHOISE_1_A_PIN 					  GPIO_PIN_5
#define CHOISE_1_B_PORT						GPIOB
#define CHOISE_1_B_PIN 					  GPIO_PIN_0

#define CHOISE_0_A								1
#define CHOISE_0_B								0
#define CHOISE_1_A								1
#define CHOISE_1_B								0

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/

/* USER CODE BEGIN Private defines */
extern I2C_HandleTypeDef hi2c1;

extern SPI_HandleTypeDef hspi1;
#if 0  /* Legacy MS5614T voltage DAC interfaces are intentionally disabled. */
extern SPI_HandleTypeDef hspi2;
extern SPI_HandleTypeDef hspi3;
#endif

extern UART_HandleTypeDef huart4;
extern UART_HandleTypeDef huart1;
extern UART_HandleTypeDef huart6;
extern DMA_HandleTypeDef hdma_uart4_rx;
extern DMA_HandleTypeDef hdma_usart1_tx;
extern DMA_HandleTypeDef hdma_usart1_rx;
extern DMA_HandleTypeDef hdma_usart6_tx;
extern DMA_HandleTypeDef hdma_usart6_rx;

extern const uint16_t Wave_DAC[Number][5];
extern const uint16_t Wave_DATA[Number][2];
extern const uint16_t Pwr_DATA[Number][2];
extern int8_t unstableFlags[Number][4];

extern uint8_t txBuffer[PACK_SIZE];
extern uint8_t aTxBuffer[USART_TX_SIZE];

extern uint32_t wave_time;
extern uint8_t workState;
#if 0  /* The DAC target is fixed to PI11210. */
extern uint8_t dacTarget;
#endif

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
