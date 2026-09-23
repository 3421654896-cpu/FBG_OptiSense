/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : usbd_cdc_if.c
  * @version        : v1.0_Cube
  * @brief          : Usb device for Virtual Com Port.
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

/* Includes ------------------------------------------------------------------*/
#include "usbd_cdc_if.h"

/* USER CODE BEGIN INCLUDE */
#include "main.h"

/* USER CODE END INCLUDE */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* Private variables ---------------------------------------------------------*/

/* USER CODE END PV */

/** @addtogroup STM32_USB_OTG_DEVICE_LIBRARY
  * @brief Usb device library.
  * @{
  */

/** @addtogroup USBD_CDC_IF
  * @{
  */

/** @defgroup USBD_CDC_IF_Private_TypesDefinitions USBD_CDC_IF_Private_TypesDefinitions
  * @brief Private types.
  * @{
  */

/* USER CODE BEGIN PRIVATE_TYPES */

/* USER CODE END PRIVATE_TYPES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Defines USBD_CDC_IF_Private_Defines
  * @brief Private defines.
  * @{
  */

/* USER CODE BEGIN PRIVATE_DEFINES */
/* USER CODE END PRIVATE_DEFINES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Macros USBD_CDC_IF_Private_Macros
  * @brief Private macros.
  * @{
  */

/* USER CODE BEGIN PRIVATE_MACRO */

/* USER CODE END PRIVATE_MACRO */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_Variables USBD_CDC_IF_Private_Variables
  * @brief Private variables.
  * @{
  */
/* Create buffer for reception and transmission           */
/* It's up to user to redefine and/or remove those define */
/** Received data over USB are stored in this buffer      */
uint8_t UserRxBufferFS[APP_RX_DATA_SIZE];

/** Data to send over USB CDC are stored in this buffer   */
uint8_t UserTxBufferFS[APP_TX_DATA_SIZE];

/* USER CODE BEGIN PRIVATE_VARIABLES */

/* USB CDC ACM line coding. */
static uint8_t cdcLineCoding[7] = {
#if BW20_RECOVERY_BRIDGE
  0x00, 0xC2, 0x01, 0x00,
#else
  0x80, 0x84, 0x1E, 0x00,
#endif
  0x00,
  0x00,
  0x08
};
static uint8_t usbAssemblyBuffer[USART_RX_SIZE];

/* USER CODE END PRIVATE_VARIABLES */

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Exported_Variables USBD_CDC_IF_Exported_Variables
  * @brief Public variables.
  * @{
  */

extern USBD_HandleTypeDef hUsbDeviceFS;

/* USER CODE BEGIN EXPORTED_VARIABLES */
volatile uint32_t usbFrameLen = 0;
volatile uint8_t usbCdcHostOpen = 0U;
#if BW20_RECOVERY_BRIDGE
extern volatile uint32_t bw20BridgeBaud;
#endif

/* USER CODE END EXPORTED_VARIABLES */

void CDC_ReleaseHostControl_FS(void)
{
  /* A physical cable removal cannot deliver the normal DTR-close request.
   * Clear only IRQ-safe scalar state here; the cooperative main loop resets
   * the transmit queue, stops the old scan and closes the SOA shutter. */
  usbCdcHostOpen = 0U;
  usbFrameLen = 0U;
}

/**
  * @}
  */

/** @defgroup USBD_CDC_IF_Private_FunctionPrototypes USBD_CDC_IF_Private_FunctionPrototypes
  * @brief Private functions declaration.
  * @{
  */

static int8_t CDC_Init_FS(void);
static int8_t CDC_DeInit_FS(void);
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length);
static int8_t CDC_Receive_FS(uint8_t* pbuf, uint32_t *Len);

/* USER CODE BEGIN PRIVATE_FUNCTIONS_DECLARATION */

/* USER CODE END PRIVATE_FUNCTIONS_DECLARATION */

/**
  * @}
  */

USBD_CDC_ItfTypeDef USBD_Interface_fops_FS =
{
  CDC_Init_FS,
  CDC_DeInit_FS,
  CDC_Control_FS,
  CDC_Receive_FS
};

/* Private functions ---------------------------------------------------------*/
/**
  * @brief  Initializes the CDC media low layer over the FS USB IP
  * @retval USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Init_FS(void)
{
  /* USER CODE BEGIN 3 */
  usbCdcHostOpen = 0U;
  usbFrameLen = 0U;
  USB_ResetQueue();
  /* Set Application Buffers */
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, UserTxBufferFS, 0);
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, UserRxBufferFS);
  return (USBD_OK);
  /* USER CODE END 3 */
}

/**
  * @brief  DeInitializes the CDC media low layer
  * @retval USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_DeInit_FS(void)
{
  /* USER CODE BEGIN 4 */
  usbCdcHostOpen = 0U;
  usbFrameLen = 0U;
  USB_ResetQueue();
  return (USBD_OK);
  /* USER CODE END 4 */
}

/**
  * @brief  Manage the CDC class requests
  * @param  cmd: Command code
  * @param  pbuf: Buffer containing command data (request parameters)
  * @param  length: Number of data to be sent (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Control_FS(uint8_t cmd, uint8_t* pbuf, uint16_t length)
{
  /* USER CODE BEGIN 5 */
  switch(cmd)
  {
    case CDC_SEND_ENCAPSULATED_COMMAND:

    break;

    case CDC_GET_ENCAPSULATED_RESPONSE:

    break;

    case CDC_SET_COMM_FEATURE:

    break;

    case CDC_GET_COMM_FEATURE:

    break;

    case CDC_CLEAR_COMM_FEATURE:

    break;

  /*******************************************************************************/
  /* Line Coding Structure                                                       */
  /*-----------------------------------------------------------------------------*/
  /* Offset | Field       | Size | Value  | Description                          */
  /* 0      | dwDTERate   |   4  | Number |Data terminal rate, in bits per second*/
  /* 4      | bCharFormat |   1  | Number | Stop bits                            */
  /*                                        0 - 1 Stop bit                       */
  /*                                        1 - 1.5 Stop bits                    */
  /*                                        2 - 2 Stop bits                      */
  /* 5      | bParityType |  1   | Number | Parity                               */
  /*                                        0 - None                             */
  /*                                        1 - Odd                              */
  /*                                        2 - Even                             */
  /*                                        3 - Mark                             */
  /*                                        4 - Space                            */
  /* 6      | bDataBits  |   1   | Number Data bits (5, 6, 7, 8 or 16).          */
  /*******************************************************************************/
    case CDC_SET_LINE_CODING:
      if (length >= sizeof(cdcLineCoding))
      {
        memcpy(cdcLineCoding, pbuf, sizeof(cdcLineCoding));
#if BW20_RECOVERY_BRIDGE
        {
          uint32_t requestedBaud = (uint32_t)pbuf[0] |
                                   ((uint32_t)pbuf[1] << 8) |
                                   ((uint32_t)pbuf[2] << 16) |
                                   ((uint32_t)pbuf[3] << 24);
          if(requestedBaud >= 9600U && requestedBaud <= 2000000U)
            bw20BridgeBaud = requestedBaud;
        }
#endif
      }
    break;

    case CDC_GET_LINE_CODING:
      if (length >= sizeof(cdcLineCoding))
      {
        memcpy(pbuf, cdcLineCoding, sizeof(cdcLineCoding));
      }
    break;

    case CDC_SET_CONTROL_LINE_STATE:
      if (pbuf != NULL)
      {
        const USBD_SetupReqTypedef *request = (const USBD_SetupReqTypedef *)(const void *)pbuf;
        usbCdcHostOpen = (request->wValue & 0x0001U) ? 1U : 0U;
        if(!usbCdcHostOpen)
        {
          usbFrameLen = 0U;
          USB_ResetQueue();
        }
      }
    break;

    case CDC_SEND_BREAK:

    break;

  default:
    break;
  }

  return (USBD_OK);
  /* USER CODE END 5 */
}

/**
  * @brief  Data received over USB OUT endpoint are sent over CDC interface
  *         through this function.
  *
  *         @note
  *         This function will issue a NAK packet on any OUT packet received on
  *         USB endpoint until exiting this function. If you exit this function
  *         before transfer is complete on CDC interface (ie. using DMA controller)
  *         it will result in receiving more data while previous ones are still
  *         not sent.
  *
  * @param  Buf: Buffer of data to be received
  * @param  Len: Number of data received (in bytes)
  * @retval Result of the operation: USBD_OK if all operations are OK else USBD_FAIL
  */
static int8_t CDC_Receive_FS(uint8_t* Buf, uint32_t *Len)
{
  /* USER CODE BEGIN 6 */
  if(Buf == NULL || Len == NULL) return (USBD_FAIL);
#if BW20_RECOVERY_BRIDGE
	if(*Len > 0U)
		HAL_UART_Transmit(&huart6, Buf, (uint16_t)*Len, 5000U);
#else
	/* Some Windows USB-CDC clients can deliver bulk OUT data before the
	 * SET_CONTROL_LINE_STATE(DTR) callback reaches this interface.  Treat an
	 * actual, complete host command as authoritative evidence that local control
	 * is active; the normal DTR-close callback still releases the board back to
	 * remote operation when the port is closed. */
	if (*Len > 0U)
	{
		usbCdcHostOpen = 1U;
	}
	if (*Len > USART_RX_SIZE || usbFrameLen > USART_RX_SIZE - *Len)
	{
		/* Never append a later packet to the tail of an overflowing frame. */
		usbFrameLen = 0U;
	}
	else
	{
			memcpy(&usbAssemblyBuffer[usbFrameLen], Buf, *Len);
			usbFrameLen += *Len;

			if (usbFrameLen == USART_RX_SIZE)
			{
					/* aRxBuffer is a one-frame mailbox.  Do not tear a command that
					 * the main loop is still executing; a complete excess command is
					 * deliberately dropped and can be retried by the host. */
					if(ReceEndFlag == 0U)
					{
						memcpy(aRxBuffer, usbAssemblyBuffer, USART_RX_SIZE);
						ReceEndFlag = 1U;
					}
					usbFrameLen = 0U;
			}
	}
#endif
	
  USBD_CDC_SetRxBuffer(&hUsbDeviceFS, &Buf[0]);
  USBD_CDC_ReceivePacket(&hUsbDeviceFS);
  return (USBD_OK);
  /* USER CODE END 6 */
}

/**
  * @brief  CDC_Transmit_FS
  *         Data to send over USB IN endpoint are sent over CDC interface
  *         through this function.
  *         @note
  *
  *
  * @param  Buf: Buffer of data to be sent
  * @param  Len: Number of data to be sent (in bytes)
  * @retval USBD_OK if all operations are OK else USBD_FAIL or USBD_BUSY
  */
uint8_t CDC_Transmit_FS(uint8_t* Buf, uint16_t Len)
{
  uint8_t result = USBD_OK;
  /* USER CODE BEGIN 7 */
  USBD_CDC_HandleTypeDef *hcdc = (USBD_CDC_HandleTypeDef*)hUsbDeviceFS.pClassData;
  if (Buf == NULL || Len == 0U || hcdc == NULL || hUsbDeviceFS.dev_state != USBD_STATE_CONFIGURED){
    return USBD_FAIL;
  }
  if (hcdc->TxState != 0U){
    return USBD_BUSY;
  }
  USBD_CDC_SetTxBuffer(&hUsbDeviceFS, Buf, Len);
  result = USBD_CDC_TransmitPacket(&hUsbDeviceFS);
  /* USER CODE END 7 */
  return result;
}

/* USER CODE BEGIN PRIVATE_FUNCTIONS_IMPLEMENTATION */

/* USER CODE END PRIVATE_FUNCTIONS_IMPLEMENTATION */

/**
  * @}
  */

/**
  * @}
  */
