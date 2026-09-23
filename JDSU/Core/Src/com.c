#include "main.h"
#include "usbd_cdc_if.h"
#include "wifi_transport.h"

extern USBD_HandleTypeDef hUsbDeviceFS;

//UART1 RX
uint8_t getData = 0;
uint8_t lastGet = 0;
volatile uint8_t ReceEndFlag = 0;
__IO uint8_t uhRxCounter = 0;
uint8_t aRxBuffer[USART_RX_SIZE] = {0};
uint8_t uartFrame[USART_RX_SIZE] = {0};

//UART6 RX
uint8_t atReceEndFlag = 0;
uint8_t atRxBuffer[AT_RX_SIZE] = {0};
uint8_t atFrame[AT_RX_SIZE] = {0};

//UART1 TX
volatile uint8_t txHead = 0;
volatile uint8_t txTail = 0;
uint16_t txCount = 0;
volatile uint8_t dma_transfer_complete = 1;
uint8_t txQueue[TX_QUEUE_SIZE][USART_TX_SIZE] = {0};
uint16_t txLen[TX_QUEUE_SIZE] = {0};
static volatile uint8_t uartQueueTxActive = 0U;
static volatile uint8_t usbQueueTxActive = 0U;
static volatile uint8_t usbQueueServiceBusy = 0U;
static volatile uint32_t txQueueDropCount = 0U;

//UART6 TX
uint8_t atTxHead = 0;
uint8_t atTxTail = 0;
uint8_t at_transfer_complete = 1;
uint8_t atTxQueue[TX_QUEUE_SIZE][USART_TX_SIZE] = {0};
uint16_t atTxLen[USART_TX_SIZE] = {0};

static uint32_t QueueEnterCritical(void)
{
	uint32_t primask = __get_PRIMASK();
	__disable_irq();
	return primask;
}

static void QueueExitCritical(uint32_t primask)
{
	if(primask == 0U) __enable_irq();
}

static uint8_t QueueCopy(const uint8_t *data, uint16_t len)
{
	uint8_t next;
	uint32_t primask;
	if(data == NULL || len == 0U || len > USART_TX_SIZE) return 0U;

	primask = QueueEnterCritical();
	next = (uint8_t)((txHead + 1U) % TX_QUEUE_SIZE);
	if(next == txTail)
	{
		if(txQueueDropCount != 0xFFFFFFFFUL) txQueueDropCount++;
		QueueExitCritical(primask);
		return 0U;
	}
	memcpy(txQueue[txHead], data, len);
	txLen[txHead] = len;
	txHead = next;
	QueueExitCritical(primask);
	return 1U;
}

void USART_Queue_Send(uint8_t *data, uint16_t len)
{
	if(QueueCopy(data, len) && dma_transfer_complete) USART_DMA_Send();
}

void USART_DMA_Send(void)
{
	if(uartQueueTxActive) return;
	if(txTail == txHead)
	{
		dma_transfer_complete = 1U;
		return;
	}

	dma_transfer_complete = 0U;
	if(HAL_UART_Transmit_DMA(&huart1, txQueue[txTail], txLen[txTail]) == HAL_OK)
		uartQueueTxActive = 1U;
	else
	{
		txTail = (uint8_t)((txTail + 1U) % TX_QUEUE_SIZE);
		dma_transfer_complete = 1U;
	}
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart->Instance == USART1)
	{
		if(uartQueueTxActive)
		{
			txTail = (uint8_t)((txTail + 1U) % TX_QUEUE_SIZE);
			uartQueueTxActive = 0U;
		}
		USART_DMA_Send();
	}
	else if(huart->Instance == USART6)
	{
		at_transfer_complete = 1;
		WifiTransport_OnUartTxComplete();
	}
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if(huart->Instance == USART1)
	{
//		HAL_UART_DMAStop(&huart1);
		
		/* Native USB owns local control while DTR is open.  Also retain the
		 * existing mailbox until the main loop has consumed it. */
		if(!usbCdcHostOpen && ReceEndFlag == 0U)
		{
			memcpy(aRxBuffer, uartFrame, USART_RX_SIZE*sizeof(uint8_t));
			ReceEndFlag = 1U;
		}
//		USART_Queue_Send(aRxBuffer, USART_RX_SIZE);

		HAL_UART_Receive_DMA(&huart1, uartFrame, USART_RX_SIZE);
	}
}

void USB_Queue_Send(uint8_t *data, uint16_t len){
	if(!usbCdcHostOpen)
	{
		/* A same-LAN raw session mirrors the native CDC byte stream so the
		 * desktop can reuse exactly the USB pages and parsers. */
		if(WifiTransport_QueueRawFrame(data, len)) return;
		USB_ResetQueue();
		return;
	}
	if(QueueCopy(data, len) && dma_transfer_complete) USB_SendNext();
}

void USB_SendNext(void){
	USBD_CDC_HandleTypeDef *hcdc;
	uint32_t primask = QueueEnterCritical();
	if(usbQueueServiceBusy)
	{
		QueueExitCritical(primask);
		return;
	}
	usbQueueServiceBusy = 1U;
	QueueExitCritical(primask);

	if(!usbCdcHostOpen)
	{
		USB_ResetQueue();
		goto done;
	}
	hcdc = (USBD_CDC_HandleTypeDef *)hUsbDeviceFS.pClassData;
	if(hcdc == NULL)
	{
		dma_transfer_complete = 1U;
		goto done;
	}

	/* The tail remains owned until the asynchronous endpoint transfer really
	 * completes.  This prevents a wrapped producer from overwriting bytes that
	 * the USB peripheral is still reading. */
	if(usbQueueTxActive)
	{
		if(hcdc->TxState != 0U) goto done;
		txTail = (uint8_t)((txTail + 1U) % TX_QUEUE_SIZE);
		usbQueueTxActive = 0U;
	}

	while(txTail != txHead)
	{
		uint8_t result;
		if(hcdc->TxState != 0U)
		{
			dma_transfer_complete = 0U;
			goto done;
		}
		dma_transfer_complete = 0U;
		result = CDC_Transmit_FS(txQueue[txTail], txLen[txTail]);
		if(result == USBD_OK)
		{
			usbQueueTxActive = 1U;
			goto done;
		}
		/* A failed endpoint submission must not wedge the ring forever. */
		txTail = (uint8_t)((txTail + 1U) % TX_QUEUE_SIZE);
	}
	dma_transfer_complete = 1U;

done:
	primask = QueueEnterCritical();
	usbQueueServiceBusy = 0U;
	QueueExitCritical(primask);
}

void USB_ResetQueue(void)
{
	uint32_t primask = QueueEnterCritical();
	txTail = txHead;
	usbQueueTxActive = 0U;
	dma_transfer_complete = 1U;
	QueueExitCritical(primask);
}

void USB_IRQHandler_Process(void)
{
    USBD_CDC_HandleTypeDef *hcdc =
        (USBD_CDC_HandleTypeDef *)hUsbDeviceFS.pClassData;

    if (hcdc != NULL &&
        hcdc->TxState == 0)
    {
        USB_SendNext();
    }
}

void HAL_UARTEx_RxEventCallback(
        UART_HandleTypeDef *huart,
        uint16_t Size)
{
    if(huart->Instance == USART6)
    {
				WifiTransport_OnUartRx(atFrame, Size);
        atReceEndFlag = 1;

        HAL_UARTEx_ReceiveToIdle_DMA(
                &huart6,
                atFrame,
                AT_RX_SIZE
        );
				__HAL_DMA_DISABLE_IT(&hdma_usart6_rx, DMA_IT_HT);
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
	if(huart->Instance == USART1)
	{
		/* USART1 is the legacy fixed-size command path.  A framing/noise/
		 * overrun error terminates its receive DMA in the HAL; re-arm it here
		 * without discarding a complete command already owned by the main loop. */
		(void)HAL_UART_Receive_DMA(&huart1, uartFrame, USART_RX_SIZE);
	}
	else if(huart->Instance == USART6)
	{
		WifiTransport_OnUartError(huart->ErrorCode);
		HAL_UARTEx_ReceiveToIdle_DMA(&huart6, atFrame, AT_RX_SIZE);
		__HAL_DMA_DISABLE_IT(&hdma_usart6_rx, DMA_IT_HT);
	}
}
