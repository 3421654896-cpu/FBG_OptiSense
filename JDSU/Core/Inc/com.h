#ifndef __USART_H
#define __USART_H

#include "stm32f2xx.h"
#include "datastruct.h"

#define TX_QUEUE_SIZE 10
#define USART_TX_SIZE 20
#define USART_RX_SIZE 808
#define PACK_SIZE 2+2+Number*12+1+4+4*15+4+2

#define AT_RX_SIZE 		2048

extern uint8_t getData;
extern uint8_t lastGet;
extern volatile uint8_t ReceEndFlag;
extern uint8_t aRxBuffer[USART_RX_SIZE];
extern uint8_t uartFrame[USART_RX_SIZE];

extern uint8_t atReceEndFlag;
extern uint8_t atRxBuffer[AT_RX_SIZE];
extern uint8_t atFrame[AT_RX_SIZE];

extern volatile uint8_t txHead;
extern volatile uint8_t txTail;
extern uint16_t txCount;
extern volatile uint8_t dma_transfer_complete;
extern uint8_t txQueue[TX_QUEUE_SIZE][USART_TX_SIZE];
extern uint16_t txLen[TX_QUEUE_SIZE];

//extern uint8_t atTxHead;
//extern uint8_t atTxTail;
extern uint8_t at_transfer_complete;
//extern uint8_t atTxQueue[TX_QUEUE_SIZE][USART_TX_SIZE];
//extern uint16_t atTxLen[USART_TX_SIZE];

void USART_Queue_Send(uint8_t *data, uint16_t len);
void USART_DMA_Send(void);

void USB_Queue_Send(uint8_t *data, uint16_t len);
void USB_SendNext(void);
void USB_ResetQueue(void);
void USB_IRQHandler_Process(void);

#endif
