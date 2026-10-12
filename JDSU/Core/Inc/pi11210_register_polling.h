#ifndef PI11210_REGISTER_POLLING_H
#define PI11210_REGISTER_POLLING_H

#include "stm32f2xx_hal.h"

/* Beta-only synchronous 7-bit, 8-bit register, two-byte transaction.
 * RM0033 EV5/EV6/EV8/EV8_2: retain all ACKs, BTF before STOP, and a separate
 * START/STOP for every DAC channel. No DMA, IRQ masking, bus overclocking,
 * reordered/merged registers or skipped identical values. */
__STATIC_FORCEINLINE HAL_StatusTypeDef PI11210_PollError(I2C_HandleTypeDef *bus,
                                                 uint32_t sr1)
{
    uint32_t error = 0U;
    if(sr1 & I2C_SR1_BERR) error |= HAL_I2C_ERROR_BERR;
    if(sr1 & I2C_SR1_ARLO) error |= HAL_I2C_ERROR_ARLO;
    if(sr1 & I2C_SR1_AF) error |= HAL_I2C_ERROR_AF;
    if(sr1 & I2C_SR1_OVR) error |= HAL_I2C_ERROR_OVR;
    if(error == 0U) return HAL_OK;
    bus->ErrorCode |= error;
    return HAL_ERROR;
}

__STATIC_FORCEINLINE HAL_StatusTypeDef PI11210_PollFlag(I2C_HandleTypeDef *bus,
        uint32_t mask, uint32_t started, uint32_t budget)
{
    for(;;)
    {
        uint32_t sr1 = bus->Instance->SR1;
        /* Errors take priority over a simultaneous ready flag, including the
         * final data byte's NACK. TXE alone does not confirm byte delivery. */
        if((sr1 & (I2C_SR1_BERR | I2C_SR1_ARLO | I2C_SR1_AF | I2C_SR1_OVR)) &&
           PI11210_PollError(bus, sr1) != HAL_OK) return HAL_ERROR;
        if((sr1 & mask) == mask) return HAL_OK;
        if((uint32_t)(DWT->CYCCNT - started) >= budget)
        {
            bus->ErrorCode |= HAL_I2C_ERROR_TIMEOUT;
            return HAL_TIMEOUT;
        }
    }
}

__STATIC_FORCEINLINE HAL_StatusTypeDef PI11210_PollIdle(I2C_HandleTypeDef *bus,
        uint32_t started, uint32_t budget)
{
    for(;;)
    {
        uint32_t sr1 = bus->Instance->SR1;
        if((sr1 & (I2C_SR1_BERR | I2C_SR1_ARLO | I2C_SR1_AF | I2C_SR1_OVR)) &&
           PI11210_PollError(bus, sr1) != HAL_OK) return HAL_ERROR;
        if(!(bus->Instance->CR1 & I2C_CR1_STOP) &&
           !(bus->Instance->SR2 & I2C_SR2_BUSY)) return HAL_OK;
        if((uint32_t)(DWT->CYCCNT - started) >= budget)
        {
            bus->ErrorCode |= HAL_I2C_ERROR_TIMEOUT;
            return HAL_TIMEOUT;
        }
    }
}

static HAL_StatusTypeDef PI11210_WriteRegisterPolling(I2C_HandleTypeDef *bus,
        uint8_t address, uint8_t reg, uint8_t data[2], uint32_t timeoutMs,
        uint8_t awaitStop)
{
    HAL_StatusTypeDef result;
    uint32_t started, budget;
    volatile uint32_t clear;
    if(bus == NULL || data == NULL || bus->Instance != I2C1 ||
       bus->State != HAL_I2C_STATE_READY || bus->Lock != HAL_UNLOCKED ||
       bus->Init.ClockSpeed != 400000U ||
       bus->Init.AddressingMode != I2C_ADDRESSINGMODE_7BIT ||
       (bus->Instance->CR2 & (I2C_CR2_DMAEN | I2C_CR2_ITERREN |
                             I2C_CR2_ITEVTEN | I2C_CR2_ITBUFEN)) ||
       !(CoreDebug->DEMCR & CoreDebug_DEMCR_TRCENA_Msk) ||
       !(DWT->CTRL & DWT_CTRL_CYCCNTENA_Msk)) return HAL_BUSY;
    if(timeoutMs == 0U || SystemCoreClock < 1000U ||
       timeoutMs > 0xFFFFFFFFUL / (SystemCoreClock / 1000U)) return HAL_ERROR;
    budget = timeoutMs * (SystemCoreClock / 1000U);
    started = DWT->CYCCNT;
    __HAL_LOCK(bus);
    bus->State = HAL_I2C_STATE_BUSY_TX;
    bus->Mode = HAL_I2C_MODE_MEM;
    bus->ErrorCode = HAL_I2C_ERROR_NONE;
    bus->pBuffPtr = data;
    bus->XferCount = 2U;
    bus->XferSize = 2U;

    result = PI11210_PollIdle(bus, started, budget);
    if(result != HAL_OK) goto finish;
    SET_BIT(bus->Instance->CR1, I2C_CR1_PE);
    CLEAR_BIT(bus->Instance->CR1, I2C_CR1_POS);
    SET_BIT(bus->Instance->CR1, I2C_CR1_START);
    result = PI11210_PollFlag(bus, I2C_SR1_SB, started, budget);
    if(result != HAL_OK) goto finish;
    bus->Instance->DR = address;
    result = PI11210_PollFlag(bus, I2C_SR1_ADDR, started, budget);
    if(result != HAL_OK) goto finish;
    /* The SR1 -> SR2 sequence is required to release the address phase. */
    clear = bus->Instance->SR1;
    clear = bus->Instance->SR2;
    (void)clear;
    result = PI11210_PollFlag(bus, I2C_SR1_TXE, started, budget);
    if(result != HAL_OK) goto finish;
    bus->Instance->DR = reg;
    result = PI11210_PollFlag(bus, I2C_SR1_TXE, started, budget);
    if(result != HAL_OK) goto finish;
    bus->Instance->DR = data[0];
    bus->pBuffPtr++;
    bus->XferCount--;
    result = PI11210_PollFlag(bus, I2C_SR1_TXE, started, budget);
    if(result != HAL_OK) goto finish;
    bus->Instance->DR = data[1];
    bus->pBuffPtr++;
    bus->XferCount--;
    result = PI11210_PollFlag(bus, I2C_SR1_BTF, started, budget);
    if(result != HAL_OK) goto finish;
    SET_BIT(bus->Instance->CR1, I2C_CR1_STOP);
    /* Like HAL Mem_Write, intermediate channels return after requesting STOP.
     * Next channel waits for actual bus idle before its separate START. The
     * last channel also confirms STOP/idle before the point's wait begins. */
    result = awaitStop ? PI11210_PollIdle(bus, started, budget) : HAL_OK;

finish:
    if(result != HAL_OK)
    {
        /* Do not send STOP after arbitration loss: another master owns it.
         * Caller asserts CLR before any bus recovery/retry. */
        if(!(bus->ErrorCode & HAL_I2C_ERROR_ARLO) &&
           (bus->Instance->SR2 & I2C_SR2_MSL))
            SET_BIT(bus->Instance->CR1, I2C_CR1_STOP);
        CLEAR_BIT(bus->Instance->SR1,
                  I2C_SR1_BERR | I2C_SR1_ARLO | I2C_SR1_AF | I2C_SR1_OVR);
    }
    bus->State = HAL_I2C_STATE_READY;
    bus->Mode = HAL_I2C_MODE_NONE;
    __HAL_UNLOCK(bus);
    return result;
}

#endif
