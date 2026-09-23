#include "main.h"

#include <math.h>
#include <string.h>

#define M1820_CONVERSION_TIME_MS          12U
#define M1820_SAMPLE_PERIOD_MS            250U
#define M1820_RETRY_PERIOD_MS             50U
#define M1820_STALE_TIMEOUT_MS            1500U
/* Main services thermal control on both sides of a scan frame.  This guard
 * prevents those adjacent calls from executing two 1-Wire slots in the same
 * inter-frame budget. */
#define M1820_COOPERATIVE_GUARD_MS         2U
#define FAN_RPM_UPDATE_MS                 1000U
#define FAN_PULSES_PER_REVOLUTION         2U
#define FAN_TACH_MIN_RPM                  300U
#define FAN_TACH_FAULT_WINDOWS            3U

#define FAN_TARGET_C                      33.0f
#define FAN_PID_BASE_PERCENT              52.0f
#define FAN_PID_KP                        24.0f
#define FAN_PID_KI                        0.55f
#define FAN_PID_KD                        2.5f
#define FAN_PID_INTEGRAL_MIN              (-30.0f)
#define FAN_PID_INTEGRAL_MAX              20.0f
#define FAN_MIN_EFFECTIVE_PERCENT         25U
#define FAN_DERIVATIVE_FILTER_TAU_S        1.5f
#define FAN_NORMAL_RISE_PERCENT_PER_S      12.0f
#define FAN_NORMAL_FALL_PERCENT_PER_S      6.0f
#define FAN_COMMAND_DEADBAND_PERCENT       0.35f

typedef enum
{
    M1820_IDLE = 0,
    M1820_CONVERT_RESET,
    M1820_CONVERT_SKIP_ROM,
    M1820_CONVERT_COMMAND,
    M1820_WAIT_CONVERSION,
    M1820_READ_RESET,
    M1820_READ_SKIP_ROM,
    M1820_READ_COMMAND,
    M1820_READ_SCRATCHPAD
} M1820State;

uint16_t temperature = 0U;
uint32_t pwm_buffer[ARR_1] = {0U};

static M1820State sensorState = M1820_IDLE;
static uint32_t conversionReadyMs = 0U;
static uint32_t nextConversionMs = 0U;
static uint32_t nextOneWireStepMs = 0U;
static uint32_t lastValidTemperatureMs = 0U;
static uint8_t sensorScratchpad[9] = {0U};
static uint8_t sensorScratchpadIndex = 0U;
static uint8_t consecutiveSensorErrors = 0U;
static uint8_t temperatureValid = 0U;
static float filteredTemperatureC = 0.0f;
static float temperatureHistory[3] = {0.0f, 0.0f, 0.0f};
static uint8_t temperatureHistoryCount = 0U;
static uint8_t temperatureHistoryIndex = 0U;

static float pidIntegral = 0.0f;
static float previousFilteredTemperatureC = 0.0f;
static float filteredDerivativeCPerS = 0.0f;
static float fanCommandPercent = 100.0f;
static uint32_t previousPidMs = 0U;
static uint8_t fanCoolingPercent = 100U;

static volatile uint32_t fanFgEdges = 0U;
static uint32_t fanFgEdgesSnapshot = 0U;
static uint32_t fanRpmUpdateMs = 0U;
static uint16_t fanRpm = 0U;
static uint8_t fanTachLowWindows = 0U;
static uint8_t fanTachFault = 0U;
static uint8_t fanPwmFault = 0U;

static void ApplyFanCommand(float requestedPercent, float dt,
        uint8_t forceFull);

static uint32_t EnterCritical(void)
{
    uint32_t primask = __get_PRIMASK();
    __disable_irq();
    return primask;
}

static void ExitCritical(uint32_t primask)
{
    if(primask == 0U) __enable_irq();
}

static uint8_t TimeReached(uint32_t now, uint32_t deadline)
{
    return ((int32_t)(now - deadline) >= 0) ? 1U : 0U;
}

void DQ_IN(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = TEMP_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(TEMP_PORT, &GPIO_InitStruct);
}

void DQ_OUT(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = TEMP_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    HAL_GPIO_Init(TEMP_PORT, &GPIO_InitStruct);
}

uint8_t M1820Z_Reset(void)
{
    uint8_t present;
    uint32_t primask = EnterCritical();

    DQ_OUT();
    HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_RESET);
    delay_us(500U);
    HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_SET);
    DQ_IN();
    delay_us(70U);
    present = (HAL_GPIO_ReadPin(TEMP_PORT, TEMP_PIN) == GPIO_PIN_RESET) ? 1U : 0U;
    delay_us(410U);
    DQ_OUT();
    HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_SET);

    ExitCritical(primask);
    return present;
}

void M1820Z_WriteBit(uint8_t bit)
{
    uint32_t primask = EnterCritical();
    DQ_OUT();
    HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_RESET);
    if(bit)
    {
        delay_us(3U);
        HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_SET);
        delay_us(67U);
    }
    else
    {
        delay_us(65U);
        HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_SET);
        delay_us(5U);
    }
    ExitCritical(primask);
}

uint8_t M1820Z_ReadBit(void)
{
    uint8_t bit;
    uint32_t primask = EnterCritical();
    DQ_OUT();
    HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_RESET);
    delay_us(2U);
    HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_SET);
    DQ_IN();
    delay_us(10U);
    bit = (HAL_GPIO_ReadPin(TEMP_PORT, TEMP_PIN) == GPIO_PIN_SET) ? 1U : 0U;
    delay_us(50U);
    DQ_OUT();
    HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_SET);
    ExitCritical(primask);
    return bit;
}

void M1820Z_WriteByte(uint8_t data)
{
    for(uint8_t index = 0U; index < 8U; index++)
    {
        M1820Z_WriteBit(data & 0x01U);
        data >>= 1;
    }
}

uint8_t M1820Z_ReadByte(void)
{
    uint8_t data = 0U;
    for(uint8_t index = 0U; index < 8U; index++)
    {
        if(M1820Z_ReadBit()) data |= (uint8_t)(1U << index);
    }
    return data;
}

static uint8_t M1820Crc8(const uint8_t *data, uint8_t length)
{
    uint8_t crc = 0U;
    for(uint8_t index = 0U; index < length; index++)
    {
        uint8_t value = data[index];
        for(uint8_t bit = 0U; bit < 8U; bit++)
        {
            uint8_t mix = (crc ^ value) & 0x01U;
            crc >>= 1;
            if(mix) crc ^= 0x8CU;
            value >>= 1;
        }
    }
    return crc;
}

static uint8_t M1820DecodeTemperature(const uint8_t *scratchpad,
        float *temperatureC)
{
    int16_t raw;

    if(scratchpad == NULL || temperatureC == NULL) return 0U;
    if(M1820Crc8(scratchpad, 8U) != scratchpad[8]) return 0U;
    raw = (int16_t)(((uint16_t)scratchpad[1] << 8) | scratchpad[0]);
    *temperatureC = 40.0f + (float)raw / 256.0f;
    if(!isfinite(*temperatureC) || *temperatureC < -70.0f || *temperatureC > 150.0f)
        return 0U;
    return 1U;
}

static void M1820ScheduleNextStep(void)
{
    /* Schedule from the end of the time slot.  Besides bounding service to
     * one primitive per frame gap, this provides ample 1-Wire recovery-high
     * time between reset/byte operations. */
    nextOneWireStepMs = HAL_GetTick() + M1820_COOPERATIVE_GUARD_MS;
}

static void M1820RecordError(uint32_t now)
{
    sensorState = M1820_IDLE;
    sensorScratchpadIndex = 0U;
    if(consecutiveSensorErrors < 255U) consecutiveSensorErrors++;
    if(consecutiveSensorErrors >= 3U) temperatureValid = 0U;
    if(!temperatureValid) ApplyFanCommand(100.0f, 0.25f, 1U);
    nextConversionMs = now + M1820_RETRY_PERIOD_MS;
    M1820ScheduleNextStep();
}

static float MedianTemperature(void)
{
    float sorted[3];
    uint8_t count = temperatureHistoryCount;
    memcpy(sorted, temperatureHistory, count * sizeof(float));
    for(uint8_t index = 1U; index < count; index++)
    {
        float value = sorted[index];
        uint8_t position = index;
        while(position > 0U && sorted[position - 1U] > value)
        {
            sorted[position] = sorted[position - 1U];
            position--;
        }
        sorted[position] = value;
    }
    return sorted[count / 2U];
}

void Set_Soft_PWM_Duty(uint8_t cooling_percent)
{
    if(cooling_percent > 100U) cooling_percent = 100U;
    fanCoolingPercent = cooling_percent;

    /* PC2 high turns Q11 on and pulls J16 PWM low.  The fan's full-speed
     * command is the opposite state: Q11 off, J16 pulled high by R129. */
    for(uint16_t index = 0U; index < ARR_1; index++)
    {
        if(index < cooling_percent)
            pwm_buffer[index] = ((uint32_t)FAN_PIN << 16U); /* PC2 reset/open */
        else
            pwm_buffer[index] = (uint32_t)FAN_PIN;         /* PC2 set/pull low */
    }
}

static void ApplyFanCommand(float requestedPercent, float dt, uint8_t forceFull)
{
    float delta;
    float maximumDelta;
    uint8_t roundedPercent;

    if(requestedPercent < (float)FAN_MIN_EFFECTIVE_PERCENT)
        requestedPercent = (float)FAN_MIN_EFFECTIVE_PERCENT;
    if(requestedPercent > 100.0f) requestedPercent = 100.0f;

    if(forceFull)
    {
        /* Temperature/sensor safety must not be delayed by the comfort ramp. */
        fanCommandPercent = 100.0f;
    }
    else
    {
        delta = requestedPercent - fanCommandPercent;
        if(delta > FAN_COMMAND_DEADBAND_PERCENT)
        {
            maximumDelta = FAN_NORMAL_RISE_PERCENT_PER_S * dt;
            fanCommandPercent += (delta > maximumDelta) ? maximumDelta : delta;
        }
        else if(delta < -FAN_COMMAND_DEADBAND_PERCENT)
        {
            maximumDelta = FAN_NORMAL_FALL_PERCENT_PER_S * dt;
            fanCommandPercent += ((-delta) > maximumDelta) ? -maximumDelta : delta;
        }
    }

    if(fanCommandPercent < (float)FAN_MIN_EFFECTIVE_PERCENT)
        fanCommandPercent = (float)FAN_MIN_EFFECTIVE_PERCENT;
    if(fanCommandPercent > 100.0f) fanCommandPercent = 100.0f;
    roundedPercent = (uint8_t)(fanCommandPercent + 0.5f);
    if(roundedPercent != fanCoolingPercent) Set_Soft_PWM_Duty(roundedPercent);
}

static void ApplyPid(uint32_t now)
{
    float dt = 0.25f;
    float rawDerivative = 0.0f;
    float derivativeAlpha;
    float error;
    float candidateIntegral;
    float proportionalAndDerivative;
    float unsaturatedOutput;
    float output;

    if(previousPidMs != 0U)
    {
        uint32_t elapsed = now - previousPidMs;
        if(elapsed > 0U && elapsed < 5000U)
        {
            dt = (float)elapsed / 1000.0f;
            rawDerivative = (filteredTemperatureC - previousFilteredTemperatureC) / dt;
        }
        else
        {
            /* Do not turn an old temperature into a derivative spike. */
            filteredDerivativeCPerS = 0.0f;
        }
    }
    previousPidMs = now;
    previousFilteredTemperatureC = filteredTemperatureC;

    derivativeAlpha = dt / (FAN_DERIVATIVE_FILTER_TAU_S + dt);
    filteredDerivativeCPerS += derivativeAlpha
            * (rawDerivative - filteredDerivativeCPerS);

    error = filteredTemperatureC - FAN_TARGET_C;
    candidateIntegral = pidIntegral + error * dt;
    if(candidateIntegral < FAN_PID_INTEGRAL_MIN)
        candidateIntegral = FAN_PID_INTEGRAL_MIN;
    if(candidateIntegral > FAN_PID_INTEGRAL_MAX)
        candidateIntegral = FAN_PID_INTEGRAL_MAX;

    proportionalAndDerivative = FAN_PID_BASE_PERCENT
           + FAN_PID_KP * error
           + FAN_PID_KD * filteredDerivativeCPerS;
    unsaturatedOutput = proportionalAndDerivative
           + FAN_PID_KI * candidateIntegral;

    /* Conditional integration prevents a long minimum/maximum interval from
     * storing an integral term that would later release as a speed step. */
    if((unsaturatedOutput < 100.0f || error < 0.0f)
       && (unsaturatedOutput > (float)FAN_MIN_EFFECTIVE_PERCENT || error > 0.0f))
        pidIntegral = candidateIntegral;

    output = proportionalAndDerivative + FAN_PID_KI * pidIntegral;
    ApplyFanCommand(output, dt, 0U);
}

static void UpdateFanRpm(uint32_t now)
{
    uint32_t elapsed = now - fanRpmUpdateMs;
    uint32_t edges;
    uint32_t delta;
    if(fanRpmUpdateMs == 0U)
    {
        fanRpmUpdateMs = now;
        fanFgEdgesSnapshot = fanFgEdges;
        return;
    }
    if(elapsed < FAN_RPM_UPDATE_MS) return;

    edges = fanFgEdges;
    delta = edges - fanFgEdgesSnapshot;
    fanFgEdgesSnapshot = edges;
    fanRpmUpdateMs = now;
    fanRpm = (uint16_t)((delta * 60000UL)
            / ((uint32_t)FAN_PULSES_PER_REVOLUTION * elapsed));

    if(fanCoolingPercent >= 50U && fanRpm < FAN_TACH_MIN_RPM)
    {
        if(fanTachLowWindows < 255U) fanTachLowWindows++;
        if(fanTachLowWindows >= FAN_TACH_FAULT_WINDOWS) fanTachFault = 1U;
    }
    else
    {
        fanTachLowWindows = 0U;
        fanTachFault = 0U;
    }
}

void ThermalControl_Init(void)
{
    uint32_t now = HAL_GetTick();
    DQ_OUT();
    HAL_GPIO_WritePin(TEMP_PORT, TEMP_PIN, GPIO_PIN_SET);
    sensorState = M1820_IDLE;
    nextConversionMs = now;
    nextOneWireStepMs = now;
    sensorScratchpadIndex = 0U;
    memset(sensorScratchpad, 0, sizeof(sensorScratchpad));
    lastValidTemperatureMs = now;
    fanRpmUpdateMs = now;
    fanPwmFault = 0U;
    pidIntegral = 0.0f;
    filteredDerivativeCPerS = 0.0f;
    fanCommandPercent = 100.0f;
    previousPidMs = 0U;
    Set_Soft_PWM_Duty(100U);
}

static void ThermalControl_UpdateSafety(uint32_t now)
{
    UpdateFanRpm(now);
    if(temperatureValid && (now - lastValidTemperatureMs) > M1820_STALE_TIMEOUT_MS)
    {
        temperatureValid = 0U;
        ApplyFanCommand(100.0f, 0.25f, 1U);
    }
}

void ThermalControl_SafetyTick(void)
{
    /* This path is safe to call between laser points.  In particular it does
     * not enter the interrupt-masked 1-Wire reset/read slots (up to about
     * 6.5 ms for a scratchpad transaction). */
    ThermalControl_UpdateSafety(HAL_GetTick());
}

void ThermalControl_Process(void)
{
    uint32_t now = HAL_GetTick();
    float sample;

    ThermalControl_UpdateSafety(now);
    if(!TimeReached(now, nextOneWireStepMs)) return;

    /* Exactly one switch arm below may enter a timed 1-Wire primitive.  The
     * slowest primitive is reset: 500 + 70 + 410 = 980 us nominal.  A byte is
     * 8 x 70 us (write) or 8 x 62 us (read).  Bus-high pauses between calls
     * are legal 1-Wire recovery time, so the complete transaction need not
     * monopolise a 6.5 ms frame boundary. */
    switch(sensorState)
    {
        case M1820_IDLE:
            if(TimeReached(now, nextConversionMs))
                sensorState = M1820_CONVERT_RESET;
            return;

        case M1820_CONVERT_RESET:
            if(!M1820Z_Reset())
            {
                M1820RecordError(now);
                return;
            }
            sensorState = M1820_CONVERT_SKIP_ROM;
            M1820ScheduleNextStep();
            return;

        case M1820_CONVERT_SKIP_ROM:
            M1820Z_WriteByte(SKIP_ROM);
            sensorState = M1820_CONVERT_COMMAND;
            M1820ScheduleNextStep();
            return;

        case M1820_CONVERT_COMMAND:
            M1820Z_WriteByte(CONVERT_T);
            conversionReadyMs = HAL_GetTick() + M1820_CONVERSION_TIME_MS;
            sensorState = M1820_WAIT_CONVERSION;
            M1820ScheduleNextStep();
            return;

        case M1820_WAIT_CONVERSION:
            if(TimeReached(now, conversionReadyMs))
                sensorState = M1820_READ_RESET;
            return;

        case M1820_READ_RESET:
            if(!M1820Z_Reset())
            {
                M1820RecordError(now);
                return;
            }
            sensorState = M1820_READ_SKIP_ROM;
            M1820ScheduleNextStep();
            return;

        case M1820_READ_SKIP_ROM:
            M1820Z_WriteByte(SKIP_ROM);
            sensorState = M1820_READ_COMMAND;
            M1820ScheduleNextStep();
            return;

        case M1820_READ_COMMAND:
            M1820Z_WriteByte(READ_SCRATCHPAD);
            sensorScratchpadIndex = 0U;
            sensorState = M1820_READ_SCRATCHPAD;
            M1820ScheduleNextStep();
            return;

        case M1820_READ_SCRATCHPAD:
            sensorScratchpad[sensorScratchpadIndex++] = M1820Z_ReadByte();
            M1820ScheduleNextStep();
            if(sensorScratchpadIndex < sizeof(sensorScratchpad)) return;

            sensorState = M1820_IDLE;
            sensorScratchpadIndex = 0U;
            if(!M1820DecodeTemperature(sensorScratchpad, &sample))
            {
                M1820RecordError(now);
                return;
            }

            consecutiveSensorErrors = 0U;
            temperatureValid = 1U;
            lastValidTemperatureMs = now;
            temperatureHistory[temperatureHistoryIndex] = sample;
            temperatureHistoryIndex = (temperatureHistoryIndex + 1U) % 3U;
            if(temperatureHistoryCount < 3U) temperatureHistoryCount++;
            filteredTemperatureC = MedianTemperature();
            temperature = (uint16_t)(filteredTemperatureC * 100.0f + 0.5f);
            ApplyPid(now);
            nextConversionMs = now + M1820_SAMPLE_PERIOD_MS;
            return;

        default:
            M1820RecordError(now);
            return;
    }
}

void ThermalControl_OnFanFgEdge(void)
{
    fanFgEdges++;
}

void ThermalControl_OnFanPwmFault(void)
{
    /* PC2 low turns the inverter MOSFET off, so the fan connector's pull-up
     * commands full speed even when the timer/DMA waveform is unavailable. */
    fanPwmFault = 1U;
    fanCommandPercent = 100.0f;
    fanCoolingPercent = 100U;
    HAL_GPIO_WritePin(FAN_PORT, FAN_PIN, GPIO_PIN_RESET);
}

float ThermalControl_GetTemperatureC(void)
{
    return filteredTemperatureC;
}

uint8_t ThermalControl_IsTemperatureValid(void)
{
    return temperatureValid;
}

uint16_t ThermalControl_GetFanDutyPermille(void)
{
    return (uint16_t)fanCoolingPercent * 10U;
}

uint16_t ThermalControl_GetFanRpm(void)
{
    return fanRpm;
}

uint16_t ThermalControl_GetStatusFlags(void)
{
    uint16_t flags = THERMAL_FLAG_PID_ACTIVE;
    if(temperatureValid) flags |= THERMAL_FLAG_SENSOR_VALID;
    else flags |= THERMAL_FLAG_SENSOR_FAULT;
    if(temperatureValid && filteredTemperatureC >= FAN_TARGET_C)
        flags |= THERMAL_FLAG_ABOVE_TARGET;
    if(fanTachFault) flags |= THERMAL_FLAG_FAN_TACH_FAULT;
    if(fanPwmFault) flags |= THERMAL_FLAG_FAN_PWM_FAULT;
    return flags;
}

uint16_t ThermalControl_GetSensorAgeMs(void)
{
    uint32_t age;
    if(!temperatureValid) return 0xFFFFU;
    age = HAL_GetTick() - lastValidTemperatureMs;
    return (age > 0xFFFEU) ? 0xFFFEU : (uint16_t)age;
}

float M1820Z_GetTmp(void)
{
    /* Legacy API retained for callers outside the scan path.  It now returns
     * only the last CRC-validated, range-checked reading. */
    ThermalControl_Process();
    return ThermalControl_GetTemperatureC();
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if(GPIO_Pin == FAN_FG_PIN) ThermalControl_OnFanFgEdge();
}
