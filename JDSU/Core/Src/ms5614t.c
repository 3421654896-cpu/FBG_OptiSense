/* AI/ms5614t.c */
#include "main.h"
#include "usbd_cdc_if.h"
#include "wifi_transport.h"
#include "stress_table.h"
#include "stress_multirate.h"
#include "firmware_version.h"
#include "candidate_route_protocol.h"

#define DAC_DELAY 3
#define TRANSITION_STEPS 6

/* Populated TIA time constants from the schematic: CH0/1 ~= 4 ms and
 * CH2/3 ~= 0.2 ms.  Reconstruct the steady value from two samples instead
 * of waiting several milliseconds at every wavelength.
 */
#define FAST_ADC_FIRST_DELAY_US 50U
#define FAST_ADC_SPACING_US 600U
#define FAST_ADC_RECHECK_SPACING_US 1000U
/* Conditional slow rechecks were A/B tested at 30- and 40-code thresholds.
 * Switching between two reconstruction paths increased center jitter, so the
 * production stress scan deliberately keeps one deterministic two-sample path. */
#define FAST_ADC_RECHECK_DIFF_CODES 0xFFFFU
#define FAST_BOUNDARY_EXTRA_DELAY_US 200U
#define PRECISION_TABLE_START_INDEX 0U
#define PRECISION_ADC_BOUNDARY_EXTRA_DELAY_US 5000U
#define PRECISION_ADC_SAMPLE_COUNT 20U
#define PRECISION_ADC_TRIM_EACH_SIDE 2U
#define PRECISION_ADC_SPACING_US 115U
#define PRECISION_ADC_SATURATION_CODE 4080U
#define STRESS_GAIN_SWITCH_SETTLE_US 5000U
#define PD_FEEDBACK_SELECTOR_2K 0U
#define PD_FEEDBACK_SELECTOR_40K 1U
#define PD_FEEDBACK_SELECTOR_5K 2U /* R118/R119 drawing says 4K; BOM note says fitted 5K. */
#define PD_FEEDBACK_SELECTOR_20K 3U
#define STRESS_WAVEFORM_RANGE_CODES 40U /* ~=24.4 mV at the 2.5 V ADC input */
#define Q15_ONE 32768L
#define CH01_40K_ALPHA_Q15 25816L /* measured 600 us two-sample gain=4.71353 */
#define CH01_20K_ALPHA_Q15 24275L /* exp(-600 us / 2000 us), 20 kOhm x 100 nF */
#define CH01_5K_ALPHA_Q15 9869L /* exp(-600 us / 500 us), 5 kOhm x 100 nF */
#define CH01_2K_ALPHA_Q15 1631L /* exp(-600 us / 200 us), 2 kOhm x 100 nF */
#define CH01_40K_RECHECK_ALPHA_Q15 25520L /* exp(-1000 us / 4000 us) */
#define CH01_20K_RECHECK_ALPHA_Q15 19875L /* exp(-1000 us / 2000 us) */
#define CH01_5K_RECHECK_ALPHA_Q15 4435L /* exp(-1000 us / 500 us) */
#define CH01_2K_RECHECK_ALPHA_Q15 221L /* exp(-1000 us / 200 us) */
#define CH23_ALPHA_Q15 12055L /* exp(-200 us / 200 us) */
#define CH23_FOCUS_SPACING_US 100U
/* Least-squares steady-state weights for samples at 0, 200, 300 and 400 us
 * with the populated CH2/3 200 us RC time constant.  The weights sum to 1.0.
 */
#define CH23_FOCUS_W0_Q15 (-9345L)
#define CH23_FOCUS_W1_Q15 10158L
#define CH23_FOCUS_W2_Q15 14624L
#define CH23_FOCUS_W3_Q15 17331L

/* PI11210 register and safety policy.  Normal laser operation is limited by
 * the user-authorized 135 mA ceiling, still below the laser's 150 mA absolute
 * maximum.  A negative host SOA value means "laser off".  OP6 is connected
 * directly to the SOA and the board has no reverse-voltage clamp or monitor;
 * therefore CLR uses the zero-current Gate function, which can guarantee the
 * laser's 3 V reverse-voltage absolute maximum is not exceeded. */
#define PI11210_I2C_TIMEOUT_MS 5U
#define PI11210_STATUS_INTERVAL_MS 1000U
#define PI11210_REG_IDAC6_CONFIG 0x06U
#define PI11210_REG_STATUS 0x0EU
#define PI11210_REG_SOFT_CONTROL 0x0FU
#define PI11210_REG_IDAC6_POLARITY 0x1FU
#define PI11210_IDAC6_GATE_CONFIG 0x4000U
#define PI11210_SOFT_RESET_COMMAND 0x8000U
#define PI11210_IDAC6_OFF_CODE 0U
#define PI11210_STATUS_PRO_TEMP 0x8000U
#define PI11210_STATUS_OVR_TEMP 0x4000U
#define PI11210_STATUS_HI_TEMP 0x2000U
#define PI11210_STATUS_PART_MASK 0x0F00U
#define PI11210_STATUS_PART_VALUE 0x0A00U

/* Hidden diagnostic command 0xFF 0xFF 0x03 0x01.  It captures the raw optical
 * step response without the normal steady-state reconstruction.  Flags=5 is
 * the temporary-route variant: two complete host rows are CRC-bound, limited
 * to the same 145 mA/current ceilings as T45, and applied without writing any
 * table or flash.  This command is additive and does not change a normal path.
 */
#define SWITCH_TEST_SAMPLE_COUNT 33U
#define SWITCH_TEST_SOURCE_SETTLE_US 100000U
/* Hidden command FF FF 03 0B + direction/tag/"SOA1" captures the SOA gate
 * response on the board.  Samples are timestamped by DWT before any USB
 * transfer, so Windows CDC scheduling and the normal 20 ms monitor cadence do
 * not limit the result.  The early offsets resolve the analogue edge while the
 * sparse tail retains enough range to observe the populated 4 ms CH0/1 TIA and
 * slower PDR/PDT recovery without allocating a large stack buffer. */
#define SOA_RESPONSE_SAMPLE_COUNT 43U
#define SOA_RESPONSE_FRAME_HEADER 48U
#define SOA_RESPONSE_RECORD_LENGTH 20U
#define SOA_RESPONSE_FRAME_LENGTH \
		(SOA_RESPONSE_FRAME_HEADER + \
		 SOA_RESPONSE_SAMPLE_COUNT * SOA_RESPONSE_RECORD_LENGTH + 6U)
#define MAX_HOST_WAVE_DELAY_US 1000000U
#define STRESS_MULTIRATE_EXTENSION_VERSION_V3 3U
#define STRESS_MULTIRATE_EXTENSION_VERSION_V4 4U
#define STRESS_MULTIRATE_EXTENSION_VERSION_V5 5U
#define STRESS_MULTIRATE_EXTENSION_VERSION_MAX \
		STRESS_MULTIRATE_EXTENSION_VERSION_V5
#define STRESS_BOARD_STATUS_TIMING_LENGTH 36U
#define STRESS_BOARD_STATUS_RUNTIME_IDENTITY_LENGTH 16U
#define STRESS_BOARD_STATUS_BASE_LENGTH \
		(STRESS_BOARD_STATUS_TIMING_LENGTH + 4U + \
		 STRESS_BOARD_STATUS_RUNTIME_IDENTITY_LENGTH)
#define STRESS_BOARD_STATUS_MULTIRATE_LENGTH \
		(STRESS_BOARD_STATUS_TIMING_LENGTH + 10U + \
		 STRESS_MULTIRATE_FRESH_BITMAP_BYTES + \
		 (2U * STRESS_TABLE_POINT_COUNT) + \
		 STRESS_BOARD_STATUS_RUNTIME_IDENTITY_LENGTH)

/* Generated together with the temperature table.  These fallbacks keep an
 * older table header buildable, but every installed auto-selection package
 * writes explicit values derived from the equal-interval acquisition. */
#ifndef TEMPERATURE_FEEDBACK_SELECTOR_CH0
#define TEMPERATURE_FEEDBACK_SELECTOR_CH0 PD_FEEDBACK_SELECTOR_40K
#endif
#ifndef TEMPERATURE_FEEDBACK_SELECTOR_CH1
#define TEMPERATURE_FEEDBACK_SELECTOR_CH1 PD_FEEDBACK_SELECTOR_40K
#endif
#ifndef TEMPERATURE_ADC_SETTLE_US
#define TEMPERATURE_ADC_SETTLE_US 12000U
#endif

#if 0  /* Legacy MS5614T frame storage. */
uint16_t frame = 0;
#endif
uint16_t adcData = 0;
/* Optional global settling time.  Automatic per-point settling is used by default. */
uint32_t wave_time = 0;

uint8_t codeBuf[2] = {0};
uint8_t readBuf[2] = {0};
uint16_t IDACData[5] = {0};
uint16_t prevDAC[5] = {0};
uint16_t uADCOriginvalues[4] = {0};
int8_t unstableFlags[Number][4] = {0};

static uint8_t prevDACValid = 0;
/* ADC/unstable arrays are frame-local and start at zero for either table. */
static uint16_t activeScanPointIndex = 0U;
static uint8_t precisionLearnedSelector[2] = {
		TEMPERATURE_FEEDBACK_SELECTOR_CH0, TEMPERATURE_FEEDBACK_SELECTOR_CH1
};
static uint8_t precisionActiveSelector[2] = {
		TEMPERATURE_FEEDBACK_SELECTOR_CH0, TEMPERATURE_FEEDBACK_SELECTOR_CH1
};
static uint8_t stressActive20k[2] = {0U, 0U};
static uint8_t stressActiveSelector[2] = {
		PD_FEEDBACK_SELECTOR_40K, PD_FEEDBACK_SELECTOR_40K
};
/* A fixed stress selection is armed only by the explicit USB command while
 * EXTRA is active.  Entering EXTRA clears the arm; clients that do not send a
 * manual selection always use fixed 40 kOhm, never amplitude autoranging. */
static uint8_t stressManualFeedbackArmed = 0U;
static uint8_t stressRequestedSelector[2] = {
		PD_FEEDBACK_SELECTOR_40K, PD_FEEDBACK_SELECTOR_40K
};
/* Operator-selected fixed feedback state used only by EXTRA/equal-interval
 * acquisition.  It is reset whenever EXTRA mode is entered, then changed by
 * the explicit 0x06 command and acknowledged before the first ADC point. */
static uint8_t extraFeedbackSelector[2] = {
		PD_FEEDBACK_SELECTOR_40K, PD_FEEDBACK_SELECTOR_40K
};
static uint8_t stressGainSettlePending = 0U;
/* Stress mode performs exactly one all-channel discovery frame after entry.
 * The resulting mask remains locked until the mode is entered again; there is
 * intentionally no background patrol of disabled channels. */
static uint8_t stressChannelDiscoveryPending = 1U;
static uint8_t stressActiveChannelMask = 0U;
static uint16_t stressDiscoveryMinimum[4] = {0xFFFFU, 0xFFFFU, 0xFFFFU, 0xFFFFU};
static uint16_t stressDiscoveryMaximum[4] = {0U, 0U, 0U, 0U};
/* Hidden command FF FF 03 02 arms one complete normal stress frame for raw
 * CH1 diagnostics.  It records the real continuous-scan analogue history but
 * does not change the production estimator or add samples. */
static uint8_t stressRawCaptureRequested = 0U;
static uint8_t stressRawCaptureActive = 0U;
/* Destructive-to-cadence, opt-in diagnostic: retain the actual sparse prefix,
 * hold one installed target, then stop and shutter. Never a normal scan frame. */
static uint32_t stressHoldRequestedTag = 0U;
static uint32_t stressHoldActiveTag = 0U;
static uint16_t stressHoldRequestedPoint = 0U;
static uint16_t stressHoldActivePoint = 0U;
/* FF FF 03 04: run the real production reads at a naturally fresh target,
 * then hold that same DAC for a directly paired stable teacher. */
static uint8_t stressHoldRequestedProduction = 0U;
static uint8_t stressHoldActiveProduction = 0U;
static uint32_t stressPairOriginCycles = 0U;
static uint32_t stressPairStartCycles[3];
static uint32_t stressPairEndCycles[3];
static uint16_t stressPairCodes[3];
static uint16_t stressPairEstimate = 0U;
static uint8_t stressPairEstimateValid = 0U;
static uint8_t stressPairValidMask = 0U;
static uint8_t stressPairSampleMask = 0U;
/* Request kind 1 = legacy ForceMap; 2 = preserve current scheduler plan.
 * Separate pending/active tags prevent a later command retagging a frame. */
static uint32_t stressRawRequestedTag = 0U;
static uint32_t stressRawActiveTag = 0U;
static uint16_t stressRawFirst[STRESS_TABLE_POINT_COUNT] = {0U};
static uint16_t stressRawSecond[STRESS_TABLE_POINT_COUNT] = {0U};
static uint16_t stressRawSlow[STRESS_TABLE_POINT_COUNT] = {0U};
static uint16_t stressRawEstimate[STRESS_TABLE_POINT_COUNT] = {0U};
static uint8_t stressRawRecheck[STRESS_TABLE_POINT_COUNT] = {0U};
/* End-of-conversion offset from the frame start for every current CH1 value.
 * Cached rows stay UINT32_MAX.  The host must use these offsets instead of
 * treating MAP/SURVEY points as if they were sampled simultaneously. */
static uint32_t stressSampleOffsetUs[STRESS_TABLE_POINT_COUNT];
static uint8_t stressSampleTimingOverflow = 0U;

/* Multirate scanning is armed explicitly while EXTRA is safe/idle.  It is not
 * persisted and is cleared on every exit from stress mode, so an old client
 * always receives genuine MAP45 frames rather than unknowingly consuming
 * cached points as current samples. */
static StressMultirateState_t stressMultirateState = {0};
static uint8_t stressMultirateArmed = 0U;
/* Volatile session capability negotiated by the explicit EXTRA-mode arm.
 * Zero means that no reduced-profile session is authorised. */
static volatile uint8_t stressMultirateNegotiatedVersion = 0U;
static uint8_t stressMultirateMapPeriod = STRESS_MULTIRATE_DEFAULT_MAP_PERIOD;
static uint16_t stressMultirateActivateCodes =
		STRESS_MULTIRATE_DEFAULT_ACTIVATE_CODES;
static uint16_t stressMultirateReleaseCodes =
		STRESS_MULTIRATE_DEFAULT_RELEASE_CODES;

static void stressMultirateDisarmAndClear(void)
{
		stressHoldRequestedTag = 0U;
		stressHoldActiveTag = 0U;
		stressHoldRequestedProduction = 0U;
		stressHoldActiveProduction = 0U;
		stressRawCaptureRequested = 0U;
		stressRawCaptureActive = 0U;
		stressRawRequestedTag = 0U;
		stressRawActiveTag = 0U;
		stressMultirateArmed = 0U;
		stressMultirateNegotiatedVersion = 0U;
		StressMultirate_Configure(&stressMultirateState,
				0U,
				stressMultirateMapPeriod,
				stressMultirateActivateCodes,
				stressMultirateReleaseCodes);
		StressMultirate_SetSingleTrackEnabled(&stressMultirateState, 0U);
		StressMultirate_StartSession(&stressMultirateState);
}

static PI11210_Status_t pi11210Status = {0};
static uint32_t pi11210NextStatusTick = 0U;
/* Sequence carried by the optional USB/LAN raw-frame status extension.  It is
 * advanced only after a complete normal scan frame has been accepted by the
 * selected transport, so aborted acquisitions and the diagnostic raw-capture
 * replacement frame do not create false gaps on the host. */
static uint32_t scanFrameSequence = 0U;

typedef struct
{
		uint32_t startCycle;
		uint32_t startTick;
		uint32_t coreClockHz;
		uint8_t cycleCounterAvailable;
} ScanFrameTimer_t;

static const PI11210_Channeld_t tableDACChannels[5] = {
		IDAC5, IDAC6, IDAC1, IDAC4, IDAC7
};

/* Use the same transition path in stress, temperature, calibration and
 * single-value operation: GAIN, SOA, PHASE, WAVE_A, WAVE_B. */
static const uint8_t tableDACUpdateOrder[5] = {0, 1, 2, 3, 4};

static const uint32_t switchTestOffsetsUs[SWITCH_TEST_SAMPLE_COUNT] = {
		0U, 25U, 50U, 75U, 100U, 150U, 200U, 300U, 400U, 500U,
		650U, 800U, 1000U, 1250U, 1500U, 2000U, 2500U, 3000U,
		4000U, 5000U, 6000U, 8000U, 10000U, 12000U, 16000U, 20000U,
		25000U, 30000U, 40000U, 50000U, 60000U, 80000U, 100000U
};

static const uint32_t soaResponseOffsetsUs[SOA_RESPONSE_SAMPLE_COUNT] = {
		0U, 25U, 50U, 75U, 100U, 150U, 200U, 300U, 400U, 500U,
		650U, 800U, 1000U, 1250U, 1500U, 2000U, 2500U, 3000U,
		4000U, 5000U, 6000U, 8000U, 10000U, 12000U, 16000U,
		20000U, 25000U, 30000U, 40000U, 50000U, 60000U, 80000U,
		100000U, 125000U, 150000U, 175000U, 200000U, 250000U,
		300000U, 350000U, 400000U, 450000U, 500000U
};

float tempData = 0;
uint16_t tempInt = 0;
uint16_t tempDec = 0;

uint16_t start_wave = 0;
uint16_t end_wave = 0;

HAL_StatusTypeDef dacRet = HAL_TIMEOUT;
HAL_StatusTypeDef dacrRet = HAL_TIMEOUT;

void delay_us(__IO uint32_t us){
		uint32_t cyclesPerUs;
		if(us == 0U) return;
		/* SysTick is HAL's millisecond time base.  Reconfiguring/disabling it
		 * here used to freeze all non-blocking thermal, CH224Q and Wi-Fi
		 * deadlines after the first laser delay.  DWT provides an independent
		 * wrap-safe microsecond delay without disturbing HAL_GetTick(). */
		CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
		DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
		cyclesPerUs = SystemCoreClock / 1000000U;
		if(cyclesPerUs == 0U) cyclesPerUs = 1U;
		/* Keep each interval well below one CYCCNT wrap.  Besides making the
		 * public delay helpers correct for large values, this prevents the
		 * multiplication from overflowing at 120 MHz. */
		while(us > 0U)
		{
				uint32_t chunkUs = (us > 1000000U) ? 1000000U : us;
				uint32_t cycles = chunkUs * cyclesPerUs;
				uint32_t start = DWT->CYCCNT;
				while((uint32_t)(DWT->CYCCNT - start) < cycles) { }
				us -= chunkUs;
		}
}

void delay_ms(__IO uint32_t ms){
	while(ms > 0U)
	{
		uint32_t chunkMs = (ms > 1000U) ? 1000U : ms;
		delay_us(chunkMs * 1000U);
		ms -= chunkMs;
	}
}

void delay_s(__IO uint32_t s){
	while(s-- > 0U) delay_us(1000000U);
}

inline void short_delay(volatile uint32_t n)
{
    while (n--) __NOP();
}

static void ScanFrameTimer_Start(ScanFrameTimer_t *timer)
{
		uint32_t coreClockHz = SystemCoreClock;
		if(coreClockHz == 0U)
		{
				/* SystemCoreClock is normally refreshed during clock setup.  Recover a
				 * valid conversion factor if an alternate startup path omitted it. */
				SystemCoreClockUpdate();
				coreClockHz = SystemCoreClock;
		}
		if(coreClockHz == 0U) coreClockHz = 1U;

		CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
		DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
		__DSB();
		__ISB();
		timer->coreClockHz = coreClockHz;
		timer->startTick = HAL_GetTick();
		timer->startCycle = DWT->CYCCNT;
		timer->cycleCounterAvailable =
				((DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) == 0U) ? 1U : 0U;
}

static uint32_t ScanFrameTimer_ElapsedUs(const ScanFrameTimer_t *timer)
{
		const uint64_t cycleWrap = 0x100000000ULL;
		uint32_t elapsedMs = (uint32_t)(HAL_GetTick() - timer->startTick);
		uint64_t elapsedCycles;
		uint64_t tickEstimateCycles;
		uint64_t wholeSeconds;
		uint64_t remainderCycles;
		uint64_t elapsedUs;

		if(!timer->cycleCounterAvailable)
		{
				elapsedUs = (uint64_t)elapsedMs * 1000ULL;
				return (elapsedUs > 0xFFFFFFFFULL)
						? 0xFFFFFFFFU : (uint32_t)elapsedUs;
		}

		/* Unsigned subtraction is exact across one 32-bit CYCCNT wrap.  HAL's
		 * millisecond clock supplies the wrap count for a pathologically long
		 * scan that spans multiple ~35.8 s wraps at 120 MHz.  Rounding to the
		 * nearest wrap is unambiguous because HAL tick uncertainty is <1 ms. */
		elapsedCycles = (uint32_t)(DWT->CYCCNT - timer->startCycle);
		tickEstimateCycles =
				((uint64_t)elapsedMs * timer->coreClockHz + 500ULL) / 1000ULL;
		if(tickEstimateCycles > elapsedCycles)
		{
				uint64_t wraps =
						(tickEstimateCycles - elapsedCycles + cycleWrap / 2ULL)
						/ cycleWrap;
				elapsedCycles += wraps * cycleWrap;
		}

		/* Split quotient/remainder before multiplying by 1e6 so conversion
		 * remains overflow-safe even if a future table makes a very long scan. */
		wholeSeconds = elapsedCycles / timer->coreClockHz;
		remainderCycles = elapsedCycles % timer->coreClockHz;
		elapsedUs = wholeSeconds * 1000000ULL
				+ (remainderCycles * 1000000ULL + timer->coreClockHz / 2U)
				/ timer->coreClockHz;
		return (elapsedUs > 0xFFFFFFFFULL)
				? 0xFFFFFFFFU : (uint32_t)elapsedUs;
}

/* SPI2 ?? 16bit */
#if 0  /* Legacy MS5614T SPI write implementation retained for reference. */
static void SPI2_Send16(uint16_t data)
{
    HAL_SPI_Transmit(&hspi2, (uint8_t*)&data, 1, HAL_MAX_DELAY);
}

/* SPI3 ?? 16bit */
static void SPI3_Send16(uint16_t data)
{
    HAL_SPI_Transmit(&hspi3, (uint8_t*)&data, 1, HAL_MAX_DELAY);
}

/* ??:D15..D12 = A1 A0 PWR SPD, D11..D0 = code */
static uint16_t MakeFrame(MS5614T_Channel_t ch, uint16_t code, MS5614T_Speed_t spd, MS5614T_Power_t pwr)
{
    if (code > 4095u) code = 4095u;

    return (uint16_t)((((uint16_t)ch  & 0x03u) << 14) |
                      (((uint16_t)pwr & 0x01u) << 13) |
                      (((uint16_t)spd & 0x01u) << 12) |
                      (code & 0x0FFFu));
}

void MS5614T_SetCode(MS5614T_Channel_t ch, uint16_t code, MS5614T_Speed_t spd, MS5614T_Power_t pwr)
{
    frame = MakeFrame(ch, code, spd, pwr);

    if (pwr == MS5614T_POWERDOWN) DAC1_PD_LOW(); else DAC1_PD_HIGH();
    DAC1_LDAC_LOW();

    DAC1_FS_HIGH();
    short_delay(DAC_DELAY);

    DAC1_CS_LOW();
    short_delay(DAC_DELAY);

    DAC1_FS_LOW();
    short_delay(DAC_DELAY);

    SPI2_Send16(frame);

    short_delay(DAC_DELAY);
    DAC1_FS_HIGH();
    short_delay(DAC_DELAY);
    DAC1_CS_HIGH();
}

void MS5614T2_SetCode(MS5614T_Channel_t ch, uint16_t code, MS5614T_Speed_t spd, MS5614T_Power_t pwr)
{
		frame = MakeFrame(ch, code, spd, pwr);

    if (pwr == MS5614T_POWERDOWN) DAC2_PD_LOW(); else DAC2_PD_HIGH();
    DAC2_LDAC_LOW();

    DAC2_FS_HIGH();
    short_delay(DAC_DELAY);

    DAC2_CS_LOW();
    short_delay(DAC_DELAY);

    DAC2_FS_LOW();
    short_delay(DAC_DELAY);

    SPI3_Send16(frame);

    short_delay(DAC_DELAY);
    DAC2_FS_HIGH();
    short_delay(DAC_DELAY);
    DAC2_CS_HIGH();
}
#endif

static void PI11210_CountI2CError(void)
{
		if(pi11210Status.i2cErrorCount < 0xFFFFU)
				pi11210Status.i2cErrorCount++;
}

void PI11210_EmergencyShutter(void)
{
		/* CLR is independent of I2C.  Once register 0x06 has been configured it
		 * is the datasheet Gate function; before that its POR Ignore function is
		 * still harmless, and preserving the high level makes the next successful
		 * initialization remain gated. */
		HAL_GPIO_WritePin(PI11210_CLR_PORT, PI11210_CLR_PIN, GPIO_PIN_SET);
		pi11210Status.soaMode = PI11210_SOA_SHUTTER;
}

static void PI11210_ForceShutter(void)
{
		PI11210_EmergencyShutter();
}

static HAL_StatusTypeDef PI11210_RecoverI2CBus(void)
{
		GPIO_InitTypeDef gpio = {0};
		HAL_StatusTypeDef result;

		(void)HAL_I2C_DeInit(&hi2c1);
		__HAL_RCC_GPIOB_CLK_ENABLE();
		gpio.Pin = GPIO_PIN_6 | GPIO_PIN_7;
		gpio.Mode = GPIO_MODE_OUTPUT_OD;
		gpio.Pull = GPIO_PULLUP;
		gpio.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
		HAL_GPIO_Init(GPIOB, &gpio);
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6 | GPIO_PIN_7, GPIO_PIN_SET);
		delay_us(5U);

		/* Clock a slave out of a truncated byte, then generate an explicit STOP. */
		for(uint8_t pulse = 0U; pulse < 9U &&
				HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_7) == GPIO_PIN_RESET; pulse++)
		{
				HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET);
				delay_us(5U);
				HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET);
				delay_us(5U);
		}
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_7, GPIO_PIN_RESET);
		delay_us(5U);
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET);
		delay_us(5U);
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_7, GPIO_PIN_SET);
		delay_us(5U);

		HAL_GPIO_DeInit(GPIOB, GPIO_PIN_6 | GPIO_PIN_7);
		result = HAL_I2C_Init(&hi2c1);
		if(result == HAL_OK) pi11210Status.busRecovered = 1U;
		return result;
}

static HAL_StatusTypeDef PI11210_WriteRegisterRaw(uint8_t registerAddress,
		uint16_t value)
{
		HAL_StatusTypeDef result = HAL_ERROR;
		uint8_t data[2];
		data[0] = (uint8_t)(value >> 8);
		data[1] = (uint8_t)value;

		for(uint8_t attempt = 0U; attempt < 2U; attempt++)
		{
				result = HAL_I2C_Mem_Write(&hi2c1,
						(uint16_t)(IDAC_7BIT_ADDR << 1),
						(uint16_t)(registerAddress << 1),
						I2C_MEMADD_SIZE_8BIT, data, 2U,
						PI11210_I2C_TIMEOUT_MS);
				if(result == HAL_OK)
				{
						pi11210Status.online = 1U;
						return HAL_OK;
				}
				PI11210_CountI2CError();
				pi11210Status.online = 0U;
				if(attempt == 0U && PI11210_RecoverI2CBus() != HAL_OK) break;
		}
		pi11210Status.initialized = 0U;
		PI11210_ForceShutter();
		return result;
}

static HAL_StatusTypeDef PI11210_ReadStatusRaw(void)
{
		HAL_StatusTypeDef result = HAL_ERROR;
		uint8_t data[2] = {0U, 0U};

		for(uint8_t attempt = 0U; attempt < 2U; attempt++)
		{
				result = HAL_I2C_Mem_Read(&hi2c1,
						(uint16_t)(IDAC_7BIT_ADDR << 1),
						(uint16_t)(PI11210_REG_STATUS << 1),
						I2C_MEMADD_SIZE_8BIT, data, 2U,
						PI11210_I2C_TIMEOUT_MS);
				if(result == HAL_OK)
				{
						pi11210Status.rawStatus = ((uint16_t)data[0] << 8) | data[1];
						pi11210Status.online = 1U;
						pi11210Status.partValid =
								((pi11210Status.rawStatus & PI11210_STATUS_PART_MASK) ==
								 PI11210_STATUS_PART_VALUE) ? 1U : 0U;
						if(pi11210Status.partValid) return HAL_OK;
						pi11210Status.initialized = 0U;
						PI11210_ForceShutter();
						return HAL_ERROR;
				}
				PI11210_CountI2CError();
				pi11210Status.online = 0U;
				if(attempt == 0U && PI11210_RecoverI2CBus() != HAL_OK) break;
		}
		pi11210Status.initialized = 0U;
		PI11210_ForceShutter();
		return result;
}

HAL_StatusTypeDef PI11210_Init(void)
{
		HAL_StatusTypeDef result;
		uint8_t preserveShutter =
				(HAL_GPIO_ReadPin(PI11210_CLR_PORT, PI11210_CLR_PIN) == GPIO_PIN_SET)
				? 1U : 0U;
		pi11210Status.initialized = 0U;

		result = PI11210_ReadStatusRaw();
		if(result != HAL_OK)
		{
				/* A malformed or interrupted block transaction can leave the DAC
				 * acknowledging I2C while returning an invalid PART_ID.  Recover once
				 * with the datasheet-defined POR-equivalent software reset.  CLR is
				 * asserted first and kept asserted; reset also zeros every DAC code. */
				if(!pi11210Status.online || pi11210Status.partValid) return result;
				PI11210_ForceShutter();
				if(PI11210_WriteRegisterRaw(PI11210_REG_SOFT_CONTROL,
						PI11210_SOFT_RESET_COMMAND) != HAL_OK) return HAL_ERROR;
				HAL_Delay(1U);
				if(PI11210_ReadStatusRaw() != HAL_OK) return HAL_ERROR;
				preserveShutter = 1U;
		}
		/* Configure the datasheet-defined zero-current Gate.  CLR stays low during
		 * normal operation; no unmonitored negative SOA voltage is generated. */
		if(PI11210_WriteRegisterRaw(PI11210_REG_IDAC6_CONFIG,
				PI11210_IDAC6_GATE_CONFIG) != HAL_OK) return HAL_ERROR;
		if(pi11210Status.rawStatus &
				(PI11210_STATUS_PRO_TEMP | PI11210_STATUS_OVR_TEMP))
		{
				PI11210_ForceShutter();
				pi11210Status.initialized = 1U;
				return HAL_ERROR;
		}
		if(PI11210_WriteRegisterRaw(PI11210_REG_IDAC6_POLARITY, 0U) != HAL_OK)
				return HAL_ERROR;

		HAL_GPIO_WritePin(PI11210_CLR_PORT, PI11210_CLR_PIN,
				preserveShutter ? GPIO_PIN_SET : GPIO_PIN_RESET);
		pi11210Status.soaMode = preserveShutter
				? PI11210_SOA_SHUTTER : PI11210_SOA_SOURCE;
		pi11210Status.initialized = 1U;
		pi11210NextStatusTick = HAL_GetTick() + PI11210_STATUS_INTERVAL_MS;
		return HAL_OK;
}

static HAL_StatusTypeDef PI11210_EnsureReady(void)
{
		if(pi11210Status.initialized && pi11210Status.online &&
				pi11210Status.partValid) return HAL_OK;
		return PI11210_Init();
}

void PI11210_Process(void)
{
		uint32_t now = HAL_GetTick();
		if((int32_t)(now - pi11210NextStatusTick) < 0) return;
		pi11210NextStatusTick = now + PI11210_STATUS_INTERVAL_MS;
		if(PI11210_ReadStatusRaw() != HAL_OK)
		{
				pi11210Status.initialized = 0U;
				PI11210_ForceShutter();
				return;
		}
		if(pi11210Status.rawStatus &
				(PI11210_STATUS_PRO_TEMP | PI11210_STATUS_OVR_TEMP))
		{
				/* An active-high hardware shutter does not depend on another I2C
				 * transaction and immediately prevents continued optical output. */
				PI11210_ForceShutter();
		}
}

PI11210_Status_t PI11210_GetStatus(void)
{
		return pi11210Status;
}

static uint16_t PI11210_LimitCode(PI11210_Channeld_t channel, uint16_t code)
{
		uint16_t maximumCode;
		switch(channel)
		{
				/* 0xFFFF is full scale.  Floor every ratio so code quantisation
				 * cannot exceed the user/laser ceiling by even one LSB. */
				case IDAC5: maximumCode = 63351U; break; /* GAIN <= 145 mA */
				case IDAC6: maximumCode = 63351U; break; /* SOA source <= 145 mA */
				case IDAC1: maximumCode = 32767U; break; /* PHASE <= 10 mA */
				case IDAC4: maximumCode = 24575U; break; /* WAVE_A <= 30 mA */
				case IDAC7: maximumCode = 24575U; break; /* WAVE_B <= 30 mA */
				default: return 0U;
		}
		return (code > maximumCode) ? maximumCode : code;
}

HAL_StatusTypeDef PI11210_SetSOAShutter(uint8_t shutterEnabled)
{
		if(shutterEnabled)
		{
				HAL_StatusTypeDef result = PI11210_EnsureReady();
				/* Assert CLR even if status verification failed.  If the shutter was
				 * configured earlier, this remains the safest state during an I2C fault. */
				PI11210_ForceShutter();
				return result;
		}
		else
		{
				if(PI11210_EnsureReady() != HAL_OK) return HAL_ERROR;
				if(pi11210Status.rawStatus &
						(PI11210_STATUS_PRO_TEMP | PI11210_STATUS_OVR_TEMP)) return HAL_ERROR;
				HAL_GPIO_WritePin(PI11210_CLR_PORT, PI11210_CLR_PIN, GPIO_PIN_RESET);
				pi11210Status.soaMode = PI11210_SOA_SOURCE;
		}
		return HAL_OK;
}

HAL_StatusTypeDef PI11210_SetCode(PI11210_Channeld_t channel, uint16_t code)
{
		HAL_StatusTypeDef result;

		switch(channel)
		{
				case IDAC5:
				case IDAC6:
				case IDAC1:
				case IDAC4:
				case IDAC7:
						break;
				default: return HAL_ERROR;
		}
		code = PI11210_LimitCode(channel, code);
		if(PI11210_EnsureReady() != HAL_OK) return HAL_ERROR;
		if(pi11210Status.rawStatus &
				(PI11210_STATUS_PRO_TEMP | PI11210_STATUS_OVR_TEMP)) return HAL_ERROR;

		/* If the zero-current gate is active, update the positive SOA code first and only
		 * then release CLR.  This prevents a stale high source code from appearing
		 * during the off/source transition. */
		result = PI11210_WriteRegisterRaw((uint8_t)channel, code);
		dacRet = result;
		if(result == HAL_OK && channel == IDAC6 &&
				pi11210Status.soaMode == PI11210_SOA_SHUTTER)
		{
				HAL_GPIO_WritePin(PI11210_CLR_PORT, PI11210_CLR_PIN, GPIO_PIN_RESET);
				pi11210Status.soaMode = PI11210_SOA_SOURCE;
		}
		return result;
}

static HAL_StatusTypeDef PI11210_ApplyChangedTableCodes(void)
{
		uint8_t allWritesOk = 1;
		for(uint8_t orderIndex = 0; orderIndex < 5; orderIndex++)
		{
				uint8_t channelIndex = tableDACUpdateOrder[orderIndex];
				if(!prevDACValid || prevDAC[channelIndex] != IDACData[channelIndex])
				{
						if(PI11210_SetCode(tableDACChannels[channelIndex], IDACData[channelIndex]) == HAL_OK)
						{
								prevDAC[channelIndex] = IDACData[channelIndex];
						}
						else
						{
								allWritesOk = 0;
						}
				}
		}
		prevDACValid = allWritesOk;
		return allWritesOk ? HAL_OK : HAL_ERROR;
}

static HAL_StatusTypeDef PI11210_ApplyCalibrationCodes(const uint16_t codes[5])
{
		uint8_t allWritesOk = 1U;
		/* The wavelength table was measured with this complete five-write order.
		 * Always rewrite all channels so the optical path depends only on the
		 * requested target row; the mode-specific settling interval follows it. */
		for(uint8_t channelIndex = 0U; channelIndex < 5U; channelIndex++)
		{
				uint16_t code = codes[channelIndex];
				if(PI11210_SetCode(tableDACChannels[channelIndex], code) == HAL_OK)
				{
						prevDAC[channelIndex] = code;
				}
				else allWritesOk = 0U;
		}
		prevDACValid = allWritesOk;
		return allWritesOk ? HAL_OK : HAL_ERROR;
}

static HAL_StatusTypeDef PI11210_ApplyFullTableRowCalibrationOrder(uint16_t tableIndex)
{
		/* Match the known-good single-value/calibration command order exactly:
		 * GAIN, SOA, PHASE, WAVE_A, WAVE_B.  All five channels are rewritten on
		 * purpose because the laser optical state is path-dependent.
		 */
		for(uint8_t channelIndex = 0; channelIndex < 5; channelIndex++)
		{
				uint16_t code = Wave_DAC[tableIndex][channelIndex];
				IDACData[channelIndex] = code;
		}
		return PI11210_ApplyCalibrationCodes(Wave_DAC[tableIndex]);
}

static HAL_StatusTypeDef PI11210_ApplyStressRowCalibrationOrder(void)
{
		uint8_t allWritesOk = 1U;
		/* The high-power stress table was calibrated through single-value mode,
		 * which always writes GAIN, SOA, PHASE, WAVE_A and WAVE_B in this exact
		 * order.  Reproduce that path for every stress point.  The former
		 * changed-channel order left SOA until last and could sample a different
		 * transient optical state even though the five final DAC codes matched.
		 * An unchanged channel cannot alter the laser state, so skip only those
		 * writes to recover the scan-rate cost without changing the transition. */
		for(uint8_t channelIndex = 0U; channelIndex < 5U; channelIndex++)
		{
				uint16_t code = IDACData[channelIndex];
				if(!prevDACValid || prevDAC[channelIndex] != code)
				{
						if(PI11210_SetCode(tableDACChannels[channelIndex], code) == HAL_OK)
						{
								prevDAC[channelIndex] = code;
						}
						else allWritesOk = 0U;
				}
		}
		prevDACValid = allWritesOk;
		return allWritesOk ? HAL_OK : HAL_ERROR;
}

static void switchTestWriteU16(uint16_t *position, uint16_t value)
{
		txBuffer[(*position)++] = (uint8_t)(value >> 8);
		txBuffer[(*position)++] = (uint8_t)value;
}

static void switchTestWriteU32(uint16_t *position, uint32_t value)
{
		txBuffer[(*position)++] = (uint8_t)(value >> 24);
		txBuffer[(*position)++] = (uint8_t)(value >> 16);
		txBuffer[(*position)++] = (uint8_t)(value >> 8);
		txBuffer[(*position)++] = (uint8_t)value;
}

static uint8_t switchTestValidRow(uint16_t index)
{
		if(index >= Number) return 0U;
		return !((Wave_DAC[index][0] == 0xFFFFU) &&
					 (Wave_DAC[index][1] == 0xFFFFU) &&
					 (Wave_DAC[index][2] == 0xFFFFU));
}

static uint8_t isTuningSectionBoundary(uint16_t tableIndex)
{
		/* First point after every discontinuity in the active temperature table,
		 * plus the first point after frame wrap.  Deriving this from Wave_DATA
		 * keeps the extra settling delay correct when automatic peak selection
		 * changes either the peak count or the points assigned to each peak. */
		if(tableIndex == 0U) return 1U;
		if(tableIndex > 99U) return 0U;
		int32_t previousPm = (int32_t)Wave_DATA[tableIndex - 1U][0] * 1000L
				+ (int32_t)Wave_DATA[tableIndex - 1U][1];
		int32_t currentPm = (int32_t)Wave_DATA[tableIndex][0] * 1000L
				+ (int32_t)Wave_DATA[tableIndex][1];
		int32_t stepPm = currentPm - previousPm;
		return (stepPm < 0L || stepPm > 500L) ? 1U : 0U;
}

static uint8_t isStressSectionBoundary(uint16_t pointIndex)
{
		if(pointIndex == 0U) return 1U;
		if(pointIndex >= STRESS_TABLE_POINT_COUNT) return 0U;
		int32_t previousPm = (int32_t)Stress_Wave_DATA[pointIndex - 1U][0] * 1000L
				+ (int32_t)Stress_Wave_DATA[pointIndex - 1U][1];
		int32_t currentPm = (int32_t)Stress_Wave_DATA[pointIndex][0] * 1000L
				+ (int32_t)Stress_Wave_DATA[pointIndex][1];
		int32_t stepPm = currentPm - previousPm;
		return (stepPm < 0L || stepPm > 500L) ? 1U : 0U;
}

static void PD_SetFeedbackSelector(uint8_t channel, uint8_t selector)
{
		/* RS2255 selector is encoded as (B << 1) | A:
		 * 0/X0=2 kOhm, 1/X1=40 kOhm, 2/X2=fitted 5 kOhm,
		 * 3/X3=20 kOhm.
		 */
		GPIO_PinState a = (selector & 0x01U) ? GPIO_PIN_SET : GPIO_PIN_RESET;
		GPIO_PinState b = (selector & 0x02U) ? GPIO_PIN_SET : GPIO_PIN_RESET;
		if(channel == 0U)
		{
				HAL_GPIO_WritePin(CHOISE_0_A_PORT, CHOISE_0_A_PIN, a);
				HAL_GPIO_WritePin(CHOISE_0_B_PORT, CHOISE_0_B_PIN, b);
		}
		else if(channel == 1U)
		{
				HAL_GPIO_WritePin(CHOISE_1_A_PORT, CHOISE_1_A_PIN, a);
				HAL_GPIO_WritePin(CHOISE_1_B_PORT, CHOISE_1_B_PIN, b);
		}
}

static uint8_t PD_ReadFeedbackSelector(uint8_t channel)
{
		GPIO_PinState a = GPIO_PIN_RESET;
		GPIO_PinState b = GPIO_PIN_RESET;
		if(channel == 0U)
		{
			a = HAL_GPIO_ReadPin(CHOISE_0_A_PORT, CHOISE_0_A_PIN);
			b = HAL_GPIO_ReadPin(CHOISE_0_B_PORT, CHOISE_0_B_PIN);
		}
		else if(channel == 1U)
		{
			a = HAL_GPIO_ReadPin(CHOISE_1_A_PORT, CHOISE_1_A_PIN);
			b = HAL_GPIO_ReadPin(CHOISE_1_B_PORT, CHOISE_1_B_PIN);
		}
		return (uint8_t)(((b == GPIO_PIN_SET) ? 0x02U : 0x00U)
				| ((a == GPIO_PIN_SET) ? 0x01U : 0x00U));
}

static void PD_SetFeedback20k(uint8_t channel, uint8_t use20k)
{
		/* Existing stress/temperature automatic gain logic intentionally keeps
		 * its validated 40/20 kOhm hysteresis. */
		PD_SetFeedbackSelector(channel, use20k
				? PD_FEEDBACK_SELECTOR_20K : PD_FEEDBACK_SELECTOR_40K);
}

static void PD_RestoreFeedback40k(void)
{
		PD_SetFeedback20k(0U, 0U);
		PD_SetFeedback20k(1U, 0U);
}

static void stressResetChannelDiscovery(void)
{
		stressChannelDiscoveryPending = 1U;
		stressActiveChannelMask = 0U;
		for(uint8_t channel = 0U; channel < 4U; channel++)
		{
				stressDiscoveryMinimum[channel] = 0xFFFFU;
				stressDiscoveryMaximum[channel] = 0U;
		}
}

static uint8_t stressSamplingChannelMask(void)
{
		/* A multirate session is an explicit CH1-only measurement contract. */
		if(StressMultirate_IsEnabled(&stressMultirateState))
				return STRESS_MULTIRATE_CHANNEL_MASK;
		return stressChannelDiscoveryPending ? 0x0FU : stressActiveChannelMask;
}

static void stressObserveChannelSample(uint8_t channel, uint16_t value)
{
		if(!stressChannelDiscoveryPending || channel >= 4U) return;
		if(value < stressDiscoveryMinimum[channel]) stressDiscoveryMinimum[channel] = value;
		if(value > stressDiscoveryMaximum[channel]) stressDiscoveryMaximum[channel] = value;
}

static void stressFinishChannelDiscovery(void)
{
		if(StressMultirate_IsEnabled(&stressMultirateState))
		{
				stressActiveChannelMask = STRESS_MULTIRATE_CHANNEL_MASK;
				stressChannelDiscoveryPending = 0U;
				return;
		}
		if(!stressChannelDiscoveryPending) return;
		uint8_t detectedMask = 0U;
		for(uint8_t channel = 0U; channel < 4U; channel++)
		{
				uint16_t minimum = stressDiscoveryMinimum[channel];
				uint16_t maximum = stressDiscoveryMaximum[channel];
				if(minimum != 0xFFFFU && maximum >= minimum
						&& (uint16_t)(maximum - minimum) >= STRESS_WAVEFORM_RANGE_CODES)
				{
						detectedMask |= (uint8_t)(1U << channel);
				}
		}
		stressActiveChannelMask = detectedMask;
		stressChannelDiscoveryPending = 0U;
}

static void stressBeginFrame(void)
{
		stressHoldActiveTag = stressHoldRequestedTag;
		stressHoldActivePoint = stressHoldRequestedPoint;
		stressHoldActiveProduction = stressHoldRequestedProduction;
		stressHoldRequestedTag = 0U;
		stressHoldRequestedProduction = 0U;
		stressPairValidMask = 0U;
		stressPairSampleMask = 0U;
		stressPairEstimate = 0U;
		stressPairEstimateValid = 0U;
		/* Drop an interrupted previous capture; never complete it using a later
		 * plan. A fresh pending request is copied below for this frame only. */
		stressRawCaptureActive = 0U;
		stressRawActiveTag = 0U;
		memset(stressSampleOffsetUs, 0xFF, sizeof(stressSampleOffsetUs));
		stressSampleTimingOverflow = 0U;
		if(stressRawCaptureRequested)
		{
				stressRawCaptureActive = stressRawCaptureRequested;
				stressRawActiveTag = stressRawRequestedTag;
				stressRawCaptureRequested = 0U;
				memset(stressRawFirst, 0, sizeof(stressRawFirst));
				memset(stressRawSecond, 0, sizeof(stressRawSecond));
				memset(stressRawSlow, 0, sizeof(stressRawSlow));
				memset(stressRawEstimate, 0, sizeof(stressRawEstimate));
				memset(stressRawRecheck, 0, sizeof(stressRawRecheck));
				/* Only the legacy request forces a map. The opt-in v2 request
				 * records actual sparse history, and marks unmeasured rows stale. */
				if(stressRawCaptureActive == 1U)
						StressMultirate_ForceMap(&stressMultirateState);
		}
		if(stressChannelDiscoveryPending)
		{
				for(uint8_t channel = 0U; channel < 4U; channel++)
				{
						stressDiscoveryMinimum[channel] = 0xFFFFU;
						stressDiscoveryMaximum[channel] = 0U;
				}
		}
		if(stressGainSettlePending)
		{
				delay_us(STRESS_GAIN_SWITCH_SETTLE_US);
				stressGainSettlePending = 0U;
		}
		(void)StressMultirate_BeginFrame(&stressMultirateState);
}

static void stressAppendCachedCh1(uint16_t pointIndex)
{
		uint16_t cached = 0U;
		/* BeginFrame may choose a reduced profile only after all 45 cache slots
		 * are valid.  Keep a defensive zero fallback, but the freshness bitmap
		 * still marks this row as non-current. */
		(void)StressMultirate_GetCachedCh1(
				&stressMultirateState, pointIndex, &cached);
		for(uint8_t channel = 0U; channel < 4U; channel++)
		{
				uADCOriginvalues[channel] = (channel == 1U) ? cached : 0U;
				txBuffer[txCount++] = (channel == 1U) ? (uint8_t)(cached >> 8) : 0U;
				txBuffer[txCount++] = (channel == 1U) ? (uint8_t)cached : 0U;
		}
}

static int32_t stressFeedbackAlphaQ15(uint8_t selector, uint8_t recheck)
{
		switch(selector)
		{
				case PD_FEEDBACK_SELECTOR_20K:
						return recheck ? CH01_20K_RECHECK_ALPHA_Q15 : CH01_20K_ALPHA_Q15;
				case PD_FEEDBACK_SELECTOR_5K:
						return recheck ? CH01_5K_RECHECK_ALPHA_Q15 : CH01_5K_ALPHA_Q15;
				case PD_FEEDBACK_SELECTOR_2K:
						return recheck ? CH01_2K_RECHECK_ALPHA_Q15 : CH01_2K_ALPHA_Q15;
				case PD_FEEDBACK_SELECTOR_40K:
				default:
						return recheck ? CH01_40K_RECHECK_ALPHA_Q15 : CH01_40K_ALPHA_Q15;
		}
}

static void precisionApplySegmentGain(uint16_t tableIndex)
{
		(void)tableIndex;
		PD_SetFeedbackSelector(0U, precisionActiveSelector[0]);
		PD_SetFeedbackSelector(1U, precisionActiveSelector[1]);
}

static uint8_t precisionNextLowerGainSelector(uint8_t selector)
{
		/* Physical gain order: 40 kOhm -> 20 kOhm -> 5 kOhm -> 2 kOhm. */
		switch(selector)
		{
				case PD_FEEDBACK_SELECTOR_40K: return PD_FEEDBACK_SELECTOR_20K;
				case PD_FEEDBACK_SELECTOR_20K: return PD_FEEDBACK_SELECTOR_5K;
				case PD_FEEDBACK_SELECTOR_5K: return PD_FEEDBACK_SELECTOR_2K;
				default: return PD_FEEDBACK_SELECTOR_2K;
		}
}

static void precisionObserveSaturation(uint16_t tableIndex)
{
		(void)tableIndex;
		/* Inherit the exact CH0/CH1 selectors used by the equal-interval source
		 * spectrum.  Only a real ADC clip may lower a complete channel by one
		 * hardware step; the mixed discovery frame is discarded and repeated. */
		if(uADCOriginvalues[0] >= PRECISION_ADC_SATURATION_CODE)
		{
				precisionLearnedSelector[0] =
						precisionNextLowerGainSelector(precisionLearnedSelector[0]);
		}
		if(uADCOriginvalues[1] >= PRECISION_ADC_SATURATION_CODE)
		{
				precisionLearnedSelector[1] =
						precisionNextLowerGainSelector(precisionLearnedSelector[1]);
		}
}

static uint16_t precisionLegacy20kMask(uint8_t selector)
{
		if(selector != PD_FEEDBACK_SELECTOR_20K) return 0U;
		if(TEMPERATURE_SEGMENT_COUNT >= 16U) return 0xFFFFU;
		return (uint16_t)((1UL << TEMPERATURE_SEGMENT_COUNT) - 1UL);
}

void runLaserSwitchTest(void)
{
		uint16_t sourceIndex = ((uint16_t)aRxBuffer[4] << 8) | aRxBuffer[5];
		uint16_t targetIndex = ((uint16_t)aRxBuffer[6] << 8) | aRxBuffer[7];
		uint8_t useFullTable = aRxBuffer[8] & 0x01U;
		uint8_t useStressTable = aRxBuffer[8] & 0x02U;
		uint8_t useCustomCodes = aRxBuffer[8] & 0x04U;
		uint32_t diagnosticTag = ((uint32_t)aRxBuffer[9] << 24) |
		        ((uint32_t)aRxBuffer[10] << 16) |
		        ((uint32_t)aRxBuffer[11] << 8) | (uint32_t)aRxBuffer[12];
		uint16_t customSourceCodes[5] = {0};
		uint16_t customTargetCodes[5] = {0};
		static const uint16_t customLimits[5] = {
				63351U, 63351U, 32767U, 24575U, 24575U
		};
		const uint16_t *sourceCodes;
		const uint16_t *targetCodes;
		uint16_t baseline[6] = {0};
		uint16_t samples[6] = {0};
		uint16_t position = 0U;
		uint32_t cyclesPerUs = SystemCoreClock / 1000000U;
		uint32_t switchStart;
		uint32_t switchEnd;
		uint32_t targetCycle;

		/* v2 diagnostic is opt-in (flags=3) and references the installed stress
		 * rows.  Flags=5 accepts only two CRC-bound, current-limited temporary
		 * rows.  Legacy flags 0/1 retain the original short table test. */
		if(useCustomCodes)
		{
				uint32_t suppliedCrc = ((uint32_t)aRxBuffer[36] << 24) |
				        ((uint32_t)aRxBuffer[37] << 16) |
				        ((uint32_t)aRxBuffer[38] << 8) | (uint32_t)aRxBuffer[39];
				if(aRxBuffer[13] != 'C' || aRxBuffer[14] != '2' || aRxBuffer[15] != 'P' ||
				   CandidateRoute_Crc(aRxBuffer, 36U) != suppliedCrc)
				{
						ClearRxBuff();
						return;
				}
				for(uint8_t channel = 0U; channel < 5U; channel++)
				{
						customSourceCodes[channel] = CandidateRoute_U16(
								aRxBuffer + 16U + 2U * channel);
						customTargetCodes[channel] = CandidateRoute_U16(
								aRxBuffer + 26U + 2U * channel);
						if(customSourceCodes[channel] > customLimits[channel] ||
						   customTargetCodes[channel] > customLimits[channel])
						{
								ClearRxBuff();
								return;
						}
				}
				for(uint16_t index = 40U; index < 808U; index++)
				{
						if(aRxBuffer[index] != 0U) { ClearRxBuff(); return; }
				}
		}
		if((aRxBuffer[8] != 0U && aRxBuffer[8] != 1U && aRxBuffer[8] != 3U &&
		    aRxBuffer[8] != 5U) ||
		   (useStressTable && (sourceIndex >= STRESS_TABLE_POINT_COUNT ||
		                       targetIndex >= STRESS_TABLE_POINT_COUNT)) ||
		   (useCustomCodes && (sourceIndex > 2000U || targetIndex > 2000U)) ||
		   (!useStressTable && !useCustomCodes && (!switchTestValidRow(sourceIndex) ||
		                        !switchTestValidRow(targetIndex))) || cyclesPerUs == 0U)
		{
				ClearRxBuff();
				return;
		}
		sourceCodes = useCustomCodes ? customSourceCodes :
		        (useStressTable ? Stress_Wave_DAC[sourceIndex] : Wave_DAC[sourceIndex]);
		targetCodes = useCustomCodes ? customTargetCodes :
		        (useStressTable ? Stress_Wave_DAC[targetIndex] : Wave_DAC[targetIndex]);

		/* Hold the source for 100 ms. Source stability is NOT verified here;
		 * the tagged curve is a pair-transition diagnostic, not sparse replay. */
		for(uint8_t channel = 0; channel < 5; channel++) IDACData[channel] = sourceCodes[channel];
		prevDACValid = 0U;
		if(((useStressTable || useCustomCodes) ? PI11210_ApplyCalibrationCodes(sourceCodes)
		                   : PI11210_ApplyChangedTableCodes()) != HAL_OK)
		{
				ClearRxBuff();
				return;
		}
		delay_us(SWITCH_TEST_SOURCE_SETTLE_US);
		ADC_ReadSix(baseline);
		if(!ADC_LastTransferOk()) { prevDACValid = 0U; ClearRxBuff(); return; }

		/* In wavelength-only mode, hold GAIN and SOA at their source values so
		 * the detector step is not caused by a simultaneous optical-power change.
		 */
		for(uint8_t channel = 0; channel < 5; channel++) IDACData[channel] = targetCodes[channel];
		if(!useFullTable)
		{
				IDACData[0] = Wave_DAC[sourceIndex][0];
				IDACData[1] = Wave_DAC[sourceIndex][1];
		}

		CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
		/* CYCCNT also drives Wi-Fi timekeeping. Never reset the shared clock;
		 * unsigned subtraction below is wrap-safe for this bounded capture. */
		DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
		switchStart = DWT->CYCCNT;
		if(useStressTable && targetIndex == STRESS_PATH_PRECONDITION_POINT_INDEX)
		{
				if(PI11210_ApplyCalibrationCodes(Stress_Path_Precondition_DAC) != HAL_OK)
				{ prevDACValid = 0U; ClearRxBuff(); return; }
				delay_us(STRESS_PATH_PRECONDITION_HOLD_US);
				if(PI11210_ApplyCalibrationCodes(targetCodes) != HAL_OK)
				{ prevDACValid = 0U; ClearRxBuff(); return; }
		}
		else if((useStressTable ? PI11210_ApplyStressRowCalibrationOrder() :
		         useCustomCodes ? PI11210_ApplyCalibrationCodes(targetCodes)
		                       : PI11210_ApplyChangedTableCodes()) != HAL_OK)
		{
				ClearRxBuff();
				return;
		}
		switchEnd = DWT->CYCCNT;

		/* Response header and pre-switch baseline.  Timestamps below are relative
		 * to the start of the first changed DAC write, so the I2C update time is
		 * included in the observed end-to-end switching time.
		 */
		txBuffer[position++] = 0xD5U;
		txBuffer[position++] = 0x5DU;
		txBuffer[position++] = useCustomCodes ? 0x03U :
		        (useStressTable ? 0x02U : 0x01U);
		txBuffer[position++] = useCustomCodes ? 0x05U :
		        (useFullTable | useStressTable);
		switchTestWriteU16(&position, sourceIndex);
		switchTestWriteU16(&position, targetIndex);
		switchTestWriteU32(&position, (switchEnd - switchStart) / cyclesPerUs);
		txBuffer[position++] = SWITCH_TEST_SAMPLE_COUNT;
		txBuffer[position++] = 6U;
		if(useStressTable || useCustomCodes)
				switchTestWriteU32(&position, diagnosticTag);
		for(uint8_t channel = 0; channel < 6; channel++) switchTestWriteU16(&position, baseline[channel]);

		for(uint8_t sampleIndex = 0; sampleIndex < SWITCH_TEST_SAMPLE_COUNT; sampleIndex++)
		{
				targetCycle = switchEnd + switchTestOffsetsUs[sampleIndex] * cyclesPerUs;
				while((int32_t)(DWT->CYCCNT - targetCycle) < 0) { }
				uint32_t sampleCycle = DWT->CYCCNT;
				ADC_ReadSix(samples);
				if(!ADC_LastTransferOk()) { prevDACValid = 0U; ClearRxBuff(); return; }
				switchTestWriteU32(&position, (sampleCycle - switchStart) / cyclesPerUs);
				for(uint8_t channel = 0; channel < 6; channel++) switchTestWriteU16(&position, samples[channel]);
		}
		txBuffer[position++] = 0x5DU;
		txBuffer[position++] = 0xD5U;

		/* Wait until the normal 20-byte telemetry packet has left, then use the
		 * same direct large-frame path as table mode.
		 */
		while(!dma_transfer_complete) { }
		dma_transfer_complete = 0U;
		if(CDC_Transmit_FS(txBuffer, position) != USBD_OK) dma_transfer_complete = 1U;

		prevDACValid = 0U;
		ClearRxBuff();
}

/* Only short waits busy-spin. Longer diagnostic holds continue safety and
 * thermal service and abort on host input or a shutter/protection transition.
 * No extra DAC writes, gain changes, or ADC reads occur before this target. */
static uint8_t fastStreamControlActive(void)
{
		return (usbCdcHostOpen || WifiTransport_IsLanRawClientActive()) ? 1U : 0U;
}

static uint8_t stressHoldWait(uint32_t start, uint32_t offsetUs, uint32_t cyclesPerUs)
{
		uint32_t lastService = DWT->CYCCNT;
		while((uint32_t)(DWT->CYCCNT - start) / cyclesPerUs < offsetUs)
		{
				if(ReceEndFlag || !fastStreamControlActive()) return 0U;
				if((uint32_t)(DWT->CYCCNT - lastService) / cyclesPerUs >= 1000U)
				{
						ThermalControl_Process();
						ThermalControl_SafetyTick();
						PI11210_Process();
						if(PI11210_GetStatus().soaMode != PI11210_SOA_SOURCE) return 0U;
						lastService = DWT->CYCCNT;
				}
		}
		return !ReceEndFlag && fastStreamControlActive() &&
		       PI11210_GetStatus().soaMode == PI11210_SOA_SOURCE;
}

/* FF FF 03 05 + R18!: bounded, USB-only experimental trajectory. This is
 * deliberately NOT a work mode or a replacement for the installed scan.
 * Run eight complete 18-row cycles, then the prefix through one target and
 * hold that SAME DAC for the teacher. Every fast value is direct CH1 ADC;
 * no cached rows, RC estimates, host currents, or table/flash writes exist.
 * FF FF 03 06 + R32! runs 32 complete cycles without the held teacher, so
 * warm-up can be separated from steady continuous-route repeatability. */
#define ROUTE_TEST_POINTS 18U
#define ROUTE_TEST_CYCLES 8U
#define ROUTE_STREAM_CYCLES 32U
#define ROUTE_TEST_HEADER 64U
#define ROUTE_TEST_TEACHERS 25U
#define ROUTE_TEST_TIMEOUT_US 2200000U
#define FAST_FLANK_STREAM_MIN_CYCLES 32U
#define FAST_FLANK_STREAM_MAX_CYCLES 1024U
#define FAST_FLANK_STREAM_WARMUP_CYCLES 16U
#define FAST_FLANK_STREAM_TIMEOUT_US 30000000U
#define FAST_FLANK_STREAM_FRAME_LENGTH 210U
#define FAST_FLANK_STREAM_END_LENGTH 42U
#define FAST_TRIPLET_STREAM_POINTS 27U
#define FAST_TRIPLET_STREAM_TIMEOUT_US 45000000U
#define FAST_TRIPLET_STREAM_FRAME_LENGTH 291U
#define FAST_CANDIDATE_V2_STREAM_FRAME_LENGTH 297U
#define FAST_CANDIDATE_V2_STREAM_END_LENGTH 48U
#define FAST_FULLMAP_STREAM_POINTS STRESS_TABLE_POINT_COUNT
#define FAST_FULLMAP_STREAM_MAX_CYCLES 65535U
#define FAST_FULLMAP_STREAM_TIMEOUT_MS 2400000U
#define FAST_FULLMAP_STREAM_FRAME_LENGTH 453U
#define FAST_FULLMAP_MIN_SPACING_US 50U
#define FAST_FULLMAP_MAX_SPACING_US FAST_ADC_SPACING_US
static const uint8_t routeTestRows[2][ROUTE_TEST_POINTS] = {
        {1,3,6,8,11,13,16,18,21,23,26,28,31,33,36,38,41,43},
        {1,3,6,8,11,13,16,18,21,23,26,28,31,33,38,36,41,43}
};
/* Three direct CH1 samples per grating.  G8 deliberately retains the
 * empirically safer descending traversal used by the 18-point route. */
static const uint8_t fastTripletRows[FAST_TRIPLET_STREAM_POINTS] = {
        1,2,3, 6,7,8, 11,12,13, 16,17,18, 21,22,23,
        26,27,28, 31,32,33, 38,37,36, 41,42,43
};
/* The full-map experiment keeps every calibrated wavelength in its installed
 * order.  Unlike the production multirate path, every transmitted value is a
 * fresh direct CH1 conversion from this same board cycle. */
static const uint8_t fastFullMapRows[FAST_FULLMAP_STREAM_POINTS] = {
        0,1,2,3,4, 5,6,7,8,9, 10,11,12,13,14,
        15,16,17,18,19, 20,21,22,23,24, 25,26,27,28,29,
        30,31,32,33,34, 35,36,37,38,39, 40,41,42,43,44
};
volatile uint32_t routeDiagnosticRequests = 0U;
volatile uint32_t routeDiagnosticAccepted = 0U;
volatile uint32_t routeDiagnosticCompleted = 0U;
volatile uint32_t routeDiagnosticLastTag = 0U;
volatile uint32_t routeDiagnosticRejectMask = 0U;

static uint32_t routeTestCrcByte(uint32_t crc, uint8_t value)
{
        crc ^= value;
        for(uint8_t bit = 0U; bit < 8U; bit++)
                crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320UL : 0U);
        return crc;
}

static uint8_t soaResponseWaitUntil(uint32_t origin, uint32_t targetUs,
		uint32_t cyclesPerUs)
{
		uint32_t lastService = DWT->CYCCNT;
		while((uint32_t)(DWT->CYCCNT - origin) / cyclesPerUs < targetUs)
		{
				if(!usbCdcHostOpen || ReceEndFlag || WifiTransport_IsOtaActive()) return 0U;
				if((uint32_t)(DWT->CYCCNT - lastService) / cyclesPerUs >= 1000U)
				{
						ThermalControl_SafetyTick();
						PI11210_Process();
						lastService = DWT->CYCCNT;
				}
		}
		return 1U;
}

/* FF FF 03 0B, byte 4: 0=source->shutter, 1=shutter->source;
 * bytes 6..9: non-zero host tag; bytes 10..13: ASCII "SOA1".
 *
 * Successful response (0xD4 0x4D) is a 48-byte header followed by 43 records.
 * Every record carries board-relative ADC start/end us and the native
 * ADC_ReadSix order CH0..CH3/PDT/PDR.
 * The laser is always left shuttered, including every failure path. */
static void runSOAResponseCapture(void)
{
		typedef char SOAResponsePacketFits[(PACK_SIZE >= SOA_RESPONSE_FRAME_LENGTH) ? 1 : -1];
		(void)sizeof(SOAResponsePacketFits);
		uint8_t direction = aRxBuffer[4];
		uint32_t tag = ((uint32_t)aRxBuffer[6] << 24) |
				((uint32_t)aRxBuffer[7] << 16) |
				((uint32_t)aRxBuffer[8] << 8) | aRxBuffer[9];
		uint32_t cyclesPerUs = SystemCoreClock / 1000000U;
		uint8_t initialMode = PI11210_GetStatus().soaMode;
		uint16_t reject = 0U;
		uint16_t baseline[6] = {0U};
		uint16_t samples[6] = {0U};
		uint16_t position = SOA_RESPONSE_FRAME_HEADER;
		uint8_t captured = 0U;
		uint8_t status = 0U;
		uint32_t switchStart = 0U, switchEnd = 0U;
		uint32_t spiErrors = ADC_GetSpiErrorCount();
		uint16_t i2cErrors = PI11210_GetStatus().i2cErrorCount;

		if(!usbCdcHostOpen) reject |= 1U;
		if(workState != EXTRA_STATE) reject |= 2U;
		if(!dma_transfer_complete) reject |= 4U;
		if(direction > 1U) reject |= 8U;
		if(!tag) reject |= 16U;
		if(memcmp(&aRxBuffer[10], "SOA1", 4U) != 0) reject |= 32U;
		if(!cyclesPerUs) reject |= 64U;
		if((direction == 0U && initialMode != PI11210_SOA_SOURCE) ||
		   (direction == 1U && initialMode != PI11210_SOA_SHUTTER)) reject |= 128U;
		if(reject)
		{
				(void)PI11210_SetSOAShutter(1U);
				memset(aTxBuffer, 0, USART_TX_SIZE);
				aTxBuffer[0] = 0xD4U; aTxBuffer[1] = 0x4DU;
				aTxBuffer[2] = 1U; aTxBuffer[3] = 0x80U;
				aTxBuffer[4] = direction; aTxBuffer[5] = initialMode;
				aTxBuffer[6] = (uint8_t)(tag >> 24); aTxBuffer[7] = (uint8_t)(tag >> 16);
				aTxBuffer[8] = (uint8_t)(tag >> 8); aTxBuffer[9] = (uint8_t)tag;
				aTxBuffer[10] = (uint8_t)(reject >> 8); aTxBuffer[11] = (uint8_t)reject;
				aTxBuffer[18] = 0x4DU; aTxBuffer[19] = 0xD4U;
				USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
				ClearTxBuff();
				ClearRxBuff();
				return;
		}

		ClearRxBuff();
		ReceEndFlag = 0U;
		memset(txBuffer, 0, SOA_RESPONSE_FRAME_LENGTH);
		ADC_ReadSix(baseline);
		if(!ADC_LastTransferOk()) status |= 2U;

		CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
		DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
		switchStart = DWT->CYCCNT;
		if(PI11210_SetSOAShutter(direction == 0U ? 1U : 0U) != HAL_OK)
				status |= 1U;
		switchEnd = DWT->CYCCNT;

		for(uint8_t sampleIndex = 0U;
				sampleIndex < SOA_RESPONSE_SAMPLE_COUNT && !status;
				sampleIndex++)
		{
				uint32_t targetUs = soaResponseOffsetsUs[sampleIndex];
				if(!soaResponseWaitUntil(switchEnd, targetUs, cyclesPerUs))
				{
						status |= 4U;
						break;
				}
				uint32_t sampleStart = DWT->CYCCNT;
				ADC_ReadSix(samples);
				uint32_t sampleEnd = DWT->CYCCNT;
				if(!ADC_LastTransferOk())
				{
						status |= 2U;
						break;
				}
				switchTestWriteU32(&position,
						(uint32_t)((sampleStart - switchStart) / cyclesPerUs));
				switchTestWriteU32(&position,
						(uint32_t)((sampleEnd - switchStart) / cyclesPerUs));
				for(uint8_t channel = 0U; channel < 6U; channel++)
						switchTestWriteU16(&position, samples[channel]);
				captured++;
		}

		uint8_t transitionMode = PI11210_GetStatus().soaMode;
		if(PI11210_SetSOAShutter(1U) != HAL_OK) status |= 8U;
		uint8_t finalMode = PI11210_GetStatus().soaMode;
		if(finalMode != PI11210_SOA_SHUTTER) status |= 8U;
		prevDACValid = 0U;

		position = 0U;
		txBuffer[position++] = 0xD4U; txBuffer[position++] = 0x4DU;
		txBuffer[position++] = 1U; txBuffer[position++] = status;
		txBuffer[position++] = direction; txBuffer[position++] = initialMode;
		txBuffer[position++] = transitionMode; txBuffer[position++] = finalMode;
		switchTestWriteU32(&position, tag);
		switchTestWriteU32(&position, APP_FIRMWARE_VERSION);
		switchTestWriteU32(&position, (switchEnd - switchStart) / cyclesPerUs);
		txBuffer[position++] = captured; txBuffer[position++] = 6U;
		txBuffer[position++] = PD_ReadFeedbackSelector(0U);
		txBuffer[position++] = PD_ReadFeedbackSelector(1U);
		switchTestWriteU32(&position, ADC_GetSpiErrorCount() - spiErrors);
		switchTestWriteU16(&position,
				(uint16_t)(PI11210_GetStatus().i2cErrorCount - i2cErrors));
		for(uint8_t channel = 0U; channel < 6U; channel++)
				switchTestWriteU16(&position, baseline[channel]);
		while(position < SOA_RESPONSE_FRAME_HEADER) txBuffer[position++] = 0U;

		uint16_t frameLength = SOA_RESPONSE_FRAME_HEADER +
				(uint16_t)captured * SOA_RESPONSE_RECORD_LENGTH + 6U;
		uint32_t crc = 0xFFFFFFFFUL;
		for(uint16_t index = 0U; index < frameLength - 6U; index++)
				crc = routeTestCrcByte(crc, txBuffer[index]);
		position = frameLength - 6U;
		switchTestWriteU32(&position, crc ^ 0xFFFFFFFFUL);
		txBuffer[position++] = 0x4DU; txBuffer[position++] = 0xD4U;

		dma_transfer_complete = 0U;
		if(CDC_Transmit_FS(txBuffer, frameLength) != USBD_OK)
			dma_transfer_complete = 1U;
		ReceEndFlag = 0U;
		ClearRxBuff();
}

static void runShortRouteDiagnostic(uint8_t streamOnly)
{
        typedef char RoutePacketFits[(PACK_SIZE >= 18502U) ? 1 : -1];
        (void)sizeof(RoutePacketFits);
        uint8_t route = aRxBuffer[4], targetPosition = aRxBuffer[5];
        uint16_t extraUs = ((uint16_t)aRxBuffer[6] << 8) | aRxBuffer[7];
        uint32_t tag = ((uint32_t)aRxBuffer[8] << 24) | ((uint32_t)aRxBuffer[9] << 16) |
                       ((uint32_t)aRxBuffer[10] << 8) | aRxBuffer[11];
        uint32_t cyclesPerUs = SystemCoreClock / 1000000U;
        routeDiagnosticRequests++;
        routeDiagnosticLastTag = tag;
        routeDiagnosticRejectMask = 0U;
        uint8_t magicInvalid = streamOnly ? (memcmp(&aRxBuffer[12], "R32!", 4U) != 0)
                                          : (memcmp(&aRxBuffer[12], "R18!", 4U) != 0);
        uint8_t targetInvalid = streamOnly ? (targetPosition != 0U)
                                           : (targetPosition >= ROUTE_TEST_POINTS);
        /* A shuttered, explicit EXTRA session and fixed known feedback are
         * prerequisites. Invalid commands must never start optical output. */
        if(!usbCdcHostOpen || workState != EXTRA_STATE || !dma_transfer_complete ||
           route >= 2U || targetInvalid || !tag || !cyclesPerUs ||
           (extraUs != 0U && extraUs != 1500U && extraUs != 4350U) ||
           magicInvalid ||
           PD_ReadFeedbackSelector(0U) != 0U || PD_ReadFeedbackSelector(1U) != 2U ||
           PI11210_GetStatus().soaMode != PI11210_SOA_SHUTTER)
        {
                uint16_t reason = (!usbCdcHostOpen ? 1U : 0U) |
                    (workState != EXTRA_STATE ? 2U : 0U) | (!dma_transfer_complete ? 4U : 0U) |
                    (route >= 2U ? 8U : 0U) | (targetInvalid ? 16U : 0U) |
                    (!tag ? 32U : 0U) | (!cyclesPerUs ? 64U : 0U) |
                    ((extraUs != 0U && extraUs != 1500U && extraUs != 4350U) ? 128U : 0U) |
                    (magicInvalid ? 256U : 0U) |
                    (PD_ReadFeedbackSelector(0U) != 0U ? 512U : 0U) |
                    (PD_ReadFeedbackSelector(1U) != 2U ? 1024U : 0U) |
                    (PI11210_GetStatus().soaMode != PI11210_SOA_SHUTTER ? 2048U : 0U);
                routeDiagnosticRejectMask = reason;
                memset(aTxBuffer, 0, USART_TX_SIZE);
                aTxBuffer[0] = 0xD9U; aTxBuffer[1] = 0x9DU; aTxBuffer[3] = 0x7FU;
                for(uint8_t byte = 0U; byte < 4U; byte++) aTxBuffer[4U+byte] = tag >> (24U-byte*8U);
                aTxBuffer[8] = reason >> 8; aTxBuffer[9] = reason;
                aTxBuffer[10] = PD_ReadFeedbackSelector(0U); aTxBuffer[11] = PD_ReadFeedbackSelector(1U);
                aTxBuffer[12] = workState; aTxBuffer[13] = PI11210_GetStatus().soaMode;
                aTxBuffer[14] = usbCdcHostOpen; aTxBuffer[15] = dma_transfer_complete;
                aTxBuffer[16] = route; aTxBuffer[17] = targetPosition;
                aTxBuffer[18] = 0x9DU; aTxBuffer[19] = 0xD9U;
                USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
                ClearTxBuff();
                return;
        }
        routeDiagnosticAccepted++;
        uint8_t cycles = streamOnly ? ROUTE_STREAM_CYCLES : ROUTE_TEST_CYCLES;
        uint8_t teacherSamples = streamOnly ? 0U : ROUTE_TEST_TEACHERS;
        uint16_t expected = cycles * ROUTE_TEST_POINTS + (streamOnly ? 0U : targetPosition + 1U);
        uint16_t length = ROUTE_TEST_HEADER + expected * 32U + teacherSamples * 20U + 6U;
        uint16_t position = 0U, completed = 0U;
        uint8_t teacherCount = 0U, ok = 1U;
        uint32_t tableCrc = 0xFFFFFFFFUL, crc = 0xFFFFFFFFUL;
        uint32_t spiErrors = ADC_GetSpiErrorCount();
        uint16_t i2cErrors = PI11210_GetStatus().i2cErrorCount;
        uint32_t origin, finalWriteEnd = 0U;
        uint16_t samples[4], six[6];
        ClearRxBuff();
        ReceEndFlag = 0U; /* A new host command now cancels; never execute it here. */
        memset(txBuffer, 0, length);
        txBuffer[0] = 0xD9U; txBuffer[1] = 0x9DU; txBuffer[2] = streamOnly ? 2U : 1U; txBuffer[3] = 1U;
        position = 4U;
        switchTestWriteU32(&position, tag);
        switchTestWriteU32(&position, APP_FIRMWARE_VERSION);
        txBuffer[12] = route; txBuffer[13] = cycles;
        txBuffer[14] = ROUTE_TEST_POINTS; txBuffer[15] = targetPosition;
        position = 16U; switchTestWriteU16(&position, extraUs);
        txBuffer[18] = PD_ReadFeedbackSelector(0U); txBuffer[19] = PD_ReadFeedbackSelector(1U);
        txBuffer[23] = streamOnly ? 2U : 1U; /* diagnostic only; never physical-bandwidth truth */
        position = 32U; switchTestWriteU16(&position, expected);
        for(uint16_t row = 0U; row < STRESS_TABLE_POINT_COUNT; row++)
                for(uint8_t channel = 0U; channel < 5U; channel++)
                {
                        uint16_t code = Stress_Wave_DAC[row][channel];
                        tableCrc = routeTestCrcByte(tableCrc, (uint8_t)(code >> 8));
                        tableCrc = routeTestCrcByte(tableCrc, (uint8_t)code);
                }
        position = 36U; switchTestWriteU32(&position, tableCrc ^ 0xFFFFFFFFUL);
        memcpy(&txBuffer[44], routeTestRows[route], ROUTE_TEST_POINTS);
        CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
        DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk; /* never reset the shared DWT clock */
        origin = DWT->CYCCNT;
        prevDACValid = 0U;
        position = ROUTE_TEST_HEADER;
        while(completed < expected && ok)
        {
                uint8_t local = completed % ROUTE_TEST_POINTS;
                uint8_t row = routeTestRows[route][local];
                if(local == 0U)
                {
                        ThermalControl_Process();
                        WifiTransport_Process();
                }
                ThermalControl_SafetyTick();
                PI11210_Process();
                if(!usbCdcHostOpen || ReceEndFlag || WifiTransport_IsOtaActive() ||
                   (completed && PI11210_GetStatus().soaMode != PI11210_SOA_SOURCE) ||
                   (DWT->CYCCNT - origin) / cyclesPerUs > ROUTE_TEST_TIMEOUT_US)
                { ok = 0U; break; }
                uint32_t writeStart = DWT->CYCCNT;
                for(uint8_t channel = 0U; channel < 5U; channel++)
                        IDACData[channel] = Stress_Wave_DAC[row][channel];
                if(PI11210_ApplyStressRowCalibrationOrder() != HAL_OK) { ok = 0U; break; }
                finalWriteEnd = DWT->CYCCNT;
                uint32_t firstDelayUs = FAST_ADC_FIRST_DELAY_US + ((row == 38U) ? extraUs : 0U);
                if(!stressHoldWait(finalWriteEnd, firstDelayUs, cyclesPerUs)) { ok = 0U; break; }
                uint32_t firstStart = DWT->CYCCNT;
                ADC_ReadMask(0x02U, samples);
                uint32_t firstEnd = DWT->CYCCNT;
                uint16_t firstCode = samples[1];
                if(!ADC_LastTransferOk() ||
                   !stressHoldWait(firstEnd, FAST_ADC_SPACING_US, cyclesPerUs)) { ok = 0U; break; }
                uint32_t secondStart = DWT->CYCCNT;
                ADC_ReadMask(0x02U, samples);
                uint32_t secondEnd = DWT->CYCCNT;
                if(!ADC_LastTransferOk()) { ok = 0U; break; }
                switchTestWriteU16(&position, row);
                txBuffer[position++] = completed / ROUTE_TEST_POINTS;
                txBuffer[position++] = local;
                switchTestWriteU32(&position, (writeStart - origin) / cyclesPerUs);
                switchTestWriteU32(&position, (finalWriteEnd - origin) / cyclesPerUs);
                switchTestWriteU32(&position, (firstStart - origin) / cyclesPerUs);
                switchTestWriteU32(&position, (firstEnd - origin) / cyclesPerUs);
                switchTestWriteU16(&position, firstCode);
                switchTestWriteU32(&position, (secondStart - origin) / cyclesPerUs);
                switchTestWriteU32(&position, (secondEnd - origin) / cyclesPerUs);
                switchTestWriteU16(&position, samples[1]);
                completed++;
        }
        /* No DAC or feedback rewrites from the final fast pair through all
         * 25 later direct-ADC windows. Optical wavelength is NOT measured here. */
        position = ROUTE_TEST_HEADER + expected * 32U;
        for(uint8_t sample = 0U; sample < teacherSamples && ok; sample++)
        {
                if(!stressHoldWait(finalWriteEnd, 300000U + sample * 20000U, cyclesPerUs) ||
                   (DWT->CYCCNT - origin) / cyclesPerUs > ROUTE_TEST_TIMEOUT_US)
                { ok = 0U; break; }
                uint32_t start = DWT->CYCCNT;
                ADC_ReadSix(six);
                uint32_t end = DWT->CYCCNT;
                if(!ADC_LastTransferOk()) { ok = 0U; break; }
                switchTestWriteU32(&position, (start - finalWriteEnd) / cyclesPerUs);
                switchTestWriteU32(&position, (end - finalWriteEnd) / cyclesPerUs);
                for(uint8_t channel = 0U; channel < 6U; channel++) switchTestWriteU16(&position, six[channel]);
                teacherCount++;
        }
        position = 20U; switchTestWriteU16(&position, completed);
        txBuffer[22] = teacherCount;
        position = 24U; switchTestWriteU32(&position, (DWT->CYCCNT - origin) / cyclesPerUs);
        switchTestWriteU32(&position, (finalWriteEnd - origin) / cyclesPerUs);
        position = 34U; switchTestWriteU16(&position, PI11210_GetStatus().i2cErrorCount - i2cErrors);
        position = 40U; switchTestWriteU32(&position, ADC_GetSpiErrorCount() - spiErrors);
        /* Automatic shutdown precedes serialization/transmit on EVERY exit
         * after acceptance, including unplug, host cancellation, and SPI/I2C error. */
        if(PI11210_SetSOAShutter(1U) != HAL_OK) ok = 0U;
        txBuffer[62] = PI11210_GetStatus().soaMode == PI11210_SOA_SHUTTER;
        txBuffer[3] = (ok && txBuffer[62] && completed == expected &&
                       teacherCount == teacherSamples) ? 0U : 1U;
        if(txBuffer[3] == 0U) routeDiagnosticCompleted++;
        ApplyWorkState(EXTRA_STATE);
        prevDACValid = 0U;
        txCount = 4U;
        for(uint16_t index = 0U; index < length - 6U; index++) crc = routeTestCrcByte(crc, txBuffer[index]);
        position = length - 6U; switchTestWriteU32(&position, crc ^ 0xFFFFFFFFUL);
        txBuffer[position++] = 0x9DU; txBuffer[position++] = 0xD9U;
        if(usbCdcHostOpen && dma_transfer_complete)
        {
                dma_transfer_complete = 0U;
                if(CDC_Transmit_FS(txBuffer, length) != USBD_OK) dma_transfer_complete = 1U;
        }
}

/* FF FF 03 07 + F15!: bounded real-time CH1 flank stream.
 * FF FF 03 08 + F27!: bounded real-time CH1 left/centre/right stream.
 *
 * This is intentionally an opt-in USB measurement path, not a replacement for
 * the legacy 45-row spectrum. It replays only the empirically validated 18
 * flank rows and keeps G8 right->left. Every record carries both direct ADC
 * reads and board-relative sample time. Cycles 0..15 are explicitly marked
 * below-minimum-warm-up because the physical laser showed at least about
 * 0.33 s of start-up drift. The flag never certifies absolute optical
 * stability; the host must maintain a causal per-session baseline.
 * No RC estimate, digital scaling, cached sample or host-supplied DAC code is
 * present. The stream is finite and every exit asserts the SOA shutter. */
static void runFastFlankStream(void)
{
		uint8_t customStream = aRxBuffer[3] == 0x0AU;
		uint8_t temporary45Stream = customStream &&
				(aRxBuffer[4] == 3U || aRxBuffer[4] == 4U || aRxBuffer[4] == 5U);
		uint8_t wideTemporary = 0U;
		uint8_t guardedCustomStream = 0U;
		uint8_t tripletStream = (aRxBuffer[3] == 0x08U) || customStream;
		uint8_t fullMapStream = aRxBuffer[3] == 0x09U;
		static CandidateRoute candidateRoute;
		static const uint8_t candidateLocalRows[TEMPORARY_ROUTE_COUNT] = {
				0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
				27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
		};
		const uint8_t *streamRows = fullMapStream ? fastFullMapRows
				: (customStream ? candidateLocalRows : (tripletStream ? fastTripletRows : routeTestRows[1]));
		uint8_t pointCount = fullMapStream ? FAST_FULLMAP_STREAM_POINTS
				: (tripletStream ? FAST_TRIPLET_STREAM_POINTS : ROUTE_TEST_POINTS);
		uint16_t frameLength = fullMapStream ? FAST_FULLMAP_STREAM_FRAME_LENGTH
				: (tripletStream ? FAST_TRIPLET_STREAM_FRAME_LENGTH
				                   : FAST_FLANK_STREAM_FRAME_LENGTH);
		uint32_t streamTimeoutMs = fullMapStream ? FAST_FULLMAP_STREAM_TIMEOUT_MS
				: ((tripletStream ? FAST_TRIPLET_STREAM_TIMEOUT_US
				                  : FAST_FLANK_STREAM_TIMEOUT_US) / 1000U);
		uint8_t route = aRxBuffer[4];
		uint16_t extraUs = ((uint16_t)aRxBuffer[6] << 8) | aRxBuffer[7];
		uint32_t tag = ((uint32_t)aRxBuffer[8] << 24) |
				((uint32_t)aRxBuffer[9] << 16) |
				((uint32_t)aRxBuffer[10] << 8) | aRxBuffer[11];
		uint16_t requestedCycles = ((uint16_t)aRxBuffer[16] << 8) | aRxBuffer[17];
		uint32_t cyclesPerUs = SystemCoreClock / 1000000U;
		/* F18 opt-in spacing: marker 'S', big-endian us. Zero legacy tail
		 * retains 600 us; other routes cannot accidentally enable this. */
		uint8_t explicitFlankSpacing = (aRxBuffer[3] == 0x07U && aRxBuffer[19] == 0x53U);
		/* Explicit F18 commissioning needs time for baseline and safe retract.
		 * Keep legacy limits untouched; still finite with a hard 60 s bound. */
		if(explicitFlankSpacing) streamTimeoutMs = 60000U;
		uint16_t sampleSpacingUs = fullMapStream
				? (((uint16_t)aRxBuffer[19] << 8) | aRxBuffer[20])
				: (explicitFlankSpacing ? (((uint16_t)aRxBuffer[20] << 8) | aRxBuffer[21]) : FAST_ADC_SPACING_US);
		/* Opt-in F45 gain qualification. Legacy requests remain fixed at 5k.
		 * Gain must already be set and read back while shuttered; this command
		 * never changes it during the scan. Each frame reports actual selectors. */
		uint8_t explicitFullMapGain = fullMapStream && aRxBuffer[21] == 0x47U;
		uint8_t requestedStreamSelector1 = explicitFullMapGain ? aRxBuffer[22] : 2U;
		uint8_t tailInvalid = 0U;
		uint8_t streamOverLan = (!usbCdcHostOpen &&
				WifiTransport_IsLanRawClientActive()) ? 1U : 0U;
		uint16_t reject = 0U;
		uint32_t tableCrc = 0xFFFFFFFFUL;
		uint16_t tailStart = customStream
				? (temporary45Stream ? 666U : (aRxBuffer[4] == 2U ? 368U : 352U))
				: (fullMapStream ? (explicitFullMapGain ? 23U : 21U) : (explicitFlankSpacing ? 22U : 19U));
		for(uint16_t index = tailStart; index < USART_RX_SIZE; index++)
				if(aRxBuffer[index] != 0U) { tailInvalid = 1U; break; }
		if(!fastStreamControlActive()) reject |= 1U;
		if(workState != EXTRA_STATE) reject |= 2U;
		if(!dma_transfer_complete) reject |= 4U;
		uint8_t temporaryCountValid = route == 3U
				? aRxBuffer[5] == TEMPORARY_ROUTE_COUNT
				: (aRxBuffer[5] >= TEMPORARY_ROUTE_MIN_COUNT &&
				   aRxBuffer[5] <= TEMPORARY_ROUTE_COUNT && aRxBuffer[5] % 5U == 0U);
		if((customStream ? (route != 1U && route != 2U && route != 3U && route != 4U && route != 5U)
		                 : (route != 1U)) ||
		   aRxBuffer[5] != (customStream
				? (temporary45Stream ? aRxBuffer[5] : 27U) : 0U) ||
		   (temporary45Stream && !temporaryCountValid)) reject |= 8U;
		uint16_t maximumCycles = fullMapStream ? FAST_FULLMAP_STREAM_MAX_CYCLES
				: (explicitFlankSpacing ? 2048U : FAST_FLANK_STREAM_MAX_CYCLES);
		uint8_t continuousTemporary = temporary45Stream && requestedCycles == 0U;
		if((!customStream && extraUs != (fullMapStream ? STRESS_PATH_PRECONDITION_HOLD_US : 1500U)) ||
		   (!continuousTemporary && requestedCycles < FAST_FLANK_STREAM_MIN_CYCLES) ||
		   requestedCycles > maximumCycles ||
		   aRxBuffer[18] != FAST_FLANK_STREAM_WARMUP_CYCLES || tailInvalid)
				reject |= 16U;
		if((fullMapStream || explicitFlankSpacing) &&
		   (sampleSpacingUs < FAST_FULLMAP_MIN_SPACING_US ||
		    sampleSpacingUs > FAST_FULLMAP_MAX_SPACING_US ||
		    (sampleSpacingUs % 25U) != 0U)) reject |= 16U;
		if(!tag) reject |= 32U;
		if(!cyclesPerUs) reject |= 64U;
		if(customStream)
		{
				/* The operator-selected temporary v3/v4 route is carried over
				 * the authenticated same-LAN RAW owner exactly like CDC.  Keep
				 * the older commissioning-only custom routes USB-local, while
				 * still applying the same CRC, current-limit, order and timing
				 * validation to every temporary packet before any DAC write. */
				if((streamOverLan && !temporary45Stream) ||
				   !CandidateRoute_Parse(aRxBuffer, USART_RX_SIZE, &candidateRoute))
					reject |= 256U;
				else
				{
					sampleSpacingUs = candidateRoute.spacingUs;
					guardedCustomStream = candidateRoute.protocolVersion == 2U;
					if(guardedCustomStream) frameLength = FAST_CANDIDATE_V2_STREAM_FRAME_LENGTH;
					if(temporary45Stream)
					{
						pointCount = candidateRoute.pointCount;
						wideTemporary = candidateRoute.totalWaitUs > 40500UL;
						frameLength = (uint16_t)(46U + pointCount *
								(wideTemporary ? 13U : 9U));
						/* Slow selected scans are finite unless cycles is zero.
						 * Their budget follows the requested dwell time, not 15 Hz. */
						streamTimeoutMs = 5000UL + requestedCycles *
								(candidateRoute.totalWaitUs / 1000UL + 100UL);
					}
				}
		}
		else if(fullMapStream)
		{
				if(memcmp(&aRxBuffer[12], "F45!", 4U) != 0) reject |= 256U;
		}
		else if(tripletStream)
		{
				if(memcmp(&aRxBuffer[12], "F27!", 4U) != 0) reject |= 256U;
		}
		else if(memcmp(&aRxBuffer[12], "F15!", 4U) != 0) reject |= 256U;
		if(PD_ReadFeedbackSelector(0U) != 0U) reject |= 512U;
		uint8_t selectedAdcChannel = (temporary45Stream && route == 5U)
				? candidateRoute.adcChannel : 1U;
		uint8_t selectedFeedback = temporary45Stream
				? candidateRoute.feedbackSelector : requestedStreamSelector1;
		uint8_t expectedSelector0 = selectedAdcChannel == 0U ? selectedFeedback : 0U;
		uint8_t expectedSelector1 = selectedAdcChannel == 1U ? selectedFeedback : 0U;
		if(expectedSelector0 > 3U || expectedSelector1 > 3U ||
		   PD_ReadFeedbackSelector(0U) != expectedSelector0 ||
		   PD_ReadFeedbackSelector(1U) != expectedSelector1) reject |= 1024U;
		if(PI11210_GetStatus().soaMode != PI11210_SOA_SHUTTER) reject |= 2048U;
		if(reject)
		{
				memset(aTxBuffer, 0, USART_TX_SIZE);
				aTxBuffer[0] = 0xD9U; aTxBuffer[1] = 0x9DU;
				aTxBuffer[2] = customStream
						? (temporary45Stream ? 0x0EU : (aRxBuffer[4] == 2U ? 0x0CU : 0x0AU))
						: 0x03U;
				aTxBuffer[3] = 0x7FU;
				for(uint8_t byte = 0U; byte < 4U; byte++)
						aTxBuffer[4U + byte] = (uint8_t)(tag >> (24U - byte * 8U));
				aTxBuffer[8] = (uint8_t)(reject >> 8); aTxBuffer[9] = (uint8_t)reject;
				aTxBuffer[10] = PD_ReadFeedbackSelector(0U);
				aTxBuffer[11] = PD_ReadFeedbackSelector(1U);
				aTxBuffer[12] = workState;
				aTxBuffer[13] = PI11210_GetStatus().soaMode;
				aTxBuffer[14] = fastStreamControlActive(); aTxBuffer[15] = dma_transfer_complete;
				aTxBuffer[16] = route; aTxBuffer[17] = aRxBuffer[18];
				aTxBuffer[18] = 0x9DU; aTxBuffer[19] = 0xD9U;
				USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
				ClearTxBuff();
				return;
		}

		for(uint16_t row = 0U; row < STRESS_TABLE_POINT_COUNT; row++)
				for(uint8_t channel = 0U; channel < 5U; channel++)
				{
						uint16_t code = Stress_Wave_DAC[row][channel];
						tableCrc = routeTestCrcByte(tableCrc, (uint8_t)(code >> 8));
						tableCrc = routeTestCrcByte(tableCrc, (uint8_t)code);
				}
		tableCrc ^= 0xFFFFFFFFUL;
		if(customStream) tableCrc = candidateRoute.tableCrc;
		/* ApplyWorkState(EXTRA_STATE) restores the safe default feedback at exit.
		 * Preserve the actual acquisition selectors for terminal provenance. */
		uint8_t streamSelector0 = PD_ReadFeedbackSelector(0U);
		uint8_t streamSelector1 = PD_ReadFeedbackSelector(1U);
		ClearRxBuff();
		ReceEndFlag = 0U;
		prevDACValid = 0U;
		CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
		DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
		uint32_t origin = DWT->CYCCNT;
		uint32_t previousCycleStart = origin;
		uint64_t streamElapsedCycles = 0ULL;
		uint32_t streamStartTick = HAL_GetTick();
		uint32_t completed = 0U;
		uint16_t stopReason = 0U;
		uint32_t spiErrors = ADC_GetSpiErrorCount();
		uint16_t i2cErrors = PI11210_GetStatus().i2cErrorCount;

		while((continuousTemporary || completed < requestedCycles) && !stopReason)
		{
				/* Never overwrite the shared transmit buffer before USB owns the
				 * previous frame. The bounded wait remains protection-aware. */
				uint32_t waitStart = DWT->CYCCNT;
				while(!dma_transfer_complete)
				{
						ThermalControl_SafetyTick();
						PI11210_Process();
						if(!fastStreamControlActive()) { stopReason |= 2U; break; }
						if(ReceEndFlag) { stopReason |= 1U; break; }
						if(WifiTransport_IsOtaActive()) { stopReason |= 4U; break; }
						if(PI11210_GetStatus().soaMode != PI11210_SOA_SOURCE)
								{ stopReason |= 8U; break; }
						if((DWT->CYCCNT - waitStart) / cyclesPerUs > 50000U)
								{ stopReason |= 128U; break; }
				}
				if(stopReason) break;
				if(!continuousTemporary &&
				   (uint32_t)(HAL_GetTick() - streamStartTick) > streamTimeoutMs)
						{ stopReason |= 16U; break; }

				ThermalControl_Process();
				ThermalControl_SafetyTick();
				PI11210_Process();
				WifiTransport_SetLocalControlActive(usbCdcHostOpen);
				WifiTransport_Process();
				if(WifiTransport_IsOtaActive()) { stopReason |= 4U; break; }
				uint32_t cycleStart = DWT->CYCCNT;
				/* DWT_CYCCNT wraps about every 35.8 s at 120 MHz. Accumulate
				 * unsigned deltas so an extended F45 session keeps monotonic
				 * board timestamps without changing the 32-bit wire field. */
				streamElapsedCycles += (uint32_t)(cycleStart - previousCycleStart);
				previousCycleStart = cycleStart;
				uint16_t firstCodes[FAST_FULLMAP_STREAM_POINTS];
				uint16_t secondCodes[FAST_FULLMAP_STREAM_POINTS];
				uint32_t writeEndUs[FAST_FULLMAP_STREAM_POINTS];
				uint32_t secondEndUs[FAST_FULLMAP_STREAM_POINTS];
				uint16_t samples[4];
				uint16_t guardWriteEndUs = 0U;

				for(uint8_t local = 0U; local < pointCount; local++)
				{
						uint8_t row = streamRows[local];
						ThermalControl_SafetyTick();
						PI11210_Process();
						if(!fastStreamControlActive()) { stopReason |= 2U; break; }
						if(ReceEndFlag) { stopReason |= 1U; break; }
						if(WifiTransport_IsOtaActive()) { stopReason |= 4U; break; }
						if(local && PI11210_GetStatus().soaMode != PI11210_SOA_SOURCE)
								{ stopReason |= 8U; break; }
						if(guardedCustomStream && local == candidateRoute.guardBeforeLocal)
						{
								if(PI11210_ApplyCalibrationCodes(candidateRoute.guardCodes) != HAL_OK)
										{ stopReason |= 32U; break; }
								uint32_t guardEnd = DWT->CYCCNT;
								uint32_t relativeGuardUs = (guardEnd - cycleStart) / cyclesPerUs;
								if(relativeGuardUs > 0xFFFFU) { stopReason |= 16U; break; }
								guardWriteEndUs = (uint16_t)relativeGuardUs;
								if(!stressHoldWait(guardEnd, candidateRoute.guardHoldUs, cyclesPerUs))
										{ stopReason |= ReceEndFlag ? 1U : 8U; break; }
						}
						for(uint8_t channel = 0U; channel < 5U; channel++)
								IDACData[channel] = customStream ? candidateRoute.codes[local][channel] : Stress_Wave_DAC[row][channel];
						if(customStream)
						{
								if(PI11210_ApplyCalibrationCodes(candidateRoute.codes[local]) != HAL_OK)
										{ stopReason |= 32U; break; }
						}
						else if(fullMapStream && row == STRESS_PATH_PRECONDITION_POINT_INDEX)
						{
								/* Preserve the already-qualified full-band predecessor path for
								 * this cavity-sensitive wavelength. Both hidden and target writes
								 * use the calibrated production order. */
								if(PI11210_ApplyCalibrationCodes(Stress_Path_Precondition_DAC) != HAL_OK)
										{ stopReason |= 32U; break; }
								uint32_t preconditionEnd = DWT->CYCCNT;
								if(!stressHoldWait(preconditionEnd,
										STRESS_PATH_PRECONDITION_HOLD_US, cyclesPerUs))
										{ stopReason |= ReceEndFlag ? 1U : 8U; break; }
								if(PI11210_ApplyCalibrationCodes(Stress_Wave_DAC[row]) != HAL_OK)
										{ stopReason |= 32U; break; }
						}
						else if(PI11210_ApplyStressRowCalibrationOrder() != HAL_OK)
								{ stopReason |= 32U; break; }
						uint32_t writeEnd = DWT->CYCCNT;
						uint32_t relativeWriteUs = (writeEnd - cycleStart) / cyclesPerUs;
						if(!wideTemporary && relativeWriteUs > 0xFFFFU) { stopReason |= 16U; break; }
						writeEndUs[local] = relativeWriteUs;
						uint32_t firstDelayUs = customStream ? candidateRoute.firstDelayUs : FAST_ADC_FIRST_DELAY_US;
						if(temporary45Stream) firstDelayUs = candidateRoute.pointDelayUs[local];
						if(fullMapStream && isStressSectionBoundary(row))
								firstDelayUs += FAST_BOUNDARY_EXTRA_DELAY_US;
						else if(!customStream && !fullMapStream && row == 38U) firstDelayUs += extraUs;
						if(!stressHoldWait(writeEnd, firstDelayUs,
								cyclesPerUs)) { stopReason |= ReceEndFlag ? 1U : 8U; break; }
						ADC_ReadMask((uint8_t)(1U << selectedAdcChannel), samples);
						firstCodes[local] = samples[selectedAdcChannel];
						if(!ADC_LastTransferOk()) { stopReason |= 64U; break; }
						uint32_t firstEnd = DWT->CYCCNT;
						if(!stressHoldWait(firstEnd, sampleSpacingUs, cyclesPerUs))
								{ stopReason |= ReceEndFlag ? 1U : 8U; break; }
						ADC_ReadMask((uint8_t)(1U << selectedAdcChannel), samples);
						secondCodes[local] = samples[selectedAdcChannel];
						if(!ADC_LastTransferOk()) { stopReason |= 64U; break; }
						uint32_t relativeSecondUs = (DWT->CYCCNT - cycleStart) / cyclesPerUs;
						if(!wideTemporary && relativeSecondUs > 0xFFFFU) { stopReason |= 16U; break; }
						secondEndUs[local] = relativeSecondUs;
				}
				if(stopReason) break;

				uint32_t cycleElapsedUs = (DWT->CYCCNT - cycleStart) / cyclesPerUs;
				uint16_t position = 0U;
				uint32_t crc = 0xFFFFFFFFUL;
				memset(txBuffer, 0, frameLength);
				txBuffer[position++] = 0xD9U; txBuffer[position++] = 0x9DU;
				txBuffer[position++] = customStream
						? (temporary45Stream ? 0x0EU : (guardedCustomStream ? 0x0CU : 0x0AU))
						: 0x03U;
				txBuffer[position++] =
						((completed >= FAST_FLANK_STREAM_WARMUP_CYCLES) ? 1U : 0U) |
						(wideTemporary ? 2U : 0U) |
						((temporary45Stream && route == 5U)
						 ? (uint8_t)(selectedAdcChannel << 2) : 0U);
				switchTestWriteU32(&position, tag);
				switchTestWriteU32(&position, APP_FIRMWARE_VERSION);
				switchTestWriteU32(&position, completed);
				switchTestWriteU32(&position,
						(uint32_t)(streamElapsedCycles / cyclesPerUs));
				switchTestWriteU32(&position, cycleElapsedUs);
				switchTestWriteU32(&position, tableCrc);
				txBuffer[position++] = customStream ? (uint8_t)(candidateRoute.firstDelayUs / 25U) : 1U;
				txBuffer[position++] = pointCount;
				txBuffer[position++] = FAST_FLANK_STREAM_WARMUP_CYCLES;
				txBuffer[position++] = (fullMapStream || customStream || explicitFlankSpacing) ? (uint8_t)(sampleSpacingUs / 25U) : 0U;
				txBuffer[position++] = streamSelector0;
				txBuffer[position++] = streamSelector1;
				switchTestWriteU16(&position,
						(uint16_t)(PI11210_GetStatus().i2cErrorCount - i2cErrors));
				switchTestWriteU32(&position, ADC_GetSpiErrorCount() - spiErrors);
				memcpy(&txBuffer[position], streamRows, pointCount);
				position += pointCount;
				for(uint8_t local = 0U; local < pointCount; local++)
				{
						switchTestWriteU16(&position, firstCodes[local]);
						switchTestWriteU16(&position, secondCodes[local]);
						if(wideTemporary)
						{
								switchTestWriteU32(&position, writeEndUs[local]);
								switchTestWriteU32(&position, secondEndUs[local]);
						}
						else
						{
								switchTestWriteU16(&position, (uint16_t)writeEndUs[local]);
								switchTestWriteU16(&position, (uint16_t)secondEndUs[local]);
						}
				}
				if(guardedCustomStream)
				{
						txBuffer[position++] = candidateRoute.protocolVersion;
						txBuffer[position++] = candidateRoute.guardBeforeLocal;
						switchTestWriteU16(&position, candidateRoute.guardHoldUs);
						switchTestWriteU16(&position, candidateRoute.guardIndex);
						switchTestWriteU16(&position, guardWriteEndUs);
				}
				for(uint16_t index = 0U; index < frameLength - 6U; index++)
						crc = routeTestCrcByte(crc, txBuffer[index]);
				position = frameLength - 6U;
				switchTestWriteU32(&position, crc ^ 0xFFFFFFFFUL);
				txBuffer[position++] = 0x9DU; txBuffer[position++] = 0xD9U;
				if(streamOverLan)
				{
						uint32_t queueWait = DWT->CYCCNT;
						uint8_t queued = 0U;
						while(!queued)
						{
								queued = fullMapStream
										? WifiTransport_QueueFastFullMapFrame(txBuffer, frameLength)
										: WifiTransport_QueueRawFrame(txBuffer, frameLength);
								if(queued) break;
								ThermalControl_SafetyTick();
								PI11210_Process();
								WifiTransport_Process();
								if(!WifiTransport_IsLanRawClientActive())
										{ stopReason |= 2U; break; }
								if(ReceEndFlag) { stopReason |= 1U; break; }
								if(WifiTransport_IsOtaActive()) { stopReason |= 4U; break; }
								if((DWT->CYCCNT - queueWait) / cyclesPerUs > 100000U)
										{ stopReason |= 128U; break; }
						}
						if(stopReason || !queued) break;
				}
				else
				{
						dma_transfer_complete = 0U;
						if(CDC_Transmit_FS(txBuffer, frameLength) != USBD_OK)
						{
								dma_transfer_complete = 1U;
								stopReason |= 128U;
								break;
						}
				}
				completed++;
		}

		/* Wait only for an accepted transfer, then close light before publishing
		 * terminal status. A lost USB host still reaches the shutter. */
		uint32_t finalWait = DWT->CYCCNT;
		if(streamOverLan)
		{
				while(!WifiTransport_FlushFastFullMapFrames() &&
				      WifiTransport_IsLanRawClientActive() &&
				      (DWT->CYCCNT - finalWait) / cyclesPerUs <= 1000000U)
				{
						ThermalControl_SafetyTick();
						PI11210_Process();
						WifiTransport_Process();
				}
		}
		while(((streamOverLan && !WifiTransport_IsLanRawQueueIdle()) ||
		       (!streamOverLan && !dma_transfer_complete)) &&
		      fastStreamControlActive() &&
		      (DWT->CYCCNT - finalWait) / cyclesPerUs <= 1000000U)
		{
				ThermalControl_SafetyTick();
				PI11210_Process();
				if(streamOverLan) WifiTransport_Process();
		}
		if((streamOverLan && !WifiTransport_IsLanRawQueueIdle()) ||
		   (!streamOverLan && !dma_transfer_complete)) stopReason |= 128U;
		if(PI11210_SetSOAShutter(1U) != HAL_OK) stopReason |= 8U;
		uint8_t shuttered = PI11210_GetStatus().soaMode == PI11210_SOA_SHUTTER;
		if(!shuttered) stopReason |= 8U;
		ApplyWorkState(EXTRA_STATE);
		prevDACValid = 0U;
		txCount = 4U;
		ReceEndFlag = 0U;
		ClearRxBuff();
		if(fastStreamControlActive() &&
		   (streamOverLan || dma_transfer_complete))
		{
				uint16_t position = 0U;
				uint32_t crc = 0xFFFFFFFFUL;
				uint16_t endLength = guardedCustomStream
						? FAST_CANDIDATE_V2_STREAM_END_LENGTH : FAST_FLANK_STREAM_END_LENGTH;
				memset(txBuffer, 0, endLength);
				txBuffer[position++] = 0xD9U; txBuffer[position++] = 0x9DU;
				txBuffer[position++] = customStream
						? (temporary45Stream ? 0x0FU : (guardedCustomStream ? 0x0DU : 0x0BU))
						: 0x04U;
				txBuffer[position++] = (completed == requestedCycles && !stopReason) ? 0U : 1U;
				switchTestWriteU32(&position, tag);
				switchTestWriteU32(&position, APP_FIRMWARE_VERSION);
				switchTestWriteU16(&position, (uint16_t)completed);
				switchTestWriteU16(&position, requestedCycles);
				switchTestWriteU16(&position, stopReason);
				txBuffer[position++] = 1U; txBuffer[position++] = shuttered;
				txBuffer[position++] = streamSelector0;
				txBuffer[position++] = streamSelector1;
				switchTestWriteU16(&position,
						(uint16_t)(PI11210_GetStatus().i2cErrorCount - i2cErrors));
				switchTestWriteU32(&position, ADC_GetSpiErrorCount() - spiErrors);
				streamElapsedCycles +=
						(uint32_t)(DWT->CYCCNT - previousCycleStart);
				switchTestWriteU32(&position,
						(uint32_t)(streamElapsedCycles / cyclesPerUs));
				switchTestWriteU32(&position, tableCrc);
				if(guardedCustomStream)
				{
						txBuffer[position++] = candidateRoute.protocolVersion;
						txBuffer[position++] = candidateRoute.guardBeforeLocal;
						switchTestWriteU16(&position, candidateRoute.guardHoldUs);
						switchTestWriteU16(&position, candidateRoute.guardIndex);
				}
				for(uint16_t index = 0U; index < endLength - 6U; index++)
						crc = routeTestCrcByte(crc, txBuffer[index]);
				position = endLength - 6U;
				switchTestWriteU32(&position, crc ^ 0xFFFFFFFFUL);
				txBuffer[position++] = 0x9DU; txBuffer[position++] = 0xD9U;
				if(streamOverLan)
				{
						uint32_t terminalWait = DWT->CYCCNT;
						while(!WifiTransport_QueueRawFrame(
								txBuffer, endLength) &&
						      WifiTransport_IsLanRawClientActive() &&
						      (DWT->CYCCNT - terminalWait) / cyclesPerUs <= 1000000U)
						{
								ThermalControl_SafetyTick();
								PI11210_Process();
								WifiTransport_Process();
						}
						while(!WifiTransport_IsLanRawQueueIdle() &&
						      WifiTransport_IsLanRawClientActive() &&
						      (DWT->CYCCNT - terminalWait) / cyclesPerUs <= 2000000U)
						{
								ThermalControl_SafetyTick();
								PI11210_Process();
								WifiTransport_Process();
						}
				}
				else
				{
						dma_transfer_complete = 0U;
						if(CDC_Transmit_FS(txBuffer, endLength) != USBD_OK)
								dma_transfer_complete = 1U;
						else
						{
								uint32_t terminalWait = DWT->CYCCNT;
								while(!dma_transfer_complete && usbCdcHostOpen &&
								      (DWT->CYCCNT - terminalWait) / cyclesPerUs <= 50000U)
								{
										ThermalControl_SafetyTick();
										PI11210_Process();
								}
						}
				}
		}
}

static void runStressHeldTrajectory(const ScanFrameTimer_t *frameTimer)
{
		/* 44-byte header + 58 (start, end, six ADC) records + trailer. */
		typedef char HoldPacketFits[(PACK_SIZE >= 1206U) ? 1 : -1];
		(void)sizeof(HoldPacketFits);
		uint32_t targetWriteEnd = DWT->CYCCNT;
		uint32_t cyclesPerUs = SystemCoreClock / 1000000U;
		const StressMultiratePlan_t *plan = StressMultirate_CurrentPlan(&stressMultirateState);
		uint16_t position = 0U;
		uint8_t ok = cyclesPerUs != 0U;
		uint16_t samples[6];
		memset(txBuffer, 0, 1206U);
		txBuffer[position++] = 0xD7U;
		txBuffer[position++] = 0x7DU;
		txBuffer[position++] = 1U;
		txBuffer[position++] = 1U; /* fail closed until all samples and shutter succeed */
		switchTestWriteU32(&position, stressHoldActiveTag);
		switchTestWriteU32(&position, APP_FIRMWARE_VERSION);
		switchTestWriteU16(&position, stressHoldActivePoint);
		txBuffer[position++] = stressMultirateNegotiatedVersion;
		txBuffer[position++] = plan->profile;
		txBuffer[position++] = plan->primarySegment;
		txBuffer[position++] = plan->secondarySegment;
		txBuffer[position++] = plan->freshCount;
		txBuffer[position++] = (plan->mapAgeFrames & 0x7FU) |
		        (plan->bandwidthDiscontinuity ? 0x80U : 0U);
		switchTestWriteU32(&position, plan->planSequence);
		switchTestWriteU32(&position, frameTimer->startTick);
		switchTestWriteU32(&position, ScanFrameTimer_ElapsedUs(frameTimer));
		txBuffer[position++] = PD_ReadFeedbackSelector(0U);
		txBuffer[position++] = PD_ReadFeedbackSelector(1U);
		txBuffer[position++] = SWITCH_TEST_SAMPLE_COUNT;
		txBuffer[position++] = 25U;
		txBuffer[position++] = 6U;
		txBuffer[position++] = 1U; /* explicit interrupted-scan diagnostic */
		for(uint8_t index = 0U; index < STRESS_MULTIRATE_FRESH_BITMAP_BYTES; index++)
				txBuffer[position++] = plan->freshBitmap[index];
		for(uint8_t sample = 0U; sample < SWITCH_TEST_SAMPLE_COUNT + 25U && ok; sample++)
		{
				uint32_t requestedUs = sample < SWITCH_TEST_SAMPLE_COUNT
				        ? switchTestOffsetsUs[sample]
				        : 300000U + (sample - SWITCH_TEST_SAMPLE_COUNT) * 20000U;
				if(!stressHoldWait(targetWriteEnd, requestedUs, cyclesPerUs)) { ok = 0U; break; }
				uint32_t startedUs = (DWT->CYCCNT - targetWriteEnd) / cyclesPerUs;
				ADC_ReadSix(samples);
				uint32_t endedUs = (DWT->CYCCNT - targetWriteEnd) / cyclesPerUs;
				if(!ADC_LastTransferOk()) { ok = 0U; break; }
				switchTestWriteU32(&position, startedUs);
				switchTestWriteU32(&position, endedUs);
				for(uint8_t channel = 0U; channel < 6U; channel++)
						switchTestWriteU16(&position, samples[channel]);
		}
		/* End this session on success AND failure; no partial frame, cached
		 * values, or post-hold optical history enters the real-time estimator. */
		if(PI11210_SetSOAShutter(1U) != HAL_OK) ok = 0U;
		ApplyWorkState(EXTRA_STATE);
		prevDACValid = 0U;
		txCount = 4U;
		txBuffer[3] = ok ? 0U : 1U;
		txBuffer[1204] = 0x7DU;
		txBuffer[1205] = 0xD7U;
		dma_transfer_complete = 0U;
		if(CDC_Transmit_FS(txBuffer, 1206U) != USBD_OK) dma_transfer_complete = 1U;
}

static void runStressProductionTeacher(const ScanFrameTimer_t *frameTimer)
{
		/* Header 52 + 3 production CH1 (start,end,code) records (30) +
		 * one causal post-pair six-channel state (20) + 25 teachers (500) + 2. */
		typedef char PairTeacherPacketFits[(PACK_SIZE >= 604U) ? 1 : -1];
		(void)sizeof(PairTeacherPacketFits);
		const StressMultiratePlan_t *plan = StressMultirate_CurrentPlan(&stressMultirateState);
		uint32_t cyclesPerUs = SystemCoreClock / 1000000U;
		uint16_t position = 0U;
		uint16_t samples[6];
		uint8_t ok = (cyclesPerUs && stressPairEstimateValid &&
		        stressPairSampleMask == STRESS_MULTIRATE_CHANNEL_MASK &&
		        (stressPairValidMask == 3U || stressPairValidMask == 7U) &&
		        StressMultirate_ShouldSample(plan, stressHoldActivePoint));
		memset(txBuffer, 0, 604U);
		txBuffer[position++] = 0xD8U;
		txBuffer[position++] = 0x8DU;
		txBuffer[position++] = 1U;
		txBuffer[position++] = 1U;
		switchTestWriteU32(&position, stressHoldActiveTag);
		switchTestWriteU32(&position, APP_FIRMWARE_VERSION);
		switchTestWriteU16(&position, stressHoldActivePoint);
		txBuffer[position++] = stressMultirateNegotiatedVersion;
		txBuffer[position++] = plan->profile;
		txBuffer[position++] = plan->primarySegment;
		txBuffer[position++] = plan->secondarySegment;
		txBuffer[position++] = plan->freshCount;
		txBuffer[position++] = (plan->mapAgeFrames & 0x7FU) |
		        (plan->bandwidthDiscontinuity ? 0x80U : 0U);
		switchTestWriteU32(&position, plan->planSequence);
		switchTestWriteU32(&position, frameTimer->startTick);
		switchTestWriteU32(&position, ScanFrameTimer_ElapsedUs(frameTimer));
		txBuffer[position++] = PD_ReadFeedbackSelector(0U);
		txBuffer[position++] = PD_ReadFeedbackSelector(1U);
		txBuffer[position++] = stressPairValidMask;
		txBuffer[position++] = 25U;
		txBuffer[position++] = 6U;
		txBuffer[position++] = 1U; /* interrupted scan, not a normal dynamic frame */
		for(uint8_t index = 0U; index < STRESS_MULTIRATE_FRESH_BITMAP_BYTES; index++)
				txBuffer[position++] = plan->freshBitmap[index];
		switchTestWriteU32(&position, wave_time);
		switchTestWriteU16(&position, stressPairEstimate);
		txBuffer[position++] = stressPairSampleMask;
		txBuffer[position++] = 1U; /* same target held; no DAC rewrite */
		for(uint8_t index = 0U; index < 3U; index++)
		{
				uint8_t present = ok && (stressPairValidMask & (1U << index));
				switchTestWriteU32(&position, present ?
				        (stressPairStartCycles[index] - stressPairOriginCycles) / cyclesPerUs : 0U);
				switchTestWriteU32(&position, present ?
				        (stressPairEndCycles[index] - stressPairOriginCycles) / cyclesPerUs : 0U);
				switchTestWriteU16(&position, present ? stressPairCodes[index] : 0U);
		}
		/* The extra optical-state read is strictly AFTER all production reads.
		 * It is a future-model feature, not already available in normal frames. */
		for(uint8_t index = 0U; index < 26U && ok; index++)
		{
				uint32_t requestedUs = index ? 300000U + (index - 1U) * 20000U : 0U;
				if(!stressHoldWait(stressPairOriginCycles, requestedUs, cyclesPerUs)) { ok = 0U; break; }
				uint32_t startUs = (DWT->CYCCNT - stressPairOriginCycles) / cyclesPerUs;
				ADC_ReadSix(samples);
				uint32_t endUs = (DWT->CYCCNT - stressPairOriginCycles) / cyclesPerUs;
				if(!ADC_LastTransferOk()) { ok = 0U; break; }
				switchTestWriteU32(&position, startUs);
				switchTestWriteU32(&position, endUs);
				for(uint8_t channel = 0U; channel < 6U; channel++)
						switchTestWriteU16(&position, samples[channel]);
		}
		if(PI11210_SetSOAShutter(1U) != HAL_OK) ok = 0U;
		ApplyWorkState(EXTRA_STATE);
		prevDACValid = 0U;
		txCount = 4U;
		txBuffer[3] = ok ? 0U : 1U;
		txBuffer[602] = 0x8DU;
		txBuffer[603] = 0xD8U;
		dma_transfer_complete = 0U;
		if(CDC_Transmit_FS(txBuffer, 604U) != USBD_OK) dma_transfer_complete = 1U;
}

static void sendStressRawCapture(uint16_t pointCount, uint32_t frameStartMs,
                                uint32_t elapsedUs)
{
		uint16_t position = 0U;
		uint8_t preservePlan = (stressRawCaptureActive == 2U);
		if(pointCount > STRESS_TABLE_POINT_COUNT) pointCount = STRESS_TABLE_POINT_COUNT;
		txBuffer[position++] = 0xD6U;
		txBuffer[position++] = 0x6DU;
		txBuffer[position++] = preservePlan ? 0x02U : 0x01U;
		switchTestWriteU16(&position, pointCount);
		txBuffer[position++] = stressActiveChannelMask;
		if(preservePlan)
		{
				const StressMultiratePlan_t *plan =
						StressMultirate_CurrentPlan(&stressMultirateState);
				switchTestWriteU32(&position, stressRawActiveTag);
				switchTestWriteU32(&position, plan->planSequence);
				switchTestWriteU32(&position, frameStartMs);
				switchTestWriteU32(&position, elapsedUs);
				txBuffer[position++] = stressMultirateNegotiatedVersion;
				txBuffer[position++] = plan->profile;
				txBuffer[position++] = plan->primarySegment;
				txBuffer[position++] = plan->secondarySegment;
				txBuffer[position++] = plan->freshCount;
				txBuffer[position++] = (plan->mapAgeFrames & 0x7FU) |
						(plan->bandwidthDiscontinuity ? 0x80U : 0U);
				for(uint8_t index = 0U; index < STRESS_MULTIRATE_FRESH_BITMAP_BYTES; index++)
						txBuffer[position++] = plan->freshBitmap[index];
				txBuffer[position++] = PD_ReadFeedbackSelector(0U);
				txBuffer[position++] = PD_ReadFeedbackSelector(1U);
				switchTestWriteU32(&position, APP_FIRMWARE_VERSION);
		}
		for(uint16_t point = 0U; point < pointCount; point++)
		{
				switchTestWriteU16(&position, stressRawFirst[point]);
				switchTestWriteU16(&position, stressRawSecond[point]);
				switchTestWriteU16(&position, stressRawSlow[point]);
				switchTestWriteU16(&position, stressRawEstimate[point]);
				txBuffer[position++] = stressRawRecheck[point];
				if(preservePlan) switchTestWriteU32(&position, stressSampleOffsetUs[point]);
		}
		txBuffer[position++] = 0x6DU;
		txBuffer[position++] = 0xD6U;

		stressRawCaptureActive = 0U;
		stressRawActiveTag = 0U;
		if(usbCdcHostOpen)
		{
				dma_transfer_complete = 0U;
				if(CDC_Transmit_FS(txBuffer, position) != USBD_OK)
						dma_transfer_complete = 1U;
		}
		else dma_transfer_complete = 1U;
}

void write_ms5614t_table(void){
		if(!dma_transfer_complete) return;

		int i;
	  int j;
		ScanFrameTimer_t frameTimer;
		uint8_t Head = 0xFF;
		ScanFrameTimer_Start(&frameTimer);
		uint32_t adcErrorCountAtFrameStart = ADC_GetSpiErrorCount();
		uint16_t tableStartIndex = (workState == PRECISION_TABLE_STATE)
				? PRECISION_TABLE_START_INDEX : 0U;
		uint16_t tablePointCount = (workState == TABLE_STATE)
				? STRESS_TABLE_POINT_COUNT : TEMPERATURE_TABLE_POINT_COUNT;
		if(workState == PRECISION_TABLE_STATE)
		{
				precisionActiveSelector[0] = precisionLearnedSelector[0];
				precisionActiveSelector[1] = precisionLearnedSelector[1];
		}
		else if(workState == TABLE_STATE)
		{
				stressBeginFrame();
		}
	
//		memset(txBuffer, 0, PACK_SIZE*sizeof(uint8_t));
		txBuffer[0] = 0xEE;
		txBuffer[1] = 0xEE;
		txCount = 4;
	
		/* Temperature mode scans all table rows but only converts CH0/CH1.
		 * CH2/CH3 protocol fields are kept as zero for compatibility.  Their
		 * former ADC time is reinvested in a denser CH0/CH1 trimmed mean.
		 */
		for (i = tableStartIndex; i <= (int)(tableStartIndex + tablePointCount);)
		{
				/* Keep only bounded safety/control work inside the latency-sensitive
				 * stress frame.  Cooperative thermal 1-Wire steps are serviced
				 * immediately before/after the frame by main; the constant-time tick
				 * still enforces stale-sensor/full-fan state.  CH224Q is hardware-
				 * strapped and its compatibility service is side-effect free.
				 * Precision mode can take well over a second, so retain its original
				 * full maintenance cadence rather than delaying sensor conversions.
				 * PI11210 polling remains in both modes because it can assert the
				 * hardware SOA shutter on a protection fault. */
				if(workState == TABLE_STATE)
				{
						ThermalControl_SafetyTick();
				}
				else
				{
						ThermalControl_Process();
						CH224Q_Process();
				}
				PI11210_Process();
				/* The radio state machine is cooperative and bounded: this call may
				 * start a USART6 DMA transfer, but never waits for it. */
				WifiTransport_SetLocalControlActive(usbCdcHostOpen);
				WifiTransport_Process();
				if(WifiTransport_IsOtaActive())
				{
					/* Abort at a point boundary; main then closes the SOA shutter
					 * and services only thermal/power/radio state machines. */
					prevDACValid = 0U;
					txCount = 4U;
					break;
				}
				if(workState != TABLE_STATE && workState != PRECISION_TABLE_STATE) break;
				if(ReceEndFlag==1 && aRxBuffer[0] == Head && aRxBuffer[1] == Head)
				{
						modify_table_loop();
						ReceEndFlag = 0U;
				}
//				checkTemp(workState);
			  // 提前把一个波长的通道数据取出来
			  if((uint16_t)i >= tableStartIndex + tablePointCount)
			  {
					IDACData[0] = 0xFFFFU;
					IDACData[1] = 0xFFFFU;
					IDACData[2] = 0xFFFFU;
			  }
			  else for(j = 0; j < 5; j++)
			  {
					  IDACData[j] = (workState == TABLE_STATE)
							? Stress_Wave_DAC[(uint16_t)i - tableStartIndex][j]
							: Wave_DAC[i][j];
				}
				i++;
			
				// 波长数据0x00就跳过
				if ((IDACData[0] == 0xFFFF) && (IDACData[1] == 0xFFFF) && (IDACData[2] == 0xFFFF)){
//						uint8_t p1 = Find_Peaks(adc1, peaks1, i-1);
//						uint8_t p2 = Find_Peaks(adc2, peaks2, i-1);
//						uint8_t p3 = Find_Peaks(adc3, peaks3, i-1);
//						uint8_t p4 = Find_Peaks(adc4, peaks4, i-1);
						uint8_t p1=0,p2=0,p3=0,p4=0;
						if(workState == PRECISION_TABLE_STATE &&
								(precisionLearnedSelector[0] != precisionActiveSelector[0] ||
								 precisionLearnedSelector[1] != precisionActiveSelector[1]))
						{
								/* Do not publish a spectrum acquired with a selector that
								 * clipped.  The next invocation repeats the complete frame
								 * using the lower fixed gain for that channel.
								 */
								prevDACValid = 0U;
								return;
						}
				
						uint16_t pointCount = (uint16_t)((i - 1) - tableStartIndex);
						txBuffer[2] = (pointCount >> 8) & 0xFF;
						txBuffer[3] = pointCount & 0xFF;
						if(workState == TABLE_STATE &&
								StressMultirate_IsEnabled(&stressMultirateState) &&
								stressSampleTimingOverflow)
						{
								/* A 2 us uint16 timestamp covers 131.068 ms.  Never
								 * publish a reduced frame with ambiguous sample times;
								 * fall back to a fresh, unoptimised MAP45 session. */
								stressMultirateDisarmAndClear();
								prevDACValid = 0U;
								txCount = 4U;
								return;
						}
						if(workState == TABLE_STATE)
								StressMultirate_EndFrame(&stressMultirateState);
					
						FillPeaks(p1,p2,p3,p4);
					
						sampleTemperature();
						if(workState == TABLE_STATE) stressFinishChannelDiscovery();
						uint16_t gainMask0 = (workState == PRECISION_TABLE_STATE)
								? precisionLegacy20kMask(precisionActiveSelector[0])
								: (stressActive20k[0] ? 1U : 0U);
						uint16_t gainMask1 = (workState == PRECISION_TABLE_STATE)
								? precisionLegacy20kMask(precisionActiveSelector[1])
								: (stressActive20k[1] ? 1U : 0U);
						uint8_t activeChannelMask = (workState == PRECISION_TABLE_STATE)
								? 0x03U : stressActiveChannelMask;
						/* Four trailer bytes carry independent, up-to-16-section 20 kOhm masks. */
						txBuffer[txCount++] = (uint8_t)(gainMask0 >> 8);
						txBuffer[txCount++] = (uint8_t)gainMask0;
						txBuffer[txCount++] = (uint8_t)(gainMask1 >> 8);
						txBuffer[txCount++] = (uint8_t)gainMask1;
						/* Optional v1 board-status extension.  Legacy GUI versions ignore
						 * these bytes because the original four gain-mask bytes remain first.
						 * CH224Q values describe the negotiated PD ceiling, not live load.
						 * Byte 17 is the locked ADC/plot channel mask.  The next seven
						 * bytes report verified PI11210 state, its I2C error count and
						 * the actual CH0/CH1 feedback selectors.  Bytes 24..27 identify
						 * the application version.  Bytes 28..35 append frame sequence and
						 * board acquisition duration without changing the v1 prefix.
						 * A tagged 16-byte final block binds the raw USB frame to the
						 * wavelength table, boot session and MCU uptime. */
						{
								CH224Q_Status pd = CH224Q_GetStatus();
								PI11210_Status_t pi = PI11210_GetStatus();
								uint8_t pdFlags = (pd.online ? 0x01U : 0U)
										| (pd.current_valid ? 0x02U : 0U)
										| ((pd.i2c_address == 0x23U) ? 0x04U : 0U);
								uint16_t fanRpm = ThermalControl_GetFanRpm();
								uint16_t fanDuty = ThermalControl_GetFanDutyPermille();
								uint16_t thermalFlags = ThermalControl_GetStatusFlags();
								uint32_t acquisitionDurationUs =
										ScanFrameTimer_ElapsedUs(&frameTimer);
								txBuffer[txCount++] = 0xB5U;
								txBuffer[txCount++] = 0x4DU;
								txBuffer[txCount++] = 1U;
								txBuffer[txCount++] = (workState == TABLE_STATE &&
										StressMultirate_IsEnabled(&stressMultirateState))
										? STRESS_BOARD_STATUS_MULTIRATE_LENGTH
										: STRESS_BOARD_STATUS_BASE_LENGTH;
								txBuffer[txCount++] = pdFlags;
								txBuffer[txCount++] = pd.protocol_status;
								txBuffer[txCount++] = (uint8_t)(pd.requested_voltage_mV >> 8);
								txBuffer[txCount++] = (uint8_t)pd.requested_voltage_mV;
								txBuffer[txCount++] = (uint8_t)(pd.available_current_mA >> 8);
								txBuffer[txCount++] = (uint8_t)pd.available_current_mA;
								txBuffer[txCount++] = (uint8_t)(pd.power_limit_mW >> 24);
								txBuffer[txCount++] = (uint8_t)(pd.power_limit_mW >> 16);
								txBuffer[txCount++] = (uint8_t)(pd.power_limit_mW >> 8);
								txBuffer[txCount++] = (uint8_t)pd.power_limit_mW;
								txBuffer[txCount++] = (uint8_t)(fanRpm >> 8);
								txBuffer[txCount++] = (uint8_t)fanRpm;
								txBuffer[txCount++] = (uint8_t)(fanDuty >> 8);
								txBuffer[txCount++] = (uint8_t)fanDuty;
								txBuffer[txCount++] = (uint8_t)(thermalFlags >> 8);
								txBuffer[txCount++] = (uint8_t)thermalFlags;
								txBuffer[txCount++] = activeChannelMask;
								txBuffer[txCount++] = (pi.online ? 0x01U : 0U)
										| (pi.partValid ? 0x02U : 0U)
										| (pi.initialized ? 0x04U : 0U)
										| ((pi.rawStatus & PI11210_STATUS_HI_TEMP) ? 0x08U : 0U)
										| ((pi.rawStatus & PI11210_STATUS_OVR_TEMP) ? 0x10U : 0U)
										| ((pi.rawStatus & PI11210_STATUS_PRO_TEMP) ? 0x20U : 0U)
										| (pi.busRecovered ? 0x40U : 0U)
										| ((pi.soaMode == PI11210_SOA_SHUTTER) ? 0x80U : 0U);
								txBuffer[txCount++] = (uint8_t)(pi.rawStatus >> 8);
								txBuffer[txCount++] = (uint8_t)pi.rawStatus;
								txBuffer[txCount++] = (uint8_t)(pi.i2cErrorCount >> 8);
								txBuffer[txCount++] = (uint8_t)pi.i2cErrorCount;
								txBuffer[txCount++] = (workState == PRECISION_TABLE_STATE)
										? precisionActiveSelector[0] : PD_ReadFeedbackSelector(0U);
								txBuffer[txCount++] = (workState == PRECISION_TABLE_STATE)
										? precisionActiveSelector[1] : PD_ReadFeedbackSelector(1U);
								txBuffer[txCount++] = (uint8_t)(APP_FIRMWARE_VERSION >> 24);
								txBuffer[txCount++] = (uint8_t)(APP_FIRMWARE_VERSION >> 16);
								txBuffer[txCount++] = (uint8_t)(APP_FIRMWARE_VERSION >> 8);
								txBuffer[txCount++] = (uint8_t)APP_FIRMWARE_VERSION;
								txBuffer[txCount++] = (uint8_t)(scanFrameSequence >> 24);
								txBuffer[txCount++] = (uint8_t)(scanFrameSequence >> 16);
								txBuffer[txCount++] = (uint8_t)(scanFrameSequence >> 8);
								txBuffer[txCount++] = (uint8_t)scanFrameSequence;
								txBuffer[txCount++] = (uint8_t)(acquisitionDurationUs >> 24);
								txBuffer[txCount++] = (uint8_t)(acquisitionDurationUs >> 16);
								txBuffer[txCount++] = (uint8_t)(acquisitionDurationUs >> 8);
								txBuffer[txCount++] = (uint8_t)acquisitionDurationUs;
								if(workState == TABLE_STATE &&
										StressMultirate_IsEnabled(&stressMultirateState))
								{
										const StressMultiratePlan_t *plan =
												StressMultirate_CurrentPlan(&stressMultirateState);
								txBuffer[txCount++] = stressMultirateNegotiatedVersion;
										txBuffer[txCount++] = plan->profile;
										txBuffer[txCount++] = plan->primarySegment;
										txBuffer[txCount++] = plan->secondarySegment;
										txBuffer[txCount++] = plan->freshCount;
										txBuffer[txCount++] = (uint8_t)(
												(plan->mapAgeFrames &
												 STRESS_MULTIRATE_MAP_AGE_MASK) |
												(plan->bandwidthDiscontinuity
												 ? STRESS_MULTIRATE_BANDWIDTH_GAP_FLAG : 0U));
										for(uint8_t byte = 0U;
												byte < STRESS_MULTIRATE_FRESH_BITMAP_BYTES; byte++)
												txBuffer[txCount++] = plan->freshBitmap[byte];
										txBuffer[txCount++] = (uint8_t)(frameTimer.startTick >> 24);
										txBuffer[txCount++] = (uint8_t)(frameTimer.startTick >> 16);
										txBuffer[txCount++] = (uint8_t)(frameTimer.startTick >> 8);
										txBuffer[txCount++] = (uint8_t)frameTimer.startTick;
										for(uint16_t point = 0U;
												point < STRESS_TABLE_POINT_COUNT; point++)
										{
												uint16_t ticks = STRESS_MULTIRATE_SAMPLE_TIME_UNAVAILABLE;
												if(StressMultirate_ShouldSample(plan, point))
												{
														uint32_t offsetUs = stressSampleOffsetUs[point];
												ticks = (uint16_t)((offsetUs +
														STRESS_MULTIRATE_SAMPLE_TIME_UNIT_US - 1U) /
														STRESS_MULTIRATE_SAMPLE_TIME_UNIT_US);
											}
										txBuffer[txCount++] = (uint8_t)(ticks >> 8);
										txBuffer[txCount++] = (uint8_t)ticks;
									}
								}
								else
								{
									/* Full MAP/precision frames do not carry a freshness
									 * schedule, but expose their true MCU frame-start tick
									 * so host cadence can be separated from network arrival. */
									txBuffer[txCount++] =
											(uint8_t)(frameTimer.startTick >> 24);
									txBuffer[txCount++] =
											(uint8_t)(frameTimer.startTick >> 16);
									txBuffer[txCount++] =
											(uint8_t)(frameTimer.startTick >> 8);
									txBuffer[txCount++] = (uint8_t)frameTimer.startTick;
								}
								/* Tagged runtime identity is always last, so future fields can
								 * be added without shifting the established schedule offsets. */
								{
									uint32_t tableCrc = WifiTransport_GetTableCrc(workState);
									uint32_t bootId = WifiTransport_GetBootId();
									uint32_t uptimeMs = HAL_GetTick();
									txBuffer[txCount++] = 0xC7U;
									txBuffer[txCount++] = 0x49U;
									txBuffer[txCount++] = 1U;
									txBuffer[txCount++] = 12U;
									txBuffer[txCount++] = (uint8_t)(tableCrc >> 24);
									txBuffer[txCount++] = (uint8_t)(tableCrc >> 16);
									txBuffer[txCount++] = (uint8_t)(tableCrc >> 8);
									txBuffer[txCount++] = (uint8_t)tableCrc;
									txBuffer[txCount++] = (uint8_t)(bootId >> 24);
									txBuffer[txCount++] = (uint8_t)(bootId >> 16);
									txBuffer[txCount++] = (uint8_t)(bootId >> 8);
									txBuffer[txCount++] = (uint8_t)bootId;
									txBuffer[txCount++] = (uint8_t)(uptimeMs >> 24);
									txBuffer[txCount++] = (uint8_t)(uptimeMs >> 16);
									txBuffer[txCount++] = (uint8_t)(uptimeMs >> 8);
									txBuffer[txCount++] = (uint8_t)uptimeMs;
								}
					}
						WifiTransport_QueueScanFrame(txBuffer,
								pointCount,
								workState,
								(int32_t)(tempData * 1000.0f),
								gainMask0,
								gainMask1,
								activeChannelMask);
						if(workState == TABLE_STATE && stressRawCaptureActive)
								sendStressRawCapture(pointCount, frameTimer.startTick,
										ScanFrameTimer_ElapsedUs(&frameTimer));
						else
						{
								uint8_t published = sendTxBuffer(pointCount, p1, p2, p3, p4);
								if(published) scanFrameSequence++;
								else if(workState == TABLE_STATE &&
										StressMultirate_IsEnabled(&stressMultirateState))
								{
										/* A lost USB/LAN owner invalidates every cached row.
										 * The next owner sees a genuine MAP45 unless it safely
										 * re-arms multirate acquisition from EXTRA. */
										stressMultirateDisarmAndClear();
								}
						}
						/* Apply any provisioning packet that arrived at the last table
						 * row, then arm maintenance for the following inter-frame gap. */
						WifiTransport_Process();
						WifiTransport_NotifyFrameBoundary();

						/* Remote scan-mode changes become visible only after the complete
						 * old-mode frame above has been snapshotted for USB.  The transport
						 * then drops that obsolete wireless snapshot and waits for the first
						 * complete new-mode frame before publishing APPLIED. */
						{
								uint8_t requestedMode;
								uint32_t requestId;
								if(WifiTransport_TakePendingMode(workState, 1U,
										&requestedMode, &requestId))
								{
										ApplyWorkState(requestedMode);
										WifiTransport_OnRemoteModeApplied(requestedMode, requestId);
								}
						}
						break;
				}

#if 0  /* Legacy selectable MS5614T path retained for reference. */
				if(dacTarget == 0)
				{
						MS5614T2_SetCode(MS5614T_DAC_A,IDACData[0],MS5614T_SPEED_FAST,MS5614T_NORMAL);
						MS5614T2_SetCode(MS5614T_DAC_C,IDACData[1],MS5614T_SPEED_FAST,MS5614T_NORMAL);
						MS5614T2_SetCode(MS5614T_DAC_B,IDACData[2],MS5614T_SPEED_FAST,MS5614T_NORMAL);
						MS5614T_SetCode(MS5614T_DAC_A,IDACData[3],MS5614T_SPEED_FAST,MS5614T_NORMAL);
						MS5614T_SetCode(MS5614T_DAC_C,IDACData[4],MS5614T_SPEED_FAST,MS5614T_NORMAL);
				}
#endif
				activeScanPointIndex = (uint16_t)(i - 1) - tableStartIndex;
				if(workState == PRECISION_TABLE_STATE)
				{
						precisionApplySegmentGain(i - 1);
						/* A sparse temperature table does not necessarily approach a row
						 * through the same cavity mode as the complete 2001-point scan.
						 * Reproduce that calibrated path with the immediate audited
						 * predecessor.  It is an optical-state command only: no ADC sample
						 * or plotted point is produced for it. */
						if(PI11210_ApplyCalibrationCodes(
								Temperature_Precondition_DAC[activeScanPointIndex]) != HAL_OK)
						{
								prevDACValid = 0U;
								return;
						}
						delay_us(TEMPERATURE_PRECONDITION_HOLD_US);
						/* Rewrite the target in the same five-channel calibration order,
						 * then apply the normal temperature settling/sample window. */
						if(PI11210_ApplyFullTableRowCalibrationOrder(i - 1) != HAL_OK)
						{
								prevDACValid = 0U;
								return;
						}
				}
				if(workState == TABLE_STATE)
				{
						/* One path-sensitive target may be certified offline with the exact
						 * previous-sparse -> immediate-full-band-predecessor -> target path.
						 * The generated table enables only that certified point.  Its hidden
						 * predecessor has no ADC read and consumes no plotted/table point;
						 * the fixed hold remains inside whole-frame timing telemetry. */
						if(activeScanPointIndex == STRESS_PATH_PRECONDITION_POINT_INDEX)
						{
								if(PI11210_ApplyCalibrationCodes(
										Stress_Path_Precondition_DAC) != HAL_OK)
								{
										prevDACValid = 0U;
										return;
								}
								delay_us(STRESS_PATH_PRECONDITION_HOLD_US);
								/* Rewrite all five target channels, including unchanged GAIN,
								 * exactly as equal-interval and single-value acquisition do. */
								if(PI11210_ApplyCalibrationCodes(
										Stress_Wave_DAC[activeScanPointIndex]) != HAL_OK)
								{
										prevDACValid = 0U;
										return;
								}
						}
						else if(PI11210_ApplyStressRowCalibrationOrder() != HAL_OK)
						{
								prevDACValid = 0U;
								return;
						}
				}
				else if(workState != PRECISION_TABLE_STATE &&
						PI11210_ApplyChangedTableCodes() != HAL_OK)
				{
						prevDACValid = 0U;
						return;
				}
				
//				sampleVoltage();
				if(workState == TABLE_STATE && stressHoldActiveTag &&
				   activeScanPointIndex == stressHoldActivePoint)
				{
						if(!stressHoldActiveProduction)
						{
								runStressHeldTrajectory(&frameTimer);
								return;
						}
						stressPairOriginCycles = DWT->CYCCNT;
						if(!StressMultirate_ShouldSample(
						        StressMultirate_CurrentPlan(&stressMultirateState), activeScanPointIndex))
						{
								runStressProductionTeacher(&frameTimer); /* fail closed; no forced fresh row */
								return;
						}
				}
				
				if(workState == PRECISION_TABLE_STATE)
				{
						sampleVoltagePrecision(i - 1);
						if(ADC_GetSpiErrorCount() != adcErrorCountAtFrameStart)
						{
								prevDACValid = 0U;
								txCount = 4U;
								return;
						}
						precisionObserveSaturation(i - 1);
				}
				else if(workState == TABLE_STATE &&
						StressMultirate_IsEnabled(&stressMultirateState) &&
						!StressMultirate_ShouldSample(
								StressMultirate_CurrentPlan(&stressMultirateState),
								activeScanPointIndex))
				{
						/* The complete calibrated 45-row DAC path above is still replayed.
						 * Only this ADC conversion is skipped; the transmitted value comes
						 * from cache and is explicitly non-fresh in the extension bitmap. */
						stressAppendCachedCh1(activeScanPointIndex);
				}
				else
				{
						sampleVoltageStable(i - 1);// i was incremented after loading this wavelength
						if(stressHoldActiveProduction && stressHoldActiveTag &&
						   activeScanPointIndex == stressHoldActivePoint)
						{
								if(ADC_GetSpiErrorCount() != adcErrorCountAtFrameStart)
										stressPairValidMask = 0U;
								runStressProductionTeacher(&frameTimer);
								return;
						}
						if(ADC_GetSpiErrorCount() != adcErrorCountAtFrameStart)
						{
								prevDACValid = 0U;
								txCount = 4U;
								return;
						}
						if(workState == TABLE_STATE &&
								StressMultirate_IsEnabled(&stressMultirateState))
						{
								uint32_t sampleOffsetUs =
										ScanFrameTimer_ElapsedUs(&frameTimer);
								if(sampleOffsetUs > STRESS_MULTIRATE_SAMPLE_TIME_MAX_US)
										stressSampleTimingOverflow = 1U;
								else
										stressSampleOffsetUs[activeScanPointIndex] = sampleOffsetUs;
								StressMultirate_RecordCh1(&stressMultirateState,
										activeScanPointIndex, uADCOriginvalues[1]);
						}
				}
				
//				M1820Z_GetTmp();
				
//				adc1[i] = uADCOriginvalues[0];
//				adc2[i] = uADCOriginvalues[1];
//				adc3[i] = uADCOriginvalues[2];
//				adc4[i] = uADCOriginvalues[3];
				
				//delay_us(1);
		}
}

void write_ms5614t_manual(void){
		uint8_t Head = 0xFF;
		static uint32_t lastTempTick = 0U;
		uint32_t nowTick = HAL_GetTick();
		
//		USART_Queue_Send(&ReceEndFlag, 1);
		/* Match EXTRA mode's 2 Hz temperature cadence.  Unsigned subtraction
		 * keeps the elapsed-time check correct across the HAL tick wrap. */
		if((uint32_t)(nowTick - lastTempTick) >= 500U)
		{
				lastTempTick = nowTick;
				checkTemp(workState);
		}
	
		if(ReceEndFlag==0) return;
		
		if (aRxBuffer[0] == Head && aRxBuffer[1] == Head)
		{
				if(aRxBuffer[2] == 0x00){
						scanWave();
				}
				else if(aRxBuffer[2] == 0x01){
						modify_table_loop();
				}
		}
		if ((aRxBuffer[0] != Head) && (aRxBuffer[0] != 0x00))
		{
				ClearRxBuff();
		}
		/* Keep the mailbox occupied until every command byte has been consumed.
		 * USB/UART callbacks drop a complete replacement frame while this flag is
		 * set, so a new command cannot tear the one being executed. */
		ReceEndFlag = 0U;
}

void write_ms5614t_extra(void){
		uint8_t Head = 0xFF;
		static uint32_t lastRtTick = 0U;
		static uint32_t lastTempTick = 0U;
		uint32_t nowTick = HAL_GetTick();
		/* Own a pending mailbox before doing any monitor ADC/USB work. An IRQ
		 * may deliver a command inside checkRT/checkTemp; leave that newly
		 * arrived mailbox intact for the next invocation, never clear it here. */
		if(ReceEndFlag)
		{
				if(aRxBuffer[0] == Head && aRxBuffer[1] == Head)
				{
						if(aRxBuffer[2] == 0x03U &&
						   (aRxBuffer[3] == 0x05U || aRxBuffer[3] == 0x06U))
						{
								if(!dma_transfer_complete) return;
								runShortRouteDiagnostic(aRxBuffer[3] == 0x06U);
						}
						else if(aRxBuffer[2] == 0x03U &&
						        (aRxBuffer[3] == 0x07U || aRxBuffer[3] == 0x08U ||
						         aRxBuffer[3] == 0x09U || aRxBuffer[3] == 0x0AU))
						{
								if(!dma_transfer_complete) return;
								runFastFlankStream();
						}
						else if(aRxBuffer[2] == 0x03U && aRxBuffer[3] == 0x0BU)
						{
								if(!dma_transfer_complete) return;
								runSOAResponseCapture();
						}
						else if(aRxBuffer[2] == 0x00) singleValue();
						else if(aRxBuffer[2] == 0x01) modify_table_loop();
						else if(aRxBuffer[2] == 0x03 && aRxBuffer[3] == 0x01)
								runLaserSwitchTest();
				}
				if((aRxBuffer[0] != Head) && (aRxBuffer[0] != 0x00)) ClearRxBuff();
				ReceEndFlag = 0U;
				return;
		}
	
		/* Temperature evolves slowly; 2 Hz is ample for the direct-mode panel
		 * and prevents its 20-byte status frame from flooding USB. */
		if((uint32_t)(nowTick - lastTempTick) >= 500U)
		{
				lastTempTick = nowTick;
				checkTemp(workState);
		}
		/* The desktop renders direct-monitor data at 20 Hz.  A 50 Hz producer
		 * leaves feedback headroom without monopolising the ADC/USB path with
		 * thousands of redundant frames per second. */
		if((uint32_t)(nowTick - lastRtTick) >= 20U)
		{
				lastRtTick = nowTick;
				checkRT();
		}
	
}

void ApplyWorkState(uint8_t newState)
{
		if(newState > PRECISION_TABLE_STATE) return;
		stressHoldRequestedTag = 0U;
		stressHoldActiveTag = 0U;
		stressRawCaptureRequested = 0U;
		stressRawCaptureActive = 0U;
		stressRawRequestedTag = 0U;
		stressRawActiveTag = 0U;
		workState = newState;
		switch(workState)
		{
			case TABLE_STATE:
			{
					StressMultirate_Configure(&stressMultirateState,
							stressMultirateArmed,
							stressMultirateMapPeriod,
							stressMultirateActivateCodes,
							stressMultirateReleaseCodes);
					/* v3 remains the fixed SURVEY9/TRACK13 contract.  Only a
					 * successfully negotiated v4 session may emit profile 3. */
					StressMultirate_SetSingleTrackEnabled(&stressMultirateState,
							(uint8_t)(stressMultirateArmed &&
							 stressMultirateNegotiatedVersion >=
							 STRESS_MULTIRATE_EXTENSION_VERSION_V4));
					StressMultirate_SetG8RightSentinel(&stressMultirateState,
							(uint8_t)(stressMultirateArmed &&
							 stressMultirateNegotiatedVersion ==
							 STRESS_MULTIRATE_EXTENSION_VERSION_V5));
					StressMultirate_StartSession(&stressMultirateState);
					for(uint8_t channel = 0U; channel < 2U; channel++)
					{
							stressActiveSelector[channel] = stressManualFeedbackArmed
									? stressRequestedSelector[channel]
									: PD_FEEDBACK_SELECTOR_40K;
							stressActive20k[channel] =
									(stressActiveSelector[channel] == PD_FEEDBACK_SELECTOR_20K)
									? 1U : 0U;
							PD_SetFeedbackSelector(channel, stressActiveSelector[channel]);
					}
					stressGainSettlePending = 1U;
					if(StressMultirate_IsEnabled(&stressMultirateState))
					{
							stressActiveChannelMask = STRESS_MULTIRATE_CHANNEL_MASK;
							stressChannelDiscoveryPending = 0U;
					}
					else stressResetChannelDiscovery();
					LED_MANUAL_LOW();
					LED_TABLE_HIGH();
					prevDACValid = 0;
					break;
			}
			case MANUAL_STATE:
			{
					stressGainSettlePending = 0U;
					stressMultirateDisarmAndClear();
					PD_RestoreFeedback40k();
					LED_TABLE_LOW();
					LED_MANUAL_HIGH();
					memset(unstableFlags, 0, sizeof(unstableFlags));
					break;
			}
			case EXTRA_STATE:
			{
					stressGainSettlePending = 0U;
					stressMultirateDisarmAndClear();
					stressManualFeedbackArmed = 0U;
					extraFeedbackSelector[0] = PD_FEEDBACK_SELECTOR_40K;
					extraFeedbackSelector[1] = PD_FEEDBACK_SELECTOR_40K;
					PD_RestoreFeedback40k();
					LED_TABLE_HIGH();
					LED_MANUAL_HIGH();
					memset(unstableFlags, 0, sizeof(unstableFlags));
					break;
			}
			case PRECISION_TABLE_STATE:
			{
					stressGainSettlePending = 0U;
					stressMultirateDisarmAndClear();
					precisionLearnedSelector[0] = TEMPERATURE_FEEDBACK_SELECTOR_CH0;
					precisionLearnedSelector[1] = TEMPERATURE_FEEDBACK_SELECTOR_CH1;
					precisionActiveSelector[0] = precisionLearnedSelector[0];
					precisionActiveSelector[1] = precisionLearnedSelector[1];
					PD_SetFeedbackSelector(0U, precisionActiveSelector[0]);
					PD_SetFeedbackSelector(1U, precisionActiveSelector[1]);
					LED_MANUAL_HIGH();
					LED_TABLE_HIGH();
					prevDACValid = 0;
					memset(unstableFlags, 0, sizeof(unstableFlags));
					break;
			}
		}
}

void modify_table_loop(void){
		if(aRxBuffer[2] == 0x03U && (aRxBuffer[3] == 0x03U || aRxBuffer[3] == 0x04U))
		{
				uint16_t point = ((uint16_t)aRxBuffer[4] << 8) | aRxBuffer[5];
				uint32_t tag = ((uint32_t)aRxBuffer[6] << 24) |
				        ((uint32_t)aRxBuffer[7] << 16) |
				        ((uint32_t)aRxBuffer[8] << 8) | aRxBuffer[9];
				if(usbCdcHostOpen && workState == TABLE_STATE &&
				   StressMultirate_IsEnabled(&stressMultirateState) &&
				   (stressMultirateNegotiatedVersion == STRESS_MULTIRATE_EXTENSION_VERSION_V4 ||
				    stressMultirateNegotiatedVersion == STRESS_MULTIRATE_EXTENSION_VERSION_V5) &&
				   point < STRESS_TABLE_POINT_COUNT && tag &&
				   !stressHoldRequestedTag && !stressHoldActiveTag &&
				   !stressRawCaptureRequested && !stressRawCaptureActive)
				{
						stressHoldRequestedPoint = point;
						stressHoldRequestedTag = tag;
						stressHoldRequestedProduction = aRxBuffer[3] == 0x04U;
				}
				ClearRxBuff();
				return;
		}
		if(aRxBuffer[2] == 0x03U && aRxBuffer[3] == 0x02U)
		{
				/* Arm the next complete frame, never the partially scanned frame in
				 * which this command happened to arrive. */
				if(workState == TABLE_STATE && aRxBuffer[4] <= 1U &&
				   !stressRawCaptureRequested && !stressHoldRequestedTag && !stressHoldActiveTag &&
				   (aRxBuffer[4] == 0U || StressMultirate_IsEnabled(&stressMultirateState)))
				{
						uint32_t tag = ((uint32_t)aRxBuffer[5] << 24) |
								((uint32_t)aRxBuffer[6] << 16) |
								((uint32_t)aRxBuffer[7] << 8) | aRxBuffer[8];
						if(aRxBuffer[4] == 0U || tag != 0U)
						{
								stressRawRequestedTag = tag;
								stressRawCaptureRequested = aRxBuffer[4] ? 2U : 1U;
						}
				}
				ClearRxBuff();
				return;
		}
		/* The existing USB receive path always assembles exactly 808 bytes, so
		 * provisioning can share it without altering any legacy command size. */
		if(WifiTransport_HandleUsbStatusQuery(aRxBuffer, USART_RX_SIZE) ||
		   WifiTransport_HandleUsbConfig(aRxBuffer, USART_RX_SIZE))
		{
				ClearRxBuff();
				return;
		}
		// 0x01 change wave_time
		if(aRxBuffer[3] == 0x01){
				uint32_t requestedDelay = ((uint32_t)aRxBuffer[7] << 24)
						| ((uint32_t)aRxBuffer[8] << 16)
						| ((uint32_t)aRxBuffer[9] << 8)
						| (uint32_t)aRxBuffer[10];
				wave_time = (requestedDelay > MAX_HOST_WAVE_DELAY_US)
						? MAX_HOST_WAVE_DELAY_US : requestedDelay;
//				uint8_t waveArray[2] = {(wave_time>>8)&0xFF, wave_time&0xFF};
//				USART_Queue_Send(waveArray, 2);
		}
		// 0x02 switch workState
		else if(aRxBuffer[3] == 0x02){
				if(aRxBuffer[8] <= PRECISION_TABLE_STATE)
				{
						ApplyWorkState(aRxBuffer[8]);
						WifiTransport_OnLocalModeChanged(workState);
				}
		}
		else if(aRxBuffer[3] == 0x03){
#if 0  /* DAC selection is disabled; PI11210 is the only target. */
				dacTarget = aRxBuffer[8];
#endif
		}
		else if(aRxBuffer[3] == 0x04){
				getFilterDiff();
		}
		else if(aRxBuffer[3] == 0x05){
				getWaveRange();
		}
		else if(aRxBuffer[3] == 0x06){
				/* Fixed CH0/CH1 transimpedance control for the 2001-point
				 * equal-interval scan.  The reply is based on the electrical
				 * choseA/B pin levels, not on a software variable echoed back. */
				uint8_t requested0 = aRxBuffer[4];
				uint8_t requested1 = aRxBuffer[5];
				uint8_t accepted = (workState == EXTRA_STATE
						&& requested0 <= PD_FEEDBACK_SELECTOR_20K
						&& requested1 <= PD_FEEDBACK_SELECTOR_20K) ? 1U : 0U;
				if(accepted)
				{
						PD_SetFeedbackSelector(0U, requested0);
						PD_SetFeedbackSelector(1U, requested1);
						delay_us(STRESS_GAIN_SWITCH_SETTLE_US);
				}
				extraFeedbackSelector[0] = PD_ReadFeedbackSelector(0U);
				extraFeedbackSelector[1] = PD_ReadFeedbackSelector(1U);
				accepted = (uint8_t)(accepted
						&& extraFeedbackSelector[0] == requested0
						&& extraFeedbackSelector[1] == requested1);
				aTxBuffer[0] = 0xFFU;
				aTxBuffer[1] = 0xFFU;
				aTxBuffer[2] = EXTRA_STATE;
				aTxBuffer[3] = 0x03U;
				aTxBuffer[4] = extraFeedbackSelector[0];
				aTxBuffer[5] = extraFeedbackSelector[1];
				aTxBuffer[6] = accepted ? 0x21U : 0xE1U;
				USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
				ClearTxBuff();
		}
		else if(aRxBuffer[3] == 0x07){
				/* Arm a fixed, operator-selected CH0/CH1 transimpedance for the
				 * next stress session.  Configuration is accepted only from the
				 * safe EXTRA state and is verified from the physical choseA/B pins. */
				uint8_t requested0 = aRxBuffer[4];
				uint8_t requested1 = aRxBuffer[5];
				uint8_t accepted = (workState == EXTRA_STATE
						&& requested0 <= PD_FEEDBACK_SELECTOR_20K
						&& requested1 <= PD_FEEDBACK_SELECTOR_20K) ? 1U : 0U;
				if(accepted)
				{
						PD_SetFeedbackSelector(0U, requested0);
						PD_SetFeedbackSelector(1U, requested1);
						delay_us(STRESS_GAIN_SWITCH_SETTLE_US);
						accepted = (uint8_t)(PD_ReadFeedbackSelector(0U) == requested0
								&& PD_ReadFeedbackSelector(1U) == requested1);
				}
				if(accepted)
				{
						stressRequestedSelector[0] = requested0;
						stressRequestedSelector[1] = requested1;
						stressManualFeedbackArmed = 1U;
				}
				else stressManualFeedbackArmed = 0U;
				aTxBuffer[0] = 0xFFU;
				aTxBuffer[1] = 0xFFU;
				aTxBuffer[2] = EXTRA_STATE;
				aTxBuffer[3] = 0x04U;
				aTxBuffer[4] = PD_ReadFeedbackSelector(0U);
				aTxBuffer[5] = PD_ReadFeedbackSelector(1U);
				aTxBuffer[6] = accepted ? 0x21U : 0xE1U;
				USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
				ClearTxBuff();
		}
		else if(aRxBuffer[3] == 0x08U){
				/* Opt-in CH1 multirate arm for the next stress session.
				 *   byte4 enable
				 *   byte5 complete-MAP period; zero means first MAP only
				 *   byte6..7 activation threshold (ADC codes)
				 *   byte8..9 release threshold (ADC codes)
				 *   byte10 requested schedule version (v3 compatibility or v4)
				 * It is accepted only while EXTRA is idle and is never persisted. */
				uint8_t enable = aRxBuffer[4];
				uint8_t mapPeriod = aRxBuffer[5];
				uint16_t activate = ((uint16_t)aRxBuffer[6] << 8) | aRxBuffer[7];
				uint16_t release = ((uint16_t)aRxBuffer[8] << 8) | aRxBuffer[9];
				uint8_t requestedVersion = aRxBuffer[10];
				uint8_t versionSupported = (uint8_t)(
						requestedVersion == STRESS_MULTIRATE_EXTENSION_VERSION_V3 ||
						requestedVersion == STRESS_MULTIRATE_EXTENSION_VERSION_V4 ||
						requestedVersion == STRESS_MULTIRATE_EXTENSION_VERSION_V5);
				uint8_t accepted = (uint8_t)(workState == EXTRA_STATE &&
						enable <= 1U && versionSupported);
				if(enable && accepted)
				{
						accepted = (uint8_t)(accepted &&
								(mapPeriod == STRESS_MULTIRATE_CONTINUOUS_MAP_PERIOD ||
								 mapPeriod == STRESS_MULTIRATE_REFERENCE_MAP_PERIOD ||
								 (mapPeriod >= STRESS_MULTIRATE_MIN_MAP_PERIOD &&
								  mapPeriod <= STRESS_MULTIRATE_MAX_MAP_PERIOD)) &&
								activate > 0U && release <= activate);
				}
				if(accepted)
				{
						if(enable)
						{
								stressMultirateArmed = 1U;
								stressMultirateNegotiatedVersion = requestedVersion;
								stressMultirateMapPeriod = mapPeriod;
								stressMultirateActivateCodes = activate;
								stressMultirateReleaseCodes = release;
								StressMultirate_SetSingleTrackEnabled(
										&stressMultirateState,
										(uint8_t)(requestedVersion >=
												STRESS_MULTIRATE_EXTENSION_VERSION_V4));
						}
						else stressMultirateDisarmAndClear();
				}
				else stressMultirateDisarmAndClear();
				aTxBuffer[0] = 0xFFU;
				aTxBuffer[1] = 0xFFU;
				aTxBuffer[2] = EXTRA_STATE;
				aTxBuffer[3] = 0x05U;
				aTxBuffer[4] = stressMultirateArmed;
				aTxBuffer[5] = stressMultirateMapPeriod;
				aTxBuffer[6] = (uint8_t)(stressMultirateActivateCodes >> 8);
				aTxBuffer[7] = (uint8_t)stressMultirateActivateCodes;
				aTxBuffer[8] = (uint8_t)(stressMultirateReleaseCodes >> 8);
				aTxBuffer[9] = (uint8_t)stressMultirateReleaseCodes;
				aTxBuffer[10] = accepted ? 0x21U : 0xE1U;
				/* Accepted arm echoes the negotiated version.  Disarm and rejected
				 * requests echo the requested byte without retaining a capability. */
				aTxBuffer[11] = (accepted && enable)
						? stressMultirateNegotiatedVersion : requestedVersion;
				USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
				ClearTxBuff();
		}
		ClearRxBuff();
}

void getFilterDiff(void){
		uint16_t getDiffIdx = 4U;
		uint16_t validateIdx = 4U;
	
		uint8_t dlow = 0, dhigh = 0;
		/* Validate all four variable-length lists before touching the destination.
		 * A damaged 808-byte command must not read past aRxBuffer or partially
		 * replace the previous filter configuration. */
		for(uint8_t channel = 0U; channel < 4U; channel++)
		{
				uint8_t count;
				if(validateIdx >= USART_RX_SIZE) return;
				count = aRxBuffer[validateIdx++];
				if((uint16_t)count > (uint16_t)((USART_RX_SIZE - validateIdx) / 2U))
						return;
				validateIdx = (uint16_t)(validateIdx + (uint16_t)count * 2U);
		}
		memset(unstableFlags, 0, sizeof(unstableFlags));
	
		uint8_t diffs1 = aRxBuffer[getDiffIdx++];
		for(uint8_t i=0;i<diffs1;i++){
				dhigh = aRxBuffer[getDiffIdx++];
				dlow = aRxBuffer[getDiffIdx++];
				uint16_t index = ((uint16_t)dhigh << 8) | dlow;
				if(index < Number) unstableFlags[index][0] = 1;
		}
		
		uint8_t diffs2 = aRxBuffer[getDiffIdx++];
		for(uint8_t i=0;i<diffs2;i++){
				dhigh = aRxBuffer[getDiffIdx++];
				dlow = aRxBuffer[getDiffIdx++];
				uint16_t index = ((uint16_t)dhigh << 8) | dlow;
				if(index < Number) unstableFlags[index][1] = 1;
		}
		
		uint8_t diffs3 = aRxBuffer[getDiffIdx++];
		for(uint8_t i=0;i<diffs3;i++){
				dhigh = aRxBuffer[getDiffIdx++];
				dlow = aRxBuffer[getDiffIdx++];
				uint16_t index = ((uint16_t)dhigh << 8) | dlow;
				if(index < Number) unstableFlags[index][2] = 1;
		}
		
		uint8_t diffs4 = aRxBuffer[getDiffIdx++];
		for(uint8_t i=0;i<diffs4;i++){
				dhigh = aRxBuffer[getDiffIdx++];
				dlow = aRxBuffer[getDiffIdx++];
				uint16_t index = ((uint16_t)dhigh << 8) | dlow;
				if(index < Number) unstableFlags[index][3] = 1;
		}
}

void getWaveRange(void){
		uint16_t getResIdx = 4;
		
		int l_high = aRxBuffer[getResIdx++];
		int l_low = aRxBuffer[getResIdx++];
	
		int r_high = aRxBuffer[getResIdx++];
		int r_low = aRxBuffer[getResIdx++];
	
		start_wave = (l_high<<8)+l_low;
		end_wave = (r_high<<8)+r_low;
}

void sampleVoltage(void){
		delay_us(wave_time);
		for(uint8_t adc_idx=0;adc_idx<4;adc_idx++){
				adcData = ADC_Write_Read(adc_idx) & 0x0FFF;
				uADCOriginvalues[adc_idx] = adcData;
				txBuffer[txCount++] = (adcData >> 8) & 0xFF;
				txBuffer[txCount++] = adcData & 0xFF;
				txBuffer[txCount++] = 0;
		}
}

void sampleVoltageStable(uint16_t i){
		(void)i;
		uint16_t first[4] = {0};
		uint16_t second[4] = {0};
		uint16_t focusThird[4] = {0};
		uint16_t focusFourth[4] = {0};
		uint16_t slowRecheck[4] = {0};
		uint8_t recheck = 0;
		uint8_t sampleMask = stressSamplingChannelMask();
		uint8_t pairedProbe = stressHoldActiveProduction && stressHoldActiveTag &&
		        activeScanPointIndex == stressHoldActivePoint;
		if(pairedProbe) stressPairSampleMask = sampleMask;
		uint8_t focusLowGain =
				((sampleMask & 0x04U) && unstableFlags[activeScanPointIndex][2])
				|| ((sampleMask & 0x08U) && unstableFlags[activeScanPointIndex][3]);
		if(sampleMask == 0U)
		{
				for(uint8_t adc_idx = 0U; adc_idx < 4U; adc_idx++)
				{
						uADCOriginvalues[adc_idx] = 0U;
						txBuffer[txCount++] = 0U;
						txBuffer[txCount++] = 0U;
				}
				return;
		}
		uint32_t firstDelayUs = wave_time + FAST_ADC_FIRST_DELAY_US;
		if(isStressSectionBoundary(activeScanPointIndex))
				firstDelayUs += FAST_BOUNDARY_EXTRA_DELAY_US;

		delay_us(firstDelayUs);
		if(pairedProbe) stressPairStartCycles[0] = DWT->CYCCNT;
		ADC_ReadMask(sampleMask, first);
		if(!ADC_LastTransferOk()) return;
		if(pairedProbe)
		{
				stressPairEndCycles[0] = DWT->CYCCNT;
				stressPairCodes[0] = first[1];
				stressPairValidMask |= 1U;
		}
		delay_us(FAST_ADC_SPACING_US);
		if(pairedProbe) stressPairStartCycles[1] = DWT->CYCCNT;
		ADC_ReadMask(sampleMask, second);
		if(!ADC_LastTransferOk()) return;
		if(pairedProbe)
		{
				stressPairEndCycles[1] = DWT->CYCCNT;
				stressPairCodes[1] = second[1];
				stressPairValidMask |= 2U;
		}
		if(stressRawCaptureActive && activeScanPointIndex < STRESS_TABLE_POINT_COUNT)
		{
				stressRawFirst[activeScanPointIndex] = first[1];
				stressRawSecond[activeScanPointIndex] = second[1];
		}
		/* Either populated high-gain channel can request the slower RC pair;
		 * discovery is not biased toward CH1. */
		uint16_t slowChannelDifference = 0U;
		for(uint8_t channel = 0U; channel < 2U; channel++)
		{
				if((sampleMask & (uint8_t)(1U << channel)) == 0U) continue;
				uint16_t difference = (second[channel] >= first[channel])
						? (second[channel] - first[channel])
						: (first[channel] - second[channel]);
				if(difference > slowChannelDifference) slowChannelDifference = difference;
		}
		if(slowChannelDifference > FAST_ADC_RECHECK_DIFF_CODES) recheck = 1;
		if(focusLowGain)
		{
				delay_us(CH23_FOCUS_SPACING_US);
				ADC_ReadMask(sampleMask, focusThird);
				if(!ADC_LastTransferOk()) return;
				delay_us(CH23_FOCUS_SPACING_US);
				ADC_ReadMask(sampleMask, focusFourth);
				if(!ADC_LastTransferOk()) return;
		}
		if(recheck)
		{
			delay_us(FAST_ADC_RECHECK_SPACING_US
						- (focusLowGain ? (2U * CH23_FOCUS_SPACING_US) : 0U));
				if(pairedProbe) stressPairStartCycles[2] = DWT->CYCCNT;
				ADC_ReadMask(sampleMask, slowRecheck);
				if(!ADC_LastTransferOk()) return;
				if(pairedProbe)
				{
						stressPairEndCycles[2] = DWT->CYCCNT;
						stressPairCodes[2] = slowRecheck[1];
						stressPairValidMask |= 4U;
				}
		}
		if(stressRawCaptureActive && activeScanPointIndex < STRESS_TABLE_POINT_COUNT)
		{
				stressRawSlow[activeScanPointIndex] = slowRecheck[1];
				stressRawRecheck[activeScanPointIndex] = recheck;
		}

		for(uint8_t adc_idx = 0; adc_idx < 4; adc_idx++)
		{
				if((sampleMask & (uint8_t)(1U << adc_idx)) == 0U)
				{
						uADCOriginvalues[adc_idx] = 0U;
						txBuffer[txCount++] = 0U;
						txBuffer[txCount++] = 0U;
						continue;
				}
				uint16_t sample1 = first[adc_idx];
				uint16_t sample2 = second[adc_idx];
				/* CH0/CH1 use a switched transimpedance resistor.  Select the RC
				 * reconstruction coefficient from the actual fixed/automatic
				 * 40/20/5/2 kOhm state. */
				int32_t alpha = (adc_idx < 2U)
						? stressFeedbackAlphaQ15(stressActiveSelector[adc_idx], 0U)
						: CH23_ALPHA_Q15;
				int32_t estimate;
				if(focusLowGain && adc_idx >= 2)
				{
						int32_t weighted =
								((int32_t)first[adc_idx] * CH23_FOCUS_W0_Q15)
								+ ((int32_t)second[adc_idx] * CH23_FOCUS_W1_Q15)
								+ ((int32_t)focusThird[adc_idx] * CH23_FOCUS_W2_Q15)
								+ ((int32_t)focusFourth[adc_idx] * CH23_FOCUS_W3_Q15);
						estimate = (weighted >= 0)
								? ((weighted + Q15_ONE / 2) / Q15_ONE)
								: ((weighted - Q15_ONE / 2) / Q15_ONE);
				}
				else
				{
				if(recheck && adc_idx < 2)
				{
						sample1 = second[adc_idx];
						sample2 = slowRecheck[adc_idx];
						alpha = stressFeedbackAlphaQ15(
								stressActiveSelector[adc_idx], 1U);
				}
				int32_t denominator = Q15_ONE - alpha;
				int32_t numerator = ((int32_t)sample2 * Q15_ONE)
						- ((int32_t)sample1 * alpha);
				if(numerator >= 0)
				{
						estimate = (numerator + denominator / 2) / denominator;
				}
				else
				{
						estimate = (numerator - denominator / 2) / denominator;
				}
				}
				if(estimate < 0) estimate = 0;
				if(estimate > 4095) estimate = 4095;
				adcData = (uint16_t)estimate;
				if(pairedProbe && adc_idx == 1U)
				{
						stressPairEstimate = adcData;
						stressPairEstimateValid = 1U;
				}
				uADCOriginvalues[adc_idx] = adcData;
				if(stressRawCaptureActive && adc_idx == 1U
						&& activeScanPointIndex < STRESS_TABLE_POINT_COUNT)
				{
						stressRawEstimate[activeScanPointIndex] = adcData;
				}
				stressObserveChannelSample(adc_idx, adcData);
				txBuffer[txCount++] = (adcData >> 8) & 0xFF;
				txBuffer[txCount++] = adcData & 0xFF;
		}
}

void sampleVoltagePrecision(uint16_t i){
		uint16_t samples[PRECISION_ADC_SAMPLE_COUNT][2] = {0};
		uint32_t settleUs = wave_time + TEMPERATURE_ADC_SETTLE_US;
		if(isTuningSectionBoundary(i))
		{
				settleUs += PRECISION_ADC_BOUNDARY_EXTRA_DELAY_US;
		}

		delay_us(settleUs);
		for(uint8_t sampleIndex = 0; sampleIndex < PRECISION_ADC_SAMPLE_COUNT; sampleIndex++)
		{
				/* The reported local collapse was reproduced with direct EXTRA
				 * reads and proved to be a laser cavity-path effect, not an ADC
				 * channel-pipeline error.  Keep the validated two-channel pipeline
				 * transaction here to minimize temperature-frame acquisition time. */
				ADC_ReadTwo(samples[sampleIndex]);
				if(!ADC_LastTransferOk()) return;
				if(sampleIndex + 1U < PRECISION_ADC_SAMPLE_COUNT)
				{
						delay_us(PRECISION_ADC_SPACING_US);
				}
		}

		for(uint8_t adcIndex = 0; adcIndex < 2U; adcIndex++)
		{
				uint16_t sorted[PRECISION_ADC_SAMPLE_COUNT];
				for(uint8_t sampleIndex = 0; sampleIndex < PRECISION_ADC_SAMPLE_COUNT; sampleIndex++)
				{
						sorted[sampleIndex] = samples[sampleIndex][adcIndex] & 0x0FFFU;
				}
				for(uint8_t index = 1U; index < PRECISION_ADC_SAMPLE_COUNT; index++)
				{
						uint16_t value = sorted[index];
						uint8_t position = index;
						while(position > 0U && sorted[position - 1U] > value)
						{
								sorted[position] = sorted[position - 1U];
								position--;
						}
						sorted[position] = value;
				}
				uint32_t sum = 0U;
				for(uint8_t index = PRECISION_ADC_TRIM_EACH_SIDE;
						index < PRECISION_ADC_SAMPLE_COUNT - PRECISION_ADC_TRIM_EACH_SIDE;
						index++) sum += sorted[index];
				uint8_t kept = PRECISION_ADC_SAMPLE_COUNT
						- 2U * PRECISION_ADC_TRIM_EACH_SIDE;
				adcData = (uint16_t)((sum + kept / 2U) / kept);
				uADCOriginvalues[adcIndex] = adcData;
				txBuffer[txCount++] = (adcData >> 8) & 0xFFU;
				txBuffer[txCount++] = adcData & 0xFFU;
		}
		/* Preserve the legacy four-channel USB point stride without touching
		 * the CH2/CH3 ADC inputs in temperature mode. */
		uADCOriginvalues[2] = 0U;
		uADCOriginvalues[3] = 0U;
		for(uint8_t adcIndex = 2U; adcIndex < 4U; adcIndex++)
		{
				txBuffer[txCount++] = 0U;
				txBuffer[txCount++] = 0U;
		}
}

void sampleTemperature(void){
		/* Full 1-Wire service runs between frames in main.  Snapshot its cached
		 * value here so a due scratchpad read cannot stretch an optical frame.
		 * The fast tick keeps the stale-data/full-fan fail-safe current. */
		ThermalControl_SafetyTick();
		if(ThermalControl_IsTemperatureValid())
			tempData = ThermalControl_GetTemperatureC();
		else
			tempData = 0.0f;
		tempInt = (int)tempData;
		tempDec = (int)((tempData-tempInt)*10000);
		txBuffer[txCount++] = (tempInt>>8) & 0xFF;
		txBuffer[txCount++] = tempInt & 0xFF;
		txBuffer[txCount++] = (tempDec>>8) & 0xFF;
		txBuffer[txCount++] = tempDec & 0xFF;
}

uint8_t sendTxBuffer(int dac_size, int p1, int p2, int p3, int p4){
		(void)dac_size; (void)p1; (void)p2; (void)p3; (void)p4;
		uint8_t published = 0U;
		txBuffer[txCount++] = 0xFF;
		txBuffer[txCount++] = 0xEF;
		uint16_t floating_size = txCount;
		
		if(usbCdcHostOpen)
		{
			dma_transfer_complete = 0;
			if(CDC_Transmit_FS(txBuffer, floating_size) == USBD_OK)
				published = 1U;
			else
				dma_transfer_complete = 1;
		}
		else
		{
			dma_transfer_complete = 1;
			published = WifiTransport_QueueRawScanFrame(
					txBuffer,
					floating_size,
					(uint16_t)dac_size,
					(workState == PRECISION_TABLE_STATE)
							? 0x03U : stressActiveChannelMask);
		}
		return published;
}

void ClearTxBuff(){
		for(uint8_t i=2;i<USART_TX_SIZE;i++){
				aTxBuffer[i] = 0;
		}
}

void ClearRxBuff(void){
		for (uint16_t i = 0; i < USART_RX_SIZE; i++)
	  {
				aRxBuffer[i] = 0;
		}
}

void checkTemp(uint8_t mode){
		tempData = M1820Z_GetTmp();
		tempInt = (int)tempData;
		tempDec = (int)((tempData-tempInt)*10000);
	
		aTxBuffer[2] = mode;// the mode of this tx
		aTxBuffer[3] = 0x01;// 0x01 refer temperature return
		aTxBuffer[4] = (tempInt>>8) & 0xFF;
		aTxBuffer[5] = tempInt & 0xFF;
		aTxBuffer[6] = (tempDec>>8) & 0xFF;
		aTxBuffer[7] = tempDec & 0xFF; 
	
		USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
		ClearTxBuff();
}

void scanWave(void){
#if 0  /* Legacy DAC selection retained for protocol reference. */
		if(aRxBuffer[3] == 0x00){
				scanWave_U();
		}
		else if(aRxBuffer[3] == 0x01){
				scanWave_I();
		}
#endif
		scanWave_I();
}

#if 0  /* Legacy MS5614T scan path retained for reference. */
void scanWave_U(void){
		uint16_t WriteData = 0;
		for(uint8_t i = 0; i < 5; i++)
		{
				WriteData = ((aRxBuffer[4 + 2 * i] << 8) + aRxBuffer[5 + 2 * i]);
				switch(i){
					case 0:MS5614T2_SetCode(MS5614T_DAC_A, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
					case 1:MS5614T2_SetCode(MS5614T_DAC_C, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
					case 2:MS5614T2_SetCode(MS5614T_DAC_B, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
					case 3:MS5614T_SetCode(MS5614T_DAC_A, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
					case 4:MS5614T_SetCode(MS5614T_DAC_C, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
				}			
		}
		
		uint8_t txIdx = 2;
		uint8_t sa = 1;
		aTxBuffer[txIdx++] = MANUAL_STATE;// the mode of this tx
		aTxBuffer[txIdx++] = 0x00;// 0x00 refer scan wave return
		aTxBuffer[txIdx++] = 0x21;// return flag
		
		adcData = ADC_Write_Read_Stable(6, &sa, 1) & 0x0FFF;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
		aTxBuffer[txIdx++] = sa;
		
		sa = 1;
		adcData = ADC_Write_Read_Stable(7, &sa, 1) & 0x0FFF;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
		aTxBuffer[txIdx++] = sa;
		
		USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
		ClearRxBuff();
		ClearTxBuff();
}
#endif

void scanWave_I(void){
		uint16_t WriteData = 0;
		uint8_t allWritesOk = 1U;
		for(uint8_t i = 0; i < 5; i++)
		{
				WriteData = ((aRxBuffer[4 + 2 * i] << 8) + aRxBuffer[5 + 2 * i]);
				switch(i){
					case 0:if(PI11210_SetCode(IDAC5, WriteData) != HAL_OK) allWritesOk = 0U;break;//GAIN
					case 1:if(PI11210_SetCode(IDAC6, WriteData) != HAL_OK) allWritesOk = 0U;break;//SOA
					case 2:if(PI11210_SetCode(IDAC1, WriteData) != HAL_OK) allWritesOk = 0U;break;//PHASE
					case 3:if(PI11210_SetCode(IDAC4, WriteData) != HAL_OK) allWritesOk = 0U;break;//WAVEA
					case 4:if(PI11210_SetCode(IDAC7, WriteData) != HAL_OK) allWritesOk = 0U;break;//WAVEB
				}			
		}
		
		uint8_t txIdx = 2;
		uint8_t sa = 1;
		aTxBuffer[txIdx++] = MANUAL_STATE;// the mode of this tx
		aTxBuffer[txIdx++] = 0x00;// 0x00 refer scan wave return
		aTxBuffer[txIdx++] = allWritesOk ? 0x21U : 0xE1U;// return flag
		
		adcData = allWritesOk ? (ADC_Write_Read_Stable(6, &sa, 1) & 0x0FFF) : 0U;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
		aTxBuffer[txIdx++] = sa;
		
		sa = 1;
		adcData = allWritesOk ? (ADC_Write_Read_Stable(7, &sa, 1) & 0x0FFF) : 0U;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
		aTxBuffer[txIdx++] = sa;
		
		USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
		ClearRxBuff();
		ClearTxBuff();
}

void singleValue(void){
#if 0  /* Legacy DAC selection retained for protocol reference. */
		if(aRxBuffer[3] == 0x00){
				singleValue_U();
		}
		else if(aRxBuffer[3] == 0x01){
				singleValue_I();
		}
#endif
		singleValue_I();
}

#if 0  /* Legacy MS5614T single-value path retained for reference. */
void singleValue_U(void){
		uint16_t WriteData = 0;
		uint8_t txIdx = 2;
		uint8_t highRecv = 0, lowRecv = 0;
		aTxBuffer[txIdx++] = EXTRA_STATE;
		aTxBuffer[txIdx++] = 0x00;
	
		for(uint8_t i = 0; i < 5; i++)
		{
				highRecv = aRxBuffer[4+2*i];
				lowRecv = aRxBuffer[5+2*i];
				WriteData = (highRecv << 8) + lowRecv;
				aTxBuffer[txIdx++] = highRecv;
				aTxBuffer[txIdx++] = lowRecv;
				switch(i){
					case 0:MS5614T2_SetCode(MS5614T_DAC_A, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
					case 1:MS5614T2_SetCode(MS5614T_DAC_C, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
					case 2:MS5614T2_SetCode(MS5614T_DAC_B, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
					case 3:MS5614T_SetCode(MS5614T_DAC_A, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
					case 4:MS5614T_SetCode(MS5614T_DAC_C, WriteData, MS5614T_SPEED_FAST, MS5614T_NORMAL);break;
				}		
		}
		USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
		ClearRxBuff();
		ClearTxBuff();
}
#endif

void singleValue_I(void){
		uint16_t values[5];
		uint8_t allWritesOk = 1U;
		uint8_t requestedMode = (aRxBuffer[14] == PI11210_SOA_SHUTTER)
				? PI11210_SOA_SHUTTER : PI11210_SOA_SOURCE;
		const PI11210_Channeld_t channels[5] = {
				IDAC5, IDAC6, IDAC1, IDAC4, IDAC7
		};

		aTxBuffer[2] = EXTRA_STATE;
		aTxBuffer[3] = 0x00;
		for(uint8_t i = 0U; i < 5U; i++)
		{
				uint16_t requested = ((uint16_t)aRxBuffer[4U + 2U * i] << 8)
						| aRxBuffer[5U + 2U * i];
				values[i] = PI11210_LimitCode(channels[i], requested);
		}

		if(requestedMode == PI11210_SOA_SHUTTER)
		{
				/* Assert the hardware zero-current gate before changing any other
				 * bias.  A malformed host packet cannot request reverse current. */
				values[1] = PI11210_IDAC6_OFF_CODE;
				if(PI11210_SetSOAShutter(1U) != HAL_OK) allWritesOk = 0U;
		}

		for(uint8_t i = 0U; i < 5U; i++)
		{
				if(requestedMode == PI11210_SOA_SHUTTER && i == 1U) continue;
				if(PI11210_SetCode(channels[i], values[i]) != HAL_OK) allWritesOk = 0U;
		}

		for(uint8_t i = 0U; i < 5U; i++)
		{
				aTxBuffer[4U + 2U * i] = (uint8_t)(values[i] >> 8);
				aTxBuffer[5U + 2U * i] = (uint8_t)values[i];
		}
		{
				PI11210_Status_t pi = PI11210_GetStatus();
				aTxBuffer[14] = pi.soaMode;
				aTxBuffer[15] = 0x80U
						| (pi.online ? 0x01U : 0U)
						| (pi.partValid ? 0x02U : 0U)
						| (allWritesOk ? 0x04U : 0U)
						| ((pi.rawStatus & PI11210_STATUS_HI_TEMP) ? 0x08U : 0U)
						| ((pi.rawStatus & PI11210_STATUS_OVR_TEMP) ? 0x10U : 0U)
						| ((pi.rawStatus & PI11210_STATUS_PRO_TEMP) ? 0x20U : 0U)
						| (pi.busRecovered ? 0x40U : 0U);
				aTxBuffer[16] = (uint8_t)(pi.rawStatus >> 8);
				aTxBuffer[17] = (uint8_t)pi.rawStatus;
				aTxBuffer[18] = (uint8_t)(pi.i2cErrorCount >> 8);
				aTxBuffer[19] = (uint8_t)pi.i2cErrorCount;
		}
		prevDACValid = 0U;
		USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
		ClearRxBuff();
		ClearTxBuff();
}

void checkRT(void){
		uint8_t txIdx = 2;
		uint8_t sa = 0;
		aTxBuffer[txIdx++] = EXTRA_STATE;
		aTxBuffer[txIdx++] = 0x02;
	
		adcData = ADC_Write_Read_Stable(6, &sa, 1) & 0x0FFF;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
		
		adcData = ADC_Write_Read_Stable(7, &sa, 1) & 0x0FFF;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
	
		adcData = ADC_Write_Read_Stable(0, &sa, 1) & 0x0FFF;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
		
		adcData = ADC_Write_Read_Stable(1, &sa, 1) & 0x0FFF;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
	
		adcData = ADC_Write_Read_Stable(2, &sa, 1) & 0x0FFF;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
		
		adcData = ADC_Write_Read_Stable(3, &sa, 1) & 0x0FFF;
		aTxBuffer[txIdx++] = (adcData >> 8) & 0xFF;
		aTxBuffer[txIdx++] = (adcData) & 0xFF;
		
		USB_Queue_Send(aTxBuffer, USART_TX_SIZE);
		ClearTxBuff();
}
