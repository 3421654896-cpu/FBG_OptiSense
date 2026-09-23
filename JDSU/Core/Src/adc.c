#include "main.h"

/* The populated input network is 1 kOhm followed by 150 pF at every CHx,
 * while the TPC5121 contributes about 22 pF internally.  The datasheet's
 * matching example requires about 1.06 us acquisition time for 12-bit,
 * 1-LSB settling.  Manual selection has a two-frame pipeline, so the selected
 * input receives the following conversion frame plus this explicit CS-high
 * guard before it is sampled.  At the configured 15 MHz SCLK this comfortably
 * exceeds the example settling time without adding a millisecond-scale delay
 * to every laser wavelength.
 */
#define TPC5121_ACQUISITION_GUARD_US 1U
#define TPC5121_SPI_TIMEOUT_MS 2U
#define TPC5121_MANUAL_CH0_CONFIG_FRAME 0x1800U
#define TPC5121_MANUAL_PRIME_FRAMES 3U

uint16_t adcSPI = 0;
uint32_t adcCount = 0;
uint16_t adcStable = 0;
uint16_t adcUnstable = 0;
uint16_t adcQueue[WINDOW_SIZE] = {0};
uint16_t adcUnstableList[Number] = {0};
static uint8_t adcLastTransferOk = 1U;
static uint32_t adcSpiErrorCount = 0U;

static void ADC_BeginTransferGroup(void)
{
		adcLastTransferOk = 1U;
}

uint8_t ADC_LastTransferOk(void)
{
		return adcLastTransferOk;
}

uint32_t ADC_GetSpiErrorCount(void)
{
		return adcSpiErrorCount;
}

void Reset_ADC_Queue(void){
		for(uint8_t i=0;i<WINDOW_SIZE;i++){
				adcQueue[i] = 0;
		}
		adcCount = 0;
		adcStable = 0;
		adcUnstable = 0;
}

static void ADC_Select_Chs(){
		uint16_t Chs = (1<<0) | (1<<1) | (1<<2) | (1<<3) | (1<<6) | (1<<7);
		ADC_SPI_Cmd(Chs);
}

void ADC_LOOP_SPI_Init(void){
		__HAL_SPI_CLEAR_OVRFLAG(&hspi1);

		ADC_BeginTransferGroup();
		/* 0x8000 is specifically the first frame of the datasheet's Auto-1
		 * programming sequence.  It is valid here because the next call writes
		 * the selected-channel bitmap. */
		ADC_SPI_Cmd(0x8000);
		ADC_Select_Chs();
		ADC_Loop_Start();
}

void ADC_MANUAL_SPI_Init(void){
		__HAL_SPI_CLEAR_OVRFLAG(&hspi1);

		/* 0x8000 enters Auto-1 programming and must never be used as a reset in
		 * manual mode.  Explicitly select Manual/CH0/0..VREF and clock enough
		 * identical frames to discard the one invalid power-up result and fill
		 * the documented two-frame conversion pipeline. */
		ADC_BeginTransferGroup();
		for(uint8_t frameIndex = 0U;
				frameIndex < TPC5121_MANUAL_PRIME_FRAMES && adcLastTransferOk;
				frameIndex++)
		{
				(void)ADC_SPI_Cmd(TPC5121_MANUAL_CH0_CONFIG_FRAME);
		}
}

uint16_t ADC_SPI_Cmd(uint16_t cmdF){
		uint16_t response = 0U;
		HAL_StatusTypeDef status;

		ADC_CS_LOW();
		status = HAL_SPI_TransmitReceive(&hspi1,
				(uint8_t*)&cmdF,
				(uint8_t*)&response,
				1U,
				TPC5121_SPI_TIMEOUT_MS);
		ADC_CS_HIGH();
		delay_us(TPC5121_ACQUISITION_GUARD_US);

		if(status != HAL_OK)
		{
				adcLastTransferOk = 0U;
				if(adcSpiErrorCount != 0xFFFFFFFFU) adcSpiErrorCount++;
				(void)HAL_SPI_Abort(&hspi1);
				adcSPI = 0U;
				return 0U;
		}

		adcSPI = response;
		return adcSPI;
}

uint16_t ADC_Write_Read(uint8_t ch){
		uint16_t frame = (0x1 << 12) | (ch << 7);
		/* TPC5121 manual-mode results have a two-frame pipeline:
		 * select CHx in frame N-2, acquire it in N-1, and read it in N.
		 * Clock three identical frames so the returned conversion belongs to ch.
		 */
		ADC_BeginTransferGroup();
		ADC_SPI_Cmd(frame);
		short_delay(1);
		if(!adcLastTransferOk) return 0U;
		ADC_SPI_Cmd(frame);
		short_delay(1);
		if(!adcLastTransferOk) return 0U;
		return ADC_SPI_Cmd(frame);
}

void ADC_ReadTwo(uint16_t values[2]){
		/* Temperature mode needs only CH0/CH1.  The final two selections flush
		 * the TPC5121 two-frame pipeline; CH2/CH3 are never selected. */
		static const uint8_t sequence[4] = {0, 1, 0, 0};
		values[0] = 0U;
		values[1] = 0U;
		ADC_BeginTransferGroup();
		for(uint8_t frameIndex = 0; frameIndex < 4; frameIndex++)
		{
				uint16_t frame = (0x1U << 12) | ((uint16_t)sequence[frameIndex] << 7);
				uint16_t result = ADC_SPI_Cmd(frame);
				if(!adcLastTransferOk) break;
				if(frameIndex >= 2U)
				{
						values[frameIndex - 2U] = result & 0x0FFFU;
				}
		}
		if(!adcLastTransferOk) values[0] = values[1] = 0U;
}

void ADC_ReadMask(uint8_t channelMask, uint16_t values[4]){
		/* Pack only enabled channels into the TPC5121 pipeline.  Two trailing
		 * selections flush its two-frame latency.  Disabled outputs are always
		 * zero, so callers cannot accidentally reuse a previous conversion. */
		uint8_t selected[4] = {0U, 0U, 0U, 0U};
		uint8_t selectedCount = 0U;
		ADC_BeginTransferGroup();
		for(uint8_t channel = 0U; channel < 4U; channel++)
		{
				values[channel] = 0U;
				if(channelMask & (uint8_t)(1U << channel))
				{
						selected[selectedCount++] = channel;
				}
		}
		if(selectedCount == 0U) return;

		for(uint8_t frameIndex = 0U; frameIndex < selectedCount + 2U; frameIndex++)
		{
				uint8_t selectedIndex = (frameIndex < selectedCount) ? frameIndex : 0U;
				uint8_t channel = selected[selectedIndex];
				uint16_t frame = (0x1U << 12) | ((uint16_t)channel << 7);
				uint16_t result = ADC_SPI_Cmd(frame);
				if(!adcLastTransferOk) break;
				if(frameIndex >= 2U)
				{
						channel = selected[frameIndex - 2U];
						values[channel] = result & 0x0FFFU;
				}
		}
		if(!adcLastTransferOk)
		{
				for(uint8_t channel = 0U; channel < 4U; channel++) values[channel] = 0U;
		}
}

void ADC_ReadFour(uint16_t values[4]){
		ADC_ReadMask(0x0FU, values);
}

void ADC_ReadSix(uint16_t values[6]){
		/* Diagnostic burst order: external PD CH0..CH3, then PDT and PDR.
		 * The TPC5121 result belongs to the selection made two frames earlier.
		 */
		static const uint8_t sequence[8] = {0, 1, 2, 3, 6, 7, 0, 0};
		for(uint8_t channel = 0U; channel < 6U; channel++) values[channel] = 0U;
		ADC_BeginTransferGroup();
		for(uint8_t frameIndex = 0; frameIndex < 8; frameIndex++)
		{
				uint16_t frame = (0x1U << 12) | ((uint16_t)sequence[frameIndex] << 7);
				uint16_t result = ADC_SPI_Cmd(frame);
				if(!adcLastTransferOk) break;
				if(frameIndex >= 2)
				{
						values[frameIndex - 2] = result & 0x0FFFU;
				}
		}
		if(!adcLastTransferOk)
		{
				for(uint8_t channel = 0U; channel < 6U; channel++) values[channel] = 0U;
		}
}

uint16_t ADC_Write_Read_Stable(uint8_t ch, uint8_t* unstable, uint8_t multi){
		uint16_t frame = (0x1 << 12) | (ch << 7);
		/* Zero would skip the sampling loop and index adcCount-1.  Treat it as
		 * the documented minimum multiplier instead of returning stale memory. */
		if(multi == 0U) multi = 1U;
		Reset_ADC_Queue();
		ADC_BeginTransferGroup();
		if(unstable != NULL) *unstable = 0U;
	
		uint32_t multiStCount = multi*STABLECOUNT;
		uint32_t multiQueSize = multi*QUEUE_SIZE;
	
		while(adcStable<multiStCount){
				adcQueue[adcCount%WINDOW_SIZE] = ADC_SPI_Cmd(frame);
				if(!adcLastTransferOk)
				{
						if(unstable != NULL) *unstable = 1U;
						return 0U;
				}
				
				if(adcCount==(multiQueSize-1)){
//						uint16_t adcSum = 0;
//						for(uint16_t i=1;i<QUEUE_SIZE;i++){
//								adcSum+=adcQueue[i];
//						}
						if(unstable != NULL) *unstable = 1U;
//						return adcQueue[adcCount%QUEUE_SIZE];
						return adcQueue[adcCount%WINDOW_SIZE];
				}
			
//				if(adcCount>2 && abs((int)(adcQueue[(adcCount-1)%QUEUE_SIZE])-(int)(adcQueue[(adcCount-2)%QUEUE_SIZE]))<205) adcStable++;
//				else adcStable=0;
			
				if(adcCount>=WINDOW_SIZE && adcCount%WINDOW_SIZE==0){
//						uint16_t cur = (adcCount-WINDOW_SIZE+1)%QUEUE_SIZE;
						uint16_t cur = 0;
						uint16_t amax = adcQueue[cur];
						uint16_t amin = adcQueue[cur];
						for(uint16_t j=1;j<WINDOW_SIZE;j++){
//								cur = (cur+1)%QUEUE_SIZE;
								cur = j;
								if(adcQueue[cur]>amax) amax = adcQueue[cur];
								if(adcQueue[cur]<amin) amin = adcQueue[cur];
						}
						if(amax-amin<STABLERANGE) adcStable++;
						else adcStable=0;
				}
				adcCount++;
		}
		return adcQueue[(adcCount-1)%WINDOW_SIZE];
}

void ADC_Loop_Start(void){
		ADC_BeginTransferGroup();
		ADC_SPI_Cmd(ADC_RESET_FRAME);
		if(!adcLastTransferOk) return;
		ADC_SPI_Cmd(ADC_ENTER_FRAME);
}

uint16_t ADC_Write_Loop(void){
		ADC_BeginTransferGroup();
		return ADC_SPI_Cmd(ADC_LOOP_FRAME);
}
