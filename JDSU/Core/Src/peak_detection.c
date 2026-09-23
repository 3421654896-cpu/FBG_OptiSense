#include "main.h"

uint16_t item=0;

uint16_t adc1[Number]={0};
uint16_t adc2[Number]={0};
uint16_t adc3[Number]={0};
uint16_t adc4[Number]={0};

int adc_norvec[Number]={0};//????,??????

float peaks1[PEAKS_COUNT]={0};
float peaks2[PEAKS_COUNT]={0};
float peaks3[PEAKS_COUNT]={0};
float peaks4[PEAKS_COUNT]={0};

int int_part = 0;
int decimal_part = 0;

uint8_t Find_Peaks(uint16_t *adc_vec, float *peaks_vec, uint16_t adc_length){
	uint16_t initials[PEAKS_COUNT]={0};
	memset(peaks_vec, 0, sizeof(float)*PEAKS_COUNT);

	Find_Initial(adc_vec, initials, adc_length);
	
	uint8_t i=0;
	for(;i<PEAKS_COUNT;i++){
		//if(i<5) initials[i] = 1;
		if(initials[i]==0) break;
		int start = initials[i]-INTERVAL>=0?initials[i]-INTERVAL:0;
		int end = initials[i]+INTERVAL<adc_length?initials[i]+INTERVAL:adc_length;
		float sumxy=0;
		int sumy=0;
		for(int j=start;j<=end;j++){
			sumxy+=adc_vec[j]*(Wave_DATA[j][0]+Wave_DATA[j][1]*0.001);
			sumy+=adc_vec[j];
		}
		peaks_vec[i] = sumxy/(float)sumy;
	}
	return i;
}

void Find_Initial(uint16_t *adc_vec, uint16_t *initials, uint16_t adc_length){
	int ma = adc_vec[0], mi = adc_vec[0];
	for(int i=0;i<adc_length;i++){
		if(adc_vec[i]>ma) ma = adc_vec[i];
		if(adc_vec[i]<mi) mi = adc_vec[i];
	}
	int mean = 0;
	int gap = ma-mi;
	for(int i=0;i<adc_length;i++){
		adc_norvec[i] = (adc_vec[i]-mi);
		mean+=adc_norvec[i];
	}
	
	if(100*mean>10*adc_length*gap) return;
	uint16_t it = 0;
	for(uint16_t i=1;i<adc_length-1;i++){
		if(it>=PEAKS_COUNT) break;
		int start = i-INTERVAL>=0?i-INTERVAL:0;
		int end = i+INTERVAL<=adc_length?i+INTERVAL:adc_length;
		
		int front=-0xFFFF, back=-0xFFFF;
		for(int j=start;j<i;j++){
			if(adc_norvec[j]>front) front = adc_norvec[j];
		}
		for(int j=i+1;j<end;j++){
			if(adc_norvec[j]>back) back = adc_norvec[j];
		}
		if((adc_norvec[i]>front||adc_norvec[i]==front)&&(adc_norvec[i]>back||adc_norvec[i]==back)&&adc_norvec[i]>PEAK_THRESHOLD){
			if(10*(adc_norvec[i]-adc_norvec[i-1])>gap*5 || 10*(adc_norvec[i]-adc_norvec[i-1])>gap*5){
//				i+=INTERVAL;
				continue;
			}
			initials[it] = i;
			i+=INTERVAL;
			it++;
		}
	}
}

void FillPeaks(uint8_t p1, uint8_t p2, uint8_t p3, uint8_t p4){
		txBuffer[txCount++] = 0xAB;
	
		txBuffer[txCount++] = p1;
		int i=0;
		for(;i<p1;i++){
			int_part = peaks1[i];
			decimal_part = (peaks1[i] - int_part)*1000;
			txBuffer[txCount++] = (int_part>>8) & 0xFF;
			txBuffer[txCount++] = int_part & 0xFF;
			txBuffer[txCount++] = (decimal_part>>8) & 0xFF;
			txBuffer[txCount++] = decimal_part & 0xFF;
		}
		
		i=0;
		txBuffer[txCount++] = p2;
		for(;i<p2;i++){
			int_part = peaks2[i];
			decimal_part = (peaks2[i] - int_part)*1000;
			txBuffer[txCount++] = (int_part>>8) & 0xFF;
			txBuffer[txCount++] = int_part & 0xFF;
			txBuffer[txCount++] = (decimal_part>>8) & 0xFF;
			txBuffer[txCount++] = decimal_part & 0xFF;
		}
		
		i=0;
		txBuffer[txCount++] = p3;
		for(;i<p3;i++){
			int_part = peaks3[i];
			decimal_part = (peaks3[i] - int_part)*1000;
			txBuffer[txCount++] = (int_part>>8) & 0xFF;
			txBuffer[txCount++] = int_part & 0xFF;
			txBuffer[txCount++] = (decimal_part>>8) & 0xFF;
			txBuffer[txCount++] = decimal_part & 0xFF;
		}
		
		i=0;
		txBuffer[txCount++] = p4;
		for(;i<p4;i++){
			int_part = peaks4[i];
			decimal_part = (peaks4[i] - int_part)*1000;
			txBuffer[txCount++] = (int_part>>8) & 0xFF;
			txBuffer[txCount++] = int_part & 0xFF;
			txBuffer[txCount++] = (decimal_part>>8) & 0xFF;
			txBuffer[txCount++] = decimal_part & 0xFF;
		}
}
