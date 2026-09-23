#ifndef __CH224Q_H
#define __CH224Q_H

#include <stdint.h>

/* On this PCB CH224Q CFG1/2/3 are hard-strapped 000 for a 9 V request.  Its
 * optional I2C readback is not wired, so current_valid remains false. */
typedef struct
{
    uint8_t online;
    uint8_t i2c_address;
    uint8_t protocol_status;
    uint8_t current_valid;
    uint16_t requested_voltage_mV;
    uint16_t available_current_mA;
    uint32_t power_limit_mW;
    uint16_t communication_errors;
    uint16_t age_ms;
} CH224Q_Status;

void CH224Q_Init(void);
void CH224Q_Process(void);
CH224Q_Status CH224Q_GetStatus(void);

#endif
