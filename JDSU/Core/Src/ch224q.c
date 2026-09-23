#include "ch224q.h"

#include <string.h>

#define CH224Q_BOARD_REQUEST_MV           9000U

static CH224Q_Status status;

void CH224Q_Init(void)
{
    /* This PCB does not wire the CH224Q in its optional I2C mode.  R4/R5/R6
     * pull CFG1/CFG2/CFG3 low, so the immutable hardware request is 000 = 9 V.
     * In particular PB10/PB11 are CFG2/CFG3 straps, not SCL/SDA.  They must
     * stay in their reset (high-impedance) state: driving recovery clocks or
     * selecting an alternate function could change the negotiated voltage.
     *
     * No current/capability readback exists on this board revision.  Report
     * the requested voltage, but deliberately leave online/current_valid
     * clear rather than manufacturing telemetry. */
    memset(&status, 0, sizeof(status));
    status.requested_voltage_mV = CH224Q_BOARD_REQUEST_MV;
    status.age_ms = 0xFFFFU;
}

void CH224Q_Process(void)
{
    /* Intentionally constant-time and side-effect free on this hardware. */
}

CH224Q_Status CH224Q_GetStatus(void)
{
    return status;
}
