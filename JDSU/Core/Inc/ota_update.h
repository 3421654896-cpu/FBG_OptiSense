#ifndef __OTA_UPDATE_H
#define __OTA_UPDATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void OtaUpdate_Init(uint32_t runtime_boot_id);

/* Physical CDC-only mailbox. Never route LAN FUSB1 commands into this API.
 * QueueUsbFrame only copies in the USB ISR; all flash work runs in main. */
uint8_t OtaUpdate_QueueUsbFrame(const uint8_t *frame, uint16_t length);
void OtaUpdate_ProcessUsb(void);

uint8_t OtaUpdate_IsActive(void);
void OtaUpdate_AbortUncommittedSession(void);
#ifdef __cplusplus
}
#endif

#endif
