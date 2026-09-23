#include "stm32f205xx.h"
#include "ota_layout.h"

typedef void (*ApplicationEntry)(void);

#define FLASH_ERROR_MASK (FLASH_SR_SOP | FLASH_SR_WRPERR | FLASH_SR_PGAERR | \
                          FLASH_SR_PGPERR | FLASH_SR_PGSERR)
#define FLASH_CLEAR_MASK (FLASH_ERROR_MASK | FLASH_SR_EOP)
#define FLASH_WAIT_LIMIT  100000000UL
#define OTA_INSTALL_RETRY_UNUSED 0xFFFFFFFFUL
#define OTA_INSTALL_RETRY_ARMED  0x00000000UL

/* startup_stm32f205xx.s calls this before the C runtime.  The bootloader does
 * not enable peripherals or change clocks; the application performs its own
 * normal SystemInit after the jump. */
void SystemInit(void)
{
    SCB->VTOR = OTA_BOOT_ADDRESS;
}

static uint32_t BootCrc32(const uint8_t *data, uint32_t length)
{
    static const uint32_t nibbleTable[16] =
    {
        0x00000000UL, 0x1DB71064UL, 0x3B6E20C8UL, 0x26D930ACUL,
        0x76DC4190UL, 0x6B6B51F4UL, 0x4DB26158UL, 0x5005713CUL,
        0xEDB88320UL, 0xF00F9344UL, 0xD6D6A3E8UL, 0xCB61B38CUL,
        0x9B64C2B0UL, 0x86D3D2D4UL, 0xA00AE278UL, 0xBDBDF21CUL
    };
    uint32_t crc = 0xFFFFFFFFUL;
    uint32_t index;
    for(index = 0U; index < length; index++)
    {
        crc ^= data[index];
        crc = (crc >> 4) ^ nibbleTable[crc & 0x0FUL];
        crc = (crc >> 4) ^ nibbleTable[crc & 0x0FUL];
    }
    return crc ^ 0xFFFFFFFFUL;
}

static uint8_t VectorValid(uint32_t address)
{
    uint32_t initialSp = *(const uint32_t *)address;
    uint32_t resetVector = *(const uint32_t *)(address + 4U);
    uint32_t resetAddress = resetVector & ~1UL;
    return (initialSp >= 0x20000000UL && initialSp <= 0x20020000UL &&
            (initialSp & 3U) == 0U && (resetVector & 1U) != 0U &&
            resetAddress >= OTA_APPLICATION_ADDRESS &&
            resetAddress < OTA_APPLICATION_END) ? 1U : 0U;
}

static uint8_t MetadataValid(const OtaMetadata *metadata)
{
    OtaMetadata normalized;
    if(metadata->magic != OTA_METADATA_MAGIC_PENDING ||
       metadata->format_version != OTA_METADATA_FORMAT_VERSION ||
       metadata->target_id != OTA_TARGET_ID ||
       metadata->link_address != OTA_APPLICATION_ADDRESS ||
       metadata->staging_address != OTA_STAGING_ADDRESS ||
       metadata->image_size < 8U ||
       metadata->image_size > OTA_APPLICATION_MAX_SIZE) return 0U;
    normalized = *metadata;
    /* reserved[6] is programmed once by the bootloader before the first
     * install attempt.  The application commits it erased (all ones), so
     * normalize it for the authenticated metadata CRC on a retry boot. */
    normalized.reserved[6] = OTA_INSTALL_RETRY_UNUSED;
    return BootCrc32(((const uint8_t *)&normalized) + 4U,
                     sizeof(normalized) - 8U) == metadata->metadata_crc32 ? 1U : 0U;
}

static uint8_t FlashWait(void)
{
    uint32_t timeout = FLASH_WAIT_LIMIT;
    while((FLASH->SR & FLASH_SR_BSY) != 0U && timeout > 0U) timeout--;
    if(timeout == 0U || (FLASH->SR & FLASH_ERROR_MASK) != 0U) return 0U;
    return 1U;
}

static uint8_t FlashUnlock(void)
{
    if((FLASH->CR & FLASH_CR_LOCK) != 0U)
    {
        FLASH->KEYR = 0x45670123UL;
        FLASH->KEYR = 0xCDEF89ABUL;
    }
    return (FLASH->CR & FLASH_CR_LOCK) == 0U ? 1U : 0U;
}

static uint8_t FlashEraseSector(uint32_t sector)
{
    if(!FlashWait()) return 0U;
    FLASH->SR = FLASH_CLEAR_MASK;
    FLASH->CR = (FLASH->CR & ~(FLASH_CR_SNB | FLASH_CR_PSIZE)) |
                FLASH_CR_SER | FLASH_CR_PSIZE_1 |
                ((sector << FLASH_CR_SNB_Pos) & FLASH_CR_SNB);
    FLASH->CR |= FLASH_CR_STRT;
    if(!FlashWait()) return 0U;
    FLASH->CR &= ~(FLASH_CR_SER | FLASH_CR_SNB);
    return 1U;
}

static uint8_t FlashProgramWord(uint32_t address, uint32_t word)
{
    if(!FlashWait()) return 0U;
    FLASH->SR = FLASH_CLEAR_MASK;
    FLASH->CR = (FLASH->CR & ~FLASH_CR_PSIZE) | FLASH_CR_PG | FLASH_CR_PSIZE_1;
    *(volatile uint32_t *)address = word;
    if(!FlashWait()) return 0U;
    FLASH->CR &= ~FLASH_CR_PG;
    return (*(const uint32_t *)address == word) ? 1U : 0U;
}

static uint8_t InstallPendingImage(const OtaMetadata *metadata)
{
    uint32_t offset;
    uint32_t length = metadata->image_size;
    if(!VectorValid(OTA_STAGING_ADDRESS) ||
       BootCrc32((const uint8_t *)OTA_STAGING_ADDRESS, length) !=
           metadata->image_crc32) return 0U;
    if(!FlashUnlock()) return 0U;
    for(offset = 2U; offset <= 5U; offset++)
    {
        if(!FlashEraseSector(offset))
        {
            FLASH->CR |= FLASH_CR_LOCK;
            return 0U;
        }
    }
    for(offset = 0U; offset < length; offset += 4U)
    {
        uint32_t word = *(const uint32_t *)(OTA_STAGING_ADDRESS + offset);
        if(!FlashProgramWord(OTA_APPLICATION_ADDRESS + offset, word))
        {
            FLASH->CR |= FLASH_CR_LOCK;
            return 0U;
        }
    }
    if(!VectorValid(OTA_APPLICATION_ADDRESS) ||
       BootCrc32((const uint8_t *)OTA_APPLICATION_ADDRESS, length) !=
           metadata->image_crc32)
    {
        FLASH->CR |= FLASH_CR_LOCK;
        return 0U;
    }
    /* Clearing only the magic is an atomic 1->0 flash operation.  Until this
     * write succeeds the next reset repeats the verified staging copy. */
    if(!FlashProgramWord(OTA_METADATA_ADDRESS, 0x00000000UL))
    {
        FLASH->CR |= FLASH_CR_LOCK;
        return 0U;
    }
    FLASH->CR |= FLASH_CR_LOCK;
    return 1U;
}

static void JumpToApplication(void)
{
    uint32_t initialSp = *(const uint32_t *)OTA_APPLICATION_ADDRESS;
    uint32_t resetVector = *(const uint32_t *)(OTA_APPLICATION_ADDRESS + 4U);
    uint32_t index;
    ApplicationEntry entry = (ApplicationEntry)resetVector;
    __disable_irq();
    SysTick->CTRL = 0U;
    SysTick->LOAD = 0U;
    SysTick->VAL = 0U;
    for(index = 0U; index < 8U; index++)
    {
        NVIC->ICER[index] = 0xFFFFFFFFUL;
        NVIC->ICPR[index] = 0xFFFFFFFFUL;
    }
    SCB->VTOR = OTA_APPLICATION_ADDRESS;
    __DSB();
    __ISB();
    __set_MSP(initialSp);
    /* PRIMASK survives a branch to the application's reset handler.  The
     * application expects reset-default IRQ state for SysTick/HAL timeouts. */
    __enable_irq();
    entry();
}

int main(void)
{
    const OtaMetadata *metadata = (const OtaMetadata *)OTA_METADATA_ADDRESS;
    if(MetadataValid(metadata))
    {
        uint8_t firstAttempt =
            metadata->reserved[6] == OTA_INSTALL_RETRY_UNUSED ? 1U : 0U;

        /* A system reset preserves the already verified staging image.  Arm
         * exactly one automatic retry before touching the application.  The
         * marker is a flash-safe 1->0 write and MetadataValid deliberately
         * authenticates its erased value, so both generations validate the
         * same application-created record. */
        if(firstAttempt)
        {
            if(!FlashUnlock() ||
               !FlashProgramWord((uint32_t)(uintptr_t)&metadata->reserved[6],
                                 OTA_INSTALL_RETRY_ARMED))
                while(1) __WFI();
            FLASH->CR |= FLASH_CR_LOCK;
        }

        /* Once a valid PENDING record exists, the application sectors may
         * already have been erased or only partly programmed by this or an
         * interrupted earlier install.  A plausible vector is not proof that
         * the whole application is executable.  Never branch to it unless the
         * complete staged image was copied, CRC-verified, and the pending
         * marker was atomically cleared.  Leaving PENDING intact makes a later
         * reset retry the same already-verified staging image. */
        if(!InstallPendingImage(metadata))
        {
            if(firstAttempt)
            {
                NVIC_SystemReset();
                while(1) __WFI();
            }
            while(1) __WFI();
        }

        /* Flash erase/program state and the pre-update application's USB/Wi-Fi
         * peripheral state must not leak into the newly installed image.
         * Start a second, clean reset generation after atomically clearing
         * PENDING; the next boot takes the ordinary no-update jump path. */
        NVIC_SystemReset();
        while(1) __WFI();
    }
    if(VectorValid(OTA_APPLICATION_ADDRESS)) JumpToApplication();
    /* Recovery-safe state: no valid application exists.  SWD remains
     * available and a reset retries any still-pending verified image. */
    while(1) __WFI();
}
